#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import json
import argparse
import logging
import math
import os
import random
import time
from pathlib import Path
from torch.utils.data import Dataset
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torch.utils.checkpoint
import transformers
from datasets import load_dataset, load_from_disk
from flax import jax_utils
from flax.core.frozen_dict import unfreeze
from flax.training import train_state
from flax.training.common_utils import shard
from huggingface_hub import create_repo, upload_folder
from PIL import Image, PngImagePlugin
from torch.utils.data import IterableDataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, FlaxCLIPTextModel, set_seed
from jax.experimental import mesh_utils
from controlnet_aux import LineartDetector, PidiNetDetector, HEDdetector
from jax.sharding import PartitionSpec as P 

import PIL

from diffusers import (
    FlaxAutoencoderKL,
    FlaxControlNetModel,
    FlaxDDPMScheduler,
    FlaxDDIMScheduler,
    FlaxStableDiffusionControlNetPipeline,
    FlaxUNet2DConditionModel,
)
from diffusers.utils import check_min_version, is_wandb_available

from jax.experimental.maps import xmap
from jax.experimental.pjit import pjit
from jax.sharding import Mesh
import cv2
# To prevent an error that occurs when there are abnormally large compressed data chunk in the png image
# see more https://github.com/python-pillow/Pillow/issues/5610
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

# if is_wandb_available():
#     import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.16.0.dev0")

logger = logging.getLogger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def log_validation(pipeline, pipeline_params, controlnet_params, tokenizer, args, rng, weight_dtype):
    logger.info("Running validation...")

    pipeline_params = pipeline_params.copy()
    pipeline_params["controlnet"] = controlnet_params

    num_samples = jax.device_count()
    prng_seed = jax.random.split(rng, jax.device_count())

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []

    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        prompts = num_samples * [validation_prompt]
        prompt_ids = pipeline.prepare_text_inputs(prompts)
        prompt_ids = shard(prompt_ids)

        validation_image = Image.open(validation_image).convert("RGB")
        processed_image = pipeline.prepare_image_inputs(num_samples * [validation_image])
        processed_image = shard(processed_image)
        images = pipeline(
            prompt_ids=prompt_ids,
            image=processed_image,
            params=pipeline_params,
            prng_seed=prng_seed,
            num_inference_steps=50,
            jit=True,
        ).images

        images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
        images = pipeline.numpy_to_pil(images)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    if False:
        formatted_images = []
        for log in image_logs:
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]

            formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))
            for image in images:
                image = wandb.Image(image, caption=validation_prompt)
                formatted_images.append(image)

        wandb.log({"validation": formatted_images})
    else:
        logger.warn(f"image logging not implemented for {args.report_to}")

    return image_logs


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- controlnet
- jax-diffusers-event
inference: true
---
    """
    model_card = f"""
# controlnet- {repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--from_pt",
        action="store_true",
        help="Load the pretrained model from a PyTorch checkpoint.",
    )
    parser.add_argument(
        "--controlnet_revision",
        type=str,
        default=None,
        help="Revision of controlnet model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--profile_steps",
        type=int,
        default=0,
        help="How many training steps to profile in the beginning.",
    )
    parser.add_argument(
        "--profile_validation",
        action="store_true",
        help="Whether to profile the (last) validation.",
    )
    parser.add_argument(
        "--profile_memory",
        action="store_true",
        help="Whether to dump an initial (before training loop) and a final (at program end) memory profile.",
    )
    parser.add_argument(
        "--ccache",
        type=str,
        default=None,
        help="Enables compilation cache.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="a beautiful art work by mmdd111",
        help="instance prompt.",
    )

    parser.add_argument(
        "--controlnet_from_pt",
        action="store_true",
        help="Load the controlnet model from a PyTorch checkpoint.",
    )
    parser.add_argument(
        "--cont",
        action="store_true",
        help="Load the controlnet model from a PyTorch checkpoint.",
    )

    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
#     parser.add_argument(
#         "--train_data_dir",
#         type=str,
#         default=None,
#         help=(
#             "A folder containing the training data. Folder contents must follow the structure described in"
#             " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
#             " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
#         ),
#     )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/{timestamp}",
        help="The output directory where the model predictions and checkpoints will be written. "
        "Can contain placeholders: {timestamp}.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=("Save a checkpoint of the training state every X updates."),
    )
    parser.add_argument("--save_frequency", type=int, default=5120, help="How frequently to save")

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt for unconditional generation")
    parser.add_argument("--section0", type=int, default=0, help="section 0")
    parser.add_argument("--section1", type=int, default=0, help="section 1")
    parser.add_argument(
        "--img_folder",
        type=str,
        default="images",
        help="instance prompt.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )

    parser.add_argument(
        "--resolution2",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
#     parser.add_argument(
#         "--resolution",
#         type=int,
#         default=512,
#         help=(
#             "The resolution for input images, all the images in the train/validation dataset will be resized to this"
#             " resolution"
#         ),
#     )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution (if not set, random crop will be used)",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Whether to prompt drop",
    )
    parser.add_argument(
        "--resize",
        action="store_true",
        help="Whether to prompt drop",
    )



    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--bucketname", type=str, default='buck', help="Name of bucket.")
    parser.add_argument("--bucketdir", type=str, default='buck', help="Bucket directory.")

    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help=("log training metric every X steps to `--report_t`"),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=('The integration to report the results and logs to. Currently only supported platforms are `"wandb"`'),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--token_dir",
        type=str,
        default=None,
        help=(
            "tokenizer dir"
        ),
    )

    parser.add_argument("--streaming", action="store_true", help="To stream a large dataset from Hub.")
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training dataset. By default it will use `load_dataset` method to load a custom dataset from the folder."
            "Folder must contain a dataset script as described here https://huggingface.co/docs/datasets/dataset_script) ."
            "If `--load_from_disk` flag is passed, it will use `load_from_disk` method instead. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--load_from_disk",
        action="store_true",
        help=(
            "If True, will load a dataset that was previously saved using `save_to_disk` from `--train_data_dir`"
            "See more https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.load_from_disk"
        ),
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help=(
            "If True, will add color map to control image"
        ),
    )

    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set. Needed if `streaming` is set to True."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` and logging the images."
        ),
    )
    parser.add_argument("--wandb_entity", type=str, default=None, help=("The wandb entity to use (for teams)."))
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet_flax",
        help=("The `project` argument passed to wandb"),
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients over"
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    args.output_dir = args.output_dir.replace("{timestamp}", time.strftime("%Y%m%d_%H%M%S"))

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")
    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    def process_im_cond(self, image):
        # ... existing code ...
        # Apply random horizontal flip
        if self.flip:
            control_image = transforms.functional.hflip(control_image)
        # ... existing code ...
        return self.tformlarge(control_image)