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

from PIL import Image
import random
from google.cloud import storage
from torchvision.transforms import RandomHorizontalFlip

class FolderData(Dataset):
    def __init__(self,
        root_dir,
        token_dir,
        caption_file=None,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        postprocess=None,
        return_paths=False,
        negative_prompt="",
        restart_from=0,
        section0=0,
        section1=0,
        if_=None,
        ip=None,
        resolution=768,
        resolution2=1536,
        drop=False,
        resize=False,
        center=False,
        color=False,
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.root_dir = root_dir
        self.default_caption = default_caption
        self.return_paths = return_paths

        import glob
        self.color = color
        print("stuff--------------------->",root_dir+if_+'/*')
#         self.captions = glob.glob(root_dir+if_+'/*')
        with open(if_, "r") as f:
          l = f.readlines()
          self.captions = [json.loads(x) for x in l]

        import random
        image_transforms.extend([transforms.ToTensor(),
                                 transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = transforms.Compose(image_transforms)
        self.resolution = resolution
        print("resolution ", resolution)
#         resolution = 768
        self.tform0 = transforms.Compose(
            [
        transforms.CenterCrop(resolution),
#         transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ]
        )

        if resize:
            self.tform1 = transforms.Compose(
                [
            transforms.Resize( resolution2, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(resolution),
#             transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
                ]
            )
        elif center:
            self.tform1 = transforms.Compose(
                [ transforms.CenterCrop(resolution),
#             transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
                ]
            )

        else:
            self.tform1 = transforms.Compose(
                [
    #         transforms.Resize( resolution2, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(resolution),
#             transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
                ]
            )

        self.tokenizer = CLIPTokenizer.from_pretrained(token_dir, subfolder="tokenizer")
        self.negative_prompt = negative_prompt
        self.instance_prompt = ip
        self.drop = drop
#         self.processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
        self.processor_hed = HEDdetector.from_pretrained('lllyasviel/Annotators')
        self.processor_pidi = PidiNetDetector.from_pretrained('lllyasviel/Annotators')
        self.processor_linear = LineartDetector.from_pretrained('lllyasviel/Annotators')

        self.tformlarge = transforms.Compose(
                [
            transforms.RandomCrop(resolution),
                ]
        )

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        data = {}
        filename = self.captions[index]['file_name']
        im = Image.open(filename)
        mins = min(im.size[0] , im.size[1]) 
        
        if mins <= 512:
            im2 = self.process_im(im)
        else:
            width, height = im.size
            left = randrange(0,width - self.resolution)
            top = randrange(0,height - self.resolution)
            right = left + self.resolution
            bottom = top + self.resolution
            im = im.crop((left, top, right, bottom))

            im2 = self.process_im(im)
            
        control_image = self.process_im_cond(im)

        # Generate a random boolean value
        flip = random.random() > 0.5

        # Apply random horizontal flip with 0.5 probability
        if flip:
            flip_transform = RandomHorizontalFlip(p=1.0)
            im2 = flip_transform(im2)
            control_image = flip_transform(control_image)

        data["pixel_values"] = im2
        data['conditioning_pixel_values'] = self.conditioning_image_transforms(control_image)

        caption = self.instance_prompt + self.captions[index]['text']
        list_ = [i for i in range(100)] 
        choice = random.choice(list_)

        ids = self.tokenize_captions(caption)
        input_ids = self.tokenizer.pad(
            {"input_ids": ids}, padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt"
        ).input_ids
        data['input_ids'] = input_ids

        return data
    
    def tokenize_captions(self,captions, is_train=True):
        inputs = self.tokenizer(captions, max_length=self.tokenizer.model_max_length, padding="do_not_pad", truncation=True)
        input_ids = inputs.input_ids
        return input_ids
    
    def process_im(self, im):
        i = random.choice([0,1])
        if False:
            im = im.convert("RGB")
            return self.tform1(im)     
        else:
            im = im.convert("RGB")
            return self.tform1(im)     
    def process_im_cond(self, image):
        mins = min(image.size[0] , image.size[1]) 
        
        if mins < 512:
            image = image.resize((512,512), resample=PIL.Image.BICUBIC)

            
        listk = [i for i in range(100)]

        r = random.choice(listk)
        if r < 50:
          control_image = self.processor_hed(image,scribble=True)

        else:
          control_image = self.processor_pidi(image,scribble=True)

        if self.color:

            img = control_image.convert("RGBA")

            pixdata = img.load()

            width, height = img.size
            for y in range(height):
                for x in range(width):
                    if pixdata[x, y][0] < 50:# == (0, 0, 0, 255):
                        pixdata[x, y] = (255, 255, 255, 0)

                    else:
                        pixdata[x, y] = (255, 255, 255, 255 ) 

            img2 = image
            scales = [16 , 32 , 64]
            scale = random.choice(scales)
            img2 = img2.resize((512,512))
            imgSmall = img2.resize((scale,scale), resample=PIL.Image.BICUBIC)

            result = imgSmall.resize(img2.size, Image.NEAREST)

            result2 = result.convert("RGBA")

            background = result2
            background.paste(img,(0,0),img)

            im = background.convert("RGB")
            return self.tformlarge(im)  
#         print( type(im) ) 
        return self.tformlarge(control_image)