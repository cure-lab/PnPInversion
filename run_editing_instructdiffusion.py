# --------------------------------------------------------
# InstructDiffusion
# Based on instruct-pix2pix (https://github.com/timothybrooks/instruct-pix2pix)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn)
# --------------------------------------------------------
from __future__ import annotations

import os
import math
import random
import sys
from argparse import ArgumentParser

import argparse
import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
import json
from utils.utils import txt_draw

import requests

sys.path.append("models/InstructDiffusion/stable_diffusion")
from ldm.util import instantiate_from_config

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "b ... -> (repeat b) ...", repeat=3)
        cfg_sigma = einops.repeat(sigma, "b ... -> (repeat b) ...", repeat=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], cond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_txt_cond \
            = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return 0.5 * (out_img_cond + out_txt_cond) + \
            text_cfg_scale * (out_cond - out_img_cond) + \
                image_cfg_scale * (out_cond - out_txt_cond)

def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    model = instantiate_from_config(config.model)

    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if 'state_dict' in pl_sd:
        pl_sd = pl_sd['state_dict']
    m, u = model.load_state_dict(pl_sd, strict=False)

    print(m, u)
    return model


def mask_decode(encoded_mask,image_shape=[512,512]):
    length=image_shape[0]*image_shape[1]
    mask_array=np.zeros((length,))
    
    for i in range(0,len(encoded_mask),2):
        splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1
            
    mask_array=mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1
            
    return mask_array


def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def instruct_diffusion_edit(edit_method, image_path,model, edit_prompt, resolution=512,steps=50,cfg_text=5.0,cfg_image=1.25):
    if edit_method=="instruct-diffusion":
        input_image_ = Image.open(image_path).convert("RGB")
        width, height = input_image_.size
        factor = resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width_resize = int((width * factor) // 64) * 64
        height_resize = int((height * factor) // 64) * 64
        input_image = ImageOps.fit(input_image_, (width_resize, height_resize), method=Image.Resampling.LANCZOS)
        with torch.no_grad(), autocast("cuda"):
            cond = {}
            cond["c_crossattn"] = [model.get_learned_conditioning([edit_prompt])]
            input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
            input_image = rearrange(input_image, "h w c -> 1 c h w").to(next(model.parameters()).device)
            cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

            uncond = {}
            uncond["c_crossattn"] = [null_token]
            uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

            sigmas = model_wrap.get_sigmas(steps)

            extra_args = {
                "cond": cond,
                "uncond": uncond,
                "text_cfg_scale": cfg_text,
                "image_cfg_scale": cfg_image,
            }

            z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
            z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
            x = model.decode_first_stage(z)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
            x = 255.0 * rearrange(x, "1 c h w -> h w c")
            print(x.shape)
            edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())

            edited_image = ImageOps.fit(edited_image, (width, height), method=Image.Resampling.LANCZOS)
            image_instruct = txt_draw(f"edit prompt: {edit_prompt}")
            
            return Image.fromarray(np.concatenate((image_instruct, np.array(input_image_), np.zeros_like(image_instruct), np.array(edited_image)),axis=1))
    else:
        raise NotImplementedError(f"No edit method named {edit_method}")


image_save_paths={
    "instruct-diffusion":"instruct-diffusion",
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
    parser.add_argument('--data_path', type=str, default="data") # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="output") # the editing category that needed to run
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs = '+', type=str, default=["instruct-diffusion"]) # the editing methods that needed to run
    parser.add_argument('--checkpoint', type=str, require=True, default="data/checkpoints/v1-5-pruned-emaonly-adaption.ckpt") # the editing methods that needed to run
    args = parser.parse_args()
    
    rerun_exist_images=args.rerun_exist_images
    data_path=args.data_path
    output_path=args.output_path
    edit_category_list=args.edit_category_list
    edit_method_list=args.edit_method_list
    checkpoint=args.checkpoint
    
    with open(f"{data_path}/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)
        
    config = OmegaConf.load("models/InstructDiffusion/configs/instruct_diffusion.yaml")
    model = load_model_from_config(config, checkpoint, None)
    model.eval().cuda()

    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    for key, item in editing_instruction.items():
        
        if item["editing_type_id"] not in edit_category_list:
            continue
        
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])
        editing_instruction = item["editing_instruction"]
        blended_word = item["blended_word"].split(" ") if item["blended_word"] != "" else []
        mask = Image.fromarray(np.uint8(mask_decode(item["mask"])[:,:,np.newaxis].repeat(3,2))).convert("L")

        for edit_method in edit_method_list:
            present_image_save_path=image_path.replace(data_path, os.path.join(output_path,image_save_paths[edit_method]))
            if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
                print(f"editing image [{image_path}] with [{edit_method}]")
                setup_seed()
                torch.cuda.empty_cache()
                edited_image = instruct_diffusion_edit(edit_method,
                                         image_path=image_path,
                                         model=model,
                                         edit_prompt=editing_instruction
                                        )
                if not os.path.exists(os.path.dirname(present_image_save_path)):
                    os.makedirs(os.path.dirname(present_image_save_path))
                edited_image.save(present_image_save_path)
                
                print(f"finish")
                
            else:
                print(f"skip image [{image_path}] with [{edit_method}]")
        
        