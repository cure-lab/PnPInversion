import torch
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
import numpy as np
from PIL import Image
import os
from models.p2p.scheduler_dev import DDIMSchedulerDev
import json
import random
import argparse
from torch import autocast, inference_mode

from utils.utils import load_512,txt_draw
from models.edit_friendly_ddm.inversion_utils import inversion_forward_process, inversion_reverse_process
from models.edit_friendly_ddm.ptp_classes import AttentionReplace,AttentionRefine,AttentionStore
from models.edit_friendly_ddm.ptp_utils import register_attention_control


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


image_save_paths={
    "edit-friendly-inversion+p2p":"edit-friendly-inversion+p2p",
    }


device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')
NUM_DDIM_STEPS = 50
model_id="CompVis/stable-diffusion-v1-4"
ldm_stable = StableDiffusionPipeline.from_pretrained(
    model_id).to(device)
ldm_stable.scheduler = DDIMScheduler.from_config(model_id, subfolder = "scheduler")
ldm_stable.scheduler.set_timesteps(NUM_DDIM_STEPS)
ETA=1
SKIP=12


def edit_image_EF(edit_method,
                  image_path,
                    prompt_src,
                    prompt_tar,
                    source_guidance_scale=1,
                    target_guidance_scale=7.5,cross_replace_steps=0.4,
                    self_replace_steps=0.6
                    ):
    if edit_method=="edit-friendly-inversion+p2p":
        image_gt = load_512(image_path)
        
        image_gt = torch.from_numpy(image_gt).float() / 127.5 - 1
        image_gt = image_gt.permute(2, 0, 1).unsqueeze(0).to(device)
        with autocast("cuda"), inference_mode():
            w0 = (ldm_stable.vae.encode(image_gt).latent_dist.mode() * 0.18215).float()
            
        controller = AttentionStore()
        register_attention_control(ldm_stable, controller)
            
        wt, zs, wts = inversion_forward_process(ldm_stable, w0, etas=ETA, prompt=prompt_src, cfg_scale=source_guidance_scale, prog_bar=True, num_inference_steps=NUM_DDIM_STEPS)
        
        controller = AttentionStore()
        register_attention_control(ldm_stable, controller)
        
        x0_reconstruct, _ = inversion_reverse_process(ldm_stable, xT=wts[NUM_DDIM_STEPS-SKIP], etas=ETA, prompts=[prompt_tar], cfg_scales=[target_guidance_scale], prog_bar=True, zs=zs[:(NUM_DDIM_STEPS-SKIP)], controller=controller)

        cfg_scale_list = [source_guidance_scale, target_guidance_scale]
        prompts = [prompt_src, prompt_tar]
        if (len(prompt_src.split(" ")) == len(prompt_tar.split(" "))):
            controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, model=ldm_stable)
        else:
            # Should use Refine for target prompts with different number of tokens
            controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, model=ldm_stable)

        register_attention_control(ldm_stable, controller)
        w0, _ = inversion_reverse_process(ldm_stable, xT=wts[NUM_DDIM_STEPS-SKIP], etas=ETA, prompts=prompts, cfg_scales=cfg_scale_list, prog_bar=True, zs=zs[:(NUM_DDIM_STEPS-SKIP)], controller=controller)
        with autocast("cuda"), inference_mode():
            x0_dec = ldm_stable.vae.decode(1 / 0.18215 * w0[1].unsqueeze(0)).sample
            x0_reconstruct_edit = ldm_stable.vae.decode(1 / 0.18215 * w0[0].unsqueeze(0)).sample
            x0_reconstruct = ldm_stable.vae.decode(1 / 0.18215 * x0_reconstruct[0].unsqueeze(0)).sample
            
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
            
        return Image.fromarray(np.concatenate(
                                            (
                                                image_instruct,
                                                np.uint8((np.array(image_gt[0].permute(1,2,0).cpu().detach())/2+ 0.5)*255),
                                                np.uint8((np.array(x0_reconstruct_edit[0].permute(1,2,0).cpu().detach())/2+ 0.5)*255),
                                                np.uint8((np.array(x0_dec[0].permute(1,2,0).cpu().detach())/2+ 0.5)*255)
                                            ),
                                            1
                                            )
                            )
    else:
        raise NotImplementedError(f"No edit method named {edit_method}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
    parser.add_argument('--data_path', type=str, default="data") # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="output") # the editing category that needed to run
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs = '+', type=str, default=["edit-friendly-inversion+p2p"]) # the editing methods that needed to run
    args = parser.parse_args()
    
    rerun_exist_images=args.rerun_exist_images
    data_path=args.data_path
    output_path=args.output_path
    edit_category_list=args.edit_category_list
    edit_method_list=args.edit_method_list
    
    with open(f"{data_path}/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)
    
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
                edited_image = edit_image_EF(
                        edit_method=edit_method,
                        image_path=image_path,
                        prompt_src=original_prompt,
                        prompt_tar=editing_prompt,
                        source_guidance_scale=1,
                        target_guidance_scale=7.5,
                        cross_replace_steps=0.4,
                        self_replace_steps=0.6
                        )
                        
                if not os.path.exists(os.path.dirname(present_image_save_path)):
                    os.makedirs(os.path.dirname(present_image_save_path))
                edited_image.save(present_image_save_path)
                
                print(f"finish")
                
            else:
                print(f"skip image [{image_path}] with [{edit_method}]")
        
        