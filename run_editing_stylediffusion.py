from typing import Optional, Union, Tuple, List, Callable, Dict
import os
import torch

import numpy as np
import random
import argparse
import json
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDIMScheduler
from models.stylediffusion import global_var
from utils.utils import txt_draw
from PIL import Image

global_var._init()
global_var.set_value("USE_INITIAL_INV",False)
LOW_RESOURCE=True
global_var.set_value("LOW_RESOURCE",LOW_RESOURCE)
global_var.set_value("MAX_NUM_WORDS",77)
NUM_DDIM_STEPS = 50
global_var.set_value("NUM_DDIM_STEPS",NUM_DDIM_STEPS)
global_var.set_value("BLOCK_NUM",1)
global_var.set_value("IS_TRAIN",True)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')
global_var.set_value("device",device)


# make the DDIM inversion pipeline
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
stable = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', scheduler=scheduler, local_files_only=True).to(device)
tokenizer = stable.tokenizer
global_var.set_value("tokenizer",tokenizer)



from models.stylediffusion.inversion import VaeInversion
from models.stylediffusion.utils import  AttentionStore, EmptyControl, register_attention_control,make_controller
from models.stylediffusion import ptp_utils_v


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


# Infernce Code
@torch.no_grad()
def text2image_ldm_stable(
        model,
        prompt: List[str],
        trainer,
        controller,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        start_time=50,
        return_type='image'
):
    batch_size = len(prompt)
    register_attention_control(model, trainer, controller)
    height = width = 512

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    latent, latents = ptp_utils_v.init_latent(latent, model, height, width, generator, batch_size)
    # image_latents = [vae_inversion.latent2image(latents[0].unsqueeze(dim=0))[0]]
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        trainer.I = i
        trainer.i = i \
            if i < NUM_DDIM_STEPS * trainer.v_replace_steps else None
        context = (uncond_embeddings, text_embeddings)
        latents = ptp_utils_v.diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=LOW_RESOURCE,)
        # image_latents += [vae_inversion.latent2image(latents[0].unsqueeze(dim=0))[0]]

    # os.makedirs('latent_save', exist_ok=True)
    # for i, latent_i in enumerate(image_latents):
    #     Image.fromarray(latent_i).save(f'latent_save/Z{NUM_DDIM_STEPS - i}_bar.png')

    if return_type == 'image':
        image = ptp_utils_v.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent



def run_and_display(stable, prompts, trainer, controller, latent=None, run_baseline=False, generator=None, verbose=True):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(stable, prompts, trainer, EmptyControl(), latent=latent, run_baseline=False,
                                         generator=generator)
        print("with prompt-to-prompt")
    images, x_t = text2image_ldm_stable(stable, prompts, trainer, controller, latent=latent,
                                        num_inference_steps=NUM_DDIM_STEPS, guidance_scale=global_var.get_value("GUIDANCE_SCALE"),
                                        generator=generator)
    if verbose:
        ptp_utils_v.view_images(images)
    return images, x_t



def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def edit_image_stylediffusion_p2p(
    image_path,
    prompt_src,
    prompt_tar,
    guidance_scale=7.5,
    cross_replace_steps=0.4,
    self_replace_steps=0.6,
    blend_word=None,
    eq_params=None,
    is_replace_controller=False,
    num_inner_steps=100,
    num_epoch=1,
    tau_v=.6,
    tau_c=.6,
    tau_s=.8,
    tau_u=.5):
    global_var.set_value("GUIDANCE_SCALE",guidance_scale)
    global_var.set_value("IS_TRAIN",True)
    
    vae_inversion = VaeInversion(stable)
    (image_gt, image_rec), x_t, trainer = vae_inversion.invert(image_path, [prompt_src], verbose=True,
                                                                   num_inner_steps=num_inner_steps,
                                                                   num_epoch=num_epoch)
    
    global_var.set_value("IS_TRAIN",False)
    
    trainer.attention_store = {}
    trainer.cur_step = 0
    # (image_gt, image_rec), x_t = vae_inversion.eval_init(image_path, [prompt_src], trainer=trainer)
    
    trainer.v_replace_steps = 1.0
    controller = AttentionStore()
    image_inv_recon, x_t_recon = run_and_display(stable, [prompt_src,prompt_tar], trainer, controller, run_baseline=False, latent=x_t,
                                             verbose=False)
    
    # edit
    trainer.v_replace_steps = tau_v
    cross_replace_steps = {'default_': tau_c,}
    self_replace_steps = tau_s
    uncond_self_replace_steps = tau_u
    controller = make_controller([prompt_src,prompt_tar], 
                                 len(prompt_src.strip(" "))==len(prompt_tar.strip(" ")), 
                                 cross_replace_steps, 
                                 self_replace_steps, 
                                 uncond_self_replace_steps, 
                                 blend_word, 
                                 eq_params)
    
    image_inv_edit, x_t_edit = run_and_display(stable, 
                                     [prompt_src,prompt_tar], 
                                     trainer, 
                                     controller, 
                                     run_baseline=False, 
                                     latent=x_t,
                                     verbose=False)
    
    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
    
    out_image=Image.fromarray(np.concatenate((image_instruct,image_gt[0],image_inv_recon[0],image_inv_edit[1],),axis=1))
    
    return out_image

image_save_paths={
    "stylediffusion+p2p":"styleidffusion+p2p",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
    parser.add_argument('--data_path', type=str, default="data") # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="output") # the editing category that needed to run
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs = '+', type=str, default=["styleidffusion+p2p"]) # the editing methods that needed to run
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
                edited_image = edit_image_stylediffusion_p2p(
                        image_path=[image_path],
                        prompt_src=original_prompt,
                        prompt_tar=editing_prompt,
                        guidance_scale=7.5,
                        cross_replace_steps=0.4,
                        self_replace_steps=0.6,
                        blend_word=None,
                        eq_params=None,
                        is_replace_controller=False,
                        num_inner_steps=100,
                        num_epoch=1,
                        tau_v=.5, 
                        tau_c=.6, 
                        tau_s=.6, 
                        tau_u=.0) # disable p2pro
                    
                if not os.path.exists(os.path.dirname(present_image_save_path)):
                    os.makedirs(os.path.dirname(present_image_save_path))
                edited_image.save(present_image_save_path)
                
                print(f"finish")
                
            else:
                print(f"skip image [{image_path}] with [{edit_method}]")
        
        