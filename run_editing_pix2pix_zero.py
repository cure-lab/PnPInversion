import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import argparse
import json
from PIL import Image

from lavis.models import load_model_and_preprocess
from models.pix2pix_zero.ddim_inv import DDIMInversion
from models.pix2pix_zero.scheduler import DDIMInverseScheduler
from models.pix2pix_zero.edit_directions import construct_direction
from models.pix2pix_zero.edit_pipeline import EditingPipeline
from utils.utils import txt_draw

from diffusers import DDIMScheduler

NUM_DDIM_STEPS = 50
XA_GUIDANCE=0.1

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

# load the BLIP model
model_blip, vis_processors, _ = load_model_and_preprocess(name="blip_caption", 
                                                          model_type="base_coco", 
                                                          is_eval=True, 
                                                          device=torch.device(device))

# make the DDIM inversion pipeline
pipe = DDIMInversion.from_pretrained('CompVis/stable-diffusion-v1-4').to(device)
pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.num_inference_steps=NUM_DDIM_STEPS

edit_pipe = EditingPipeline.from_pretrained('CompVis/stable-diffusion-v1-4').to(device)
edit_pipe.scheduler = DDIMScheduler.from_config(edit_pipe.scheduler.config)
edit_pipe.scheduler.num_inference_steps=NUM_DDIM_STEPS




def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


## convert sentences to sentence embeddings
def load_sentence_embeddings(l_sentences, tokenizer, text_encoder, device=device):
    with torch.no_grad():
        l_embeddings = []
        for sent in l_sentences:
            text_inputs = tokenizer(
                    sent,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=None)[0]
            l_embeddings.append(prompt_embeds)
    return torch.concat(l_embeddings, dim=0).mean(dim=0).unsqueeze(0)


def edit_image_ddim_pix2pix_zero(image_path,
                prompt_src,
                prompt_tar,
                guidance_scale=7.5,
                image_size=[512,512]):
    image_gt = Image.open(image_path).resize(image_size, Image.Resampling.LANCZOS)
    # generate the caption
    prompt_str = model_blip.generate({"image": vis_processors["eval"](image_gt).unsqueeze(0).to(device)})[0]
    latent_list, x_inv_image, x_dec_img = pipe(
            prompt_str, 
            guidance_scale=1,
            num_inversion_steps=NUM_DDIM_STEPS,
            img=image_gt
        )
    
    inversion_latent=latent_list[-1].detach()
    
    mean_emb_src = load_sentence_embeddings([prompt_src], edit_pipe.tokenizer, edit_pipe.text_encoder, device=device)
    mean_emb_tar = load_sentence_embeddings([prompt_tar], edit_pipe.tokenizer, edit_pipe.text_encoder, device=device)
    
    rec_pil, edit_pil = edit_pipe(prompt_str,
                num_inference_steps=NUM_DDIM_STEPS,
                x_in=inversion_latent,
                edit_dir=(mean_emb_tar.mean(0)-mean_emb_src.mean(0)).unsqueeze(0),
                guidance_amount=XA_GUIDANCE,
                guidance_scale=guidance_scale,
                negative_prompt=prompt_str # use the unedited prompt for the negative prompt
        )
    
    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
    
    out_image=np.concatenate((np.array(image_instruct),np.array(image_gt),np.array(rec_pil[0]),np.array(edit_pil[0])),1)
    
    return Image.fromarray(out_image)
    

def edit_image_directinversion_pix2pix_zero(image_path,
                prompt_src,
                prompt_tar,
                guidance_scale=7.5,
                image_size=[512,512]):
    image_gt = Image.open(image_path).resize(image_size, Image.Resampling.LANCZOS)
    # generate the caption
    prompt_str = model_blip.generate({"image": vis_processors["eval"](image_gt).unsqueeze(0).to(device)})[0]
    latent_list, x_inv_image, x_dec_img = pipe(
            prompt_str, 
            guidance_scale=1,
            num_inversion_steps=NUM_DDIM_STEPS,
            img=image_gt
        )
    
    inversion_latent=latent_list[-1].detach()
    
    mean_emb_src = load_sentence_embeddings([prompt_src], edit_pipe.tokenizer, edit_pipe.text_encoder, device=device)
    mean_emb_tar = load_sentence_embeddings([prompt_tar], edit_pipe.tokenizer, edit_pipe.text_encoder, device=device)
    
    rec_pil, edit_pil = edit_pipe(prompt_str,
                num_inference_steps=NUM_DDIM_STEPS,
                x_in=inversion_latent,
                edit_dir=(mean_emb_tar.mean(0)-mean_emb_src.mean(0)).unsqueeze(0),
                guidance_amount=XA_GUIDANCE,
                guidance_scale=guidance_scale,
                negative_prompt=prompt_str, # use the unedited prompt for the negative prompt
                latent_list=latent_list
        )
    
    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
    
    out_image=np.concatenate((np.array(image_instruct),np.array(image_gt),np.array(rec_pil[0]),np.array(edit_pil[0])),1)
    
    return Image.fromarray(out_image)


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

    
image_save_paths={
    "ddim+pix2pix-zero":"ddim+pix2pix-zero",
    "directinversion+pix2pix-zero":"directinversion+pix2pix-zero",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
    parser.add_argument('--data_path', type=str, default="data") # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="output") # the editing category that needed to run
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs = '+', type=str, default=["ddim+pix2pix-zero","directinversion+pix2pix-zero"]) # the editing methods that needed to run
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
                if edit_method=="ddim+pix2pix-zero":
                    edited_image = edit_image_ddim_pix2pix_zero(
                        image_path=image_path,
                        prompt_src=original_prompt,
                        prompt_tar=editing_prompt,
                        guidance_scale=7.5,
                    )
                elif edit_method=="directinversion+pix2pix-zero":
                    edited_image = edit_image_directinversion_pix2pix_zero(
                        image_path=image_path,
                        prompt_src=original_prompt,
                        prompt_tar=editing_prompt,
                        guidance_scale=7.5,
                    )
                else:
                    raise NotImplementedError(f"No edit method named {edit_method}")
                
                
                if not os.path.exists(os.path.dirname(present_image_save_path)):
                    os.makedirs(os.path.dirname(present_image_save_path))
                edited_image.save(present_image_save_path)
                
                print(f"finish")
                
            else:
                print(f"skip image [{image_path}] with [{edit_method}]")
        
        
        