import argparse
import json
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import random
import os

from diffusers import DDIMScheduler

from models.p2p.inversion import DirectInversion
from models.masactrl.diffuser_utils import MasaCtrlPipeline
from models.masactrl.masactrl_utils import AttentionBase
from models.masactrl.masactrl_utils import regiter_attention_editor_diffusers
from models.masactrl.masactrl import MutualSelfAttentionControl
from utils.utils import load_512,txt_draw

from torchvision.io import read_image

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


def load_image(image_path, device):
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image



class MasaCtrlEditor:
    def __init__(self, method_list, device, num_ddim_steps=50) -> None:
        self.device=device
        self.method_list=method_list
        self.num_ddim_steps=num_ddim_steps
        # init model
        self.scheduler = DDIMScheduler(beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="scaled_linear",
                                    clip_sample=False,
                                    set_alpha_to_one=False)
        self.model = MasaCtrlPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", scheduler=self.scheduler).to(device)
        self.model.scheduler.set_timesteps(self.num_ddim_steps)

        
    def __call__(self, 
                edit_method,
                image_path,
                prompt_src,
                prompt_tar,
                guidance_scale,
                step=4,
                layper=10):
        if edit_method=="ddim+masactrl":
            return self.edit_image_ddim_MasaCtrl(image_path,prompt_src,prompt_tar,guidance_scale,step=step,layper=layper)
        elif edit_method=="directinversion+masactrl":
            return self.edit_image_directinversion_MasaCtrl(image_path,prompt_src,prompt_tar,guidance_scale,step=step,layper=layper)
        else:
            raise NotImplementedError(f"No edit method named {edit_method}")

    def edit_image_directinversion_MasaCtrl(self,image_path,prompt_src,prompt_tar,guidance_scale,step=4,layper=10):
        source_image=load_image(image_path, self.device)
        image_gt = load_512(image_path)
        
        prompts=["", prompt_tar]
        
        null_inversion = DirectInversion(model=self.model,
                                                num_ddim_steps=self.num_ddim_steps)
        
        _, image_enc_latent, x_stars, noise_loss_list = null_inversion.invert(
            image_gt=image_gt, prompt=prompts, guidance_scale=guidance_scale)
        x_t = x_stars[-1]
        
        # results of direct synthesis
        editor = AttentionBase()
        regiter_attention_editor_diffusers(self.model, editor)
        image_fixed = self.model([prompt_tar],
                            latents=x_t,
                            num_inference_steps=self.num_ddim_steps,
                            guidance_scale=guidance_scale,
                            noise_loss_list=None)
        
        # hijack the attention module
        editor = MutualSelfAttentionControl(step, layper)
        regiter_attention_editor_diffusers(self.model, editor)

        # inference the synthesized image
        image_masactrl = self.model(prompts,
                            latents= x_t.expand(len(prompts), -1, -1, -1),
                            guidance_scale=guidance_scale,
                            noise_loss_list=noise_loss_list)
        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        out_image=np.concatenate((
                                np.array(image_instruct),
                                ((source_image[0].permute(1,2,0).detach().cpu().numpy() * 0.5 + 0.5)*255).astype(np.uint8),
                                (image_masactrl[0].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8),
                                (image_masactrl[-1].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)),1)
        
        return Image.fromarray(out_image)
    
    def edit_image_ddim_MasaCtrl(self, image_path,prompt_src,prompt_tar,guidance_scale,step=4,layper=10):
        source_image=load_image(image_path, self.device)
        
        prompts=["", prompt_tar]
        
        start_code, latents_list = self.model.invert(source_image,
                                            "",
                                            guidance_scale=guidance_scale,
                                            num_inference_steps=self.num_ddim_steps,
                                            return_intermediates=True)
        start_code = start_code.expand(len(prompts), -1, -1, -1)
        
        # results of direct synthesis
        editor = AttentionBase()
        regiter_attention_editor_diffusers(self.model, editor)
        image_fixed = self.model([prompt_tar],
                            latents=start_code[-1:],
                            num_inference_steps=self.num_ddim_steps,
                            guidance_scale=guidance_scale)
        
        # hijack the attention module
        editor = MutualSelfAttentionControl(step, layper)
        regiter_attention_editor_diffusers(self.model, editor)

        # inference the synthesized image
        image_masactrl = self.model(prompts,
                            latents=start_code,
                            guidance_scale=guidance_scale)
        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        out_image=np.concatenate((
                                np.array(image_instruct),
                                ((source_image[0].permute(1,2,0).detach().cpu().numpy() * 0.5 + 0.5)*255).astype(np.uint8),
                                (image_masactrl[0].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8),
                                (image_masactrl[-1].permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8)),1)
        
        return Image.fromarray(out_image)




image_save_paths={
    "ddim+masactrl":"ddim+masactrl",
    "directinversion+masactrl":"directinversion+masactrl",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
    parser.add_argument('--data_path', type=str, default="data") # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="output") # the editing category that needed to run
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs = '+', type=str, default=["ddim+masactrl","directinversion+masactrl"]) # the editing methods that needed to run
    args = parser.parse_args()
    
    rerun_exist_images=args.rerun_exist_images
    data_path=args.data_path
    output_path=args.output_path
    edit_category_list=args.edit_category_list
    edit_method_list=args.edit_method_list
    
        
    masactrl_editor=MasaCtrlEditor(edit_method_list, torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') )

    
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
                edited_image = masactrl_editor(edit_method,
                                         image_path=image_path,
                                        prompt_src=original_prompt,
                                        prompt_tar=editing_prompt,
                                        guidance_scale=7.5,
                                        step=4,
                                        layper=10
                                        )
                if not os.path.exists(os.path.dirname(present_image_save_path)):
                    os.makedirs(os.path.dirname(present_image_save_path))
                edited_image.save(present_image_save_path)
                
                print(f"finish")
                
            else:
                print(f"skip image [{image_path}] with [{edit_method}]")
        
        