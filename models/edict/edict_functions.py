import torch
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import math
import imageio
from PIL import Image
import torchvision
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import datetime
import torch
import sys
import os
from torchvision import datasets
import pickle

# StableDiffusion P2P implementation originally from https://github.com/bloc97/CrossAttentionControl

# Have diffusers with hardcoded double-casting instead of float
from models.edict.my_diffusers import AutoencoderKL, UNet2DConditionModel
from models.edict.my_diffusers.schedulers.scheduling_utils import SchedulerOutput
from models.edict.my_diffusers import LMSDiscreteScheduler, PNDMScheduler, DDPMScheduler, DDIMScheduler


import random
from tqdm.auto import tqdm
from torch import autocast
from difflib import SequenceMatcher

# Build our CLIP model
model_path_clip = "openai/clip-vit-large-patch14"
clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=torch.float16)
clip = clip_model.text_model



model_path_diffusion = "CompVis/stable-diffusion-v1-4"
# Build our SD model
unet = UNet2DConditionModel.from_pretrained(model_path_diffusion, subfolder="unet", revision="fp16", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(model_path_diffusion, subfolder="vae", revision="fp16", torch_dtype=torch.float16)

# Push to devices w/ double precision
device = 'cuda'
unet.double().to(device)
vae.double().to(device)
clip.double().to(device)
print("Loaded all models")

    
def EDICT_editing(im_path,
                  base_prompt,
                  edit_prompt,
                  use_p2p=False,
                  steps=50,
                  mix_weight=0.93,
                  init_image_strength=0.8,
                  guidance_scale=3,
                 run_baseline=False):
    """
    Main call of our research, performs editing with either EDICT or DDIM
    
    Args:
        im_path: path to image to run on
        base_prompt: conditional prompt to deterministically noise with
        edit_prompt: desired text conditoining
        steps: ddim steps
        mix_weight: Weight of mixing layers.
            Higher means more consistent generations but divergence in inversion
            Lower means opposite
            This is fairly tuned and can get good results
        init_image_strength: Editing strength. Higher = more dramatic edit. 
            Typically [0.6, 0.9] is good range.
            Definitely tunable per-image/maybe best results are at a different value
        guidance_scale: classifier-free guidance scale
            3 I've found is the best for both our method and basic DDIM inversion
            Higher can result in more distorted results
        run_baseline:
            VERY IMPORTANT
            True is EDICT, False is DDIM
    Output:
        PAIR of Images (tuple)
        If run_baseline=True then [0] will be edit and [1] will be original
            This is to maintain consistently structured outputs across function calls
            The functions below will never operate on [1], leaving it unchanged
        If run_baseline=False then they will be two nearly identical edited versions
    """
    # Resize/center crop to 512x512 (Can do higher res. if desired)
    orig_im = load_im_into_format_from_path(im_path) if isinstance(im_path, str) else im_path # trust OK
    
    # compute latent pair (second one will be original latent if run_baseline=True)
    latents = coupled_stablediffusion(base_prompt,
                                     reverse=True,
                                      init_image=orig_im,
                                     init_image_strength=init_image_strength,
                                      steps=steps,
                                      mix_weight=mix_weight,
                                     guidance_scale=guidance_scale,
                                     run_baseline=run_baseline)
    # Denoise intermediate state with new conditioning
    gen = coupled_stablediffusion(edit_prompt if (not use_p2p) else base_prompt,
                                  None if (not use_p2p) else edit_prompt,
                                fixed_starting_latent=latents,
                                 init_image_strength=init_image_strength,
                                steps=steps,
                                mix_weight=mix_weight,
                                 guidance_scale=guidance_scale,
                                 run_baseline=run_baseline)
    
    return gen


def recon_test(im, steps=50, strength=1.0,
               run_baseline=False, 
               back_and_forth=False,
               prompt='',
              guidance_scale=7,
              plot=False,
              mix_weight=1):
    """
    Compute MSE Loss for images that are fully backed into latent space then decoded
    MSE computed on pixels normalized to [-1, 1]
    
    Args:
        im: a PIL Image or image path
        strength: How far to decode to 
        run_baseline: DDIM if True, EDICT if False
        prompt: Default is unconditional, this should work with baseline too. Conditional is differentiator
        guidance_scale: classifier-free guidance scale. Default to widely accepted value for generation
        mix_weight: Weight of mixing layers in EDICT
    Output:
        Single float of MSE for image and method
    """
    if isinstance(im, str): im = load_im_into_format_from_path(im) 
    
    latents = coupled_stablediffusion(prompt,
                                   reverse=True,
                                    init_image=im,
                                    steps=steps,
                                    init_image_strength=strength,
                                    run_baseline=run_baseline,
                                    back_and_forth=back_and_forth,
                                    guidance_scale=guidance_scale,
                                      mix_weight=mix_weight
                                   )
    if run_baseline:
        latents = latents[0]
    recon = coupled_stablediffusion(prompt,
                                   reverse=False,
                                    fixed_starting_latent=latents,
                                    steps=steps,
                                    init_image_strength=strength,
                                    run_baseline=run_baseline,
                                    back_and_forth=back_and_forth,
                                    guidance_scale=guidance_scale,
                                    mix_weight=mix_weight
                                   )
    recon = recon[0] if isinstance(recon, list) else recon

    
    orig_im_arr = im_to_np(im)
    recon_im_arr = im_to_np(recon)
    return np.square(orig_im_arr - recon_im_arr).mean()

def im_to_np(im):
    return np.array(im).astype(np.float64) / 255.0 * 2.0 - 1.0

def img2img_editing(im_path,
                  edit_prompt,
                  steps=50,
                  init_image_strength=0.7,
                  guidance_scale=3):
    """
    Basic SDEdit/img2img, given an image add some noise and denoise with prompt
    """
    orig_im = load_im_into_format_from_path(im_path)
    
    return baseline_stablediffusion(edit_prompt,
                                     init_image_strength=init_image_strength,
                                    steps=steps,
                                  init_image=orig_im,
                                 guidance_scale=guidance_scale)


def center_crop(im):
    width, height = im.size   # Get dimensions
    min_dim = min(width, height)
    left = (width - min_dim)/2
    top = (height - min_dim)/2
    right = (width + min_dim)/2
    bottom = (height + min_dim)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


def load_im_into_format_from_path(im_path):
    return center_crop(Image.open(im_path)).resize((512,512))


#### P2P STUFF #### 
def init_attention_weights(weight_tuples):
    tokens_length = clip_tokenizer.model_max_length
    weights = torch.ones(tokens_length)
    
    for i, w in weight_tuples:
        if i < tokens_length and i >= 0:
            weights[i] = w
    
    
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.last_attn_slice_weights = weights.to(device)
        if module_name == "CrossAttention" and "attn1" in name:
            module.last_attn_slice_weights = None
    

def init_attention_edit(tokens, tokens_edit):
    tokens_length = clip_tokenizer.model_max_length
    mask = torch.zeros(tokens_length)
    indices_target = torch.arange(tokens_length, dtype=torch.long)
    indices = torch.zeros(tokens_length, dtype=torch.long)

    tokens = tokens.input_ids.numpy()[0]
    tokens_edit = tokens_edit.input_ids.numpy()[0]
    
    for name, a0, a1, b0, b1 in SequenceMatcher(None, tokens, tokens_edit).get_opcodes():
        if b0 < tokens_length:
            if name == "equal" or (name == "replace" and a1-a0 == b1-b0):
                mask[b0:b1] = 1
                indices[b0:b1] = indices_target[a0:a1]

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.last_attn_slice_mask = mask.to(device)
            module.last_attn_slice_indices = indices.to(device)
        if module_name == "CrossAttention" and "attn1" in name:
            module.last_attn_slice_mask = None
            module.last_attn_slice_indices = None


def init_attention_func():
    def new_attention(self, query, key, value, sequence_length, dim):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = (
                torch.einsum("b i d, b j d -> b i j", query[start_idx:end_idx], key[start_idx:end_idx]) * self.scale
            )
            attn_slice = attn_slice.softmax(dim=-1)
            
            if self.use_last_attn_slice:
                if self.last_attn_slice_mask is not None:
                    new_attn_slice = torch.index_select(self.last_attn_slice, -1, self.last_attn_slice_indices)
                    attn_slice = attn_slice * (1 - self.last_attn_slice_mask) + new_attn_slice * self.last_attn_slice_mask
                else:
                    attn_slice = self.last_attn_slice
                
                self.use_last_attn_slice = False
                    
            if self.save_last_attn_slice:
                self.last_attn_slice = attn_slice
                self.save_last_attn_slice = False
                
            if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
                attn_slice = attn_slice * self.last_attn_slice_weights
                self.use_last_attn_weights = False

            attn_slice = torch.einsum("b i j, b j d -> b i d", attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.last_attn_slice = None
            module.use_last_attn_slice = False
            module.use_last_attn_weights = False
            module.save_last_attn_slice = False
            module._attention = new_attention.__get__(module, type(module))
            
def use_last_tokens_attention(use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.use_last_attn_slice = use
            
def use_last_tokens_attention_weights(use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.use_last_attn_weights = use
            
def use_last_self_attention(use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn1" in name:
            module.use_last_attn_slice = use
            
def save_last_tokens_attention(save=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.save_last_attn_slice = save
            
def save_last_self_attention(save=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn1" in name:
            module.save_last_attn_slice = save
####################################


##### BASELINE ALGORITHM, ONLY USED NOW FOR SDEDIT ####3

@torch.no_grad()
def baseline_stablediffusion(prompt="",
                    prompt_edit=None,
                             null_prompt='',
                    prompt_edit_token_weights=[],
                    prompt_edit_tokens_start=0.0,
                    prompt_edit_tokens_end=1.0,
                    prompt_edit_spatial_start=0.0,
                    prompt_edit_spatial_end=1.0,
                    clip_start=0.0,
                    clip_end=1.0,
                    guidance_scale=7,
                    steps=50,
                    seed=1,
                    width=512, height=512,
                    init_image=None, init_image_strength=0.5,
                    fixed_starting_latent = None,
                   prev_image= None,
                   grid=None,
                   clip_guidance=None,
                   clip_guidance_scale=1,
                   num_cutouts=4,
                   cut_power=1,
                   scheduler_str='lms',
                    return_latent=False,
                            one_pass=False,
                            normalize_noise_pred=False):
    width = width - width % 64
    height = height - height % 64
    
    #If seed is None, randomly select seed from 0 to 2^32-1
    if seed is None: seed = random.randrange(2**32 - 1)
    generator = torch.cuda.manual_seed(seed)
    
    #Set inference timesteps to scheduler
    scheduler_dict = {'ddim':DDIMScheduler,
                     'lms':LMSDiscreteScheduler,
                     'pndm':PNDMScheduler,
                     'ddpm':DDPMScheduler}
    scheduler_call = scheduler_dict[scheduler_str]
    if scheduler_str == 'ddim':
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                                     beta_schedule="scaled_linear",
                                     clip_sample=False, set_alpha_to_one=False)
    else:
        scheduler = scheduler_call(beta_schedule="scaled_linear",
                              num_train_timesteps=1000)

    scheduler.set_timesteps(steps)
    if prev_image is not None:
        prev_scheduler = LMSDiscreteScheduler(beta_start=0.00085,
                                         beta_end=0.012,
                                              beta_schedule="scaled_linear",
                                         num_train_timesteps=1000)
        prev_scheduler.set_timesteps(steps)
    
    #Preprocess image if it exists (img2img)
    if init_image is not None:
        init_image = init_image.resize((width, height), resample=Image.Resampling.LANCZOS)
        init_image = np.array(init_image).astype(np.float64) / 255.0 * 2.0 - 1.0
        init_image = torch.from_numpy(init_image[np.newaxis, ...].transpose(0, 3, 1, 2))

        #If there is alpha channel, composite alpha for white, as the diffusion model does not support alpha channel
        if init_image.shape[1] > 3:
            init_image = init_image[:, :3] * init_image[:, 3:] + (1 - init_image[:, 3:])

        #Move image to GPU
        init_image = init_image.to(device)

        #Encode image
        with autocast(device):
            init_latent = vae.encode(init_image).latent_dist.sample(generator=generator) * 0.18215

        t_start = steps - int(steps * init_image_strength)
            
    else:
        init_latent = torch.zeros((1, unet.in_channels, height // 8, width // 8), device=device)
        t_start = 0
    
    #Generate random normal noise
    if fixed_starting_latent is None:
        noise = torch.randn(init_latent.shape, generator=generator, device=device, dtype=unet.dtype)
        if scheduler_str == 'ddim':
            if init_image is not None:
                raise notImplementedError
                latent = scheduler.add_noise(init_latent, noise,
                                         1000 - int(1000 * init_image_strength)).to(device)
            else:
                latent = noise
        else:
            latent = scheduler.add_noise(init_latent, noise,
                                         t_start).to(device)
    else:
        latent = fixed_starting_latent
        t_start = steps - int(steps * init_image_strength)
    
    if prev_image is not None:
        #Resize and prev_image for numpy b h w c -> torch b c h w
        prev_image = prev_image.resize((width, height), resample=Image.Resampling.LANCZOS)
        prev_image = np.array(prev_image).astype(np.float64) / 255.0 * 2.0 - 1.0
        prev_image = torch.from_numpy(prev_image[np.newaxis, ...].transpose(0, 3, 1, 2))
        
        #If there is alpha channel, composite alpha for white, as the diffusion model does not support alpha channel
        if prev_image.shape[1] > 3:
            prev_image = prev_image[:, :3] * prev_image[:, 3:] + (1 - prev_image[:, 3:])
            
        #Move image to GPU
        prev_image = prev_image.to(device)
        
        #Encode image
        with autocast(device):
            prev_init_latent = vae.encode(prev_image).latent_dist.sample(generator=generator) * 0.18215
            
        t_start = steps - int(steps * init_image_strength)
        
        prev_latent = prev_scheduler.add_noise(prev_init_latent, noise, t_start).to(device)
    else:
        prev_latent = None
        
    
    #Process clip
    with autocast(device):
        tokens_unconditional = clip_tokenizer(null_prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_unconditional = clip(tokens_unconditional.input_ids.to(device)).last_hidden_state

        tokens_conditional = clip_tokenizer(prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_conditional = clip(tokens_conditional.input_ids.to(device)).last_hidden_state

        #Process prompt editing
        assert not ((prompt_edit is not None) and (prev_image is not None))
        if prompt_edit is not None:
            tokens_conditional_edit = clip_tokenizer(prompt_edit, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
            embedding_conditional_edit = clip(tokens_conditional_edit.input_ids.to(device)).last_hidden_state
            init_attention_edit(tokens_conditional, tokens_conditional_edit)
        elif prev_image is not None:
            init_attention_edit(tokens_conditional, tokens_conditional)
            
            
        init_attention_func()
        init_attention_weights(prompt_edit_token_weights)
            
        timesteps = scheduler.timesteps[t_start:]
        # print(timesteps)
        
        assert isinstance(guidance_scale, int)
        num_cycles = 1 # guidance_scale + 1
        
        last_noise_preds = None
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            t_index = t_start + i
            
            latent_model_input = latent
            if scheduler_str=='lms':
                sigma = scheduler.sigmas[t_index] # last is first and first is last
                latent_model_input = (latent_model_input / ((sigma**2 + 1) ** 0.5)).to(unet.dtype)
            else:
                assert scheduler_str in ['ddim', 'pndm', 'ddpm']

            #Predict the unconditional noise residual

            if len(t.shape) == 0:
                t = t[None].to(unet.device)
            noise_pred_uncond = unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional,
                                   ).sample

            if prev_latent is not None:
                prev_latent_model_input = prev_latent
                prev_latent_model_input = (prev_latent_model_input / ((sigma**2 + 1) ** 0.5)).to(unet.dtype)
                prev_noise_pred_uncond = unet(prev_latent_model_input, t,
                                              encoder_hidden_states=embedding_unconditional,
                                       ).sample
            # noise_pred_uncond = unet(latent_model_input, t,
            #                          encoder_hidden_states=embedding_unconditional)['sample']

            #Prepare the Cross-Attention layers
            if prompt_edit is not None or prev_latent is not None:
                save_last_tokens_attention()
                save_last_self_attention()
            else:
                #Use weights on non-edited prompt when edit is None
                use_last_tokens_attention_weights()

            #Predict the conditional noise residual and save the cross-attention layer activations
            if prev_latent is not None:
                raise NotImplementedError # I totally lost track of what this is
                prev_noise_pred_cond = unet(prev_latent_model_input, t, encoder_hidden_states=embedding_conditional,
                                      ).sample
            else:
                noise_pred_cond = unet(latent_model_input, t, encoder_hidden_states=embedding_conditional,
                                      ).sample

            #Edit the Cross-Attention layer activations
            t_scale = t / scheduler.num_train_timesteps
            if prompt_edit is not None or prev_latent is not None:
                if t_scale >= prompt_edit_tokens_start and t_scale <= prompt_edit_tokens_end:
                    use_last_tokens_attention()
                if t_scale >= prompt_edit_spatial_start and t_scale <= prompt_edit_spatial_end:
                    use_last_self_attention()

                #Use weights on edited prompt
                use_last_tokens_attention_weights()

                #Predict the edited conditional noise residual using the cross-attention masks
                if prompt_edit is not None:
                    noise_pred_cond = unet(latent_model_input, t,
                                           encoder_hidden_states=embedding_conditional_edit).sample

            #Perform guidance
            # if i%(num_cycles)==0: # cycle_i+1==num_cycles:
            """
            if cycle_i+1==num_cycles:
                noise_pred = noise_pred_uncond
            else:
                noise_pred = noise_pred_cond - noise_pred_uncond

            """
            if last_noise_preds is not None:
                # print( (last_noise_preds[0]*noise_pred_uncond).sum(), (last_noise_preds[1]*noise_pred_cond).sum())
                # print(F.cosine_similarity(last_noise_preds[0].flatten(), noise_pred_uncond.flatten(), dim=0),
                #      F.cosine_similarity(last_noise_preds[1].flatten(), noise_pred_cond.flatten(), dim=0))
                last_grad= last_noise_preds[1] - last_noise_preds[0]
                new_grad = noise_pred_cond - noise_pred_uncond
                # print( F.cosine_similarity(last_grad.flatten(), new_grad.flatten(), dim=0))
            last_noise_preds = (noise_pred_uncond, noise_pred_cond)

            use_cond_guidance = True 
            if use_cond_guidance:
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_uncond
            if clip_guidance is not None and t_scale >= clip_start and t_scale <= clip_end:
                noise_pred, latent = new_cond_fn(latent, t, t_index,
                                                 embedding_conditional, noise_pred,clip_guidance,
                                                clip_guidance_scale, 
                                                num_cutouts, 
                                                scheduler, unet,use_cutouts=True,
                                                cut_power=cut_power)
            if normalize_noise_pred:
                noise_pred = noise_pred * noise_pred_uncond.norm() /  noise_pred.norm()
            if scheduler_str == 'ddim':
                latent = forward_step(scheduler, noise_pred,
                                        t,
                                        latent).prev_sample
            else:
                latent = scheduler.step(noise_pred,
                                        t_index,
                                        latent).prev_sample

            if prev_latent is not None:
                prev_noise_pred = prev_noise_pred_uncond + guidance_scale * (prev_noise_pred_cond - prev_noise_pred_uncond)
                prev_latent = prev_scheduler.step(prev_noise_pred, t_index, prev_latent).prev_sample
            if one_pass: break

        #scale and decode the image latents with vae
        if return_latent: return latent
        latent = latent / 0.18215
        image = vae.decode(latent.to(vae.dtype)).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).round().astype("uint8")
    return Image.fromarray(image)
####################################

#### HELPER FUNCTIONS FOR OUR METHOD #####

def get_alpha_and_beta(t, scheduler):
    # want to run this for both current and previous timnestep
    if t.dtype==torch.long:
        alpha = scheduler.alphas_cumprod[t]
        return alpha, 1-alpha
    
    if t<0:
        return scheduler.final_alpha_cumprod, 1 - scheduler.final_alpha_cumprod

    
    low = t.floor().long()
    high = t.ceil().long()
    rem = t - low
    
    low_alpha = scheduler.alphas_cumprod[low]
    high_alpha = scheduler.alphas_cumprod[high]
    interpolated_alpha = low_alpha * rem + high_alpha * (1-rem)
    interpolated_beta = 1 - interpolated_alpha
    return interpolated_alpha, interpolated_beta
    

# A DDIM forward step function
def forward_step(
    self,
    model_output,
    timestep: int,
    sample,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    return_dict: bool = True,
    use_double=False,
) :
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    prev_timestep = timestep - self.config.num_train_timesteps / self.num_inference_steps
        
    if timestep > self.timesteps.max():
        raise NotImplementedError("Need to double check what the overflow is")
  
    alpha_prod_t, beta_prod_t = get_alpha_and_beta(timestep, self)
    alpha_prod_t_prev, _ = get_alpha_and_beta(prev_timestep, self)
    
    
    alpha_quotient = ((alpha_prod_t / alpha_prod_t_prev)**0.5)
    first_term =  (1./alpha_quotient) * sample
    second_term = (1./alpha_quotient) * (beta_prod_t ** 0.5) * model_output
    third_term = ((1 - alpha_prod_t_prev)**0.5) * model_output
    return first_term - second_term + third_term
                
# A DDIM reverse step function, the inverse of above
def reverse_step(
    self,
    model_output,
    timestep: int,
    sample,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    generator=None,
    return_dict: bool = True,
    use_double=False,
) :
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    prev_timestep = timestep - self.config.num_train_timesteps / self.num_inference_steps
   
    if timestep > self.timesteps.max():
        raise NotImplementedError
    else:
        alpha_prod_t = self.alphas_cumprod[timestep]
        
    alpha_prod_t, beta_prod_t = get_alpha_and_beta(timestep, self)
    alpha_prod_t_prev, _ = get_alpha_and_beta(prev_timestep, self)
    
    alpha_quotient = ((alpha_prod_t / alpha_prod_t_prev)**0.5)
    
    first_term =  alpha_quotient * sample
    second_term = ((beta_prod_t)**0.5) * model_output
    third_term = alpha_quotient * ((1 - alpha_prod_t_prev)**0.5) * model_output
    return first_term + second_term - third_term  
 



@torch.no_grad()
def latent_to_image(latent):
    image = vae.decode(latent.to(vae.dtype)/0.18215).sample
    image = prep_image_for_return(image)
    return image

def prep_image_for_return(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).round().astype("uint8")
    image = Image.fromarray(image)
    return image

#############################

##### MAIN EDICT FUNCTION #######
# Use EDICT_editing to perform calls

@torch.no_grad()
def coupled_stablediffusion(prompt="",
                           prompt_edit=None,
                            null_prompt='',
                            prompt_edit_token_weights=[],
                            prompt_edit_tokens_start=0.0,
                            prompt_edit_tokens_end=1.0,
                            prompt_edit_spatial_start=0.0,
                            prompt_edit_spatial_end=1.0,
                            guidance_scale=7.0, steps=50,
                            seed=1, width=512, height=512,
                            init_image=None, init_image_strength=1.0,
                           run_baseline=False,
                           use_lms=False,
                           leapfrog_steps=True,
                          reverse=False,
                          return_latents=False,
                          fixed_starting_latent=None,
                           beta_schedule='scaled_linear',
                            mix_weight=0.93):
    #If seed is None, randomly select seed from 0 to 2^32-1
    if seed is None: seed = random.randrange(2**32 - 1)
    generator = torch.cuda.manual_seed(seed)

    def image_to_latent(im):
        if isinstance(im, torch.Tensor):
            # assume it's the latent
            # used to avoid clipping new generation before inversion
            init_latent = im.to(device)
        else:
            #Resize and transpose for numpy b h w c -> torch b c h w
            im = im.resize((width, height), resample=Image.Resampling.LANCZOS)
            im = np.array(im).astype(np.float64) / 255.0 * 2.0 - 1.0
            # check if black and white
            if len(im.shape) < 3:
                im = np.stack([im for _ in range(3)], axis=2) # putting at end b/c channels
                
            im = torch.from_numpy(im[np.newaxis, ...].transpose(0, 3, 1, 2))

            #If there is alpha channel, composite alpha for white, as the diffusion model does not support alpha channel
            if im.shape[1] > 3:
                im = im[:, :3] * im[:, 3:] + (1 - im[:, 3:])

            #Move image to GPU
            im = im.to(device)
            #Encode image
            init_latent = vae.encode(im).latent_dist.sample(generator=generator) * 0.18215
            return init_latent
    assert not use_lms, "Can't invert LMS the same as DDIM"
    if run_baseline: leapfrog_steps=False
    #Change size to multiple of 64 to prevent size mismatches inside model
    width = width - width % 64
    height = height - height % 64
    
    
    #Preprocess image if it exists (img2img)
    if init_image is not None:
        assert reverse # want to be performing deterministic noising 
        # can take either pair (output of generative process) or single image
        if isinstance(init_image, list):
            if isinstance(init_image[0], torch.Tensor):
                init_latent = [t.clone() for t in init_image]
            else:
                init_latent = [image_to_latent(im) for im in init_image]
        else:
            init_latent = image_to_latent(init_image)
        # this is t_start for forward, t_end for reverse
        t_limit = steps - int(steps * init_image_strength)
    else:
        assert not reverse, 'Need image to reverse from'
        init_latent = torch.zeros((1, unet.in_channels, height // 8, width // 8), device=device)
        t_limit = 0
    
    if reverse:
        latent = init_latent
    else:
        #Generate random normal noise
        noise = torch.randn(init_latent.shape,
                            generator=generator,
                            device=device,
                           dtype=torch.float64)
        if fixed_starting_latent is None:
            latent = noise
        else:
            if isinstance(fixed_starting_latent, list):
                latent = [l.clone() for l in fixed_starting_latent]
            else:
                latent = fixed_starting_latent.clone()
            t_limit = steps - int(steps * init_image_strength)
    if isinstance(latent, list): # initializing from pair of images
        latent_pair = latent
    else: # initializing from noise
        latent_pair = [latent.clone(), latent.clone()]
        
    
    if steps==0:
        if init_image is not None:
            return image_to_latent(init_image)
        else:
            image = vae.decode(latent.to(vae.dtype) / 0.18215).sample
            return prep_image_for_return(image)
    
    #Set inference timesteps to scheduler
    schedulers = []
    for i in range(2):
        # num_raw_timesteps = max(1000, steps)
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                                     beta_schedule=beta_schedule,
                                  num_train_timesteps=1000,
                                     clip_sample=False,
                                  set_alpha_to_one=False)
        scheduler.set_timesteps(steps)
        schedulers.append(scheduler)
    

    # CLIP Text Embeddings
    tokens_unconditional = clip_tokenizer(null_prompt, padding="max_length",
                                          max_length=clip_tokenizer.model_max_length,
                                          truncation=True, return_tensors="pt", 
                                          return_overflowing_tokens=True)
    embedding_unconditional = clip(tokens_unconditional.input_ids.to(device)).last_hidden_state

    tokens_conditional = clip_tokenizer(prompt, padding="max_length", 
                                        max_length=clip_tokenizer.model_max_length,
                                        truncation=True, return_tensors="pt", 
                                        return_overflowing_tokens=True)
    embedding_conditional = clip(tokens_conditional.input_ids.to(device)).last_hidden_state

    #Process prompt editing (if running Prompt-to-Prompt)
    if prompt_edit is not None:
        tokens_conditional_edit = clip_tokenizer(prompt_edit, padding="max_length", 
                                                 max_length=clip_tokenizer.model_max_length,
                                                 truncation=True, return_tensors="pt", 
                                                 return_overflowing_tokens=True)
        embedding_conditional_edit = clip(tokens_conditional_edit.input_ids.to(device)).last_hidden_state

        init_attention_edit(tokens_conditional, tokens_conditional_edit)

    init_attention_func()
    init_attention_weights(prompt_edit_token_weights)

    timesteps = schedulers[0].timesteps[t_limit:]
    if reverse: timesteps = timesteps.flip(0)

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
        t_scale = t / schedulers[0].num_train_timesteps

        if (reverse) and (not run_baseline):
            # Reverse mixing layer
            new_latents = [l.clone() for l in latent_pair]
            new_latents[1] = (new_latents[1].clone() - (1-mix_weight)*new_latents[0].clone()) / mix_weight
            new_latents[0] = (new_latents[0].clone() - (1-mix_weight)*new_latents[1].clone()) / mix_weight
            latent_pair = new_latents

        # alternate EDICT steps
        for latent_i in range(2): 
            if run_baseline and latent_i==1: continue # just have one sequence for baseline
            # this modifies latent_pair[i] while using 
            # latent_pair[(i+1)%2]
            if reverse and (not run_baseline):
                if leapfrog_steps:
                    # what i would be from going other way
                    orig_i = len(timesteps) - (i+1) 
                    offset = (orig_i+1) % 2
                    latent_i = (latent_i + offset) % 2
                else:
                    # Do 1 then 0
                    latent_i = (latent_i+1)%2
            else:
                if leapfrog_steps:
                    offset = i%2
                    latent_i = (latent_i + offset) % 2

            latent_j = ((latent_i+1) % 2) if not run_baseline else latent_i

            latent_model_input = latent_pair[latent_j]
            latent_base = latent_pair[latent_i]

            #Predict the unconditional noise residual
            noise_pred_uncond = unet(latent_model_input, t, 
                                     encoder_hidden_states=embedding_unconditional).sample

            #Prepare the Cross-Attention layers
            if prompt_edit is not None:
                save_last_tokens_attention()
                save_last_self_attention()
            else:
                #Use weights on non-edited prompt when edit is None
                use_last_tokens_attention_weights()

            #Predict the conditional noise residual and save the cross-attention layer activations
            noise_pred_cond = unet(latent_model_input, t, 
                                   encoder_hidden_states=embedding_conditional).sample

            #Edit the Cross-Attention layer activations
            if prompt_edit is not None:
                t_scale = t / schedulers[0].num_train_timesteps
                if t_scale >= prompt_edit_tokens_start and t_scale <= prompt_edit_tokens_end:
                    use_last_tokens_attention()
                if t_scale >= prompt_edit_spatial_start and t_scale <= prompt_edit_spatial_end:
                    use_last_self_attention()

                #Use weights on edited prompt
                use_last_tokens_attention_weights()

                #Predict the edited conditional noise residual using the cross-attention masks
                noise_pred_cond = unet(latent_model_input,
                                       t, 
                                       encoder_hidden_states=embedding_conditional_edit).sample

            #Perform guidance
            grad = (noise_pred_cond - noise_pred_uncond)
            noise_pred = noise_pred_uncond + guidance_scale * grad


            step_call = reverse_step if reverse else forward_step
            new_latent = step_call(schedulers[latent_i],
                                      noise_pred,
                                        t,
                                        latent_base)# .prev_sample
            new_latent = new_latent.to(latent_base.dtype)

            latent_pair[latent_i] = new_latent

        if (not reverse) and (not run_baseline):
            # Mixing layer (contraction) during generative process
            new_latents = [l.clone() for l in latent_pair]
            new_latents[0] = (mix_weight*new_latents[0] + (1-mix_weight)*new_latents[1]).clone() 
            new_latents[1] = ((1-mix_weight)*new_latents[0] + (mix_weight)*new_latents[1]).clone() 
            latent_pair = new_latents

    #scale and decode the image latents with vae, can return latents instead of images
    if reverse or return_latents:
        results = [latent_pair]
        return results if len(results)>1 else results[0]
    
    # decode latents to iamges
    images = []
    for latent_i in range(2):
        latent = latent_pair[latent_i] / 0.18215
        image = vae.decode(latent.to(vae.dtype)).sample
        images.append(image)

    # Return images
    return_arr = []
    for image in images:
        image = prep_image_for_return(image)
        return_arr.append(image)
    results = [return_arr]
    return results if len(results)>1 else results[0]


