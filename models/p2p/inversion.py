import torch
import numpy as np
from PIL import Image
import torch.nn.functional as nnf
from torch.optim.adam import Adam

from models.p2p.attention_control import register_attention_control
from utils.utils import slerp_tensor, image2latent, latent2image

class NegativePromptInversion:
    
    def prev_step(self, model_output, timestep, sample):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output, timestep, sample):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    @torch.no_grad()
    def init_prompt(self, prompt):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        print("DDIM Inversion ...")
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = image2latent(self.model.vae, image)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents, latent

    def invert(self, image_gt, prompt, npi_interp=0.0):
        """
        Get DDIM Inversion of the image
        
        Parameters:
        image_gt - the gt image with a size of [512,512,3], the channel follows the rgb of PIL.Image. i.e. RGB.
        prompt - this is the prompt used for DDIM Inversion
        npi_interp - the interpolation ratio among conditional embedding and unconditional embedding
        num_ddim_steps - the number of ddim steps
        
        Returns:
            image_rec - the image reconstructed by VAE decoder with a size of [512,512,3], the channel follows the rgb of PIL.Image. i.e. RGB.
            image_rec_latent - the image latent with a size of [64,64,4]
            ddim_latents - the ddim inversion latents 50*[64,4,4], the first latent is the image_rec_latent, the last latent is noise (but in fact not pure noise)
            uncond_embeddings - the fake uncond_embeddings, in fact is cond_embedding or a interpolation among cond_embedding and uncond_embedding
        """
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        image_rec, ddim_latents, image_rec_latent = self.ddim_inversion(image_gt)
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        if npi_interp > 0.0: # do vector interpolation among cond_embedding and uncond_embedding
            cond_embeddings = slerp_tensor(npi_interp, cond_embeddings, uncond_embeddings)
        uncond_embeddings = [cond_embeddings] * self.num_ddim_steps
        return image_rec, image_rec_latent, ddim_latents, uncond_embeddings

    def __init__(self, model,num_ddim_steps):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.prompt = None
        self.context = None
        self.num_ddim_steps=num_ddim_steps




class NullInversion:
    
    def prev_step(self, model_output, timestep: int, sample):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output, timestep: int, sample):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, guidance_scale, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else guidance_scale
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], 
            padding="max_length", 
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = image2latent(self.model.vae, image)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon, guidance_scale):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        for i in range(self.num_ddim_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            t = self.model.scheduler.timesteps[i]
            if num_inner_steps!=0:
                uncond_embeddings.requires_grad = True
                optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
                latent_prev = latents[len(latents) - i - 2]
                with torch.no_grad():
                    noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
                for j in range(num_inner_steps):
                    noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                    loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_item = loss.item()
                    if loss_item < epsilon + i * 2e-5:
                        break
                
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, guidance_scale, False, context)
        return uncond_embeddings_list
    
    def invert(self, image_gt, prompt, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5):
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale)
        return image_gt, image_rec, ddim_latents, uncond_embeddings
    
    def __init__(self, model,num_ddim_steps):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.prompt = None
        self.context = None
        self.num_ddim_steps=num_ddim_steps
        


class DirectInversion:
    
    def prev_step(self, model_output, timestep: int, sample):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        difference_scale_pred_original_sample= - beta_prod_t ** 0.5  / alpha_prod_t ** 0.5
        difference_scale_pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 
        difference_scale = alpha_prod_t_prev ** 0.5 * difference_scale_pred_original_sample + difference_scale_pred_sample_direction
        
        return prev_sample,difference_scale
    
    def next_step(self, model_output, timestep: int, sample):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, guidance_scale, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else guidance_scale
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""]*len(prompt), padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        cond_embeddings=cond_embeddings[[0]]
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent
    
    @torch.no_grad()
    def ddim_null_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings=uncond_embeddings[[0]]
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, uncond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent
    
    @torch.no_grad()
    def ddim_with_guidance_scale_loop(self, latent,guidance_scale):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings=uncond_embeddings[[0]]
        cond_embeddings=cond_embeddings[[0]]
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(self.num_ddim_steps):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            uncond_noise_pred = self.get_noise_pred_single(latent, t, uncond_embeddings)
            cond_noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            noise_pred = uncond_noise_pred + guidance_scale * (cond_noise_pred - uncond_noise_pred)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = image2latent(self.model.vae, image)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents
    
    @torch.no_grad()
    def ddim_null_inversion(self, image):
        latent = image2latent(self.model.vae, image)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_null_loop(latent)
        return image_rec, ddim_latents
    
    @torch.no_grad()
    def ddim_with_guidance_scale_inversion(self, image,guidance_scale):
        latent = image2latent(self.model.vae, image)
        image_rec = latent2image(self.model.vae, latent)[0]
        ddim_latents = self.ddim_with_guidance_scale_loop(latent,guidance_scale)
        return image_rec, ddim_latents

    def offset_calculate(self, latents, num_inner_steps, epsilon, guidance_scale):
        noise_loss_list = []
        latent_cur = torch.concat([latents[-1]]*(self.context.shape[0]//2))
        for i in range(self.num_ddim_steps):            
            latent_prev = torch.concat([latents[len(latents) - i - 2]]*latent_cur.shape[0])
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(torch.concat([latent_cur]*2), t, self.context)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec, _ = self.prev_step(noise_pred_w_guidance, t, latent_cur)
                loss = latent_prev - latents_prev_rec
                
            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss
            
        return noise_loss_list
    
    def invert(self, image_gt, prompt, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5):
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        noise_loss_list = self.offset_calculate(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale)
        return image_gt, image_rec, ddim_latents, noise_loss_list
    
    def invert_without_attn_controller(self, image_gt, prompt, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5):
        self.init_prompt(prompt)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        noise_loss_list = self.offset_calculate(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale)
        return image_gt, image_rec, ddim_latents, noise_loss_list
    
    def invert_with_guidance_scale_vary_guidance(self, image_gt, prompt, inverse_guidance_scale, forward_guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5):
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        
        image_rec, ddim_latents = self.ddim_with_guidance_scale_inversion(image_gt,inverse_guidance_scale)
        
        noise_loss_list = self.offset_calculate(ddim_latents, num_inner_steps, early_stop_epsilon,forward_guidance_scale)
        return image_gt, image_rec, ddim_latents, noise_loss_list

    def null_latent_calculate(self, latents, num_inner_steps, epsilon, guidance_scale):
        noise_loss_list = []
        latent_cur = torch.concat([latents[-1]]*(self.context.shape[0]//2))
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        for i in range(self.num_ddim_steps):            
            latent_prev = torch.concat([latents[len(latents) - i - 2]]*latent_cur.shape[0])
            t = self.model.scheduler.timesteps[i]

            if num_inner_steps!=0:
                uncond_embeddings = uncond_embeddings.clone().detach()
                uncond_embeddings.requires_grad = True
                optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
                for j in range(num_inner_steps):
                    latents_input = torch.cat([latent_cur] * 2)
                    noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=torch.cat([uncond_embeddings, cond_embeddings]))["sample"]
                    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
                    
                    latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)[0]
                    
                    loss = nnf.mse_loss(latents_prev_rec[[0]], latent_prev[[0]])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_item = loss.item()

                    if loss_item < epsilon + i * 2e-5:
                        break
                    
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(torch.concat([latent_cur]*2), t, self.context)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec, _ = self.prev_step(noise_pred_w_guidance, t, latent_cur)
                
                latent_cur = self.get_noise_pred(latent_cur, t,guidance_scale, False, torch.cat([uncond_embeddings, cond_embeddings]))[0]
                loss = latent_cur - latents_prev_rec
                
            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss
            
        return noise_loss_list
        
    
    def invert_null_latent(self, image_gt, prompt, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5):
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        latent_list = self.null_latent_calculate(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale)
        return image_gt, image_rec, ddim_latents, latent_list
    
    def offset_calculate_not_full(self, latents, num_inner_steps, epsilon, guidance_scale,scale):
        noise_loss_list = []
        latent_cur = torch.concat([latents[-1]]*(self.context.shape[0]//2))
        for i in range(self.num_ddim_steps):            
            latent_prev = torch.concat([latents[len(latents) - i - 2]]*latent_cur.shape[0])
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(torch.concat([latent_cur]*2), t, self.context)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec, _ = self.prev_step(noise_pred_w_guidance, t, latent_cur)
                loss = latent_prev - latents_prev_rec
                loss=loss*scale
                
            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss
            
        return noise_loss_list
        
    def invert_not_full(self, image_gt, prompt, guidance_scale, num_inner_steps=10, early_stop_epsilon=1e-5,scale=1.):
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        noise_loss_list = self.offset_calculate_not_full(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale,scale)
        return image_gt, image_rec, ddim_latents, noise_loss_list
    
    def offset_calculate_skip_step(self, latents, num_inner_steps, epsilon, guidance_scale,skip_step):
        noise_loss_list = []
        latent_cur = torch.concat([latents[-1]]*(self.context.shape[0]//2))
        for i in range(self.num_ddim_steps):            
            latent_prev = torch.concat([latents[len(latents) - i - 2]]*latent_cur.shape[0])
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred = self.get_noise_pred_single(torch.concat([latent_cur]*2), t, self.context)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred_w_guidance = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec, _ = self.prev_step(noise_pred_w_guidance, t, latent_cur)
                if (i%skip_step)==0:
                    loss = latent_prev - latents_prev_rec
                else:
                    loss=torch.zeros_like(latent_prev)
                
            noise_loss_list.append(loss.detach())
            latent_cur = latents_prev_rec + loss
            
        return noise_loss_list
    
    
    def invert_skip_step(self, image_gt, prompt, guidance_scale, skip_step,num_inner_steps=10, early_stop_epsilon=1e-5,scale=1.):
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        
        noise_loss_list = self.offset_calculate_skip_step(ddim_latents, num_inner_steps, early_stop_epsilon,guidance_scale,skip_step)
        return image_gt, image_rec, ddim_latents, noise_loss_list
    
    
    def __init__(self, model,num_ddim_steps):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.prompt = None
        self.context = None
        self.num_ddim_steps=num_ddim_steps
        
       