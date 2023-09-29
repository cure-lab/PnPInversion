
from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
import numpy as np
from models.stylediffusion import global_var
from models.stylediffusion.utils import register_attention_control, image_grid,load_512,Trainer
from diffusers import DDIMScheduler
from PIL import Image
from tqdm import tqdm
from torch.optim.adam import Adam
import torch.nn.functional as nnf
from models.stylediffusion import ptp_utils_v

device=global_var.get_value("device")
NUM_DDIM_STEPS=global_var.get_value("NUM_DDIM_STEPS")
USE_INITIAL_INV=global_var.get_value("USE_INITIAL_INV")

class VaeInversion:

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
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

    def get_noise_pred(self, latents, t, is_forward=True, context=None, trainer=None):
        if context is None:
            context = self.context
        uncond_embeddings, cond_embeddings = context
        GUIDANCE_SCALE=global_var.get_value("GUIDANCE_SCALE")
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        trainer.uncond = True
        noise_pred_uncond = self.model.unet(latents, t, encoder_hidden_states=uncond_embeddings)["sample"]
        trainer.uncond = False
        noise_prediction_text = self.model.unet(latents, t, encoder_hidden_states=cond_embeddings)["sample"]
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(0, 3, 1, 2).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: List[str]):
        uncond_input = self.model.tokenizer(
            [""] * len(prompt), padding="max_length", max_length=self.model.tokenizer.model_max_length,
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
    def ddim_loop(self, latent, trainer=None):
        # store cross-attn during the ddim inversion
        if trainer:
            register_attention_control(self.model, trainer, None)

        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            if trainer:
                trainer.cur_att_layer = 32  # w=1, skip uncond attn layer
                trainer.attention_store = {}
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
            if trainer:
                attn_store = {}
                for key, value in trainer.attention_store.items():
                    if 'down_cross' in key or 'up_cross' in key:
                        attn_store[key] = [v for v in value if v.shape[1]==16**2]
                self.ddim_inv_attn += [attn_store]  # A*(0), A*(1), ... A*(T-1)

        # trainer.attention_store = sum(self.ddim_inv_attn)
        if trainer:
            trainer.attention_store = {}
            for ddim_inv_attn in self.ddim_inv_attn:
                if len(trainer.attention_store) == 0:
                    trainer.attention_store = ddim_inv_attn
                else:
                    for key in trainer.attention_store:
                        for i in range(len(trainer.attention_store[key])):
                            trainer.attention_store[key][i] += ddim_inv_attn[key][i]
            # A*(T) = A*(T-1)
            self.ddim_inv_attn += [self.ddim_inv_attn[-1]]

        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image, trainer=None):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent, trainer)
        return image_rec, ddim_latents

    def optimization(self, trainer, latents, image, num_inner_steps, num_epoch, epsilon):
        # torch.cuda.empty_cache()
        cross_attn_keys = self.ddim_inv_attn[0].keys()

        register_attention_control(self.model, trainer, None)

        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(0, 3, 1, 2).to(device)
        trainer.image = image

        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        x = np.linspace(0, NUM_DDIM_STEPS - 1, NUM_DDIM_STEPS)
        NUM_INNER_STEPS = np.ceil(num_inner_steps * np.exp(-.1 * x))
        bar = tqdm(total=int(np.sum(NUM_INNER_STEPS)), colour='red', ncols=100)
        for epoch in range(num_epoch):
            latent_cur = latents[-1]
            for i in range(NUM_DDIM_STEPS):
                num_inner_steps = int(NUM_INNER_STEPS[i])
                trainer.i = i
                if epoch == 0 and i > 0:
                    trainer.copy_params_and_buffers(trainer.embedding[i-1], trainer.embedding[i])
                embedding_i = trainer.embedding[i]
                optimizer = Adam(embedding_i.parameters(), lr=1e-2 * (1. - i / 100.))
                embedding_i.requires_grad_(True)
                latent_prev = latents[len(latents) - i - 2]
                t = self.model.scheduler.timesteps[i]
                with torch.no_grad():
                    trainer.uncond = True
                    noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                for j in range(num_inner_steps):
                    trainer.uncond = False
                    trainer.cur_att_layer = trainer.num_uncond_att_layers
                    trainer.attention_store = {}
                    # latent loss
                    noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
                    GUIDANCE_SCALE=global_var.get_value("GUIDANCE_SCALE")
                    noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                    latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                    latent_loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                    # cross-attn loss
                    for attn_key in list(trainer.attention_store.keys()):
                        if attn_key in cross_attn_keys:
                            trainer.attention_store[attn_key] = [attn for attn in trainer.attention_store[attn_key] if attn.shape[1]==16**2]
                        else:
                            del trainer.attention_store[attn_key]
                    attn_loss = torch.tensor(.0).to(device)
                    for key in cross_attn_keys:
                        if 'cross' in key:
                            for attn_gt, attn in zip(self.ddim_inv_attn[NUM_DDIM_STEPS - i][key], trainer.attention_store[key]):
                                attn_loss += nnf.mse_loss(attn_gt, attn)
                    # loss
                    loss = (latent_loss + attn_loss) 
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_item = loss.item()
                    bar.desc = f"Epoch[{epoch+1}/{num_epoch}, t={i}, iter={num_inner_steps}]"
                    bar.set_postfix(loss=loss_item)
                    bar.update()
                    if loss_item < epsilon + i * 2e-5:
                        break
                for j in range(j + 1, num_inner_steps):
                    bar.update()
                with torch.no_grad():
                    trainer.attention_store = {}
                    context = (uncond_embeddings, cond_embeddings)
                    latent_cur = self.get_noise_pred(latent_cur, t, False, context, trainer)
                embedding_i.requires_grad_(False)
            with torch.no_grad():
                image_inv = ptp_utils_v.latent2image(self.model.vae, latent_cur).squeeze()
                if len(image_inv.shape) == 3:
                    image_inv = image_inv[np.newaxis, :]
                image_inv = image_grid(image_inv, grid_size=(1, image_inv.shape[0]))
                # Image.fromarray(image_inv).save(f'ptp-epoch{epoch}-{args.idx}.png')
        bar.close()
        return trainer

    def invert(self, image_path: List[str], prompt: List[str], offsets=(0, 0, 0, 0), verbose=False, num_inner_steps=10, num_epoch=1, early_stop_epsilon=1e-5):
        self.init_prompt(prompt)
        image_gt = [load_512(path, *offsets) for path in image_path]
        image_gt = np.array(image_gt)

        # clip encoder and mapping-network
        trainer = Trainer()
        trainer.ddim_inv = True
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt, trainer)
        trainer.ddim_inv = False
        if trainer.attention_store:
            # show_cross_attention(trainer, prompt, res=16, from_where=["up", "down"])
            # show_hot_cross_attention(trainer, prompt, res=16, from_where=["up", "down"])
            pass
        trainer.attention_store = {}

        if verbose:
            print("StyleDiffusion optimization...")
        trainer = self.optimization(trainer, ddim_latents, image_gt, num_inner_steps, num_epoch, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], trainer

    def eval_init(self, image_path: List[str], prompt_gt: List[str], offsets=(0, 0, 0, 0), verbose=True, trainer=None):
        self.init_prompt(prompt_gt)
        image_gt = [load_512(path, *offsets) for path in image_path]
        image_gt = np.array(image_gt)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)

        image = torch.from_numpy(image_gt).float() / 127.5 - 1
        image = image.permute(0, 3, 1, 2).to(device)
        trainer.image = image \
            if not USE_INITIAL_INV else image.expand(2, *image.shape[1:])
        return (image_gt, image_rec), ddim_latents[-1]

    def __init__(self, model):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None
        self.ddim_inv_attn = []
