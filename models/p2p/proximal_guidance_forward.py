import torch
import torch.nn.functional as F

from models.p2p.attention_control import register_attention_control
from utils.utils import init_latent

def dilate(image, kernel_size, stride=1, padding=0):
    """
    Perform dilation on a binary image using a square kernel.
    """
    # Ensure the image is binary
    assert image.max() <= 1 and image.min() >= 0
    
    # Get the maximum value in each neighborhood
    dilated_image = F.max_pool2d(image, kernel_size, stride, padding)
    
    return dilated_image

def proximal_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False,
                   edit_stage=True, prox=None, quantile=0.7,
                   image_enc=None, recon_lr=0.1, recon_t=400,
                   inversion_guidance=False, x_stars=None, i=0,
                   dilate_mask=0,):
    
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    step_kwargs = {
        'ref_image': None,
        'recon_lr': 0,
        'recon_mask': None,
    }
    mask_edit = None
    if edit_stage and prox is not None:
        if prox == 'l1':
            score_delta = noise_prediction_text - noise_pred_uncond
            if quantile > 0:
                threshold = score_delta.abs().quantile(quantile)
            else:
                threshold = -quantile  # if quantile is negative, use it as a fixed threshold
            score_delta -= score_delta.clamp(-threshold, threshold)
            score_delta = torch.where(score_delta > 0, score_delta-threshold, score_delta)
            score_delta = torch.where(score_delta < 0, score_delta+threshold, score_delta)
            if (recon_t > 0 and t < recon_t) or (recon_t < 0 and t > -recon_t):
                step_kwargs['ref_image'] = image_enc
                step_kwargs['recon_lr'] = recon_lr
                mask_edit = (score_delta.abs() > threshold).float()
                if dilate_mask > 0:
                    radius = int(dilate_mask)
                    mask_edit = dilate(mask_edit.float(), kernel_size=2*radius+1, padding=radius)
                step_kwargs['recon_mask'] = 1 - mask_edit
        elif prox == 'l0':
            score_delta = noise_prediction_text - noise_pred_uncond
            if quantile > 0:
                threshold = score_delta.abs().quantile(quantile)
            else:
                threshold = -quantile  # if quantile is negative, use it as a fixed threshold
            score_delta -= score_delta.clamp(-threshold, threshold)
            if (recon_t > 0 and t < recon_t) or (recon_t < 0 and t > -recon_t):
                step_kwargs['ref_image'] = image_enc
                step_kwargs['recon_lr'] = recon_lr
                mask_edit = (score_delta.abs() > threshold).float()
                if dilate_mask > 0:
                    radius = int(dilate_mask)
                    mask_edit = dilate(mask_edit.float(), kernel_size=2*radius+1, padding=radius)
                step_kwargs['recon_mask'] = 1 - mask_edit
        else:
            raise NotImplementedError
        noise_pred = noise_pred_uncond + guidance_scale * score_delta
    else:
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents, **step_kwargs)["prev_sample"]
    if mask_edit is not None and inversion_guidance and (recon_t > 0 and t < recon_t) or (recon_t < 0 and t > -recon_t):
        recon_mask = 1 - mask_edit
        latents = latents - recon_lr * (latents - x_stars[len(x_stars)-i-2].expand_as(latents)) * recon_mask

    latents = controller.step_callback(latents)
    return latents


@torch.no_grad()
def proximal_guidance_forward(
    model,
    prompt,
    controller,
    guidance_scale = 7.5,
    generator = None,
    latent = None,
    uncond_embeddings=None,
    edit_stage=True,
    prox=None,
    quantile=0.7,
    image_enc=None,
    recon_lr=0.1,
    recon_t=400,
    inversion_guidance=False,
    x_stars=None,
    dilate_mask=None
):
    """
        Get DDIM Forward result
        
        Parameters:
        model - the diffusion model
        prompt - forward prompt
        controller - prompt2prompt attention controller
        guidance_scale - generation guidance scale
        generator - the generator of random noisy latent (only needed if latent=None)
        latent - the x_T step latent
        uncond_embeddings - the generated unconditional embeddings
        edit_stage - doing inference
        prox - 
        quantile - quantile of the proximal guidance's threshold
        image_enc - 
        recon_lr - 
        recon_t - 
        inversion_guidance - 
        x_stars - 
        
        Returns:
            image_rec - the image reconstructed by VAE decoder with a size of [512,512,3], the channel follows the rgb of PIL.Image. i.e. RGB.
            image_rec_latent - the image latent with a size of [64,64,4]
            ddim_latents - the ddim inversion latents 50*[64,4,4], the first latent is the image_rec_latent, the last latent is noise (but in fact not pure noise)
            uncond_embeddings - the fake uncond_embeddings, in fact is cond_embedding or a interpolation among cond_embedding and uncond_embedding
    """
    batch_size = len(prompt)
    register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, 
            padding="max_length", 
            max_length=model.tokenizer.model_max_length, 
            truncation=True,
            return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    for i, t in enumerate(model.scheduler.timesteps):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = proximal_guidance_diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False,
                                           edit_stage=edit_stage, prox=prox, quantile=quantile,
                                           image_enc=image_enc, recon_lr=recon_lr, recon_t=recon_t,
                                           inversion_guidance=inversion_guidance, x_stars=x_stars, i=i,dilate_mask=dilate_mask)
        
    return latents, latent




