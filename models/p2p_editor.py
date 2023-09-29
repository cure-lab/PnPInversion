
from models.p2p.scheduler_dev import DDIMSchedulerDev
from models.p2p.inversion import NegativePromptInversion, NullInversion, DirectInversion
from models.p2p.attention_control import EmptyControl, AttentionStore, make_controller
from models.p2p.p2p_guidance_forward import p2p_guidance_forward, direct_inversion_p2p_guidance_forward, direct_inversion_p2p_guidance_forward_add_target,p2p_guidance_forward_single_branch
from models.p2p.proximal_guidance_forward import proximal_guidance_forward
from diffusers import StableDiffusionPipeline
from utils.utils import load_512, latent2image, txt_draw
from PIL import Image
import numpy as np

class P2PEditor:
    def __init__(self, method_list, device, num_ddim_steps=50) -> None:
        self.device=device
        self.method_list=method_list
        self.num_ddim_steps=num_ddim_steps
        # init model
        self.scheduler = DDIMSchedulerDev(beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="scaled_linear",
                                    clip_sample=False,
                                    set_alpha_to_one=False)
        self.ldm_stable = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", scheduler=self.scheduler).to(device)
        self.ldm_stable.scheduler.set_timesteps(self.num_ddim_steps)

        
    def __call__(self, 
                edit_method,
                image_path,
                prompt_src,
                prompt_tar,
                guidance_scale=7.5,
                proximal=None,
                quantile=0.7,
                use_reconstruction_guidance=False,
                recon_t=400,
                recon_lr=0.1,
                cross_replace_steps=0.4,
                self_replace_steps=0.6,
                blend_word=None,
                eq_params=None,
                is_replace_controller=False,
                use_inversion_guidance=False,
                dilate_mask=1,):
        if edit_method=="ddim+p2p":
            return self.edit_image_ddim(image_path, prompt_src, prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        elif edit_method in ["null-text-inversion+p2p", "null-text-inversion+p2p_a800", "null-text-inversion+p2p_3090"]:
            return self.edit_image_null_text_inversion(image_path, prompt_src, prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        elif edit_method == "ablation_null-text-inversion_single_branch+p2p":
            return self.edit_image_null_text_inversion_single_branch(image_path, prompt_src, prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        elif edit_method=="negative-prompt-inversion+p2p":
            return self.edit_image_negative_prompt_inversion(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar,
                                        guidance_scale=guidance_scale, proximal=None, quantile=quantile, use_reconstruction_guidance=use_reconstruction_guidance,
                                        recon_t=recon_t, recon_lr=recon_lr, cross_replace_steps=cross_replace_steps,
                                        self_replace_steps=self_replace_steps, blend_word=blend_word, eq_params=eq_params,
                                        is_replace_controller=is_replace_controller, use_inversion_guidance=use_inversion_guidance,
                                        dilate_mask=dilate_mask)
        elif edit_method=="directinversion+p2p":
            return self.edit_image_directinversion(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        elif edit_method in ["directinversion+p2p_guidance_0_1", "directinversion+p2p_guidance_0_5","directinversion+p2p_guidance_0_25", \
            "directinversion+p2p_guidance_0_75", "directinversion+p2p_guidance_1_1", "directinversion+p2p_guidance_1_5", "directinversion+p2p_guidance_1_25", \
                "directinversion+p2p_guidance_1_75", "directinversion+p2p_guidance_25_1", "directinversion+p2p_guidance_25_5", "directinversion+p2p_guidance_25_25", \
                    "directinversion+p2p_guidance_25_75", "directinversion+p2p_guidance_5_1", "directinversion+p2p_guidance_5_5", \
                        "directinversion+p2p_guidance_5_25", "directinversion+p2p_guidance_5_75", "directinversion+p2p_guidance_75_1", \
                            "directinversion+p2p_guidance_75_5", "directinversion+p2p_guidance_75_25", "directinversion+p2p_guidance_75_75"]:
            guidance_scale={
                "0":0,
                "1":1,
                "25":2.5,
                "5":5,
                "75":7.5}
            
            inverse_guidance_scale=guidance_scale[edit_method.split("_")[-2]]
            forward_guidance_scale=guidance_scale[edit_method.split("_")[-1]]
            return self.edit_image_directinversion_vary_guidance_scale(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar, 
                                                            inverse_guidance_scale=inverse_guidance_scale, 
                                                            forward_guidance_scale=forward_guidance_scale, 
                                                            cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                                            blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        elif edit_method=="null-text-inversion+proximal-guidance":
            return self.edit_image_null_text_inversion_proximal_guidanca(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar,
                                        guidance_scale=guidance_scale, proximal=proximal, quantile=quantile, use_reconstruction_guidance=use_reconstruction_guidance,
                                        recon_t=recon_t, recon_lr=recon_lr, cross_replace_steps=cross_replace_steps,
                                        self_replace_steps=self_replace_steps, blend_word=blend_word, eq_params=eq_params,
                                        is_replace_controller=is_replace_controller, use_inversion_guidance=use_inversion_guidance,
                                        dilate_mask=dilate_mask)
        elif edit_method=="negative-prompt-inversion+proximal-guidance":
            return self.edit_image_negative_prompt_inversion(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar,
                                        guidance_scale=guidance_scale, proximal=proximal, quantile=quantile, use_reconstruction_guidance=use_reconstruction_guidance,
                                        recon_t=recon_t, recon_lr=recon_lr, cross_replace_steps=cross_replace_steps,
                                        self_replace_steps=self_replace_steps, blend_word=blend_word, eq_params=eq_params,
                                        is_replace_controller=is_replace_controller, use_inversion_guidance=use_inversion_guidance,
                                        dilate_mask=dilate_mask)
        elif edit_method=="ablation_null-latent-inversion+p2p":
            return self.edit_image_null_latent_inversion(image_path, prompt_src, prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        elif edit_method in [
                             "ablation_directinversion_08+p2p",
                             "ablation_directinversion_04+p2p",
                             ]:
            scale=float(edit_method.split("+")[0].split("_")[-1])/10
            return self.edit_image_directinversion_not_full(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller,scale=scale)
        elif edit_method in [
                             "ablation_directinversion_interval_2+p2p",
                             "ablation_directinversion_interval_5+p2p",
                             "ablation_directinversion_interval_10+p2p",
                             "ablation_directinversion_interval_24+p2p",
                             "ablation_directinversion_interval_49+p2p",
                             ]:
            skip_step=int(edit_method.split("+")[0].split("_")[-1])
            return self.edit_image_directinversion_skip_step(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller,skip_step=skip_step)
        elif edit_method=="ablation_directinversion_add-target+p2p":
            return self.edit_image_directinversion_add_target(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        elif edit_method=="ablation_directinversion_add-source+p2p":
            return self.edit_image_directinversion_add_source(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        else:
            raise NotImplementedError(f"No edit method named {edit_method}")

    def edit_image_ddim(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = NullInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, uncond_embeddings = null_inversion.invert(
            image_gt=image_gt, prompt=prompt_src,guidance_scale=guidance_scale,num_inner_steps=0)
        x_t = x_stars[-1]

        controller = AttentionStore()
        reconstruct_latent, x_t = p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=[prompt_src], 
                                       controller=controller, 
                                       latent=x_t, 
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None, 
                                       uncond_embeddings=uncond_embeddings)
        

        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        latents, _ = p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       latent=x_t, 
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None, 
                                       uncond_embeddings=uncond_embeddings)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    def edit_image_null_text_inversion(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = NullInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, uncond_embeddings = null_inversion.invert(
            image_gt=image_gt, prompt=prompt_src,guidance_scale=guidance_scale)
        x_t = x_stars[-1]

        controller = AttentionStore()
        reconstruct_latent, x_t = p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=[prompt_src], 
                                       controller=controller, 
                                       latent=x_t, 
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None, 
                                       uncond_embeddings=uncond_embeddings)
        

        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        latents, _ = p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       latent=x_t, 
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None, 
                                       uncond_embeddings=uncond_embeddings)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    def edit_image_null_text_inversion_single_branch(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = NullInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, uncond_embeddings = null_inversion.invert(
            image_gt=image_gt, prompt=prompt_src,guidance_scale=guidance_scale)
        x_t = x_stars[-1]

        controller = AttentionStore()
        reconstruct_latent, x_t = p2p_guidance_forward_single_branch(model=self.ldm_stable, 
                                       prompt=[prompt_src], 
                                       controller=controller, 
                                       latent=x_t, 
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None, 
                                       uncond_embeddings=uncond_embeddings)
        

        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        latents, _ = p2p_guidance_forward_single_branch(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       latent=x_t, 
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None, 
                                       uncond_embeddings=uncond_embeddings)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))


    def edit_image_negative_prompt_inversion(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        proximal=None,
        quantile=0.7,
        use_reconstruction_guidance=False,
        recon_t=400,
        recon_lr=0.1,
        npi_interp=0,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
        use_inversion_guidance=False,
        dilate_mask=1,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = NegativePromptInversion(model=self.ldm_stable,
                                                num_ddim_steps=self.num_ddim_steps)
        _, image_enc_latent, x_stars, uncond_embeddings = null_inversion.invert(
            image_gt=image_gt, prompt=prompt_src, npi_interp=npi_interp)
        x_t = x_stars[-1]

        controller = AttentionStore()
        reconstruct_latent, x_t = proximal_guidance_forward(
                    model=self.ldm_stable,
                    prompt=[prompt_src],
                    controller=controller,
                    latent=x_t,
                    guidance_scale=guidance_scale,
                    generator=None,
                    uncond_embeddings=uncond_embeddings,
                    edit_stage=False,
                    prox=None,
                    quantile=quantile,
                    image_enc=None,
                    recon_lr=recon_lr,
                    recon_t=recon_t,
                    inversion_guidance=False,
                    x_stars=None,
                    dilate_mask=dilate_mask)
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        
        latents, _ = proximal_guidance_forward(
                        model=self.ldm_stable,
                        prompt=prompts,
                        controller=controller,
                        latent=x_t,
                        guidance_scale=guidance_scale,
                        generator=None,
                        uncond_embeddings=uncond_embeddings,
                        edit_stage=True,
                        prox=proximal,
                        quantile=quantile,
                        image_enc=image_enc_latent if use_reconstruction_guidance else None,
                        recon_lr=recon_lr
                            if use_reconstruction_guidance or use_inversion_guidance else 0,
                        recon_t=recon_t
                            if use_reconstruction_guidance or use_inversion_guidance else 1000,
                        x_stars=x_stars,
                        dilate_mask=dilate_mask)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)


        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    def edit_image_directinversion(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = DirectInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, noise_loss_list = null_inversion.invert(
            image_gt=image_gt, prompt=prompts,guidance_scale=guidance_scale)
        x_t = x_stars[-1]

        controller = AttentionStore()
        
        reconstruct_latent, x_t = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)
    
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        latents, _ = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    def edit_image_directinversion_vary_guidance_scale(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        inverse_guidance_scale=1,
        forward_guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = DirectInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, noise_loss_list = null_inversion.invert_with_guidance_scale_vary_guidance(
            image_gt=image_gt, prompt=prompts, inverse_guidance_scale=inverse_guidance_scale, 
            forward_guidance_scale=forward_guidance_scale)
        x_t = x_stars[-1]

        controller = AttentionStore()
        
        reconstruct_latent, x_t = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=forward_guidance_scale, 
                                       generator=None)
    
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        latents, _ = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=forward_guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        

        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    def edit_image_null_text_inversion_proximal_guidanca(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        proximal=None,
        quantile=0.7,
        use_reconstruction_guidance=False,
        recon_t=400,
        recon_lr=0.1,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
        use_inversion_guidance=False,
        dilate_mask=1,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = NullInversion(model=self.ldm_stable,
                                   num_ddim_steps=self.num_ddim_steps)
        _, image_enc_latent, x_stars, uncond_embeddings = null_inversion.invert(
            image_gt=image_gt, prompt=prompt_src,guidance_scale=guidance_scale)
        x_t = x_stars[-1]

        controller = AttentionStore()
        reconstruct_latent, x_t = proximal_guidance_forward(
                    model=self.ldm_stable,
                    prompt=[prompt_src],
                    controller=controller,
                    latent=x_t,
                    guidance_scale=guidance_scale,
                    generator=None,
                    uncond_embeddings=uncond_embeddings,
                    edit_stage=False,
                    prox=None,
                    quantile=quantile,
                    image_enc=None,
                    recon_lr=recon_lr,
                    recon_t=recon_t,
                    inversion_guidance=False,
                    x_stars=None,
                    dilate_mask=dilate_mask)
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        
        latents, _ = proximal_guidance_forward(
                        model=self.ldm_stable,
                        prompt=prompts,
                        controller=controller,
                        latent=x_t,
                        guidance_scale=guidance_scale,
                        generator=None,
                        uncond_embeddings=uncond_embeddings,
                        edit_stage=True,
                        prox=proximal,
                        quantile=quantile,
                        image_enc=image_enc_latent if use_reconstruction_guidance else None,
                        recon_lr=recon_lr
                            if use_reconstruction_guidance or use_inversion_guidance else 0,
                        recon_t=recon_t
                            if use_reconstruction_guidance or use_inversion_guidance else 1000,
                        x_stars=x_stars,
                        dilate_mask=dilate_mask)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)


        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    def edit_image_null_latent_inversion(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = DirectInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, noise_loss_list = null_inversion.invert_null_latent(
            image_gt=image_gt, prompt=prompts,guidance_scale=guidance_scale)
        x_t = x_stars[-1]

        controller = AttentionStore()
        
        reconstruct_latent, x_t = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)
    
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        latents, _ = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        

        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    def edit_image_directinversion_not_full(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
        scale=1.
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = DirectInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, noise_loss_list = null_inversion.invert_not_full(
            image_gt=image_gt, prompt=prompts,guidance_scale=guidance_scale,scale=scale)
        x_t = x_stars[-1]

        controller = AttentionStore()
        
        reconstruct_latent, x_t = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)
    
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        latents, _ = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    
    def edit_image_directinversion_skip_step(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        skip_step,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = DirectInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, noise_loss_list = null_inversion.invert_skip_step(
            image_gt=image_gt, prompt=prompts,guidance_scale=guidance_scale,skip_step=skip_step)
        x_t = x_stars[-1]

        controller = AttentionStore()
        
        reconstruct_latent, x_t = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)
    
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        latents, _ = direct_inversion_p2p_guidance_forward(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))

    def edit_image_directinversion_add_target(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = DirectInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, noise_loss_list = null_inversion.invert(
            image_gt=image_gt, prompt=prompts,guidance_scale=guidance_scale)
        x_t = x_stars[-1]

        controller = AttentionStore()
        
        reconstruct_latent, x_t = direct_inversion_p2p_guidance_forward_add_target(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)
    
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        latents, _ = direct_inversion_p2p_guidance_forward_add_target(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))


    def edit_image_directinversion_add_source(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = DirectInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, noise_loss_list = null_inversion.invert(
            image_gt=image_gt, prompt=prompts,guidance_scale=guidance_scale)
        x_t = x_stars[-1]
        
        noise_loss_list_new=[]
        
        for i in range(len(noise_loss_list)):
            noise_loss_list_new.append(noise_loss_list[i][[0]].repeat(2,1,1,1))
        
        controller = AttentionStore()
        
        reconstruct_latent, x_t = direct_inversion_p2p_guidance_forward_add_target(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list_new, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)
    
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }

        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps,
                                    device=self.device)
        
        latents, _ = direct_inversion_p2p_guidance_forward_add_target(model=self.ldm_stable, 
                                       prompt=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list_new, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        
        image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        return Image.fromarray(np.concatenate((image_instruct, image_gt, reconstruct_image,images[-1]),axis=1))
