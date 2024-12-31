import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler


width=512
height=512

latents_widht=512//8
latents_height=512//8

def generate(prompt:str,uncond_prompt:str,input_image=None,strength=0.8,do_cfg=True,cfg_scale=7.5,sampler_name="ddpm",
n_inference_steps=50,models={},seed=None,device=None,tokenizer=None):
    with torch.no_grad():
        if not(0<strength<=1):
            raise ValueError("Strength must be between 0 and 1")
        if idle_device:
            to_idle:lambda x:x.to(idle_device)
        else:
            to_idle=lambda x:x 
        generator=torch.Generator(device=device)
        if seed is None:
            generate_seed()
        else:generate.manual_seed(seed)
        clip=models["clip"]
        clip.to(device)
        if do_cfg:
            cond_tokens=tokenizer.batch_encode_plus([prompt],padding="max_length",max_length=77).input_ids
            cond_tokens=torch.tensor(cond_tokens,dtype=torch.long,device=device)
            cond_context=clip(cond_tokens)
            uncond_tokens=tokenizer.batch_encode_plus([uncond_prompt],padding="max_length",max_length=77).input_ids
            uncond_tokens=torch.tensor(uncond_tokens,dtype=torch.long,device=device)
            uncond_context=clip(uncond_tokens)
            context=torch.cat([cond_context,uncond_context])
        else:
            token=tokenizer.batch_encode_plus([prompt],padding="max_length",max_length=77).input_ids
            tokens=torch.tensor(tokens,dtype=torch.long,device=device)
            context=clip(tokens)

        to_idle(clip)
        if sampler_name=="ddpm":
            sampler=DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError("Invalid sampler name")
        latents_shape=(1,4,latents_height,latents_widht)
        if input_image:
            encoder=models["encoder"]
            encoder.to(device)
            input_image_tensor=input_image.resize((width, height))
            input_image=np.array(input_image_tensor)
            input_image_tensor=torch.tensor(input_image_tensor,dtype=torch.float32)
            input_image_tensor=rescale(input_image_tensor,(0,255),(-1,1))
            input_image_tensor=input_image_tensor.unsqueeze(0)
            input_image_tensor=input_image_tensor.permute(0,3,1,2)
            encoder_noise=torch.randn(latents_shape,generator=generator,device=device)
            latents=encoder(input_image_tensor,encoder_noise)
            sampler.set_strength(strength=strength)
            latents=sampler.add_noise(latents,sampler.timesteps[0])
            to_idle(encoder)
        else:
            latents=torch.randn(latents_shape,generator=generator,device=device)
            