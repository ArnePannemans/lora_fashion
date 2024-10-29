import os
import torch
from diffusers import FluxPipeline
from PIL import Image

"""
Attempt at doing local inference with the FLUX + LoRA. 24GB VRAM is not enough without Quantization :(
"""

base_model = "black-forest-labs/FLUX.1-dev"
lora_dir = "SW_A/lora"

pipe = FluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)
pipe.to("cuda")

checkpoint = "sweater_A_lora_v1_000001800.safetensors"

prompt = "Amateur photo of a male person walking through new york wearing a light brown SW_A sweater."

guidance_scale = 4.0
lora_scales = 1.0

pipe.load_lora_weights(os.path.join(lora_dir, checkpoint))
pipe.fuse_lora(lora_scale=lora_scales)

image = pipe(prompt, num_inference_steps=40, guidance_scale=guidance_scale).images[0]

image = Image.fromarray(image)
image.show()

