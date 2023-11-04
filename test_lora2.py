from diffusers import StableDiffusionPipeline
import torch

model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on mars"
prompt = "a photo of a cute 6-year-old girl with red hair"

image = pipe(prompt).images[0]  
    
image.save("astronaut_rides_horse.png")
