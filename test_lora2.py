# from diffusers import StableDiffusionPipeline
# import torch

# model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

# # prompt = "a photo of an astronaut riding a horse on mars"
# prompt = "a photo of a cute 6-year-old girl with red hair"

# image = pipe(prompt).images[0]  
    
# image.save("astronaut_rides_horse.png")
from huggingface_hub import snapshot_download
snapshot_download(repo_id='runwayml/stable-diffusion-v1-5', allow_patterns='tokenizer/*', cache_dir='diffusers/sd-image-variations-diffusers')
'''
import torch
from diffusers import StableDiffusionPipeline
from consistencydecoder import ConsistencyDecoder, save_image, load_image

# encode with stable diffusion vae
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, device="cuda:0"
)
pipe.vae.cuda()
decoder_consistency = ConsistencyDecoder(device="cuda:0") # Model size: 2.49 GB

image = load_image("/home/nfs/jsh/DisCo/0.png", size=(256, 256), center_crop=True)
latent = pipe.vae.encode(image.half().cuda()).latent_dist.mean
print(latent.shape)
# torch.Size([1, 4, 32, 32])
# decode with gan
sample_gan = pipe.vae.decode(latent).sample.detach()
save_image(sample_gan, "gan.png")

# decode with vae
sample_consistency = decoder_consistency(latent)
save_image(sample_consistency, "con.png")
'''