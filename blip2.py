# pip install accelerate
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
torch.backends.cudnn.enabled = False
# processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
processor = AutoProcessor.from_pretrained("/home/nfs/jsh/DisCo/diffusers/blip2")
model = Blip2ForConditionalGeneration.from_pretrained("/home/nfs/jsh/DisCo/diffusers/blip2")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image = Image.open('/home/nfs/jsh/DisCo/exp/bk/eval_step_53999/gt/00006_00jpg.png').convert('RGB')  
inputs = processor(image, return_tensors="pt").to(device)

generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

'''
prompt = "Question: what is the person in the picture like? Answer:"

inputs = processor(image, text=prompt, return_tensors="pt").to(device)

generated_ids = model.generate(**inputs, max_new_tokens=20)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)
'''

