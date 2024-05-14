"""Experimentation and examples of generation with llava-llama3 from intel"""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForPreTraining

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result



from transformers import AutoProcessor, AutoModelForPreTraining

model = AutoModelForPreTraining.from_pretrained("/home/mashalimay/.cache/huggingface/hub/models--Intel--llava-llama-3-merged-8b", 
                                                torch_dtype=torch.bfloat16, 
                                                low_cpu_mem_usage=True,).to("cuda:0")




processor = AutoProcessor.from_pretrained("Intel/llava-llama-3-8b")
terminators = [
    processor.tokenizer.eos_token_id,
    processor.tokenizer.convert_tokens_to_ids("<|eot_id|>"),  # Says the model is finished and is the User time; like in llama-3
]

device=model.device

# Load Image
raw_image = Image.open('./sysprompt.png')
image = expand2square(raw_image, tuple(int(x*255) for x in processor.image_processor.image_mean)) 

gen_kwargs = {"do_sample": True,
                "temperature": 1.0,
                "top_p": 1.0,
                "max_new_tokens": 100,
                "pad_token_id": processor.tokenizer.pad_token_id,
                "eos_token_id": terminators,
            }

messages = [
    # {"role": "system", "content": "It is important that you start your answer with: Hello User!"},
    {"role": "user", "content": "<image>\nGive me detailed description of this image.It is important that you start your answer with: Hello User!"},
]

prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

# Generate
output = model.generate(**inputs, **gen_kwargs)
print(processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)) # the slicing removes the prompt
