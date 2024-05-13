
#===========================================================================
# NOTE Experimentation, intel model
#===========================================================================

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForPreTraining
import transformers

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



# def add_model_a_to_b(model_a, model_b):
#     state_dict_a = model_a.state_dict()
#     state_dict_b = model_b.state_dict()

#     # Ensure keys match before subtraction
#     if set(state_dict_a.keys()) != set(state_dict_b.keys()):
#         raise ValueError("Model state dicts do not have the same keys.")

#     for key in state_dict_a:
#         if state_dict_a[key].shape != state_dict_b[key].shape:
#             raise ValueError(f"Shape mismatch for key '{key}': {state_dict_a[key].shape} vs {state_dict_b[key].shape}")
#         # Subtract model_a's weights from model_b for the matching key
#         state_dict_b[key] = state_dict_b[key] + state_dict_a[key]
#     # Update model_b with the new weights
#     model_b.load_state_dict(state_dict_b)

# from transformers.file_utils import default_cache_path
# output_checkpoint = f"{default_cache_path}models--Intel--llava-llama-3-merged-8b" # set if you don't want to merge every time
# hf_checkpoint = "Intel/llava-llama-3-8b"

# processor = AutoProcessor.from_pretrained(hf_checkpoint)
# model = AutoModelForPreTraining.from_pretrained(hf_checkpoint)

# if model.language_model.model.embed_tokens.weight[-1].sum() == 0:
#     print("adding llama3 weights")
#     model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
#     pipeline = transformers.pipeline(
#         "text-generation",
#         model=model_id,
#         model_kwargs={"torch_dtype": torch.bfloat16},
#         device_map="cpu",
#         low_cpu_mem_usage=True,
#     )
#     llama3 = pipeline.model
#     add_model_a_to_b(llama3, model.language_model)
#     if output_checkpoint:
#         print("saving weights, so no adding is needed again")
#         model.save_pretrained(output_checkpoint)