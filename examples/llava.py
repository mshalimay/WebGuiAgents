"""Experimentation and examples of generation with llava-llama3 from xtuner"""


#-----------------------------------------------------------
import torch
from transformers import (
    AutoProcessor,
    LlavaProcessor,
    LlavaForConditionalGeneration,
    AutoTokenizer
)
from PIL import Image

# Load model and processor
model_id = "xtuner/llava-llama-3-8b-v1_1-transformers"



processor = AutoProcessor.from_pretrained(model_id)
# tokenizer = AutoTokenizer.from_pretrained(model_id) # Tokenizer loaded is the same as with Processor

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,  
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,).to("cuda:0")

# Load Image
raw_image = Image.open('./sysprompt.png')

# visualize the special tokens already in the processor
processor.tokenizer.special_tokens_map


#-----------------------------------------------------------
# generation using terminators - correct way
#-----------------------------------------------------------
terminators = [
    processor.tokenizer.eos_token_id,
    processor.tokenizer.convert_tokens_to_ids("<|eot_id|>"),  # Says the model is finished and is the User time; like in llama-3
]

gen_kwargs = {
                "do_sample": False,
                # "temperature": 0.0,
                # "top_p": 0.0,
                "max_new_tokens": 100,
                "pad_token_id": processor.tokenizer.pad_token_id,
                "eos_token_id": terminators,
            }

messages = [
    {"role": "user", "content": "<image>\nGive me detailed description of this image. It is important that you start your answer with: Hello User!"},
]

prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)
output = model.generate(**inputs, **gen_kwargs)
print(processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=False))
# The image is a screenshot of a webpage, specifically a user manual for a website. The webpage is divided into two main sections: a header and a body. The header is a simple line of text that reads "The current web page's URL is: <http://www.example.com>".
# The body of the webpage is a list of instructions, written in a clear, black font against a white background. The instructions are organized into three main sections: "Page Operation Actions", "Tab Management Actions", and "URL Navigation Actions". Each section contains a series of steps, written in a bullet point format. The steps are numbered and indented, making it easy to follow along.
# The text of the instructions is black, which stands out against the white background of the webpage. The layout is simple and straightforward, making it easy for users to understand and follow the instructions. The webpage appears to be designed for ease of use, with clear and concise instructions for navigating the website.<|eot_id|>

#-----------------------------------------------------------
# #NOTE generation using system prompt
#-----------------------------------------------------------
# NOTE: system prompt is ignored both in this finetuned model and Intel's model

# Combines with llama-3 prompting
# https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/

template = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n<image>\n{user_prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")

terminators = [
    processor.tokenizer.eos_token_id,
    processor.tokenizer.convert_tokens_to_ids("<|eot_id|>"),  # Says the model is finished and is the User time; like in llama-3
]

gen_kwargs = {"do_sample": True, "temperature": 1.0, "top_p": 1.0,
                "max_new_tokens": 200, "pad_token_id": processor.tokenizer.pad_token_id,
                "eos_token_id": terminators,}

prompt = template.format(
    system_prompt="Start your answer with: 'Hello user, I hope you are having a wonderful day.'",
    user_prompt="Give me detailed description of this image.",
)

prompt_no_sys = template_nosysprompt.format(
    user_prompt="Give me detailed description of this image. Start your answer with: 'Hello, I hope you are having a wonderful day'",
)

inputs = processor(prompt_no_sys, raw_image, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, **gen_kwargs)
print(processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)) # the slicing removes the prompt

#-----------------------------------------------------------
#   Generation following original example:  https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers
#   Obs: I already tested that the `prompt` from the example matches the template below
#-----------------------------------------------------------
template_nosysprompt = ("<|start_header_id|>user<|end_header_id|>\n\n<image>\n{user_prompt}<|eot_id|>"
                        "<|start_header_id|>assistant<|end_header_id|>\n\n")
prompt = template_nosysprompt.format(
    user_prompt="Give me detailed description of this image.",
)

inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

gen_kwargs = {
                "do_sample": False,
                # "temperature": 0.0,
                # "top_p": 0.0,
                "max_new_tokens": 300,
                "pad_token_id": processor.tokenizer.pad_token_id,
            }

output = model.generate(**inputs, **gen_kwargs)
print(processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)) # the slicing removes the prompt

#FIXME: PROBLEM: not including terminators leads to undesired behavior:
# The image is a screenshot of a webpage, specifically a user manual for a website. The webpage is divided into two main sections: a header and a body. The header is a simple line of text that reads "The current web page's URL is: <http://www.example.com>".
# The body of the webpage is a list of instructions, written in a clear, black font against a white background. The instructions are organized into three main sections: "Page Operation Actions", "Tab Management Actions", and "URL Navigation Actions". Each section contains a series of steps, written in a bullet point format. The steps are numbered and indented, making it easy to follow along.
# The text of the instructions is black, which stands out against the white background of the webpage. The layout is simple and straightforward, making it easy for users to understand and follow the instructions. The webpage appears to be designed for ease of use, with clear and concise instructions for navigating the website.<|eot_id|><|eot_id|>user example.com<|eot_id|><|eot_id|> www.example.com<|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|> www.example.com<|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|> www.example.com<|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|> www.example.com<|eot_id|><|eot_id|><|eot_id|><|eot_id|><|eot_id|> www.example.com<|eot_id|><|eot_id|><|eot_id|> www.example.com<|eot_id|><|eot_id|> www.example.com<|eot_id|><|eot_id|> www.example.com<|eot_id|> www.example.com<|eot_id|> www.example.com<|eot_id|> www.example


#----------------------------------------------------------------
# Using LLAMA-3 tokenizer to create prompt
#----------------------------------------------------------------
from transformers import AutoTokenizer
llama3_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")


# Manual Prompt
template = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n<image>\n{user_prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")

manual_prompt = template.format(
    system_prompt="You are an intelligent model that excels on following text-based instructions and analyzing images.",
    user_prompt="Give me detailed description of this image.",
)

# Llama 3 prompt:
messages = [
    {"role": "system", "content": "You are an intelligent model that excels on following text-based instructions and analyzing images."},
    {"role": "user", "content": "<image>\nGive me detailed description of this image."},
]
llama3_prompt = llama3_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
assert llama3_prompt == manual_prompt
llama3_prompt


processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
llama3_prompt == tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


#---------------------------------------------------------------------------------
# Tests with <image> coming after 
#---------------------------------------------------------------------------------



# Load model and processor
model_id = "xtuner/llava-llama-3-8b-v1_1-transformers"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,  
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,).to("cuda:0")

# Load Image
raw_image = Image.open('./sysprompt.png')


# Create prompt
template = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
            # "<|start_header_id|>user<|end_header_id|>\n\n<image>\n{user_prompt}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}\n<image><|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")
prompt = template.format(
    system_prompt="You are an intelligent model that excels on following text-based instructions and analyzing images.",
    user_prompt="Give me detailed description of this image:",
)
print(prompt)

# Encode input
inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

terminators = [
    processor.tokenizer.eos_token_id,
    processor.tokenizer.convert_tokens_to_ids("<|eot_id|>"),  # Says the model is finished and is the User time; like in llama-3
]

gen_kwargs = {
                "do_sample": False,
                # "temperature": 0.0,
                # "top_p": 0.0,
                "max_new_tokens": 300,
                "pad_token_id": processor.tokenizer.pad_token_id,
                "eos_token_id": terminators,
            }




output = model.generate(**inputs, **gen_kwargs)
print(processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)) # the slicing removes the prompt




url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

#original llava pads with mean, HF llava pads with zeros
image = expand2square(image, tuple(int(x*255) for x in processor.image_processor.image_mean)) 
inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
# Generate
generate_ids = model.generate(**inputs, max_length=30)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)

