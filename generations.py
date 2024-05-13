"""Different examples on how to generate text with Hugging Face"""

#-------------------------------------------------------------------------------
## generate with tgi
#-------------------------------------------------------------------------------
import os
from text_generation import Client
import torch
client = Client("http://127.0.0.1:8080", timeout=60)
prompt = "How many stars ther are in the sky?"
torch.manual_seed(0)
temperature = 1.0
top_p = 0.9
max_new_tokens = 200

x = client.generate(
        prompt="How many stars ther are in the sky?",
        do_sample=False,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    ).generated_text

print(x)


#-------------------------------------------------------------------------------
## generate with pipeline and model.generate
#-------------------------------------------------------------------------------
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import os
os.environ['HF_TOKEN'] = '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'
torch.cuda.empty_cache()
model_path="meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, 
                            device_map="auto", attn_implementation="flash_attention_2",)

device = model.device
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]


torch.manual_seed(0)
prompt = "How many stars ther are in the sky?"

do_sample=True
temperature=1.0
top_p=.9
max_new_tokens=200
device = model.device

generate_kwargs = {
    "do_sample": True,
    "temperature": temperature,
    "top_p": top_p,
    "max_new_tokens": max_new_tokens,
    'return_dict_in_generate':False
}


inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

outputs = model.generate(inputs, **generate_kwargs)

outputs = model.generate(inputs, max_new_tokens=max_new_tokens, 
                         temperature=temperature, top_p=top_p, 
                         do_sample=do_sample, return_dict_in_generate=False)

generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True) # slicing removes the prompt

# using pipeline - dont generate same results as Client or manual generation
text_generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, framework="pt")
t = text_generator("How many stars ther are in the sky?", **generate_kwargs)
print(t)

#-------------------------------------------------------------------------------
## generate with vllm
#-------------------------------------------------------------------------------

from vllm import LLM, SamplingParams
import torch
model_path="meta-llama/Meta-Llama-3-8B"

prompt = "How many stars ther are in the sky?"
torch.manual_seed(0)
temperature = 1.0
top_p = 0.9
max_new_tokens = 200


torch.cuda.empty_cache()
llm = LLM(model=model_path, dtype='auto', gpu_memory_utilization=.98, max_seq_len_to_capture=4000, max_model_len=4000)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_new_tokens, seed=0)


z = llm.generate(prompt, sampling_params)

z[0].outputs[0].text
# clean GPU memory


#-------------------------------------------------------------------------------
## VLM generation - gemini-pro-vision
#-------------------------------------------------------------------------------
import google.generativeai as genai
import PIL, os

os.environ['GOOGLE_API_KEY']="AIzaSyAI0XvTWHzl2ckpwQWMRuHNuaYlS1zrcGs"
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


model = genai.GenerativeModel('models/gemini-1.5-pro-latest')   

model.generate_content("Hello, are you there?")

img = PIL.Image.open('sysprompt.png')
text_prompt = 'Do you think the instructions given in the image are in a proper format to prompt you? Why/Why not?'
model.generate_content([text_prompt, img])


# Models available 
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)

# Models metadata
genai.get_model('models/gemini-1.5-pro-latest')



