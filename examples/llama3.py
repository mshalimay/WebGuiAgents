"""Experimentation and examples of generation with LLAMA3"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


# Visualize prompt template format in multi-turn chat
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
    {"role": "assistant", "content": "I am a chatbot. How can I help you?"},
    {"role": "user", "content": "Please tell me how many stars there are in the sky."},
]

# output are token IDs
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

print(tokenizer.decode(input_ids[0], skip_special_tokens=False))

# Output is a direct string
input = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
print(input)

# remove <|begin_of_text|> from string
str_prompt = input.split("<|begin_of_text|>")[1]
input_ids2 = tokenizer.encode(str_prompt, return_tensors="pt").to(model.device)
input_ids2 - input_ids

# show special tokens map
tokenizer.special_tokens_map

# Add pad_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = "<pad>"
    model.resize_token_embeddings(len(tokenizer))

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))



#-----------------------------------------------------------
# Tests with the Tokenizer
#-----------------------------------------------------------

#-----------------------------------------------------------    
# Testing the add_special_tokens, add_bos_token, add_eos_token False settings
#-----------------------------------------------------------    

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.special_tokens_map


obs = """OBSERVATION:
[1744] link 'HP CB782A#ABA 640 Inkjet Fax Machine (Renewed)'
		[1749] StaticText '$279.49'
		[1757] button 'Add to Cart'
		[1760] button 'Add to Wish List'
		[1761] button 'Add to Compare'
URL: http://onestopmarket.com/office-products/office-electronics.html
OBJECTIVE: What is the price of HP Inkjet Fax Machine"""

# Normal encoding:
tokenizer.decode(tokenizer.encode(obs), skip_special_tokens=False)
# "<|begin_of_text|>OBSERVATION:\n[1744] link 'HP CB782A#ABA 640 Inkjet Fax Machine (Renewed)'\n\t\t[1749] StaticText '$279.49'\n\t\t[1757] button 'Add to Cart'\n\t\t[1760] button 'Add to Wish List'\n\t\t[1761] button 'Add to Compare'\nURL: http://onestopmarket.com/office-products/office-electronics.html\nOBJECTIVE: What is the price of HP Inkjet Fax Machine"


# Try to set these to false dont have effect
tokenizer.add_special_tokens = False
tokenizer.add_bos_token = False
tokenizer.add_eos_token = False
tokenizer.decode(tokenizer.encode(obs), skip_special_tokens=False)
# "<|begin_of_text|>OBSERVATION:\n[1744] link 'HP CB782A#ABA 640 Inkjet Fax Machine (Renewed)'\n\t\t[1749] StaticText '$279.49'\n\t\t[1757] button 'Add to Cart'\n\t\t[1760] button 'Add to Wish List'\n\t\t[1761] button 'Add to Compare'\nURL: http://onestopmarket.com/office-products/office-electronics.html\nOBJECTIVE: What is the price of HP Inkjet Fax Machine"

# This works:
tokenizer.decode(tokenizer.encode(obs, add_special_tokens=False), skip_special_tokens=False)


prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
tokenizer.add_special_tokens = True
tokenizer.add_bos_token = True  
tokenizer.add_eos_token = True
encoded_before = tokenizer.encode(prompt)


tokenizer.add_special_tokens = False 
tokenizer.add_bos_token = False  
tokenizer.add_eos_token = False
encoded_after = tokenizer.encode(prompt)

# DIfferent generations, still retain start_header_id, but removes begin_of_text, etc (see special_tokens_map)
tokenizer.decode(encoded_before)
tokenizer.decode(encoded_after)


tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


