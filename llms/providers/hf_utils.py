import PIL
from text_generation import Client  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration # type: ignore
import torch
from vllm import LLM, SamplingParams

global hf_model, hf_tokenizer, hf_processor, model_name


# TODO: check if needs special treatment because of 'quant'
def define_hf_model(model:str, model_path: str, vlmodel:bool, tokenizer_path:str, 
                    quant:str=None, engine:str='automodel', flash_attn:bool=True,
                    max_model_len:int=4096, num_gpus=1, dtype:str='auto'):

    torch.cuda.empty_cache()
    global hf_model, hf_tokenizer, model_name
    model_name = model.lower()

    # Load Tokenizer/Processor
    if 'llava' in model_name:
        global hf_processor
        hf_processor = AutoProcessor.from_pretrained(tokenizer_path)
        hf_tokenizer = hf_processor.tokenizer
    else:
        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load the model
    if engine=='vllm': # Deploys with VLLM engine
        print('Deploying HF model via VLLM engine.')
        hf_model = LLM(model=model_path, dtype=dtype, tokenizer=tokenizer_path,
                       gpu_memory_utilization=.98, max_model_len=max_model_len, 
                       tensor_parallel_size=num_gpus)

    elif engine=='automodel': # Deploys with Transformers AutoModel
        print('Deploying HF model via Transformers AutoModel.')
            
        kwargs = {
            'torch_dtype': dtype,
            'device_map': 'auto',
        }
        if flash_attn: kwargs.update({'attn_implementation': 'flash_attention_2'})

        # TODO: move this to YAML file and pass to the function via run.py instead
        if 'llama-3' in model_name: kwargs['torch_dtype'] = torch.bfloat16

        # Load the model
        if vlmodel:
            if 'llava' in model_name:
                hf_model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype, device_map='cuda:0')
            else:
                raise NotImplementedError(f"Model {model_name} not inplemented yet")
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        hf_model.eval()

def automodel_generate(model, tokenizer, prompt, gen_kwargs):
    device = model.device

    # Get inputs for the specific model
    if 'llava-llama-3' in model_name and 'instruct' in model_name:
        inputs = hf_processor(prompt[0], prompt[1], return_tensors='pt')
    else:
        inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Set padding token to suppress warning messages in generation
    if tokenizer.pad_token_id is not None:
        gen_kwargs.update({'pad_token_id': tokenizer.pad_token_id})
    else:
        gen_kwargs.update({'pad_token_id': tokenizer.eos_token_id})

    # Generate output tokens
    if hasattr(inputs, 'shape'):
        outputs = model.generate(inputs.to(device), **gen_kwargs)
    else:
        outputs = model.generate(**(inputs.to(device)), **gen_kwargs)

    # Index to trim the prompt from the output
    if hasattr(inputs, 'input_ids'):
        trim_prompt_idx = inputs.input_ids.shape[1]
    elif hasattr(inputs, 'shape'):
        trim_prompt_idx = inputs.shape[1]
    else:
        trim_prompt_idx = 0
        print("WARNING: Could not determine input shape. Generated answer may include the prompt")

    # Decode the output tokens. The slicing removes the prompt
    return tokenizer.decode(outputs[0][trim_prompt_idx:], skip_special_tokens=True) 

def generate_from_huggingface_completion(
    prompt: str | list[str] | list[list[str], list[PIL.Image.Image]],
    model_endpoint: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_sequences: list[str] | None = None,
    conversation_file: str | None = None,
    task_id:int | None = None,
    local: bool = False,
    engine: str = 'automodel'
) -> str:
    generation: str

    # vllm engine
    if engine=='vllm':
        gen_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "stop": stop_sequences,
            'stop_token_ids': get_terminators(model_name)
        }
        sampling_params = SamplingParams(**gen_kwargs)
        generation = hf_model.generate(prompt, sampling_params)[0].outputs[0].text

    # `Transformers` AutoModel
    elif engine=='automodel': 
        do_sample = True if (temperature != 0 or top_p>0) else False
        gen_kwargs = {
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "eos_token_id": get_terminators(model_name, stop_seqs=stop_sequences),
        }
        generation = automodel_generate(hf_model, hf_tokenizer, prompt, gen_kwargs)

    # TGI
    elif engine=='tgi':
        terminators = get_terminators(model_name, tgi=True, stop_sequences=stop_sequences)
        client = Client(model_endpoint, timeout=180)
        generation = client.generate(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            stop_sequences=terminators,
        ).generated_text

    else:
        raise ValueError(f"Engine {engine} not supported")

    # Write conversation to file
    if task_id is not None:
        if isinstance(prompt, list): # list[str] or list[list[str], list[PIL.Image.Image]]
            messages = prompt[0] if isinstance(prompt[0], (list, str)) else prompt
            if isinstance(messages, str):
                messages = [messages]
        elif isinstance(prompt, str):
            messages = [prompt]
        
        with open(f'{conversation_file}_{task_id}.txt', "a") as f:
            f.write('----------------------------------\n')
            f.write('PROMPT' + "\n")
            f.write('----------------------------------\n')
            for m in messages:
                if isinstance(m, str):
                    f.write(m + "\n")
            f.write('----------------------------------\n')
            f.write('GENERATION' + "\n")
            f.write('----------------------------------\n')
            f.write(generation + "\n\n\n\n")
    return generation


def generate_from_huggingface_fuzzy_match(
    messages: list[dict],
    model_endpoint: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    stop_sequences: list[str] | None = None,
    conversation_file: str | None = None,
    task_id:int | None = None,
    local: bool = False,
    engine: str = 'automodel',
    model:str = 'meta-llama/Meta-Llama-3-8B-Instruct',
) -> str:
    generation: str
    if 'hf_model' not in globals() and 'hf_model' not in locals():
        define_hf_model(model=model, vlmodel=False, model_path=model, tokenizer_path=model, quant=None, 
                        engine='automodel', max_model_len=None, num_gpus=1, flash_attn=True, dtype='auto')

    prompt = create_llama3_chat_input(messages, hf_tokenizer, engine)

    generation = generate_from_huggingface_completion(prompt, model_endpoint, temperature, top_p, max_new_tokens, stop_sequences, conversation_file, task_id, local, engine)
    return generation
    



def create_llama3_chat_input(messages:list[dict], tokenizer, engine:str) -> str:
    # https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
        
    input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # If deploying with VLLM or TGI, use the text string directly
    if engine == 'vllm' or engine == 'tgi':
        return input
    # Transformers AutoModel uses the embedding; need to remove <|begin_of_text|> or else will get duplicated when encoding.
    elif engine == 'automodel': 
        return input.split("<|begin_of_text|>")[1]
        #OBS: # This is just to keep the prompt creation in string format for all deployment modes;
        # could return the tokenIDs directly to use in Transformers AutoModel
    else:
        raise ValueError(f"Engine {engine} not supported.")
    


def get_terminators(model_name:str, tgi:bool=False, stop_seqs=None):
    if 'llama-3' in model_name and 'instruct' in model_name:
        # Obs: this intentionally includes llava-llama-3-8b-instruct
        terminators = [
            hf_tokenizer.eos_token_id,
            hf_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    else: 
        terminators = [hf_tokenizer.eos_token_id,]

    if stop_seqs is not None:
        terminators.extend(hf_tokenizer.encode(stop_seqs, add_special_tokens=False))

    if tgi:
        # TGI use strings
        terminators = [hf_tokenizer.convert_ids_to_tokens(token_id) for token_id in terminators]

    return terminators
    



def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = PIL.Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = PIL.Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

    
def transform_imgs_llava3_intel(imgs):
    new_imgs = []
    for img in imgs:
        image = expand2square(img, tuple(int(x*255) for x in hf_processor.image_processor.image_mean)) 
        new_imgs.append(image)
    return new_imgs