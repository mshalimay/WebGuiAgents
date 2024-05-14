"""Combines intel llava-llama-3 model with meta's llama-3 weights for use. 
Saved to HF cache directory under models--Intel--llava-llama-3-merged-8b"""

from transformers import AutoProcessor, AutoModelForPreTraining
import transformers
import torch

def add_model_a_to_b(model_a, model_b):
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()

    # Ensure keys match before subtraction
    if set(state_dict_a.keys()) != set(state_dict_b.keys()):
        raise ValueError("Model state dicts do not have the same keys.")

    for key in state_dict_a:
        if state_dict_a[key].shape != state_dict_b[key].shape:
            raise ValueError(f"Shape mismatch for key '{key}': {state_dict_a[key].shape} vs {state_dict_b[key].shape}")
        # Subtract model_a's weights from model_b for the matching key
        state_dict_b[key] = state_dict_b[key] + state_dict_a[key]
    # Update model_b with the new weights
    model_b.load_state_dict(state_dict_b)

from transformers.file_utils import default_cache_path
output_checkpoint = f"{default_cache_path}models--Intel--llava-llama-3-merged-8b" # set if you don't want to merge every time
hf_checkpoint = "Intel/llava-llama-3-8b"

processor = AutoProcessor.from_pretrained(hf_checkpoint)
model = AutoModelForPreTraining.from_pretrained(hf_checkpoint)

if model.language_model.model.embed_tokens.weight[-1].sum() == 0:
    print("adding llama3 weights")
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    llama3 = pipeline.model
    add_model_a_to_b(llama3, model.language_model)
    if output_checkpoint:
        print("saving weights, so no adding is needed again")
        model.save_pretrained(output_checkpoint)