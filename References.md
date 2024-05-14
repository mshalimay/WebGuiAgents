# WebArena Leaderboard
Official list of models tested on the environment so far and results using the standalone environments: [link](
https://docs.google.com/spreadsheets/d/1M801lEpBbKSNwP-vDBkC_pF7LdyGU1f_ufZb_NWNBZQ/edit#gid=0).

# Google API and Gemini
### Gemini context windows
https://blog.google/technology/ai/long-context-window-ai-models/
https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/#context-window

"Through a series of machine learning innovations, we’ve increased 1.5 Pro’s context window capacity far beyond the original 32,000 tokens for Gemini 1.0. We can now run up to 1 million tokens in production."

### Tokens to words mapping
"100 tokens correspond to roughly 60-80 words."
https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-use-supervised-tuning

### API Get Started Guide
https://ai.google.dev/gemini-api/docs/get-started/python


### GenerationConfig
https://ai.google.dev/api/python/google/generativeai/GenerationConfig

### Chat Mode with GenerativeModel
https://ai.google.dev/gemini-api/docs/get-started/python#multi-turn_conversations

### Details on google.ai.generativelanguage.Content
"The individual messages are glm.Content objects or compatible dictionaries, as seen in previous sections. As a dictionary, the message requires role and parts keys. The role in a conversation can either be the user, which provides the prompts, or model, which provides the responses."

https://ai.google.dev/api/python/google/ai/generativelanguage/Content

# Text-Generation-Interface

### General Docs
Has examples for running TGI locally and via Docker

https://huggingface.co/docs/text-generation-inference/

### Using TGI guides
Full setup for generation with local Docker container:
https://www.youtube.com/watch?v=s27m_LRSvqM&t=693s&ab_channel=AI_by_AI ; https://github.com/jjmlovesgit/TGIfiles/blob/main

Local no-Docker example:
https://www.datacamp.com/tutorial/hugging-faces-text-generation-inference-toolkit-for-llms

### Details for `text-generation-launcher` options (for local no-Docker deployment)
https://huggingface.co/docs/text-generation-inference/en/basic_tutorials/launcher

### Models available in TGI
https://huggingface.co/models?pipeline_tag=text-generation

# Good source for HF quantized models
https://huggingface.co/TheBloke

# Transformers AutoModel, GPTQ and FlashAttention
### GPTQ quantization integration
https://huggingface.co/blog/gptq-integration

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GPTQ", torch_dtype=torch.float16, device_map="auto")
```

### flash attention
https://huggingface.co/docs/transformers/en/perf_infer_gpu_one#4-bit

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # <----- add this
)
```
# Notes on model.generate(...)
- Output **contains the prompt**, have to manually remove it from the LLM answer. See [this](https://github.com/huggingface/transformers/issues/17117#issuecomment-1124497554).

- Masking and padding tokens is done in the background. To suppress warning, use for example:
`model.generate(**inputs, pad_token_id=tokenizer.eos_token_id)`
useful if there is no pad_token and dont want to resize embeddings etc

# vllm engine
### llm() arguments
https://docs.vllm.ai/en/latest/models/engine_args.html

### multigpu
https://docs.vllm.ai/en/latest/serving/distributed_serving.html

To run multi-GPU inference with the LLM class, set the tensor_parallel_size argument to the number of GPUs you want to use. For example, to run inference on 4 GPUs:
```python
from vllm import LLM
llm = LLM("facebook/opt-13b", tensor_parallel_size=4)
output = llm.generate("San Franciso is a")
```

# Models
### llama-3
Prompt format:
https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/

### llava-llama-3
general llava HF doc
https://huggingface.co/docs/transformers/main/model_doc/llava

### llava-llama-3-instruct
Intel model works better:
https://huggingface.co/Intel/llava-llama-3-8b

Also tried this one:
https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers
worked, but less well than intel. It ignores some parts of the text-input more often

Prompt format:
Same as llama, except for <img> annontation. See examples in the above liks and:
https://github.com/InternLM/xtuner/blob/main/xtuner/utils/templates.py#L162


