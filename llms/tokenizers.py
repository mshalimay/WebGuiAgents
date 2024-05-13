from typing import Any

import tiktoken
from transformers import AutoTokenizer, LlamaTokenizer
# 3
class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        if provider == "openai":
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        elif provider == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # REVIEW[mandrade]: I believe the below is for trimming the observation. 
            # But it don't works for all tokenizers; special tokens are still added
            # and it prevents using the tokenizer for other purposes.
            # I added a line of code in trimming for preventing the special tokens to be added.

            # turn off adding special tokens automatically
            # self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
            # self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
            # self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]
        elif 'google' in provider.lower():
            # NOTE: @modif - gemini takes text input directly
            self.tokenizer = None
    
        else:
            raise NotImplementedError

    def encode(self, text: str, add_special_tokens=True) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: list[int], skip_special_tokens=False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)
