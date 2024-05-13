import argparse
from typing import Any
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import PIL
import IPython.display

# https://github.com/google-gemini/generative-ai-python/blob/v0.3.0/google/generativeai/types/content_types.py#L53
# image types supported by Gemini
IMAGE_TYPES=(PIL.Image.Image, IPython.display.Image)

from llms import (
    generate_from_huggingface_completion,
    # generate_from_openai_chat_completion,
    # generate_from_openai_completion,
    generate_from_google_completion, #NOTE @modif
    lm_config,
)

APIInput = str | list[Any] | dict[str, Any]


def call_llm(
    lm_config: lm_config.LMConfig,
    prompt: APIInput,
    task_id: str | None = None,
) -> str:
    response: str
    if lm_config.provider == "openai":
        if lm_config.mode == "chat":
            assert isinstance(prompt, list)
            response = generate_from_openai_chat_completion(
                messages=prompt,
                model=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                top_p=lm_config.gen_config["top_p"],
                context_length=lm_config.gen_config["context_length"],
                max_tokens=lm_config.gen_config["max_tokens"],
                stop_token=None,
            )
        elif lm_config.mode == "completion":
            assert isinstance(prompt, str)
            response = generate_from_openai_completion(
                prompt=prompt,
                engine=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                max_tokens=lm_config.gen_config["max_tokens"],
                top_p=lm_config.gen_config["top_p"],
                stop_token=lm_config.gen_config["stop_token"],
            )
        else:
            raise ValueError(
                f"OpenAI models do not support mode {lm_config.mode}"
            )
    elif lm_config.provider == "huggingface":
        # assert isinstance(prompt, str)
        response = generate_from_huggingface_completion(
            prompt=prompt,
            model_endpoint=lm_config.gen_config["model_endpoint"],
            temperature=lm_config.gen_config["temperature"],
            top_p=lm_config.gen_config["top_p"],
            stop_sequences=lm_config.gen_config["stop_sequences"],
            max_new_tokens=lm_config.gen_config["max_new_tokens"],
            conversation_file=lm_config.conversation_file,
            task_id=task_id,
            local=lm_config.local,
            engine=lm_config.engine,
        )
    elif lm_config.provider == 'google':
        assert isinstance(prompt, list)

        #TODO create assert statment to verify correctness of input
        if lm_config.mode=='chat':    
            pass
        elif lm_config.mode=='completion':
            # assert all([isinstance(p, str) or isinstance(p, IMAGE_TYPES) for p in prompt])
            pass
        else:
            raise ValueError(
                f"Google models do not support mode {lm_config.mode}"
            )
            
        #NOTE: @modif
        response = generate_from_google_completion(
            prompt=prompt,
            temperature=lm_config.gen_config["temperature"],
            top_p=lm_config.gen_config["top_p"],
            stop_sequences=lm_config.gen_config["stop_sequences"],
            max_new_tokens=lm_config.gen_config["max_new_tokens"],
            conversation_file=lm_config.conversation_file,
            task_id=task_id,
            top_k=lm_config.gen_config["top_k"],
        )
    else:
        raise NotImplementedError(
            f"Provider {lm_config.provider} not implemented"
        )

    return response
