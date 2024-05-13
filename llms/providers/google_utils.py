#NOTE @modif

import logging
import os
import random
import time
from typing import Any
import PIL
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core.exceptions import InternalServerError, ResourceExhausted
import json

global google_model
def define_google_model(model_str: str):
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    global google_model
    google_model = genai.GenerativeModel(model_str)

def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 1,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple[Any] = (ResourceExhausted, InternalServerError),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:

                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e
    return wrapper


@retry_with_exponential_backoff
def generate_from_google_completion(
    prompt: list[str | PIL.Image.Image],
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    stop_sequences: list[str] | None = None,
    conversation_file: str | None = None,
    task_id:int | None = None,
    top_k:int | None=None
) -> str:

    # TODO: experiment with safety settings?
    safety_config = {
        # HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    # https://ai.google.dev/api/python/google/generativeai/GenerationConfig
    gen_kwargs = {
        'candidate_count':1,
        'max_output_tokens':max_new_tokens,
        'top_p':top_p,
        'temperature':temperature,
        'stop_sequences':stop_sequences,
    }
    if top_k is not None and top_k > 0: # Gemini models default to None, despite the documentation saying 40
        gen_kwargs['top_k'] = top_k

    gen_config = genai.types.GenerationConfig(**gen_kwargs)

    response = google_model.generate_content(
        prompt,
        safety_settings=safety_config,
        generation_config=gen_config,
    )
    answer = response.text

    # Save conversation to file
    if task_id is not None:
        with open(f'{conversation_file}_{task_id}.txt', "a") as f:
            f.write('----------------------------------\n')
            f.write('PROMPT' + "\n")
            f.write('----------------------------------\n')
            for p in prompt:
                if isinstance(p, str):
                    f.write(p + "\n")
                else:
                    f.write('Image' + "\n")
            f.write('----------------------------------\n')
            f.write('GENERATION' + "\n")
            f.write('----------------------------------\n')
            f.write(answer + "\n\n\n")
            
    return answer

def wrap_system_prompt(prompt:str, system_init:str = "System Prompt:\n", system_end:str = "", marker:str = "***"):
    """Ex.: wrap_system_prompt("Hello?", "System Prompt:\n", "", "***") -> "***System Prompt:\nHello?***"""
    return f"{marker}{system_init}{prompt}{system_end}{marker}"
    