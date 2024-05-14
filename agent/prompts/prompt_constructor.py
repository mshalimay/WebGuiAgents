import json
import re
from pathlib import Path
from typing import Any, TypedDict

from browser_env import Action, ActionParsingError, Trajectory
from browser_env.env_config import URL_MAPPINGS
from browser_env.utils import StateInfo, pil_to_b64, pil_to_google
from llms import lm_config
from llms.tokenizers import Tokenizer
from llms.utils import APIInput
from PIL import Image
from llms.providers.google_utils import wrap_system_prompt
from llms.providers.hf_utils import transform_imgs_llava3_intel, create_llama3_chat_input



class Instruction(TypedDict):
    """Instruction for constructing prompt"""
    intro: str
    examples: list[tuple[str, str]]
    template: str
    meta_data: dict[str, Any]


class PromptConstructor(object):
    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        self.instruction_path = Path(instruction_path)
        self.obs_modality = "text"
        self.lm_config = lm_config
        instruction = json.load(open(self.instruction_path))
        instruction["examples"] = [tuple(e) for e in instruction["examples"]]
        self.instruction: Instruction = instruction
        self.tokenizer = tokenizer

    def get_lm_api_input(
        self, 
        intro: str, 
        examples: list[tuple[str, str]] | list[tuple[str, str, str]], 
        current: str,
        page_screenshot_img: Image.Image | None = None,
        images: list[Image.Image] | None = None,
    ) -> APIInput:
        """Return the require format for an API"""
        message: list[dict[str, str]] | str

        #================================================================================================
        #NOTE OpenAI provider
        #================================================================================================
        if "openai" in self.lm_config.provider:
            if self.lm_config.mode == "chat":
                message = [{"role": "system", "content": intro}]
                for (x, y) in examples:
                    message.append(
                        {
                            "role": "system",
                            "name": "example_user",
                            "content": x,
                        }
                    )
                    message.append(
                        {
                            "role": "system",
                            "name": "example_assistant",
                            "content": y,
                        }
                    )
                message.append({"role": "user", "content": current})
                return message
            elif self.lm_config.mode == "completion":
                message = f"{intro}\n\n"
                message += "Here are a few examples:\n"
                for example in examples:
                    message += f"Observation\n:{example[0]}\n\n"
                    message += f"Action: {example[1]}\n\n"
                message += "Now make prediction given the observation\n\n"
                message += f"Observation\n:{current}\n\n"
                message += "Action:"
                return message
            else:
                raise ValueError(
                    f"OpenAI models do not support mode {self.lm_config.mode}"
                )
        #================================================================================================
        #NOTE Huggingface provider
        #================================================================================================
        elif "huggingface" in self.lm_config.provider:
             #-----------------------
            # LLAMA-2
            #-----------------------

            if "llama-2" in self.lm_config.model.lower():
                # https://huggingface.co/blog/llama2#how-to-prompt-llama-2
                # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L320
                if self.lm_config.mode == "chat":
                    B_INST, E_INST = "[INST]", "[/INST]"
                    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
                    BOS, EOS = "<s>", "</s>"
                    # adding the system message to be the starting of the first example
                    examples = [
                        (
                            B_SYS + intro + E_SYS + examples[0][0],
                            examples[0][1],
                        )
                    ] + examples[1:]
                    message = "".join(
                        [
                            f"{BOS}{B_INST} {x.strip()} {E_INST} {y.strip()} {EOS}"
                            for (x, y) in examples
                        ]
                    )
                    # add the current observation
                    message += f"{BOS}{B_INST} {current.strip()} {E_INST} {self.instruction['meta_data'].get('force_prefix', '')}"

                    return message
                else:
                    raise ValueError("Only chat mode is supported for Llama-2")

            #-----------------------
            # LLAVA-LLAMA-3
            #-----------------------
            elif "llava-llama-3" in self.lm_config.model.lower():
                # Prompt Format https://github.com/InternLM/xtuner/blob/main/xtuner/utils/templates.py#L162
                # Obs: same as Llama-3, except for `<image>\n` before user text-input. See `llava.py` examples.

                # TODO  Additional mesage conditional on SOM or actree
                # TODO: The prompts have <image> before the text-input. I believe also works fine after the text-input, but have to test more.
                # OBS: Have to test how the model performs with multiple images. 
                # Per `Transformers`: "the model has not been explicitly trained to process multiple images in the same prompt, although this is technically possible, you may experience inaccurate results."
                # Here each image is in a different prompt. But nonetheless.

                # Chat mode
                if self.lm_config.mode == "chat":
                    if 'instruct' not in self.lm_config.model.lower():
                        print("WARNING: Llava-3 (no instruct) should use completion prompting. See https://github.com/meta-llama/llama3/tree/main")

                
                    #-------------------------------------
                    #  Add Introduction/system prompt
                    #-------------------------------------
                    # NOTE: finetuned models available until 2024-05-13 dont deal well with system prompt. Ignores it.
                    # This is why I use the 'user' and 'understood' workaround below.

                    add_sys_msg = f"\nThe next {len(examples)} dialogues are examples. Use them to base your response.\n\n"
                    # add_sys_msg=""

                    # message = [{"role": "system", "content": f"{intro}{add_sys_msg}"}]
                    message = [{"role": "user", "content": f"****INSTRUCTIONS: {intro}{add_sys_msg} ****"}]
                    message.append({"role": "assistant", "content": "Understood."})

                    #-------------------------------------
                    #  Add examples 
                    #-------------------------------------
                    add_user_msgs = ["This is the current webpage screenshot.",
                            "Below is the webpage's accessibility tree and URL, and the objective you have to complete:\n"]
                    wrap_content = (lambda x, img: f"<image>\n" + f"{' '.join(add_user_msgs)}{x}" if img is not None else f"{add_user_msgs[1]}{x}")
                    example_imgs = []
                    for example in examples:
                        x, y = example[:2]
                        if len(example) == 3:
                            example_img = Image.open(example[2])
                            example_imgs.append(example_img)
                        message.append({"role": "user", "content": wrap_content(x, example_img)})
                        message.append({"role": "assistant", "content": y})                        

                    #-------------------------------------
                    # Add Current observation 
                    #-------------------------------------
                    message.append({"role": "user", "content": wrap_content(current, page_screenshot_img)})

                    #-------------------------------------
                    # Transform inputs
                    #-------------------------------------
                    # Transform to llama-3 prompt format
                    str_message = create_llama3_chat_input(message, tokenizer=self.tokenizer.tokenizer, 
                                                           engine=self.lm_config.engine)

                    # Images input
                    if len(example_imgs) == 0 and page_screenshot_img is None:
                        print("WARNING: No images provided for Llava-3 chat mode. The model may not perform well.")
                        all_images=None
                    else:
                        all_images = []
                        if len(example_imgs) > 0 :
                            all_images.extend(example_imgs)
                        if page_screenshot_img is not None:
                            all_images.append(page_screenshot_img)

                        all_images = transform_imgs_llava3_intel(all_images)
                    return [str_message, all_images]

                # Completion mode
                else:
                    raise ValueError("TODO: completion mode for Llama-3")

            #-----------------------
            # LLAMA-3
            #-----------------------
            elif 'llama-3' in self.lm_config.model.lower():
                # Chat mode
                if self.lm_config.mode == 'chat':
                    if 'instruct' not in self.lm_config.model.lower():
                        print("WARNING: Llama-3 (no instruct) should use completion prompting. See https://github.com/meta-llama/llama3/tree/main")
                    # Add System Prompt / Intro
                    message = [{"role": "system", "content": intro}]

                    # Add Examples
                    for (x, y) in examples:
                        message.append({"role": "user", "content": x})
                        message.append({"role": "assistant", "content": y})

                    # Add Current Observation
                    message.append({"role": "user", "content": current})

                    # Transform to llama-3 prompt format
                    str_message = create_llama3_chat_input(message, tokenizer=self.tokenizer.tokenizer,
                                                              engine=self.lm_config.engine)
                    return str_message

                # Completion mode
                else:
                    raise ValueError("TODO: completion mode for Llama-3")

            else:
                raise ValueError(f"No support for {self.lm_config.model} yet. ")

        #================================================================================================
        #NOTE GEMINI
        #================================================================================================
        elif "google" in self.lm_config.provider:
            example_hint='\n\nNext I will show you some examples followed by the current information and objective to complete.'            

            # Completion Format
            if self.lm_config.mode == "completion":
                #-----------------------------------------------------
                #  Add Introduction/system prompt 
                #-----------------------------------------------------
                if self.lm_config.sys_prompt:
                    message=[wrap_system_prompt(f"{intro}{example_hint}", system_init="System Prompt:\n", system_end="", marker="***")]
                else:
                    message=[f"{intro}{example_hint}"]

                #-----------------------------------------------------
                #  Add examples 
                #-----------------------------------------------------
                # Visual Language Model
                if self.lm_config.vlm:  
                    for example in examples:
                        x, y = example[:2]
                        example_img = None if len(example) < 3 else Image.open(example[2])

                        message.append(f"{x}\n")              # Ex. accesibility tree
                        if example_img is not None:           # Ex. images
                            message.extend(
                                [
                                    "IMAGES:",
                                    "(1) current page screenshot:",
                                    pil_to_google(example_img),
                                ]
                            )
                        message.append(f"ACTION:\n {y}")      # Ex. actions

                # Text-only model
                else: 
                    for (x, y) in examples:
                        message.append(f"{x}\n")
                        message.append(f"ACTION:\n {y}")

                #-----------------------------------------------------
                # Add Current observation 
                #-----------------------------------------------------
                # Visual Language Model
                if self.lm_config.vlm:  
                    message.append(f"Observation: {current}\n")
                    message.extend(
                        [
                            "Images:",
                            "(1) current page screenshot:",
                            pil_to_google(page_screenshot_img)[0],
                        ]
                    )
                    for image_i, image in enumerate(images):
                        message.extend(
                            [
                                f"({image_i+2}) input image {image_i+1}",
                                (image),
                            ]
                        )
                    message.append("ACTION:")

                # Text-only model
                else: 
                    message.append(f"{current}\n")
                    message.append("ACTION:")

                return message

            # Chat Format
            elif self.lm_config.mode == "chat" and not self.lm_config.vlm:  # VLM model does not work well in chat model ref @google
                if self.lm_config.sys_prompt:
                    message=[{'role':'user', 'parts': wrap_system_prompt(f"{intro}{example_hint}", 
                                                        system_init="System Prompt:\n", system_end="", marker="***")}]
                    message.append({'role':'model', 'parts': "Understood." })                    
                else:
                    message=[{"role": 'user', "parts": f"{intro}{example_hint}"}]
                    message.append({'role':'model', 'parts': "Understood." }) 

                # Examples
                for (x, y) in examples:
                    message.append({"role": 'user', "parts": x,})
                    message.append({"role": 'model', "parts": y,})

                message.append({"role": 'user', "parts": current})

                return message
            else:
                raise ValueError(
                    f"Gemini models do not support mode {self.lm_config.mode}"
                )
        else:
            raise NotImplementedError(
                f"Provider {self.lm_config.provider} not implemented"
            )

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        raise NotImplementedError

    def map_url_to_real(self, url: str) -> str:
        """Map the urls to their real world counterparts"""
        for i, j in URL_MAPPINGS.items():
            if i in url:
                url = url.replace(i, j)
        return url

    def map_url_to_local(self, url: str) -> str:
        """Map the urls to their local counterparts"""
        for i, j in URL_MAPPINGS.items():
            if j in url:
                url = url.replace(j, i)
            # https
            if j.replace("http", "https") in url:
                url = url.replace(j.replace("http", "https"), i)
        return url

    def _extract_action(self, response: str) -> str:
        raise NotImplementedError

    def extract_action(self, response: str) -> str:
        response = self._extract_action(response)
        response = self.map_url_to_local(response)
        return response


class DirectPromptConstructor(PromptConstructor):
    """The agent will direct predict the action"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        """Construct prompt given the trajectory"""
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            if self.lm_config.provider == "google":
                obs = obs[:max_obs_length]
            else:
                # REVIEW[mandrade]: modified to correctly trim the observation. See comments in Tokenizer
                obs = self.tokenizer.decode(self.tokenizer.encode(obs, add_special_tokens=False)[:max_obs_length], 
                                            skip_special_tokens=False)  # type: ignore[arg-type]


        page = state_info["info"]["page"]
        url = page.url
        previous_action_str = meta_data["action_history"][-1]

        # input x
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )

        # make sure all keywords are replaced
        assert all([f"{{k}}" not in current for k in keywords])
        prompt = self.get_lm_api_input(intro, examples, current)
        return prompt

    def _extract_action(self, response: str) -> str:
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        # REVIEW[mandrade]: problem: gets only the first ocurrence before the action splitter.
        # If the llm answer something with ``` then a correct action ```action ...```, will parse the wrong part.
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"

        # match = re.search(pattern, response)
        # if match:
        #     return match.group(1).strip()

        # Modification: get the last ocurrence of all matches, as it is instructed for LLM to return action last.
        # Alternatively: modify the action splitter to an even more rare string. Problem is complexifying too much the LLM's job.
        matchs = re.findall(pattern, response)
        if matchs:
            return matchs[-1][0].strip()

        else:
            raise ActionParsingError(
                f"Cannot parse action from response {response}"
            )


class CoTPromptConstructor(PromptConstructor):
    """The agent will perform step-by-step reasoning before the answer"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        page_screenshot_img: Image.Image | None = None,
        images: list[Image.Image] | None = None,
        meta_data: dict[str, Any] = {},
    ) -> APIInput:

        # Parse instruction data.
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        obs = state_info["observation"][self.obs_modality]

        # Trim observation to `max_obs_length` // observation = accessibility tree for example.
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            if self.lm_config.provider == "google":
                obs = obs[:max_obs_length]
            else:
                # REVIEW[mandrade]: modified to correctly trim the observation. See comments in Tokenizer
                obs = self.tokenizer.decode(self.tokenizer.encode(obs, add_special_tokens=False)[:max_obs_length], 
                                            skip_special_tokens=False)  # type: ignore[arg-type]

        # Current state
        page = state_info["info"]["page"]
        url = page.url
        previous_action_str = meta_data["action_history"][-1]
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )
        assert all([f"{{k}}" not in current for k in keywords])

        # Creates prompt specific to each LLM.
        prompt = self.get_lm_api_input(
            intro, examples, current, page_screenshot_img, images
        )
        return prompt

    #OBS: this is actually what is ran wehn calls extract_action
    def _extract_action(self, response: str) -> str:
        # find the first occurence of action
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            raise ActionParsingError(
                f'Cannot find the answer phrase "{self.answer_phrase}" in "{response}"'
            )



#FIXME: incorporate openAI to the prompt constructor above

class MultimodalCoTPromptConstructor(CoTPromptConstructor):
    """The agent will perform step-by-step reasoning before the answer"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        page_screenshot_img: Image.Image,
        images: list[Image.Image],
        meta_data: dict[str, Any] = {},
    ) -> APIInput:
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            if self.lm_config.provider == "google":
                print("NOTE: This is a Gemini model, so we use characters instead of tokens for max_obs_length.")
                obs = obs[:max_obs_length]
            else:
                obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        page = state_info["info"]["page"]
        url = page.url
        previous_action_str = meta_data["action_history"][-1]
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )

        assert all([f"{{k}}" not in current for k in keywords])

        prompt = self.get_lm_api_input(
            intro, examples, current, page_screenshot_img, images
        )
        return prompt

    def get_lm_api_input(
        self,
        intro: str,
        examples: list[tuple[str, str, str]],
        current: str,
        page_screenshot_img: Image.Image,
        images: list[Image.Image],
    ) -> APIInput:
        """Return the require format for an API"""
        message: list[dict[str, str]] | str | list[str | Image.Image]
        if "openai" in self.lm_config.provider:
            if self.lm_config.mode == "chat":
                message = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": intro}],
                    }
                ]
                for (x, y, z) in examples:
                    example_img = Image.open(z)
                    message.append(
                        {
                            "role": "system",
                            "name": "example_user",
                            "content": [
                                {"type": "text", "text": x},
                                {
                                    "type": "text",
                                    "text": "IMAGES: (1) current page screenshot",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": pil_to_b64(example_img)
                                    },
                                },
                            ],
                        }
                    )
                    message.append(
                        {
                            "role": "system",
                            "name": "example_assistant",
                            "content": [{"type": "text", "text": y}],
                        }
                    )

                # Encode images and page_screenshot_img as base64 strings.
                current_prompt = current
                content = [
                    {
                        "type": "text",
                        "text": "IMAGES: (1) current page screenshot",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(page_screenshot_img)},
                    },
                ]
                for image_i, image in enumerate(images):
                    content.extend(
                        [
                            {
                                "type": "text",
                                "text": f"({image_i+2}) input image {image_i+1}",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": pil_to_b64(image)},
                            },
                        ]
                    )
                content = [{"type": "text", "text": current_prompt}] + content

                message.append({"role": "user", "content": content})
                return message
            else:
                raise ValueError(
                    f"GPT-4V models do not support mode {self.lm_config.mode}"
                )
   
