"""Script to run end-to-end evaluation on the benchmark"""
import pandas as pd

#NOTE: new @mandrade
import os

os.environ[
    "CLASSIFIEDS"
    ] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:9980"

os.environ[
    "CLASSIFIEDS_RESET_TOKEN"
    ] = "4b61655535e7ed388f0d40a93600254c"

os.environ[
    "SHOPPING"
        ] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7770"
os.environ[
            "SHOPPING_ADMIN"
        ] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7780/admin/admin/"
os.environ[
            "REDDIT"
        ] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:9999"
os.environ[
            "GITLAB"
        ] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8023"
os.environ[
        "MAP"
    ] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000"
os.environ[
        "WIKIPEDIA"
    ] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
os.environ[
        "HOMEPAGE"
    ] = "PASS"  # The home page is not currently hosted in the demo site

import functools
import select
import signal
import sys
import requests
import torch
import yaml
from llms.providers.hf_utils import define_hf_model
from llms.providers.google_utils import define_google_model

# export TF_CPP_MIN_LOG_LEVEL=2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

observation_lens = {}


# end new @mandrade


import argparse
import glob
import json
import logging
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List

import openai
import requests
import torch
from beartype import beartype
from PIL import Image

from agent import (
    PromptAgent,
    construct_agent,
)
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from browser_env.auto_login import get_site_comb_from_filepath

from evaluation_harness import evaluator_router, image_utils

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=[
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "html",
            "image",
            "image_som",
        ],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=2048)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When concesecutive parsing failure exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When consecutive repeated actions exceed this threshold, the agent will terminate early.",
        type=int,
        default=5,
    )

    parser.add_argument("--test_config_base_dir", type=str, default='config_files/test_webarena')

    parser.add_argument(
        "--eval_captioning_model_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run eval captioning model on. By default, runs it on CPU.",
    )
    parser.add_argument(
        "--eval_captioning_model",
        type=str,
        default=None,
        choices=["Salesforce/blip2-flan-t5-xl"],
        help="Captioning backbone for VQA-type evals.",
    )
    parser.add_argument(
        "--captioning_model",
        type=str,
        default=None,
        choices=["Salesforce/blip2-flan-t5-xl", "llava-hf/llava-1.5-7b-hf"],
        help="Captioning backbone for accessibility tree alt text.",
    )

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=3840,
    )
    parser.add_argument(
        "--model_endpoint", # if using TGI, have to deploy in the address provided here
        help="Endpoint where the model is being deployed. Defaults to localhost:8080.",
        type=str,
        default="http://127.0.0.1:8080", 
    )

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=910)

    # logging related
    parser.add_argument("--result_dir", type=str, default="")


    #REVIEW[mandrade]: added arguments
    parser.add_argument('--deployment_mode', choices=["tgi", "vllm", "automodel"], default="automodel",
                        help='Deployment mode for the hugging-face models. If TGI, model_endpoint should be provided.')
    
    parser.add_argument("--vlm", action="store_true",
                        help='Uses vision-language models.')

    parser.add_argument('--swap_tasks', action="store_true", 
                        help='Swap tasks requiring OpenAI')

    parser.add_argument('--render_screenshot', action="store_true")

    parser.add_argument('--sys_prompt', action="store_true",
                            help='Adds system prompt hints to gemini models')

    parser.add_argument('--top_k', type=int, default=0)

    parser.add_argument('--num_gpus', type=int, default=1)

    parser.add_argument('--local', action='store_true', 
        help='Deploy a local TGI server running on --model_endpoint. Requires text-generation-launcher installation.')

    parser.add_argument('--max_model_len', type=int, default=-1, 
        help='Model context length in vLLM. Use if needs to reduce GPU memory.')

    parser.add_argument('--warmup', type=int, default=0)

    parser.add_argument('--eager', action="store_true", 
        help='Force eager mode if using vLLM. Uses less GPU memory, but less efficient.')

    parser.add_argument('--flash_attn', action="store_true", 
        help='Uses flash attention in AutoModel engine.')

    parser.add_argument('--fuzzy_match_provider', type=str, default='openai',
            choices=['openai', 'google', 'huggingface'],
            help="LLM provider for fuzzy matching evaluation. If not provided, uses GPT4-Turbo. If GPT not available, \
            uses Gemini Pro 1.5 or 1.0, whichever is available. If 'huggingface', uses LLAMA-3-Instruct-8b.")
    
    args = parser.parse_args()
    # check the whether the action space is compatible with the observation space
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type
        not in [
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "image_som",
        ]
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


@beartype
def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to stop early"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [
                action["action_type"] == ActionTypes.NONE
                for action in last_k_actions
            ]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all(
                [
                    is_equivalent(action, last_action)
                    for action in last_k_actions
                ]
            ):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if (
            sum([is_equivalent(action, last_action) for action in action_seq])
            >= k
        ):
            return True, f"Same typing action for {k} times"

    return False, ""


def test(
    args: argparse.Namespace,
    agent,
    config_file_list: list[str],
    caption_img_fn = None,
    eval_caption_img_fn = None,
) -> None:

    scores = []
    max_steps = args.max_steps

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }
    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
        captioning_fn=caption_img_fn,
    )
    # iterate over each task as defined in config_files jsons
    for config_file in config_file_list:
        try:
            render_helper = RenderHelper(
                config_file, args.result_dir, args.action_set_tag
            )

            # Get task data
            with open(config_file) as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                image_paths = _c.get("image", None)
                images = []
                # Load input images for the task, if any.
                if image_paths is not None:
                    if isinstance(image_paths, str):
                        image_paths = [image_paths]
                    for image_path in image_paths:
                        # Load image either from the web or from a local path.
                        if image_path.startswith("http"):
                            input_image = Image.open(requests.get(image_path, stream=True).raw)
                        else:
                            input_image = Image.open(image_path)
                        images.append(input_image)

                # automatically login
                if _c["storage_state"]:
                    cookie_file_name = os.path.basename(_c["storage_state"])
                    comb = get_site_comb_from_filepath(cookie_file_name)
                    temp_dir = tempfile.mkdtemp()
                    # subprocess to renew the cookie
                    subprocess.run(
                        [
                            "python",
                            "browser_env/auto_login.py",
                            "--auth_folder",
                            temp_dir,
                            "--site_list",
                            *comb,
                        ]
                    )
                    _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                    assert os.path.exists(_c["storage_state"])
                    # update the config file
                    config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                    with open(config_file, "w") as f:
                        json.dump(_c, f)

            # Log info
            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            # Prepare agent and environment
            agent.reset(config_file)
            trajectory: Trajectory = []
            obs, info = env.reset(options={"config_file": config_file})
            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory.append(state_info)

            #REVIEW[mandrade] Log observation length
            observation_lens[task_id]={}
            observation_lens[task_id]['raw']=[]
            observation_lens[task_id]['raw'].append(len(obs['text']))
            if agent.prompt_constructor.tokenizer.tokenizer is not None:
                observation_lens[task_id]['tokenized']=[]
                observation_lens[task_id]['tokenized'].append(len(agent.prompt_constructor.tokenizer(obs['text'])))

            meta_data = {"action_history": ["None"]}
            while True:
                early_stop_flag, stop_info = early_stop(
                    trajectory, max_steps, early_stop_thresholds
                )

                if early_stop_flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                else:
                    try:
                        task_id = None if args.warmup > 0 else task_id # Dont save conversation if warmup
                        # Construct prompt, get LLM answer, transform into action (eg: scroll)
                        action = agent.next_action(
                            trajectory,
                            intent,
                            images=images,
                            meta_data=meta_data,
                            task_id=task_id, 
                        )
                    except ValueError as e:
                        # If action is not valid, create a stop action
                        action = create_stop_action(f"ERROR: {str(e)}")
                        logger.info(f"[Error] {str(e)}")

                trajectory.append(action)

                action_str = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=args.action_set_tag,
                    prompt_constructor=agent.prompt_constructor
                    if isinstance(agent, PromptAgent)
                    else None,
                )
                render_helper.render(
                    action, state_info, meta_data, args.render_screenshot
                )
                meta_data["action_history"].append(action_str)

                if action["action_type"] == ActionTypes.STOP:
                    #REVIEW[mandrade]: for no N/A prompting, a impossible action will typically return a stop action with empty answer.
                    # But this will be evaluated as fail in fuzzy_match. Either adjust prompting (but problems of early stopping) or use less stringent evaluation.
                    # If action string contains only space-like characters or empty, replace for the raw prediction or N/A.
                    # Use raw prediction for fuzzy match of tasks with ref answers like: 'There is no airport within 5 km of'
                    # E.g. original: There is no airport within... next action is ```stop [ ]``` -> action_str = stop [ ] -> answer = ""
                    if check_fuzzy_match(config_file='', config=_c):
                        test_string = re.sub(r"\s+", "", action_str)
                        if test_string == "stop[]":
                            if check_na_ref_answer(_c):
                                trajectory[-1]["answer"] = "n/a"  # answer = "" -> 'n/a'
                            else:
                                action_splitter = agent.prompt_constructor.instruction["meta_data"]["action_splitter"]
                                trajectory[-1]["answer"] = action['raw_prediction'].split(action_splitter)[0].strip().lower() # answer = "" -> there is no airport within... next action is 
                    break

                obs, _, terminated, _, info = env.step(action)
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                #REVIEW[mandrade] Log observation length
                observation_lens[task_id]['raw'].append(len(obs['text']))
                if agent.prompt_constructor.tokenizer.tokenizer is not None:
                    observation_lens[task_id]['tokenized'].append(len(agent.prompt_constructor.tokenizer(obs['text'])))

                if terminated:
                    # add a action place holder
                    trajectory.append(create_stop_action(""))
                    break

            # NOTE: eval_caption_image_fn is used for running eval_vqa functions.
            evaluator = evaluator_router(
                config_file, captioning_fn=eval_caption_img_fn, fuzzy_match_prov=args.fuzzy_match_provider
            )
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page,
                client=env.get_page_client(env.page),
            )

            # Print Pass/Fail
            if score == 1:
                logger.info(f"[Result] (PASS) {config_file}")
            else:
                logger.info(f"[Result] (FAIL) {config_file}")

            # Store scores and save trace
            #REVIEW[mandrade]: added warmup
            if args.warmup == 0:
                scores.append(score)
            
                if args.save_trace_enabled:
                    env.save_trace(
                        Path(args.result_dir) / "traces" / f"{task_id}.zip"
                    )
            else:
                logger.info(f"[Warmup] {args.warmup}")
                args.warmup -= 1

            # Compute average observation length
            if task_id in observation_lens:
                avg_raw = sum(observation_lens[task_id]['raw']) / len(observation_lens[task_id]['raw'])
                logger.info(f"[Avg Obs Len] Task {task_id}: Raw: {avg_raw}")
                if len(observation_lens[task_id])==2:
                    avg_tokenized = sum(observation_lens[task_id]['tokenized']) / len(observation_lens[task_id]['tokenized'])
                    logger.info(f"[Avg Obs Len] Task {task_id}: Tokenized: {avg_tokenized}")

            # End of Task (while)

        # End of Test (for task in tasks)

        except openai._exceptions.OpenAIError as e:
            logger.info(f"[OpenAI Error] {repr(e)}")
        except Exception as e:
            logger.info(f"[Unhandled Error] {repr(e)}]")
            import traceback

            # write to error file
            with open(Path(args.result_dir) / "error.txt", "a") as f:
                f.write(f"[Config file]: {config_file}\n")
                f.write(f"[Unhandled Error] {repr(e)}\n")
                f.write(traceback.format_exc())  # write stack trace to file

        # save observation lengths
        pd.DataFrame(observation_lens).to_csv(f"{args.result_dir}/observation_lens.csv")
        render_helper.close()

    env.close()
    logger.info(f"Average score: {sum(scores) / len(scores)}")

def check_na_ref_answer(config_file):
    reference_answers = config_file['eval']["reference_answers"]
    if reference_answers is not None:
        ref_answers = [item for item in reference_answers.values()]
        return any(["N/A" in ans for ans in ref_answers])
    return False


def prepare(args: argparse.Namespace) -> None:

    # Convert prompt python files to json
    from agent.prompts import to_json
    to_json.run()

    # Prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = (
            f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        )
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if args.save_trace_enabled and not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    # Log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")

    if args.max_model_len < 0:
        args.max_model_len = None
    if args.top_k <= 0:
        args.top_k = None

    args.provider = args.provider.strip().lower()

    if args.deployment_mode=='vllm' and args.vlm:
        args.deployment_mode=='automodel'
        logger.info("vLLM doesn't support Vision Language models yet. Defaulting to Transformers AutoModel.")


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [
        os.path.basename(f).split(".")[0].split("_")[1] for f in result_files
    ]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")

#REVIEW[mandrade]: added some auxiliary functions
def check_fuzzy_match(config_file:str=None, config=None):
    if config is None:
        config = json.load(open(config_file))
    if config['eval']['reference_answers'] is not None:
        if 'fuzzy_match' in config['eval']['reference_answers']:
            return True
    return False

def swap_tasks(config_list, test_config_base_dir, st_idx, ed_idx, verbose=True):
    """Remove tasks that require openAI and swap by the next tasks available"""
    # TODO: select tasks to keep proportion balanced
    num_tasks = ed_idx - st_idx + 1
    temp_list = [config for config in config_list if not check_fuzzy_match(config)]
    new_tasks = []
    idx = ed_idx

    n_tasks_add = num_tasks - len(temp_list)
    while n_tasks_add > 0:
        config_file = os.path.join(test_config_base_dir, f"{idx}.json")
        if not check_fuzzy_match(config_file=config_file):
            new_tasks.append(config_file)
            n_tasks_add -= 1
        idx += 1

    if verbose:
        print("\n\nThe following tasks were removed for requiring OpenAI key:")
        print(list(set(config_list) - set(temp_list)))
        print("The following tasks were added:")
        print(list(set(new_tasks) - set(temp_list)))
        print("\n\n")
    temp_list.extend(new_tasks)
    return temp_list

def load_model_config(model_repo_file:str):
    with open(model_repo_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


# Auxiliary functions for local TGI deployment

def deploy_tgi(model_path, quantize, num_shard):
    print(f'\n\nDeploying TGI locally for model {model_path}\n')
    process = subprocess.Popen(['text-generation-launcher', '--model-id', model_path,
                                '--quantize', quantize, '--num-shard', str(num_shard),
                                '--port' '8080' '--master-port' '8080' '--master-addr' 'localhost'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print("TGI has been started and is running in the background.")
    print(f"Process ID: {process.pid}")
    return process

def wait_tgi(model_endpoint, max_mins=10, timeout_secs=60, process=None, try_again_sec=30):
    from text_generation import Client  # Assuming this is correctly imported
    client = Client(model_endpoint, timeout=timeout_secs)
    running = False
    start_time = time.time()

    # Convert file objects to file descriptors
    stdout_fd = process.stdout.fileno()
    stderr_fd = process.stderr.fileno()
    download_flag=False

    while not running and (time.time() - start_time) / 60 < max_mins:
        try:
            print(client.generate(prompt='Hello world').generated_text)
            running = True
        except:
            print(f"TGI didn't responde after {timeout_secs}. Trying again in {try_again_sec} seconds")
            time.sleep(try_again_sec)
            pass

        # Print tgi process output without blocking
        rlist, _, _ = select.select([stdout_fd, stderr_fd], [], [], 0)
        for fd in rlist:
            if fd == stdout_fd:
                output = os.read(stdout_fd, 1024)  # Read up to 1024 bytes
                if output:
                    out_str = output.decode('utf-8', errors='ignore').strip()
                    print(out_str)

                    if 'Starting download process' in out_str:
                        download_flag=True
                    if 'Skipping download.' in out_str:
                        download_flag=False
                else:
                    if download_flag:
                        print('TGI is downloading the model. Sleeping for 5 minutes')
                        max_mins += 5
                        time.sleep(60*5)
                            
            elif fd == stderr_fd:
                error_output = os.read(stderr_fd, 1024)  # Read up to 1024 bytes
                if error_output:
                    print(error_output.decode('utf-8', errors='ignore').strip(), file=sys.stderr)


    if not running:
        # raise TimeoutError(f"TGI Model {model_endpoint} not running after {max_mins} minutes")
        print(f"TGI Model {model_endpoint} not answring after {max_mins} minutes. Exiting...")
        terminate_process(process)
        sys.exit(0)
    else:
        print(f"\n\nTGI Model {model_endpoint} running\n\n")

def terminate_process(process):
    if process and process.poll() is None:
        print(f"Terminating TGI process...{process.pid}")
        os.kill(process.pid, signal.SIGTERM)
        process.wait()
        print("Process terminated")

def signal_handler(process, sig, frame):
    print(f'SIGINT received, terminating the process...{process.pid}')
    if process and process.poll() is None:  # Check if the process is still running
        os.kill(process.pid, signal.SIGTERM)
        process.wait()
        print("Process terminated gracefully.")
    sys.exit(0)

def set_seed(seed=0):
    import random
    import numpy
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    


def define_captioning_fn(args):
    #TODO: Put the args.observation parsing in prepare(args) function

    # Define captioning function
    caption_image_fn = eval_caption_image_fn = None

    if args.captioning_model is not None:
        if args.observation_type in [
            "accessibility_tree_with_captioner",
            "image_som",
        ]:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            caption_image_fn = image_utils.get_captioning_fn(device, dtype, args.captioning_model)
        else:
            caption_image_fn = None
            args.captioning_model=None

    if args.eval_captioning_model is not None:
        # Load a (possibly different) captioning model for running VQA evals.
        if (caption_image_fn is not None and args.eval_captioning_model == args.captioning_model):
            eval_caption_image_fn = caption_image_fn
        else:
            dtype = torch.float16 if (torch.cuda.is_available() and 
                            args.eval_captioning_model_device == "cuda") else torch.float32

            eval_caption_image_fn = image_utils.get_captioning_fn(
                args.eval_captioning_model_device,
                dtype,
                args.eval_captioning_model,
            )
        
    return caption_image_fn, eval_caption_image_fn

if __name__ == "__main__":
    try:
        # set_seed(0)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        args = config()
        args.sleep_after_execution = 2.5
        prepare(args)

        ## FIXME @debug interactive
        # os.environ['GOOGLE_API_KEY'] = 'AIzaSyCmqYPObitVV81ARC7_tofzArttNt63Y84'    
        # # args.model='llama-3/8b-instruct'; args.provider='huggingface'
        # args.model="gemini-pro-1.0"; args.provider="google"
        # args.test_start_idx=0
        # args.test_end_idx=26
        # args.model_endpoint='http://127.0.0.1:8080'
        # args.render=False
        # args.slow_mo=0 if not args.render else 100
        # args.mode='chat'
        # args.instruction_path='./agent/prompts/jsons/p_cot_id_actree_3s.json'
        # args.result_dir='results/debug'
        # args.sys_prompt=True
        # args.local=False
        # args.vllm=False
        # args.warmup=0
        # args.swap_tasks=True
        # args.deployment_mode='automodel'
        ## end of @debug    

        # print full path to result_dir
        print(f"\nResults will be saved in {os.path.abspath(args.result_dir)}\n")

        # Retrieve task json files
        test_file_list = []
        st_idx = args.test_start_idx
        ed_idx = args.test_end_idx
        for i in range(st_idx, ed_idx):
            test_file_list.append(os.path.join(args.test_config_base_dir, f"{i}.json"))

        # check if any task requires OpenAI and substitute for other tasks if specified
        if args.swap_tasks:
            test_file_list = swap_tasks(test_file_list, args.test_config_base_dir, st_idx, ed_idx, verbose=False)

        if "debug" not in args.result_dir:
            test_file_list = get_unfinished(test_file_list, args.result_dir)

        if len(test_file_list) == 0:
            logger.info("No task left to run")
            sys.exit(0)
        if args.warmup > 0:
            # repeat the first task for warmup
            for _ in range(args.warmup):
                test_file_list.insert(0, test_file_list[0])
            # print(test_file_list)
        print(f"Total {len(test_file_list)} tasks left")

        # file to save conversations
        args.conversation_file = f"{args.result_dir}/conversations"

        # Get model configurations # TODO: Better integration with arguments passed through command line
        model_config = load_model_config('./model_repo.yaml')
        if 'huggingface' in args.provider.lower():
            args.model_path = model_config['models'][args.model]['model_path']
            args.tokenizer_path = model_config['models'][args.model]['tokenizer_path']
            args.quant = model_config['models'][args.model]['quant']
            model_save_location = model_config['general']['hf_models_location']

        if 'google' in args.provider.lower():
            args.model_path = model_config['models'][args.model]['model_path']
            args.tokenizer_path=None

        # Save args file to results folder
        os.makedirs(args.result_dir, exist_ok=True)
        with open(f'{args.result_dir}/args.json', "w") as f:
            json.dump(vars(args), f, indent=4)        

        # Define captioning function, if specified
        caption_img_fn, eval_caption_img_fn = define_captioning_fn(args)

        # Build agent, including prompt constructor
        agent_captioning_fn = caption_img_fn if args.observation_type == "accessibility_tree_with_captioner" else None
        agent = construct_agent(args=args, 
                        captioning_fn=agent_captioning_fn) # NOTE: captioning_fn here is used for captioning input images.

        # Load the model.
        if 'google' in args.provider.lower():
            define_google_model(args.model_path)

        elif 'huggingface' in args.provider.lower():
            if args.deployment_mode == 'automodel' or args.deployment_mode == 'vllm':
                define_hf_model(model=args.model, vlmodel=args.vlm, model_path=args.model_path, tokenizer_path=args.tokenizer_path, quant=args.quant, 
                        engine=args.deployment_mode, max_model_len=args.max_model_len, num_gpus=args.num_gpus, flash_attn=args.flash_attn)

            elif args.deployment_mode == 'tgi' and args.local:
                # Deploys TGI locally on subprocess. Not Docker.
                tgi_process = deploy_tgi(args.model_path, args.quant, args.num_shards, model_save_location)
                # Set up signal handler to terminate process if script stopped with ctrl+c
                handler = functools.partial(signal_handler, tgi_process)
                signal.signal(signal.SIGINT, handler)
                # wait for TGI deployment
                wait_tgi(args.model_endpoint, max_mins=10, timeout_secs=60, process=tgi_process, try_again_sec=10)
        else:
            raise ValueError(f"Provider {args.provider} not supported")

        # Evaluate
        test(args, agent, test_file_list, caption_img_fn=caption_img_fn, eval_caption_img_fn=eval_caption_img_fn)

        # save log file to results folder
        os.rename(LOG_FILE_NAME, f"{args.result_dir}/log.txt")

    except Exception as e:
        # print traceback
        import traceback
        traceback.print_exc()  # This prints the full traceback to stderr
        if 'tgi_process' in locals():
            terminate_process(tgi_process)
            sys.exit(0)
