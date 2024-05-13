# google api key
export GOOGLE_API_KEY="AIzaSyAI0XvTWHzl2ckpwQWMRuHNuaYlS1zrcGs"

# website urls for testing
export CLASSIFIEDS="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:9980"
export CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
export SHOPPING="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7770"
export SHOPPING_ADMIN="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7780/admin"
export REDDIT="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:9999"
export GITLAB="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8023"
export MAP="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000"
export WIKIPEDIA="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export HOMEPAGE="PASS"  # The home page is not currently hosted in the demo site

# suppress some tensorflow warnings
export TF_CPP_MIN_LOG_LEVEL=2

# aux function to create flags
get_flag() {
    local flag=$1
    local condition=$2
    if [ "$condition" = true ]; then
        echo "--$flag"
    else
        echo ""
    fi
}
#---------------------------------------------
# parametrization for end to end eval
#---------------------------------------------

# Models
model=gemini-pro-1.0; provider=google   
# model='llama-3/8b-instruct'; provider='huggingface'
# model='llava-llama-3/8b-instruct'; provider='huggingface'
vlm=$(get_flag 'vlm' false)   # if true, use visual language model / input


# Prompting
mode='chat' # 'completion', 'chat'
instruction='p_cot_id_actree_2s_no_na'      # p_cot_id_actree_2s_no_na, p_cot_id_actree_3s, p_multimodal_cot_id_actree_3s
sys_prompt=$(get_flag 'sys_prompt' true)    # for GEMINI: if true, adds system prompt hint 

# Generation                              
temperature=0.9                             # Default: 1.0 for GPT | 0.9 Gemini-pro | 0.6 for others (see VisualWebArena).
top_p=1.0                                   # Default: 0.9 for GPT | 1.0 Gemini-Pro | 0.9 llama-3 | 0.95 for others (see VisualWebArena).
max_tokens=500                              # Max tokens to generate. Default: 384.
context_length=0                            # Used in open AI. Default: 0.
top_k=40                                    # Used in gemini. Default: 0 (uses gemini default). Obs: gemini defaults to None, despite documentation saying default is 40.

# Input size parameters 
max_obs_length=15360                         # In tokens. # Default: 3840 | 640 for models with small ctx | 15360 chars for Gemini-Pro 
current_viewport_only=$(get_flag 'current_viewport_only' true)   # Default: true
viewport_width=1280                         # Default: 1280  
viewport_height=1024                        # Default: 720 for small context window models | 2048 for large context window models

# Tasks
test_start_idx=8
test_end_idx=9
swap_tasks=$(get_flag 'swap_tasks' false)   # if true, swap fuzzy_match tasks by the tasks immediately after them

# Execution params
deployment_mode='automodel'                 # For Hugging Face models; deploys with 'tgi', 'automodel', 'vllm'
flash_attn=$(get_flag 'flash_attn' true)    # autmodel-only: if true, uses flash attention
model_endpoint='http://127.0.0.1:8080'      # tgi-only: example: 'http://127.0.0.1:8080'
local=$(get_flag 'local' false)             # tgi-only: if true, will deploy a local tgi server
eager=$(get_flag 'eager' false)             # vllm engine only. Eager mode in Transformers. True uses less memory, but slower.
max_model_len=-1                            # vllm engine only. If -1, use the default max model length for the model.

# Max steps, early stopping
max_steps=30
parsing_failure_th=3
repeating_action_failure_th=5
max_retry=1

#paths
result_dir=./results/${provider}/${model}_${mode}$(date +%Y%m%d_%H_%M)
instruction_path=./agent/prompts/jsons/${instruction}.json
test_config_base_dir=config_files/test_webarena

# rendering parameters
render_screenshot=$(get_flag 'render_screenshot' true)
save_trace_enabled=$(get_flag 'save_trace_enabled' false)
render=$(get_flag 'render' false)                           # shows the browser window
[ "$render" = '--render' ] && slow_mo=100 || slow_mo=0      # slow_mo=100 if rendering the browser

#------------------------------------------------------------------------------
# End to End evaluation
#-----------------------------------------------------------------------------
# make results directory if it doesn't exist
mkdir -p $result_dir

# remove html files from result_dir, or else will think tasks are complete
rm -f $result_dir/*.html

# end to end eval on sample jsons
python3 run.py \
    --instruction_path $instruction_path\
    --result_dir $result_dir \
    --test_start_idx $test_start_idx \
    --test_end_idx $test_end_idx \
    --provider $provider \
    --model $model \
    $vlm \
    --mode $mode \
    --slow_mo $slow_mo \
    --temperature $temperature \
    --top_p $top_p \
    --max_tokens $max_tokens \
    --context_length $context_length \
    --max_retry $max_retry \
    --max_obs_length $max_obs_length \
    --model_endpoint $model_endpoint \
    --viewport_width $viewport_width \
    --viewport_height $viewport_height \
    --max_steps $max_steps \
    --parsing_failure_th $parsing_failure_th \
    --repeating_action_failure_th $repeating_action_failure_th \
    --deployment_mode $deployment_mode \
    $swap_tasks \
    $render \
    $render_screenshot \
    $save_trace_enabled \
    $current_viewport_only \
    $sys_prompt \
    $local \
    $eager \
    --max_model_len $max_model_len \
    --test_config_base_dir $test_config_base_dir \


