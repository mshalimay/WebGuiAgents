#!/bin/bash
yaml_file="./model_repo.yaml"

model_name="llama-3/8b-instruct"

# Extract information from the YAML file
hf_endpoint=$(yq e ".hf_models.${model_name}.hf_endpoint" $yaml_file)
quant=$(yq e ".hf_models.${model_name}.quant" $yaml_file)
hf_models_location=$(yq e ".general.hf_models_location" $yaml_file)

# hf_endpoint="meta-llama/Meta-Llama-3-8B"
# quant=""
# hf_models_location="/home/mashalimay/.cache/huggingface/hub"  <- this is the typicasl location where HF save the models

max_input_tokens=1000
max_total_tokens=1001 
max_batch_total_tokens=$max_total_tokens
max_batch_size=1        # Important: set this to 1 because tgi will stress test and you can run out of GPU memory. See TGI notes.
container_name="hf-tgi"
num_cpus=4
num_shards=1

# Start the Docker container, mounting to the model repository
docker run --rm --entrypoint /bin/bash -itd \
  --name $container_name \
  -v $hf_models_location:/data \
  --gpus 'all' \
   --cpus=$num_cpus \
  -p 8080:8080 \
  -e HF_TOKEN=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 \
   ghcr.io/huggingface/text-generation-inference:latest \

# Check if the container started successfully
if [ $? -eq 0 ]; then
    echo "Container started successfully!"
else
    echo "Failed to start the container."
    exit 1
fi

# Deploy TGI server on the container

# obs: cannot use quantize and dtype at the same time
if [ "$quant" = "" ]; then
    docker exec $container_name bash -c "text-generation-launcher \
            --model-id $hf_endpoint --num-shard $num_shards --dtype float16 --tokenizer-config-path $hf_endpoint\
             --max-batch-size $max_batch_size --max-input-tokens $max_input_tokens --max-total-tokens $max_total_tokens \
             --max-batch-total-tokens $max_batch_total_tokens \
             --trust-remote-code --cuda-memory-fraction 0.98 --port 8080 --master-port 8080 --master-addr localhost"
else
    docker exec $container_name bash -c "text-generation-launcher \
            --model-id $hf_endpoint --num-shard $num_shards --quantize $quant  --max-batch-size $max_batch_size\
            --trust-remote-code --cuda-memory-fraction 0.98 --port 8080 --master-port 8080 --master-addr localhost"
fi

# Check if the text-generation-launcher command was successful
if [ $? -eq 0 ]; then
    echo "text-generation-launcher ran successfully!"
else
    echo "Failed to run text-generation-launcher."
    echo "Stopping the Docker container..."
    docker stop $container_name
    exit 1
fi



