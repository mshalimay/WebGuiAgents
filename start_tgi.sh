yaml_file="model_repo.yaml" 

model_name=llama-2-7b-instruct-awq
num_shard=1

# Extracting configuration
hf_endpoint=$(yq e ".hf_models.$model_name.hf_endpoint" $yaml_file)
quant=$(yq e ".hf_models.$model_name.quant" $yaml_file)
hf_models_location=$(yq e ".hf_models.$model_name.hf_models_location" $yaml_file)

cd hf_models_location

text-generation-launcher --model-id $hf_endpoint --quantize $quant --num-shard $num_shard \
        --port 3000 --master-port 3000 --master-addr localhost