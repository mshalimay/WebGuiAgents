# WebGuiAgents
This repo is heavily based on [WebArena](https://github.com/web-arena-x/webarena) and [VisualWebArena](https://github.com/web-arena-x/visualwebarena) <link> . See other references in `References.md`.

# Setup Steps
```bash
python -m venv webguiagents
source webguiagents/bin/activate
pip install -r requirements.txt
playwright install
pip install -e 
pip install -q -U google-generativeai
```

See [here](https://ai.google.dev/gemini-api/docs/get-started/python
) for more details on setting Google API key.

If want to use Intel's `llava-llama-3` model, first run `prepare_llava_intel.py` to combine the weights of `llava` and Meta's `llama-3`. 
- Obs: This requires holding both models in RAM with FP16 precision, so if the process gets killed it might be because your are running out of RAM.


## Optional: Install FlashAttention for faster inference.

```bash
pip install flash-attn
```


## Optional: TGI installation - local
1) Install Rust if dont have
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
2) install Protoc if dont have
```bash
PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP
```

3) Clone TGI repository
```bash
git clone https://github.com/huggingface/text-generation-inference.git
cd text-generation-inference/
BUILD_EXTENSIONS=False make install
```

If complaint contains  that ```The system library `openssl` required by crate `openssl-sys` was not found.```, install: 

```bash
sudo apt-get install libssl-dev
```

4) Running TGI locally. See `start_tgi.sh` to deploy TGI locally.


## Optional: TGI Docker Setup Steps
TODO


# Usage
- `run.py` contain the code for end-to-end evaluation.
    - To see all options, please run `python run.py --h`
    - Alternatively, check `run.sh` for usage of each option

- `run.sh` contains commands to replicate results, and describing/showing how to use each of the relevant arguments in `run.py`.

## Models
To use a specific model, select among the options in `models.yaml` and pass through the `--model <model_name>` parameter. 

For example, 

```bash
python run.py
--model llama-2/7b-instruct-gptq --provider huggingface <other parameters>
```

Before using a new model, add the it in `models.yaml` with a name of your choice and the relevant attributes. 

## Model Deployment: 
The original codebase used `Transformers` text-generation-interface (TGI) to serve the LLM / VLM models. Since cloud usage is typically paid and TGI is a bit cumbersome to deploy locally, I added some code to facilitate local deployment and also alternative modes to serve the model.

Below the three options to pass to `python run.py --deployment_mode`

1) Transformers automodel engine (DEFAULT)
- `--deployment_mode automodel` to use HuggingFace's automodel engine.
- Setting `--flashattn` uses FlashAttention. 
    - Obs: requires to install FlashAttention, such as in `pip install flash-attn`
- This choice is typically the more fail-proof and is efficient, specially if using `flash-attn`.

2) vLLM engine
- `--deploymnet_mode vllm` uses [vLLM engine](https://blog.vllm.ai/2023/06/20/vllm.html) to serve the model. 
- Requires installation of `vllm` as in: `pip install vllm`

3) TGI. 
I added options to serve the model locally via a Docker container or a local installation of TGI. 
- `--deployment_mode tgi` uses TGI for deployment.
- Setting `--local`, `run.py` will deploy a subprocess to serve the model via `text-generation-launcher`.
- Alternatively, and typically more fail-proof, serve the model via a Docker container and provide the endpoint in `--model_endpoint <endpoint>`. 
    - Use `start_tgi_docker.sh` to deploy TGI via Docker and update the model endpoint in `run.sh` accordingly.


# Examples
TODO
