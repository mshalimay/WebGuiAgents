# Setup Steps
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install
pip install -e 
pip install -q -U google-generativeai
```

See [here](https://ai.google.dev/gemini-api/docs/get-started/python
) for more details on setting Google API installation key.

If want to use Intel's `llava-llama-3` model, needs to run `prepare_llava_intel.py` to combine the weights of `llava` and Meta's `llama-3`. 
- Obs: This requires holding both models in RAM with FP16 precision, so if the process gets killed it might be because your are running out of RAM.


# Optional: Install FlashAttention for faster inference.

```bash
pip install flash-attn
```


# Optional: TGI installation - local
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


# Optional: TGI Docker Setup Steps
TODO
