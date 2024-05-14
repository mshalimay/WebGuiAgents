

## dev installation

Original instructions have little bugs
1) Create a .cache directory for mypy
2) Adjust `mypy` command to include the path to the cache directory and to use the `agent` folder (not agents)

```bash
mkdir ./.cache
mypy --cache-dir='./.cache' --install-types --non-interactive browser_env agent evaluation_harness
```


pip install -q -U google-generativeai

install vertex ai:
pip install --upgrade google-cloud-aiplatform
