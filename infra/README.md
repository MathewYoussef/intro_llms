# intro_llms container workflow

This repo uses a container-first setup with two services:
- `lab` for experiments (Jupyter + CLI)
- `vllm` for serving the Nemotron model

Both containers share the same host cache (`~/data/hf`) to avoid re-downloading weights.

## One-time setup

1) Create a local env file:
```bash
cp .env.example .env
```

2) Update `.env` to your UID/GID and data path:
```bash
UID=$(id -u)
GID=$(id -g)
DATA_DIR=$HOME/data
```

## Start the lab container

```bash
docker compose -f infra/compose.yml --env-file .env up lab
```

Jupyter will bind to `http://localhost:8888` and print the token in logs.

### Lab image (SOP B)

The `lab` service builds `intro-llms/lab:cu130-jupyter` on top of your existing
`isaac-gr00t:*jupyterlab*` image (layer-reused; only the delta is new).

**Why this exists:** it lets the course have a dedicated, reproducible environment
(`intro-llms/lab`) without modifying your original gr00t images. Conceptually:

- Base image: gr00t (already built/cached locally)
- Course image: gr00t + course-specific deps (a small “delta” layer)

Docker images are immutable; building `intro-llms/lab` does **not** “contaminate” gr00t.

By default, `intro-llms/lab` installs **no extra packages** beyond what gr00t already has.
To opt-in:
- set `INSTALL_COURSE_DEPS=1` in `.env`
- rebuild: `docker compose -f infra/compose.yml --env-file .env build lab`

### Adding a new dependency to the lab image

If you discover you need an extra package (call it “X”), do this:

1) Add it to `infra/lab/requirements-course.txt`
2) Set `INSTALL_COURSE_DEPS=1` in `.env` (if not already)
3) Rebuild the lab image:
```bash
docker compose -f infra/compose.yml --env-file .env build lab
```

Notes:
- Avoid relying on `pip install X` inside a running container as the long-term workflow.
  It’s fine for a quick test, but bake it into the image for repeatability.
- Rebuilding requires network access (pip downloads) unless the deps are already cached
  inside Docker layers.

### Nemotron note (Transformers local)

Nemotron uses custom model code on Hugging Face (`trust_remote_code=True`) and requires
extra runtime deps like `mamba-ssm` (and typically `causal-conv1d`). Those are included
in `infra/lab/requirements-course.txt`.

If you see an error like `undefined symbol ... selective_scan_cuda...`, it usually means
`mamba-ssm` was installed from a prebuilt wheel that doesn't match the container's
PyTorch/CUDA. This repo forces source builds for `mamba-ssm` and `causal-conv1d` to
compile them against the base image.

Note: the base image uses the Jetson AI Lab pip index (`pypi.jetson-ai-lab.io`) by
default. Some packages (like `mamba-ssm`) may not be available there, so the lab image
build adds `https://pypi.org/simple` as an extra index.

## Start the vLLM server

```bash
docker compose -f infra/compose.yml --env-file .env up vllm
```

The server exposes an OpenAI-compatible API on `http://localhost:8000/v1`.

## Notes

- The repo is mounted into `/workspace` in the lab container.
- Model cache is mounted at `/data/hf`, with `HF_HOME=/data/hf` and `HF_HUB_CACHE=/data/hf/hub`.
- If you want to change the model, set `MODEL_ID` in `.env`.
