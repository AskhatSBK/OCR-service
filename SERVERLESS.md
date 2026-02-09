# RunPod Serverless Setup

Guide for running the OCR Service with RunPod Serverless vLLM instead of a local GPU.

## Step 1: Build and Push Docker Image to Docker Hub

RunPod Serverless pulls worker images from a container registry. You need to build and push the vLLM worker image.

### 1.1 Login to Docker Hub

```bash
docker login
```

### 1.2 Build the vLLM Worker Image

```bash
docker build -t YOUR_DOCKERHUB_USERNAME/dots-ocr-vllm:latest -f Dockerfile.vllm .
```

Or if you're using the official vLLM image as a base, tag and push it with your model baked in:

```bash
docker pull vllm/vllm-openai:latest

docker tag vllm/vllm-openai:latest YOUR_DOCKERHUB_USERNAME/dots-ocr-vllm:latest
```

### 1.3 Push to Docker Hub

```bash
docker push YOUR_DOCKERHUB_USERNAME/dots-ocr-vllm:latest
```

The image must be **public** on Docker Hub, or you need to configure RunPod with registry credentials.

## Step 2: Create a RunPod Serverless Endpoint

1. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless) and create a new endpoint
2. Set **Container Image** to your Docker Hub image: `YOUR_DOCKERHUB_USERNAME/dots-ocr-vllm:latest`
3. Select a **GPU** with enough VRAM for the model (e.g. A40, A100)

### Environment Variables on RunPod

When configuring the endpoint, add these environment variables:

| Variable | Value | Description |
|---|---|---|
| `MODEL_NAME` | `rednote-hilab/dots.ocr` | HuggingFace model to serve |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for tensor parallelism |
| `GPU_MEMORY_UTILIZATION` | `0.9` | Fraction of GPU memory to use |
| `CHAT_TEMPLATE_CONTENT_FORMAT` | `string` | Content format for chat template |
| `SERVED_MODEL_NAME` | `model` | Model name exposed via the API (clients use this) |
| `HF_TOKEN` | `your_hf_token` | HuggingFace token (if model is gated) |

4. Note your **Endpoint ID** — the alphanumeric string from `https://api.runpod.ai/v2/{ENDPOINT_ID}/...`
5. Get your **API Key** from [RunPod Settings](https://www.runpod.io/console/user/settings)

## Step 3: Configure the Service

```bash
cp OCR/.env.example OCR/.env
```

Edit `OCR/.env`:

```env
DOTS_OCR_BASE_URL=https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1
DOTS_OCR_API_KEY=your_runpod_api_key
DOTS_OCR_MODEL=model
PROMPT_MODE=prompt_ocr
OCR_MODE=DOTS
ROOT_PATH=/
```

| Variable | Description |
|---|---|
| `DOTS_OCR_BASE_URL` | Full RunPod OpenAI-compatible URL. Overrides `DOTS_OCR_HOST`/`DOTS_OCR_PORT`. |
| `DOTS_OCR_API_KEY` | Your RunPod API key. |
| `DOTS_OCR_MODEL` | Must match `SERVED_MODEL_NAME` configured on the RunPod endpoint. |

## Step 4: Run Service Only

No local GPU needed — inference is handled by RunPod:

```bash
docker compose up service --build
```

## Verify RunPod Endpoint

Before sending files, check that the endpoint is reachable:

```bash
# List available models
curl -s -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/openai/v1/models

# Check endpoint health
curl -s -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/health
```

The `/models` response tells you the exact model name to use in `DOTS_OCR_MODEL`.

## Troubleshooting

### 500 Internal Server Error from RunPod

- **Model name mismatch**: `DOTS_OCR_MODEL` in `.env` must match `SERVED_MODEL_NAME` on RunPod. Query `/models` to confirm.
- **Worker not ready**: Serverless workers scale to zero when idle. The first request may take 30s-2min for cold start. Check `/health` for worker status.
- **Out of VRAM**: The selected GPU may not have enough memory. Check RunPod endpoint logs.

### Connection Refused / Timeout

- Verify `DOTS_OCR_BASE_URL` includes `/openai/v1` at the end.
- Verify `DOTS_OCR_API_KEY` is valid.

### Switching Back to Local vLLM

Clear `DOTS_OCR_BASE_URL` to fall back to `DOTS_OCR_HOST`/`DOTS_OCR_PORT`:

```env
DOTS_OCR_BASE_URL=
DOTS_OCR_HOST=dots-ocr
DOTS_OCR_PORT=8000
DOTS_OCR_MODEL=model
```
