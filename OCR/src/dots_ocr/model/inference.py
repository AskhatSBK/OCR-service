import threading

from openai import OpenAI, APIError, APITimeoutError, APIConnectionError

from src.dots_ocr.utils.image_utils import PILimage_to_base64


class InferenceError(RuntimeError):
    """vLLM backend failed or returned nothing usable."""


class InferenceTimeout(InferenceError):
    """vLLM backend did not answer in time."""


_clients: dict[tuple, OpenAI] = {}
_clients_lock = threading.Lock()


def get_client(base_url: str, api_key: str, max_retries: int) -> OpenAI:
    """Return a process-wide OpenAI client so HTTP connections are pooled
    instead of re-created for every page. Timeouts are set per call."""
    key = (base_url, api_key, max_retries)
    with _clients_lock:
        client = _clients.get(key)
        if client is None:
            client = OpenAI(
                api_key=api_key or "0",
                base_url=base_url,
                max_retries=max_retries,
            )
            _clients[key] = client
        return client


def inference_with_vllm(
        image,
        prompt,
        base_url,
        api_key="0",
        temperature=0.1,
        top_p=0.9,
        max_completion_tokens=32768,
        model_name='model',
        timeout=180.0,
        max_retries=2,
        ) -> str:
    client = get_client(base_url, api_key, max_retries)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": PILimage_to_base64(image)},
                },
                # if no "<|img|><|imgpad|><|endofimg|>" here, vllm v1 will add "\n" here
                {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"},
            ],
        }
    ]
    try:
        response = client.chat.completions.create(
            messages=messages,
            model=model_name,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
        )
    except APITimeoutError as e:
        raise InferenceTimeout(f"vLLM request timed out after {timeout:.0f}s") from e
    except (APIConnectionError, APIError) as e:
        raise InferenceError(f"vLLM request failed: {e}") from e

    if not response.choices:
        raise InferenceError("vLLM returned no choices")
    content = response.choices[0].message.content
    if content is None:
        raise InferenceError("vLLM returned empty content")
    return content
