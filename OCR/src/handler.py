import os
import re
import base64
import tempfile

import runpod
from loguru import logger

from src.dots_ocr.parser import DotsOCRParser
from src.settings import settings


ALLOWED_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png', '.heic', '.doc', '.docx', '.heif'}

parser = DotsOCRParser(
    ip=settings.DOTS_OCR_HOST,
    port=settings.DOTS_OCR_PORT,
    model_name=settings.DOTS_OCR_MODEL,
    base_url=settings.DOTS_OCR_BASE_URL,
    api_key=settings.DOTS_OCR_API_KEY,
)


def _clean_content(content: str) -> str:
    pattern = r"data:image\/[a-zA-Z]+;base64,[A-Za-z0-9+/=\s]+"
    return re.sub(pattern, "", content)


def handler(job):
    """
    RunPod Serverless handler for OCR processing.

    Input schema:
    {
        "input": {
            "file_base64": "<base64-encoded file content>",
            "filename": "document.pdf",
            "prompt_mode": "prompt_ocr"  // optional, defaults to settings.PROMPT_MODE
        }
    }
    """
    job_input = job["input"]

    file_base64 = job_input.get("file_base64")
    filename = job_input.get("filename")
    prompt_mode = job_input.get("prompt_mode", settings.PROMPT_MODE)

    if not file_base64 or not filename:
        return {"error": "Both 'file_base64' and 'filename' are required."}

    file_extension = os.path.splitext(filename)[1].lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        return {
            "error": f"Unsupported file format. Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"
        }

    try:
        file_bytes = base64.b64decode(file_base64)
    except Exception:
        return {"error": "Invalid base64 encoding for 'file_base64'."}

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        try:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

            with tempfile.TemporaryDirectory() as temp_output_dir:
                results = parser.parse_file(
                    input_path=temp_file_path,
                    output_dir=temp_output_dir,
                    fitz_preprocess=True,
                    prompt_mode=prompt_mode,
                )

                response_data = {
                    "filename": filename,
                    "results": [],
                }

                for result in results:
                    result_data = {
                        "page_no": result.get("page_no", 0),
                    }

                    if "md_content_path" in result and os.path.exists(result["md_content_path"]):
                        with open(result["md_content_path"], "r", encoding="utf-8") as f:
                            content = _clean_content(f.read())
                            result_data["markdown_content"] = content

                    response_data["results"].append(result_data)

                return response_data

        except Exception as e:
            logger.exception(f"Error parsing file {filename}: {str(e)}")
            return {"error": f"Error parsing file: {str(e)}"}
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


runpod.serverless.start({"handler": handler})
