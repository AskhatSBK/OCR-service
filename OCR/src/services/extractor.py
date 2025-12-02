from abc import ABC, abstractmethod
import fitz
import httpx
from loguru import logger
from src.settings import settings
import re

class AbstractExtractor(ABC):


    @staticmethod
    @abstractmethod
    async def _download_file(file_id: int):
        raise NotImplementedError


    @abstractmethod
    async def extract(self, attachments: list[dict]) -> dict:
        raise NotImplementedError



class Extractor(AbstractExtractor):

    @staticmethod

    async def extract(self, attachments: list[dict]) -> dict:
        raise NotImplementedError

class DotsOcrExtractor(Extractor):
    def __init__(self, dots_ocr_host: str, dots_ocr_port: int, dots_ocr_model: str) -> None:
        self.dots_ocr_host = dots_ocr_host
        self.dots_ocr_port = dots_ocr_port
        self.dots_ocr_model = dots_ocr_model
        self.binaries = []

    async def _set_binaries(self, attachments: list[dict]) -> None:
        files_data = []
        for file in attachments:
            filename = file['name']
            file_id = file['id']
            binary, mimetype = await self._download_file(file_id)
            files_data.append({
                'filename': filename,
                'binary': binary,
            })
        self.binaries = files_data


    async def _process_file_with_dots_ocr(self, file_bytes: bytes, filename: str):
        """Process file using DotsOCR and return extracted text"""
        try:
            from src.dots_ocr.parser import DotsOCRParser
            from src.dots_ocr.utils.format_transformer import layoutjson2md
            from src.dots_ocr.utils.doc_utils import load_images_from_pdf
            from src.dots_ocr.utils.image_utils import fetch_image
            from src.dots_ocr.utils.prompts import dict_promptmode_to_prompt
            from src.dots_ocr.utils.layout_utils import post_process_output
            from src.dots_ocr.model.inference import inference_with_vllm
            from src.dots_ocr.utils.image_utils import smart_resize
            from PIL import Image
            import tempfile
            import os

            allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            file_extension = os.path.splitext(filename)[1].lower()

            if file_extension not in allowed_extensions:
                raise Exception(
                   f"Неподдерживаемый формат файла. Поддерживаемые форматы: {', '.join(allowed_extensions)}"
                )

            # Создание временного файла
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                try:
                    # Запись загруженного файла во временный файл
                    temp_file.write(file_bytes)
                    temp_file_path = temp_file.name

                    # Создание временной директории для результатов
                    with tempfile.TemporaryDirectory() as temp_output_dir:
                        # Инициализация парсера с настройками из settings
                        parser = DotsOCRParser(
                            ip=settings.DOTS_OCR_HOST,
                            port=settings.DOTS_OCR_PORT,
                            model_name=settings.DOTS_OCR_MODEL,
                        )

                        # Парсинг файла
                        results = parser.parse_file(
                            input_path=temp_file_path,
                            output_dir=temp_output_dir,
                            fitz_preprocess=True,
                            prompt_mode=settings.PROMPT_MODE,
                        )
                        response_data = {}
                        for result in results:
                            page =  result.get("page_no", (0, )),
                            page = page[0] + 1

                            if "md_content_path" in result and os.path.exists(result["md_content_path"]):
                                with open(result["md_content_path"], "r", encoding="utf-8") as f:
                                    content = f.read()
                                    cleaned_content = self._clean_content(content)
                                    response_data[f"{filename}_page_{page}{file_extension}"] = cleaned_content

                        return response_data

                except Exception as e:
                    logger.exception(f"Ошибка при парсинге файла {filename}: {str(e)}")
                    raise Exception(f"Ошибка при парсинге файла {filename}: {str(e)}")
                finally:
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
        except Exception as e:
            logger.exception(f"Ошибка при парсинге файла {filename}: {str(e)}")
            raise Exception(f"Ошибка при парсинге файла {filename}: {str(e)}")

    @staticmethod
    def _clean_content(content: str) -> str:
        pattern = r"data:image\/[a-zA-Z]+;base64,[A-Za-z0-9+/=\s]+"
        cleaned = re.sub(pattern, "", content)
        return cleaned


    async def extract(self, attachments: list[dict]) -> dict:
        await self._set_binaries(attachments)
        results = {}

        for file in self.binaries:
            filename = file["filename"]
            binary = file["binary"]

            pages = [(filename, binary)]

            for page_filename, content in pages:
                contents = await self._process_file_with_dots_ocr(content, page_filename)
                results.update(contents)
        return results