import os
import tempfile
from typing import Any

from fastapi import APIRouter, Body, HTTPException, UploadFile, File, Form
from loguru import logger
from src.dots_ocr.parser import DotsOCRParser
from src.services.extractor import DotsOcrExtractor
from src.settings import settings



router = APIRouter(tags=["v1"])

@router.post("/parse-file")
async def parse_file(
    file: UploadFile = File(...),
):
    """
    Парсинг файла с помощью DotsOCR
    Поддерживаемые форматы:'.pdf','.jpg', '.jpeg', '.png', '.heic', '.doc', '.docx', '.heic', '.heif'
    """
    # Валидация типа файла
    allowed_extensions = {'.pdf','.jpg', '.jpeg', '.png', '.heic', '.doc', '.docx', '.heic', '.heif'}
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат файла. Поддерживаемые форматы: {', '.join(allowed_extensions)}"
        )
    


    
    # Создание временного файла
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        try:
            # Запись загруженного файла во временный файл
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
            
            # Создание временной директории для результатов
            with tempfile.TemporaryDirectory() as temp_output_dir:
                # Инициализация парсера с настройками из settings
                parser = DotsOCRParser(
                    ip=settings.DOTS_OCR_HOST,
                    port=settings.DOTS_OCR_PORT,
                    model_name=settings.DOTS_OCR_MODEL,
                    base_url=settings.DOTS_OCR_BASE_URL,
                    api_key=settings.DOTS_OCR_API_KEY,
                )
                
                # Парсинг файла
                results = parser.parse_file(
                    input_path=temp_file_path,
                    output_dir=temp_output_dir,
                    fitz_preprocess=True,
                    prompt_mode=settings.PROMPT_MODE,
                )
                
                # Подготовка ответа
                response_data = {
                    "filename": file.filename,
                    "results": []
                }
                
                # Обработка результатов
                for result in results:
                    result_data = {
                        "page_no": result.get("page_no", 0),
                    }

                    if "md_content_path" in result and os.path.exists(result["md_content_path"]):
                        with open(result["md_content_path"], "r", encoding="utf-8") as f:
                            content = DotsOcrExtractor._clean_content(f.read())
                            result_data["markdown_content"] = content


                    response_data["results"].append(result_data)
                
                return response_data
                
        except Exception as e:
            logger.exception(f"Ошибка при парсинге файла {file.filename}: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Ошибка при парсинге файла: {str(e)}"
            )
        finally:
            # Удаление временного файла
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
