import os
import json
import time
from multiprocessing.pool import ThreadPool
import argparse

import fitz
from loguru import logger


from src.dots_ocr.model.inference import inference_with_vllm, InferenceTimeout
from src.dots_ocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from src.dots_ocr.utils.image_utils import get_image_by_fitz_doc, fetch_image, smart_resize
from src.dots_ocr.utils.doc_utils import fitz_doc_to_image
from src.dots_ocr.utils.prompts import dict_promptmode_to_prompt
from src.dots_ocr.utils.layout_utils import post_process_output, draw_layout_on_image, pre_process_bboxes
from src.dots_ocr.utils.format_transformer import layoutjson2md

from .preprocess import preprocess_path

class UnsupportedDocumentError(ValueError):
    """Input file type cannot be parsed."""


class DotsOCRParser:
    """
    parse image or pdf file
    """
    
    def __init__(self,
            ip='0.0.0.0',
            port=8005,
            model_name='model',
            temperature=0.1,
            top_p=1.0,
            max_completion_tokens=16384,
            num_thread=8,
            dpi = 200,
            output_dir="./output",
            min_pixels=None,
            max_pixels=None,
            use_hf=False,
            base_url='',
            api_key='0',
            timeout=180.0,
            max_retries=2,
            office_timeout=120,
            deadline=None,
        ):
        self.dpi = dpi
        # Absolute time.monotonic() after which no new page is sent to vLLM.
        self.deadline = deadline
        self.office_timeout = office_timeout

        # default args for vllm server
        self.ip = ip
        self.port = port
        self.base_url = base_url or f"http://{ip}:{port}/v1"
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.model_name = model_name
        # default args for inference
        self.temperature = temperature
        self.top_p = top_p
        self.max_completion_tokens = max_completion_tokens
        self.num_thread = num_thread
        self.output_dir = output_dir
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self.use_hf = use_hf
        if self.use_hf:
            # self._load_hf_model()
            # print(f"use hf model, num_thread will be set to 1")
            ...
        assert self.min_pixels is None or self.min_pixels >= MIN_PIXELS
        assert self.max_pixels is None or self.max_pixels <= MAX_PIXELS

    # def _load_hf_model(self):
    #     import torch
    #     from transformers import AutoModelForCausalLM, AutoProcessor
    #     from qwen_vl_utils import process_vision_info
    #
    #     model_path = "./weights/DotsOCR"
    #     self.model = AutoModelForCausalLM.from_pretrained(
    #         model_path,
    #         attn_implementation="flash_attention_2",
    #         torch_dtype=torch.bfloat16,
    #         device_map="auto",
    #         trust_remote_code=True
    #     )
    #     self.processor = AutoProcessor.from_pretrained(model_path,  trust_remote_code=True,use_fast=True)
    #     self.process_vision_info = process_vision_info

    def _inference_with_hf(self, image, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=24000)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response

    def _inference_with_vllm(self, image, prompt):
        timeout = self.timeout
        if self.deadline is not None:
            remaining = self.deadline - time.monotonic()
            if remaining <= 1:
                raise InferenceTimeout("request deadline exceeded before all pages were OCR'd")
            timeout = min(timeout, remaining)
        response = inference_with_vllm(
            image,
            prompt,
            base_url=self.base_url,
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=self.max_completion_tokens,
            timeout=timeout,
            max_retries=self.max_retries,
        )
        return response

    def get_prompt(self, prompt_mode, bbox=None, origin_image=None, image=None, min_pixels=None, max_pixels=None):
        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == 'prompt_grounding_ocr':
            assert bbox is not None
            bboxes = [bbox]
            bbox = pre_process_bboxes(origin_image, bboxes, input_width=image.width, input_height=image.height, min_pixels=min_pixels, max_pixels=max_pixels)[0]
            prompt = prompt + str(bbox)
        return prompt

    # def post_process_results(self, response, prompt_mode, save_dir, save_name, origin_image, image, min_pixels, max_pixels)
    def _parse_single_image(
        self, 
        origin_image, 
        prompt_mode, 
        save_dir, 
        save_name, 
        source="image", 
        page_idx=0, 
        bbox=None,
        fitz_preprocess=False,
        ):
        min_pixels, max_pixels = self.min_pixels, self.max_pixels
        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or MIN_PIXELS  # preprocess image to the final input
            max_pixels = max_pixels or MAX_PIXELS
        if min_pixels is not None: assert min_pixels >= MIN_PIXELS, f"min_pixels should >= {MIN_PIXELS}"
        if max_pixels is not None: assert max_pixels <= MAX_PIXELS, f"max_pixels should <= {MAX_PIXELS}"

        if source == 'image' and fitz_preprocess:
            image = get_image_by_fitz_doc(origin_image, target_dpi=self.dpi)
            image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            image = fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)
        input_height, input_width = smart_resize(image.height, image.width)
        prompt = self.get_prompt(prompt_mode, bbox, origin_image, image, min_pixels=min_pixels, max_pixels=max_pixels)
        if self.use_hf:
            response = self._inference_with_hf(image, prompt)
        else:
            response = self._inference_with_vllm(image, prompt)
        result = {'page_no': page_idx,
            "input_height": input_height,
            "input_width": input_width
        }
        if source == 'pdf':
            save_name = f"{save_name}_page_{page_idx}"
        if prompt_mode in ['prompt_layout_all_en', 'prompt_layout_only_en', 'prompt_grounding_ocr']:
            cells, filtered = post_process_output(
                response, 
                prompt_mode, 
                origin_image, 
                image,
                min_pixels=min_pixels, 
                max_pixels=max_pixels,
                )
            if filtered and prompt_mode != 'prompt_layout_only_en':  # model output json failed, use filtered process
                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding="utf-8") as w:
                    json.dump(response, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                origin_image.save(image_layout_path)
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })

                md_file_path = os.path.join(save_dir, f"{save_name}.md")
                with open(md_file_path, "w", encoding="utf-8") as md_file:
                    md_file.write(cells)
                result.update({
                    'md_content_path': md_file_path
                })
                result.update({
                    'filtered': True
                })
            else:
                try:
                    image_with_layout = draw_layout_on_image(origin_image, cells)
                except Exception as e:
                    logger.warning(f"Error drawing layout on image: {e}")
                    image_with_layout = origin_image

                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w', encoding="utf-8") as w:
                    json.dump(cells, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                image_with_layout.save(image_layout_path)
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })
                if prompt_mode != "prompt_layout_only_en":  # no text md when detection only
                    md_content = layoutjson2md(origin_image, cells, text_key='text')
                    md_content_no_hf = layoutjson2md(origin_image, cells, text_key='text', no_page_hf=True) # used for clean output or metric of omnidocbench、olmbench 
                    md_file_path = os.path.join(save_dir, f"{save_name}.md")
                    with open(md_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content)
                    md_nohf_file_path = os.path.join(save_dir, f"{save_name}_nohf.md")
                    with open(md_nohf_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content_no_hf)
                    result.update({
                        'md_content_path': md_file_path,
                        'md_content_nohf_path': md_nohf_file_path,
                    })
        else:
            md_content = response
            result['md_content'] = md_content
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)
            result.update({
                'md_content_path': md_file_path,
            })

        return result
    
    def parse_image(self, input_path, filename, prompt_mode, save_dir, bbox=None, fitz_preprocess=False):
        origin_image = fetch_image(input_path)
        result = self._parse_single_image(origin_image, prompt_mode, save_dir, filename, source="image", bbox=bbox, fitz_preprocess=fitz_preprocess)
        result['file_path'] = input_path
        return [result]
        
    def parse_pdf(self, input_path, filename, prompt_mode, save_dir, max_pages=None):
        with fitz.open(input_path) as doc:
            total_pages = doc.page_count
            pages_to_parse = total_pages if max_pages is None else min(total_pages, max_pages)
            logger.info(f"Parsing PDF with {total_pages} pages ({pages_to_parse} to OCR)")
            if pages_to_parse == 0:
                return [], total_pages

            def _execute_task(page_idx):
                # Render lazily inside the worker so only `num_thread` page
                # bitmaps live in memory at once, not the whole document.
                with fitz.open(input_path) as worker_doc:
                    image = fitz_doc_to_image(worker_doc[page_idx], target_dpi=self.dpi)
                return self._parse_single_image(
                    origin_image=image,
                    prompt_mode=prompt_mode,
                    save_dir=save_dir,
                    save_name=filename,
                    source="pdf",
                    page_idx=page_idx,
                )

            num_thread = 1 if self.use_hf else min(pages_to_parse, self.num_thread)
            with ThreadPool(num_thread) as pool:
                # imap (not imap_unordered) keeps page order; the first page
                # failure re-raises here and aborts the whole document.
                results = list(pool.imap(_execute_task, range(pages_to_parse)))

        for result in results:
            result['file_path'] = input_path
        return results, total_pages

    def parse_file(self,
        input_path,
        output_dir="",
        prompt_mode="prompt_layout_all_en",
        bbox=None,
        fitz_preprocess=False,
        max_pages=None,
        ):
        """Parse a document. Returns (results, total_pages)."""
        output_dir = output_dir or self.output_dir
        output_dir = os.path.abspath(output_dir)

        # Preprocess: convert DOC/DOCX to PDF, HEIC/HEIF to PNG, normalize extensions
        input_path = preprocess_path(
            input_path, workdir=output_dir, cleanup=False, office_timeout_sec=self.office_timeout,
        )

        filename, file_ext = os.path.splitext(os.path.basename(input_path))
        save_dir = os.path.join(output_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        if file_ext == '.pdf':
            results, total_pages = self.parse_pdf(input_path, filename, prompt_mode, save_dir, max_pages=max_pages)
        elif file_ext in image_extensions:
            results = self.parse_image(input_path, filename, prompt_mode, save_dir, bbox=bbox, fitz_preprocess=fitz_preprocess)
            total_pages = 1
        else:
            raise UnsupportedDocumentError(
                f"file extension {file_ext} not supported, supported extensions are {sorted(image_extensions)} and .pdf"
            )

        return results, total_pages



def main():
    prompts = list(dict_promptmode_to_prompt.keys())
    parser = argparse.ArgumentParser(
        description="dots.ocr Multilingual Document Layout Parser",
    )
    
    parser.add_argument(
        "input_path", type=str,
        help="Input PDF/image file path"
    )
    
    parser.add_argument(
        "--output", type=str, default="./output",
        help="Output directory (default: ./output)"
    )
    
    parser.add_argument(
        "--prompt", choices=prompts, type=str, default="prompt_layout_all_en",
        help="prompt to query the model, different prompts for different tasks"
    )
    parser.add_argument(
        '--bbox', 
        type=int, 
        nargs=4, 
        metavar=('x1', 'y1', 'x2', 'y2'),
        help='should give this argument if you want to prompt_grounding_ocr'
    )
    parser.add_argument(
        "--ip", type=str, default="localhost",
        help=""
    )
    parser.add_argument(
        "--port", type=int, default=8008,
        help=""
    )
    parser.add_argument(
        "--model_name", type=str, default="model",
        help=""
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help=""
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0,
        help=""
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help=""
    )
    parser.add_argument(
        "--max_completion_tokens", type=int, default=16384,
        help=""
    )
    parser.add_argument(
        "--num_thread", type=int, default=16,
        help=""
    )
    parser.add_argument(
        "--no_fitz_preprocess", action='store_true',
        help="False will use tikz dpi upsample pipeline, good for images which has been render with low dpi, but maybe result in higher computational costs"
    )
    parser.add_argument(
        "--min_pixels", type=int, default=None,
        help=""
    )
    parser.add_argument(
        "--max_pixels", type=int, default=None,
        help=""
    )
    parser.add_argument(
        "--use_hf", type=bool, default=False,
        help=""
    )
    args = parser.parse_args()

    normalized_input_path = preprocess_path(args.input_path, workdir=None, cleanup=True)

    dots_ocr_parser = DotsOCRParser(
        ip=args.ip,
        port=args.port,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_completion_tokens,
        num_thread=args.num_thread,
        dpi=args.dpi,
        output_dir=args.output, 
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_hf=args.use_hf,
    )

    fitz_preprocess = not args.no_fitz_preprocess
    if fitz_preprocess:
        print(f"Using fitz preprocess for image input, check the change of the image pixels")
    result, _ = dots_ocr_parser.parse_file(
        normalized_input_path, 
        prompt_mode=args.prompt,
        bbox=args.bbox,
        fitz_preprocess=fitz_preprocess,
        )
    


if __name__ == "__main__":
    main()
