"""Download OCR model weights to <repo>/model/weights/<dir>, the path compose.yml mounts.

    python dots.ocr/tools/download_model.py                       # dots.mocr -> model/weights/DotsMOCR
    python dots.ocr/tools/download_model.py -n rednote-hilab/dots.ocr -d DotsOCR

Folder names must not contain '.' (upstream model-loader quirk).
"""
from argparse import ArgumentParser
import os


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--type', '-t', type=str, default="huggingface", choices=["huggingface", "modelscope"])
    parser.add_argument('--name', '-n', type=str, default="dots-studio/dots.mocr")
    parser.add_argument('--dir', '-d', type=str, default="DotsMOCR",
                        help="Folder under model/weights; must match OCR_MODEL_DIR in compose")
    args = parser.parse_args()
    if "." in args.dir:
        parser.error("--dir must not contain '.'")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_dir = os.path.join(repo_root, "model", "weights", args.dir)
    os.makedirs(model_dir, exist_ok=True)
    if args.type == "huggingface":
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=args.name, local_dir=model_dir)
    else:
        from modelscope import snapshot_download
        snapshot_download(repo_id=args.name, local_dir=model_dir)

    print(f"model downloaded to {model_dir}")
