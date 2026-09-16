"""
Setup Stage 2: Model Downloader

Run this AFTER setup/install.py. Downloads:

  1. FLUX.2 Klein base model(s) from HuggingFace (black-forest-labs) — several GB each.
     4B is ~8GB, 9B is ~18GB. You only need the one(s) you plan to train.
  2. The Qwen3 text encoder(s) the fork's Flux2Klein model classes expect
     (Qwen/Qwen3-4B for the 4B model, Qwen/Qwen3-8B for the 9B model) —
     these download automatically via HF cache on first training run too,
     but pre-fetching here avoids a long silent-looking pause on your first
     smoke test.
  3. The small (~300MB total) face-detection/upscale models this repo's own
     ingestion pipeline needs: YOLOFace (face detection), ArcFace w600k_r50
     (identity embeddings), 4x-UltraSharp (ESRGAN upscaler).

Some of these repos are gated on HuggingFace (require accepting a license on
the model page + a HF access token). If a download fails with a 401/403, go
accept the license at the printed URL, then set HF_TOKEN env var (or run
`huggingface-cli login`) and re-run this script — already-downloaded files
are skipped, so it's safe to re-run.

Usage:
    python setup/download_models.py --model 4b        # only FLUX.2 Klein 4B (~8GB)
    python setup/download_models.py --model 9b        # only FLUX.2 Klein 9B (~18GB)
    python setup/download_models.py --model both       # both (~26GB)
    python setup/download_models.py --skip-base-model  # only the small face/upscale models
"""
import os
import sys
import json
import argparse
import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FACE_MODELS = {
    "yoloface_8n.onnx": "https://github.com/facefusion/facefusion-assets/releases/download/models/yoloface_8n.onnx",
    "arcface_w600k_r50.onnx": "https://huggingface.co/facefusion/models-3.0.0/resolve/main/arcface_w600k_r50.onnx",
}
UPSCALE_MODELS = {
    "4x-UltraSharp.pth": "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth",
}

HF_KLEIN_4B_REPO = "black-forest-labs/FLUX.2-klein-base-4B"
HF_KLEIN_4B_FILE = "flux-2-klein-base-4b.safetensors"
HF_KLEIN_9B_REPO = "black-forest-labs/FLUX.2-klein-base-9B"
HF_KLEIN_9B_FILE = "flux-2-klein-base-9b.safetensors"
HF_QWEN3_4B_REPO = "Qwen/Qwen3-4B"
HF_QWEN3_8B_REPO = "Qwen/Qwen3-8B"


def load_paths():
    paths_file = os.path.join(REPO_ROOT, "setup", "paths.json")
    if not os.path.exists(paths_file):
        raise FileNotFoundError("setup/paths.json not found. Run 'python setup/install.py' first.")
    with open(paths_file, "r", encoding="utf-8") as f:
        return json.load(f)


def download_url(url, dest_path, chunk_size=1024 * 1024):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"  Already exists, skipping: {dest_path}")
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"  Downloading {url}\n    -> {dest_path}")
    tmp_path = dest_path + ".partial"
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r    {downloaded/1e6:.0f}MB / {total/1e6:.0f}MB ({pct:.1f}%)", end="", flush=True)
    print()
    os.replace(tmp_path, dest_path)


def download_hf_file(repo_id, filename, dest_dir, token=None):
    """Downloads a single file from a HuggingFace repo using huggingface_hub
    (handles Xet/LFS storage backends correctly, unlike a raw requests.get)."""
    from huggingface_hub import hf_hub_download
    dest_path = os.path.join(dest_dir, filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"  Already exists, skipping: {dest_path}")
        return dest_path
    print(f"  Downloading {repo_id}/{filename} ...")
    try:
        local_path = hf_hub_download(
            repo_id=repo_id, filename=filename, token=token,
            local_dir=dest_dir, local_dir_use_symlinks=False,
        )
        return local_path
    except Exception as e:
        print(f"  ERROR downloading {repo_id}/{filename}: {e}")
        print(f"  If this is a 401/403: accept the license at "
              f"https://huggingface.co/{repo_id} while logged in, then set "
              f"HF_TOKEN env var (or run `huggingface-cli login`) and re-run this script.")
        raise


def prefetch_hf_snapshot(repo_id, token=None):
    """Warms the shared HF cache for a text-encoder repo (used by name at
    training time via transformers' from_pretrained, not copied into models_root)."""
    from huggingface_hub import snapshot_download
    print(f"  Pre-fetching {repo_id} into the shared HuggingFace cache ...")
    try:
        snapshot_download(repo_id=repo_id, token=token)
    except Exception as e:
        print(f"  WARNING: could not pre-fetch {repo_id} ({e}). "
              f"It will be downloaded automatically on your first training run instead.")


def main():
    parser = argparse.ArgumentParser(description="Downloads base models + small face/upscale models")
    parser.add_argument("--model", default="4b", choices=["4b", "9b", "both", "none"],
                         help="Which FLUX.2 Klein base model(s) to download (default: 4b)")
    parser.add_argument("--skip-base-model", action="store_true", help="Skip FLUX.2 Klein base model download entirely")
    parser.add_argument("--skip-face-models", action="store_true", help="Skip small face/upscale model download")
    parser.add_argument("--hf-token", default=None, help="HuggingFace access token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    paths = load_paths()
    models_root = paths["models_root"]
    token = args.hf_token or os.environ.get("HF_TOKEN")

    if not args.skip_base_model and args.model != "none":
        want_4b = args.model in ("4b", "both")
        want_9b = args.model in ("9b", "both")

        if want_4b:
            print("\n--- FLUX.2 Klein 4B base model (~8GB) ---")
            dest_dir = os.path.join(models_root, "flux2_klein_4b_local")
            os.makedirs(dest_dir, exist_ok=True)
            download_hf_file(HF_KLEIN_4B_REPO, HF_KLEIN_4B_FILE, dest_dir, token)
            print("--- Pre-fetching Qwen3-4B text encoder (used by the 4B model) ---")
            prefetch_hf_snapshot(HF_QWEN3_4B_REPO, token)

        if want_9b:
            print("\n--- FLUX.2 Klein 9B base model (~18GB) ---")
            dest_dir = os.path.join(models_root, "flux2_klein_9b_local")
            os.makedirs(dest_dir, exist_ok=True)
            download_hf_file(HF_KLEIN_9B_REPO, HF_KLEIN_9B_FILE, dest_dir, token)
            print("--- Pre-fetching Qwen3-8B text encoder (used by the 9B model) ---")
            prefetch_hf_snapshot(HF_QWEN3_8B_REPO, token)

    if not args.skip_face_models:
        print("\n--- Face detection / identity / upscale models (~300MB total) ---")
        face_dir = os.path.join(REPO_ROOT, "models", "face")
        upscale_dir = os.path.join(REPO_ROOT, "models", "upscale")
        for fname, url in FACE_MODELS.items():
            download_url(url, os.path.join(face_dir, fname))
        for fname, url in UPSCALE_MODELS.items():
            download_url(url, os.path.join(upscale_dir, fname))

    print("\n" + "=" * 70)
    print(" MODEL DOWNLOAD COMPLETE")
    print("=" * 70)
    print("""
NEXT STEP:
  python -m pipeline.run_full_pipeline --input_dir "<photos folder>" --subject_name mysubject
  (or double-click RUN_FULL_PIPELINE.bat on Windows)
""")


if __name__ == "__main__":
    main()
