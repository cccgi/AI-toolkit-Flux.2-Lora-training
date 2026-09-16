"""
Setup Stage 1: Automated Installer

Run this FIRST on a fresh machine with nothing installed yet. It will:

  1. Verify Python 3.10-3.12 is available (3.13+ breaks insightface/onnxruntime/
     torchcodec wheels on Windows — see ai-toolkit-perceptual's own README).
  2. Clone the ai-toolkit-perceptual fork (adds ArcFace identity-loss +
     depth-consistency training support on top of vanilla Ostris ai-toolkit)
     next to this repo, unless it's already there.
  3. Create a dedicated venv for it and install torch + requirements.txt
     (auto-detects your GPU: cu126 for most NVIDIA cards, cu128 for RTX 50-series/
     Blackwell — the cu126 build has no kernels for Blackwell and training dies
     with "CUDA error: no kernel image is available for execution on the device").
  4. Create a lightweight venv for THIS repo's own pipeline scripts (ingestion,
     captioning, evaluation — needs onnxruntime/opencv/spandrel, NOT torch/GPU
     training deps, so it installs fast and stays isolated from the trainer env).
  5. Write setup/paths.json recording every resolved path, so the rest of the
     pipeline (pipeline/*.py) never has to guess or hardcode anything.

This does NOT download the multi-GB base models — run
setup/download_models.py after this finishes for that.

Usage:
    python setup/install.py [--ai-toolkit-dir PATH] [--gpu {auto,cu126,cu128,cpu}]
"""
import os
import sys
import json
import shutil
import argparse
import subprocess
import venv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORK_URL = "https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual.git"
MIN_PY = (3, 10)
MAX_PY = (3, 12)


def run(cmd, **kwargs):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def find_compatible_python():
    """Finds a Python 3.10-3.12 interpreter. Prefers the running interpreter if
    it's already in range, else searches common Windows/conda install spots."""
    v = sys.version_info
    if MIN_PY <= (v.major, v.minor) <= MAX_PY:
        return sys.executable

    candidates = []
    if os.name == "nt":
        for base in [
            r"C:\Python312", r"C:\Python311", r"C:\Python310",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python312"),
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python311"),
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python310"),
        ]:
            p = os.path.join(base, "python.exe")
            if os.path.exists(p):
                candidates.append(p)
        for exe in ["python3.12", "python3.11", "python3.10"]:
            found = shutil.which(exe)
            if found:
                candidates.append(found)
    else:
        for exe in ["python3.12", "python3.11", "python3.10"]:
            found = shutil.which(exe)
            if found:
                candidates.append(found)

    for c in candidates:
        try:
            out = subprocess.run([c, "--version"], capture_output=True, text=True, timeout=10)
            print(f"Found candidate: {c} -> {out.stdout.strip()}{out.stderr.strip()}")
            return c
        except Exception:
            continue

    return None


def detect_gpu():
    """Best-effort GPU family detection for choosing the right torch CUDA build."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        name = out.stdout.strip().lower()
        print(f"Detected GPU: {out.stdout.strip()}")
        # RTX 50-series / Blackwell needs cu128, not cu126
        if any(tag in name for tag in ["rtx 50", "5090", "5080", "5070", "5060", "blackwell", "b100", "b200"]):
            return "cu128"
        if name:
            return "cu126"
    except Exception as e:
        print(f"Could not run nvidia-smi ({e}) — assuming no NVIDIA GPU / CPU only.")
    return "cpu"


def venv_python(venv_dir):
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def create_venv(target_python, venv_dir):
    if os.path.exists(venv_python(venv_dir)):
        print(f"venv already exists at {venv_dir}, reusing it.")
        return
    print(f"Creating venv at {venv_dir} using {target_python} ...")
    run([target_python, "-m", "venv", venv_dir])


def install_ai_toolkit_perceptual(ai_toolkit_dir, target_python, gpu_choice):
    if os.path.exists(os.path.join(ai_toolkit_dir, "run.py")):
        print(f"ai-toolkit-perceptual already present at {ai_toolkit_dir}, skipping clone.")
    else:
        parent = os.path.dirname(ai_toolkit_dir)
        os.makedirs(parent, exist_ok=True)
        run(["git", "clone", FORK_URL, ai_toolkit_dir])

    venv_dir = os.path.join(ai_toolkit_dir, "venv")
    create_venv(target_python, venv_dir)
    py = venv_python(venv_dir)

    gpu = detect_gpu() if gpu_choice == "auto" else gpu_choice
    print(f"Installing torch for GPU profile: {gpu}")

    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    if gpu == "cu128":
        run([py, "-m", "pip", "install", "--no-cache-dir",
             "torch==2.11.0", "torchvision", "torchaudio",
             "--index-url", "https://download.pytorch.org/whl/cu128"])
    elif gpu == "cu126":
        run([py, "-m", "pip", "install", "--no-cache-dir",
             "torch==2.7.0", "torchvision==0.22.0", "torchaudio==2.7.0",
             "--index-url", "https://download.pytorch.org/whl/cu126"])
    else:
        print("WARNING: No NVIDIA GPU detected. Installing CPU-only torch — "
              "training will be extremely slow to the point of impractical. "
              "This is fine for testing the pipeline end-to-end on tiny "
              "step counts, not for real training runs.")
        run([py, "-m", "pip", "install", "--no-cache-dir", "torch", "torchvision", "torchaudio"])

    req_path = os.path.join(ai_toolkit_dir, "requirements.txt")
    run([py, "-m", "pip", "install", "-r", req_path])

    return py, gpu


def install_pipeline_venv(target_python):
    """Lightweight venv for this repo's own ingestion/captioning/eval scripts."""
    venv_dir = os.path.join(REPO_ROOT, ".venv")
    create_venv(target_python, venv_dir)
    py = venv_python(venv_dir)
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    req_path = os.path.join(REPO_ROOT, "setup", "requirements-pipeline.txt")
    run([py, "-m", "pip", "install", "-r", req_path])
    return py


def main():
    parser = argparse.ArgumentParser(description="Automated installer for the Flux.2 Klein ArcFace LoRA pipeline")
    parser.add_argument("--ai-toolkit-dir", default=None,
                         help="Where to clone/find ai-toolkit-perceptual (default: sibling folder next to this repo)")
    parser.add_argument("--models-root", default=None,
                         help="Where base models will live (default: <ai-toolkit-dir>/../ai-toolkit-models)")
    parser.add_argument("--datasets-root", default=None,
                         help="Where prepared training datasets will live (default: <repo>/datasets)")
    parser.add_argument("--output-root", default=None,
                         help="Where training run outputs will be saved (default: <ai-toolkit-dir>/output)")
    parser.add_argument("--gpu", default="auto", choices=["auto", "cu126", "cu128", "cpu"],
                         help="Torch CUDA build to install (auto-detects by default)")
    parser.add_argument("--skip-ai-toolkit-install", action="store_true",
                         help="Skip cloning/installing ai-toolkit-perceptual (use if you already have it fully set up — you'll need to fill paths.json manually)")
    args = parser.parse_args()

    print("=" * 70)
    print(" Flux.2 Klein ArcFace LoRA Pipeline — Automated Installer")
    print("=" * 70)

    py_for_venvs = find_compatible_python()
    if not py_for_venvs:
        print("\nERROR: No compatible Python (3.10, 3.11, or 3.12) found.")
        print("Install Python 3.12 from https://www.python.org/downloads/ and re-run.")
        print("(Python 3.13+ breaks insightface/onnxruntime/torchcodec wheels on Windows.)")
        sys.exit(1)
    print(f"Using {py_for_venvs} for venv creation.")

    ai_toolkit_dir = args.ai_toolkit_dir or os.path.join(os.path.dirname(REPO_ROOT), "ai-toolkit-perceptual")
    ai_toolkit_dir = os.path.abspath(ai_toolkit_dir)
    models_root = args.models_root or os.path.join(os.path.dirname(ai_toolkit_dir), "ai-toolkit-models")
    models_root = os.path.abspath(models_root)
    datasets_root = args.datasets_root or os.path.join(REPO_ROOT, "datasets")
    output_root = args.output_root or os.path.join(ai_toolkit_dir, "output")

    perceptual_python = None
    gpu_used = args.gpu
    if not args.skip_ai_toolkit_install:
        print(f"\n--- Installing ai-toolkit-perceptual at {ai_toolkit_dir} ---")
        perceptual_python, gpu_used = install_ai_toolkit_perceptual(ai_toolkit_dir, py_for_venvs, args.gpu)
    else:
        perceptual_python = venv_python(os.path.join(ai_toolkit_dir, "venv"))
        print(f"\nSkipping ai-toolkit-perceptual install. Assuming venv python at: {perceptual_python}")

    print(f"\n--- Installing this repo's pipeline venv ---")
    pipeline_python = install_pipeline_venv(py_for_venvs)

    os.makedirs(datasets_root, exist_ok=True)
    os.makedirs(models_root, exist_ok=True)
    os.makedirs(output_root, exist_ok=True)

    paths = {
        "ai_toolkit_perceptual_root": ai_toolkit_dir,
        "ai_toolkit_perceptual_python": perceptual_python,
        "pipeline_python": pipeline_python,
        "models_root": models_root,
        "datasets_root": datasets_root,
        "output_root": output_root,
        "gpu_profile": gpu_used,
    }
    paths_file = os.path.join(REPO_ROOT, "setup", "paths.json")
    with open(paths_file, "w", encoding="utf-8") as f:
        json.dump(paths, f, indent=2)

    print("\n" + "=" * 70)
    print(" INSTALL COMPLETE")
    print("=" * 70)
    print(json.dumps(paths, indent=2))
    print(f"\nSaved to: {paths_file}")
    print("""
NEXT STEPS:
  1. Run: python setup/download_models.py
     (downloads FLUX.2 Klein base model(s) + face/upscale models — several GB)
  2. Run: python -m pipeline.run_full_pipeline --input_dir "<photos folder>" --subject_name mysubject
     (or double-click RUN_FULL_PIPELINE.bat on Windows for a guided prompt flow)
""")


if __name__ == "__main__":
    main()
