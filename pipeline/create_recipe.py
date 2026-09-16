"""
Stage 3: 1-Click Recipe & Config Engine (ArcFace-aware, portable)

Generates training configs targeting the ai-toolkit-perceptual fork
(https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual), which
adds ArcFace identity-anchor + depth-consistency losses on top of Ostris ai-toolkit.

Everything below encodes lessons learned across many real training attempts on
FLUX.2 Klein 4B/9B (see docs/LESSONS_LEARNED.md for the full forensic history):

  - LoRA rank must be proportional to model size to actually steer identity:
    rank64 works well for 4B and is the recommended baseline for 9B too
    (NOT rank32 — that under-powers 9B's much deeper cross-attention stack
    relative to 4B, and 9B ends up looking WORSE than 4B despite being bigger).
  - Train at NATIVE resolution (1024), not downscaled (768) — 768 caps identity
    precision, especially on 9B where fine facial detail matters more.
  - identity_loss_weight sweet spot is ~0.08-0.10. Above ~0.10-0.12 on a small
    (~20 image) dataset the ArcFace loss dominates the diffusion loss,
    producing facial distortions (misaligned pupils, jagged eye contours,
    "carved" plastic-skin artifacts) and killing prompt flexibility.
  - CRITICAL PITFALL: on 9B, `low_vram: true` + `gradient_checkpointing: true`
    together cause a catastrophic ~16-minute-per-step slowdown (confirmed via
    nvidia-smi — GPU sits at low utilization, thrashing CPU<->GPU). The fix is
    `low_vram: false` + `layer_offloading: true` instead, which uses proper
    block-swap CPU offload without that penalty (smoke-tested clean at ~32s/
    step for 9B rank64 @ 1024 on a single RTX 3090).
  - `diff_output_preservation` conflicts with `cache_text_embeddings` (raises
    ValueError) AND crashes on backward-through-graph-twice when combined with
    ArcFace's own anchors — leave it OFF; it doesn't add anything ArcFace loss
    doesn't already provide.
  - `layer_offloading: true` auto-converts qtype 'qfloat8' -> 'float8'
    internally (fork behavior, not a bug) — no manual action needed.
  - Always run the 1-step smoke config first. A real crash on step 1 saves
    hours vs. discovering it 90 minutes into a 2000-step run.

Paths to your ai-toolkit-perceptual install, conda env, and base models are
read from setup/paths.json (written by setup/install.py during setup) so this
script works on any machine after a normal install — no hardcoded personal
drive letters.
"""
import os
import sys
import json
import yaml
import argparse

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATHS_FILE = os.path.join(_REPO_ROOT, "setup", "paths.json")


def load_paths():
    if not os.path.exists(_PATHS_FILE):
        raise FileNotFoundError(
            f"{_PATHS_FILE} not found. Run 'python setup/install.py' first — "
            f"it detects/creates your ai-toolkit-perceptual install, conda env, "
            f"and base model locations, and writes this file."
        )
    with open(_PATHS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_model_presets(paths):
    """Model presets built from the resolved install paths in paths.json."""
    perceptual_root = paths["ai_toolkit_perceptual_root"]
    models_root = paths["models_root"]
    return {
        "flux2_klein_4b": {
            "arch": "flux2_klein_4b",
            "name_or_path": os.path.join(models_root, "flux2_klein_4b_local"),
            "te_name_or_path": None,  # defaults to Qwen/Qwen3-4B (auto HF download on first run)
            "quantize": True, "qtype": "qfloat8",
            "quantize_te": True, "qtype_te": "qfloat8",
            "low_vram": True, "layer_offloading": False,
            "default_lr": 1e-4, "default_steps": 1500,
            "default_rank": 64,
            "default_identity_loss_weight": 0.08,
            "default_resolution": 1024,
            "vram_gate_mb": 6000,
        },
        "flux2_klein_4b_strong": {
            # Higher-capacity variant for pushing identity harder without
            # crossing the ~0.10-0.12 ArcFace over-bake threshold.
            "arch": "flux2_klein_4b",
            "name_or_path": os.path.join(models_root, "flux2_klein_4b_local"),
            "te_name_or_path": None,
            "quantize": True, "qtype": "qfloat8",
            "quantize_te": True, "qtype_te": "qfloat8",
            "low_vram": True, "layer_offloading": False,
            "default_lr": 1e-4, "default_steps": 1800,
            "default_rank": 96,
            "default_identity_loss_weight": 0.09,
            "default_resolution": 1024,
            "vram_gate_mb": 6000,
        },
        "flux2_klein_9b": {
            "arch": "flux2_klein_9b",
            "name_or_path": os.path.join(models_root, "flux2_klein_9b_local"),
            "te_name_or_path": None,  # defaults to Qwen/Qwen3-8B (auto HF download on first run)
            "quantize": True, "qtype": "qfloat8",
            "quantize_te": True, "qtype_te": "qfloat8",
            # CRITICAL: low_vram=False + layer_offloading=True on 9B, NOT
            # low_vram=True + gradient_checkpointing — see module docstring.
            "low_vram": False, "layer_offloading": True,
            "default_lr": 1e-4, "default_steps": 2000,
            "default_rank": 64,  # proportional to 4B's rank64, not a weaker rank32
            "default_identity_loss_weight": 0.10,  # 9B tends to resist ID loss more than 4B
            "default_resolution": 1024,
            "vram_gate_mb": 6000,
        },
    }


# Multi-domain evaluation prompts spanning studio and natural holdout probes
DEFAULT_SAMPLE_PROMPTS_TEMPLATE = [
    "{trigger} {class_noun}, black blazer over a low-cut top, watch on left wrist, deep red studio backdrop, soft studio lighting, professional portrait",
    "{trigger} {class_noun}, seated at a cafe window, holding a coffee cup, overcast daylight, medium shot",
    "{trigger} {class_noun}, close frontal portrait, neutral expression, grey backdrop, soft studio lighting, looking at camera",
    "{trigger} {class_noun}, soft warm smile, looking at camera, casual outdoor lighting, medium close-up portrait",
    "{trigger} {class_noun}, seated in car interior, casual natural daylight, looking at camera, authentic candid portrait",
]


def create_recipe(
    model_key,
    dataset_dir,
    trigger="tkn",
    class_noun="person",
    steps=None,
    lr=None,
    rank=None,
    identity_loss_weight=None,
    mask_min_value=0.10,
    save_every=100,
    sample_every=100,
    job_name=None,
    config_out_dir=None,
    output_root=None,
    paths=None,
):
    paths = paths or load_paths()
    presets = get_model_presets(paths)
    if model_key not in presets:
        raise ValueError(f"Model '{model_key}' not supported. Available: {list(presets.keys())}")

    preset = presets[model_key]
    steps = steps or preset["default_steps"]
    lr = lr or preset["default_lr"]
    rank = rank or preset["default_rank"]
    identity_loss_weight = identity_loss_weight if identity_loss_weight is not None else preset["default_identity_loss_weight"]
    resolution = preset["default_resolution"]

    perceptual_root = paths["ai_toolkit_perceptual_root"]
    config_out_dir = config_out_dir or os.path.join(perceptual_root, "config", "auto")
    output_root = output_root or paths.get("output_root") or os.path.join(_REPO_ROOT, "output")

    train_folder = os.path.join(dataset_dir, "training") if os.path.exists(os.path.join(dataset_dir, "training")) else dataset_dir
    mask_folder = os.path.join(dataset_dir, "masks") if os.path.exists(os.path.join(dataset_dir, "masks")) else None

    if not job_name:
        dataset_name = os.path.basename(os.path.normpath(dataset_dir))
        job_name = f"{dataset_name}_{model_key}_r{rank}_{steps}steps"

    config_path = os.path.join(config_out_dir, f"{job_name}.yaml")
    smoke_config_path = os.path.join(config_out_dir, f"{job_name}_smoke.yaml")
    bat_path = os.path.join(perceptual_root, f"start_{job_name}.bat")
    sh_path = os.path.join(perceptual_root, f"start_{job_name}.sh")

    sample_prompts = [p.format(trigger=trigger, class_noun=class_noun) for p in DEFAULT_SAMPLE_PROMPTS_TEMPLATE]

    dataset_dict = {
        "folder_path": train_folder,
        "mask_min_value": float(mask_min_value),
        "default_caption": f"{trigger} {class_noun}",
        "caption_ext": "txt",
        "caption_dropout_rate": 0.05,
        "cache_latents_to_disk": True,
        "is_reg": False,
        "network_weight": 1,
        "resolution": [resolution],
        "controls": [],
        "shrink_video_to_frames": True,
        "num_frames": 1,
        "flip_x": False,
        "flip_y": False,
        "num_repeats": [1],
        "diffusion_loss_weight": 1,
        "depth_loss_weight": 0.1,
        "identity_loss_weight": identity_loss_weight,
        "loss_split": "sum",
    }
    if mask_folder and os.path.exists(mask_folder):
        dataset_dict["mask_path"] = mask_folder

    model_config = {
        "name_or_path": preset["name_or_path"],
        "quantize": preset["quantize"],
        "qtype": preset["qtype"],
        "quantize_te": preset["quantize_te"],
        "qtype_te": preset["qtype_te"],
        "arch": preset["arch"],
        "low_vram": preset["low_vram"],
        "model_kwargs": {"match_target_res": False},
        "layer_offloading": preset["layer_offloading"],
    }
    if preset.get("te_name_or_path"):
        model_config["te_name_or_path"] = preset["te_name_or_path"]

    def build_process(n_steps, is_smoke):
        return {
            "type": "diffusion_trainer",
            "training_folder": output_root,
            "sqlite_db_path": "./aitk_db.db",
            "device": "cuda",
            "trigger_word": trigger,
            "performance_log_every": 10,
            "network": {
                "type": "lora",
                "linear": rank, "linear_alpha": rank,
                "conv": 0, "conv_alpha": 0,
                "network_kwargs": {"ignore_if_contains": []},
            },
            "save": {
                "dtype": "bf16",
                "save_every": 1 if is_smoke else save_every,
                "max_step_saves_to_keep": 1 if is_smoke else 30,
                "save_format": "diffusers",
                "push_to_hub": False,
            },
            "datasets": [dataset_dict],
            "train": {
                "weight_noise": {"enabled": True, "mode": "relative", "sigma": 0.0125, "log_every": 1},
                "max_grad_norm": 1,
                "batch_size": 1,
                "bypass_guidance_embedding": False,
                "steps": n_steps,
                "gradient_accumulation": 1,
                "train_unet": True,
                "train_text_encoder": False,
                "gradient_checkpointing": True,
                "noise_scheduler": "flowmatch",
                "optimizer": "adamw8bit",
                "timestep_type": "weighted",
                "content_or_style": "balanced",
                "optimizer_params": {"weight_decay": 0.0001},
                "unload_text_encoder": False,
                "cache_text_embeddings": False,  # must stay False; conflicts w/ diff_output_preservation & ArcFace anchors
                "lr": lr,
                "lr_scheduler": "cosine",
                "lr_scheduler_params": {"total_iters": n_steps},
                "ema_config": {"use_ema": False},
                "skip_first_sample": is_smoke,
                "force_first_sample": not is_smoke,
                "disable_sampling": is_smoke,
                "dtype": "bf16",
                "diff_output_preservation": False,  # see module docstring — do not enable
                "diff_output_preservation_multiplier": 1.0,
                "diff_output_preservation_class": class_noun,
                "loss_type": "mse",
                "gradient_cosine_log_every": 50,
            },
            "logging": {"log_every": 10, "use_ui_logger": False},
            "model": model_config,
            "sample": {
                "sampler": "flowmatch",
                "sample_every": sample_every,
                "width": resolution, "height": resolution,
                "samples": [{"prompt": p} for p in sample_prompts],
                "neg": "",
                "seed": 42,
                "walk_seed": False,
                "guidance_scale": 4,
                "sample_steps": 25,
                "num_frames": 1,
                "fps": 1,
            },
            "face_id": {
                "enabled": True,
                "init_scale": 0.3,
                "identity_loss_weight": identity_loss_weight,
                "landmark_loss_weight": 0,
                "body_proportion_loss_weight": 0,
                "body_shape_loss_weight": 0,
                "identity_loss_min_t": 0,
                "identity_loss_max_t": 0.9,
                "identity_metrics": True,
                "identity_loss_use_average": True,
                "identity_loss_min_cos": 0.2,
                "identity_loss_preview_max_keep": 500,
            },
            "depth_consistency": {
                "loss_weight": 0.1,
                "input_size": 518,
                "preview_every": 100,
                "mask_source": "subject",
                "loss_max_t": 1,
                "model_id": "depth-anything/Depth-Anything-V2-Small-hf",
                "loss_min_t": 0,
                "grad_checkpoint": True,
            },
            "subject_mask": {"enabled": False},
        }

    prod_cfg = {
        "job": "extension",
        "config": {
            "name": job_name,
            "process": [build_process(steps, is_smoke=False)],
        },
        "meta": {"name": job_name, "version": "v1-arcface-portable"},
    }
    smoke_cfg = {
        "job": "extension",
        "config": {
            "name": f"{job_name}_smoke",
            "process": [build_process(1, is_smoke=True)],
        },
        "meta": {"name": f"{job_name}_smoke", "version": "v1-arcface-portable"},
    }

    os.makedirs(config_out_dir, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(prod_cfg, f, default_flow_style=False, sort_keys=False)
    with open(smoke_config_path, "w", encoding="utf-8") as f:
        yaml.dump(smoke_cfg, f, default_flow_style=False, sort_keys=False)

    perceptual_python = paths["ai_toolkit_perceptual_python"]
    run_output_dir = os.path.join(output_root, job_name)

    # Windows .bat launcher: VRAM gate -> smoke test -> full run
    bat_content = f"""@echo off
setlocal
title {job_name} - 1-Click ArcFace LoRA Training
cd /d "{perceptual_root}"

echo =========================================================
echo  {job_name}
echo  Model: {model_key}  Rank: {rank}  Res: {resolution}  Steps: {steps}
echo  identity_loss_weight: {identity_loss_weight}
echo =========================================================
echo.

echo [Gate] Checking GPU has enough free VRAM before starting...
"{perceptual_python}" -c "import subprocess; out=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip(); print('GPU:',out); used=int(out.split(',')[0].strip().split()[0]); exit(1) if used>{preset['vram_gate_mb']} else exit(0)"
if errorlevel 1 (
    echo.
    echo [ABORT] GPU already has more than {preset['vram_gate_mb']}MB in use.
    echo         Check nvidia-smi / Task Manager's GPU tab to confirm whether
    echo         another job is genuinely running before re-running this.
    echo         This script will NOT kill anything for you.
    echo.
    pause
    exit /b 1
)

echo [Gate] GPU is free. Running a 1-step smoke test first...
"{perceptual_python}" run.py "{smoke_config_path}" -l "{run_output_dir}_smoke\\console.log"
if errorlevel 1 (
    echo.
    echo [ABORT] Smoke test failed. Check the traceback above before
    echo         attempting the full {steps}-step run.
    echo.
    pause
    exit /b 1
)

echo.
echo [OK] Smoke test passed. Starting full run ({steps} steps)...
echo      Log: {run_output_dir}\\console.log
echo.
"{perceptual_python}" run.py "{config_path}" -l "{run_output_dir}\\console.log"

echo.
echo [DONE] Training finished or exited. Check the log above / console.log.
pause
"""
    with open(bat_path, "w", encoding="utf-8") as f:
        f.write(bat_content)

    # Linux/macOS .sh launcher equivalent
    sh_content = f"""#!/usr/bin/env bash
set -e
cd "{perceptual_root}"

echo "========================================================="
echo " {job_name}"
echo " Model: {model_key}  Rank: {rank}  Res: {resolution}  Steps: {steps}"
echo " identity_loss_weight: {identity_loss_weight}"
echo "========================================================="
echo

echo "[Gate] Checking GPU has enough free VRAM before starting..."
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$USED" -gt {preset['vram_gate_mb']} ]; then
    echo
    echo "[ABORT] GPU already has more than {preset['vram_gate_mb']}MB in use ($USED MB)."
    echo "        Check nvidia-smi to confirm whether another job is genuinely"
    echo "        running before re-running this. This script will NOT kill anything."
    exit 1
fi

echo "[Gate] GPU is free. Running a 1-step smoke test first..."
"{perceptual_python}" run.py "{smoke_config_path}" -l "{run_output_dir}_smoke/console.log"

echo
echo "[OK] Smoke test passed. Starting full run ({steps} steps)..."
echo "     Log: {run_output_dir}/console.log"
echo
"{perceptual_python}" run.py "{config_path}" -l "{run_output_dir}/console.log"

echo
echo "[DONE] Training finished or exited. Check the log above / console.log."
"""
    with open(sh_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(sh_content)
    try:
        os.chmod(sh_path, 0o755)
    except OSError:
        pass

    print("=== 1-Click ArcFace Recipe Created ===")
    print(f"Target Model: {model_key} (steps: {steps}, lr: {lr}, rank: {rank}, id_loss: {identity_loss_weight}, res: {resolution})")
    print(f"Config YAML: {config_path}")
    print(f"Smoke YAML:  {smoke_config_path}")
    print(f"Windows Launcher: {bat_path}")
    print(f"Linux/macOS Launcher: {sh_path}")
    return bat_path, sh_path, config_path, smoke_config_path, run_output_dir


if __name__ == "__main__":
    paths = load_paths()
    presets = get_model_presets(paths)

    parser = argparse.ArgumentParser(description="1-Click ArcFace Recipe & Config Generator")
    parser.add_argument("--model", default="flux2_klein_4b", choices=list(presets.keys()), help="Target model preset")
    parser.add_argument("--dataset_dir", required=True, help="Path to prepared dataset directory")
    parser.add_argument("--trigger", default="tkn", help="Trigger word (short, unique, not a real word)")
    parser.add_argument("--class_noun", default="person", help="Class noun (e.g. woman, man, person)")
    parser.add_argument("--steps", type=int, help="Total training steps")
    parser.add_argument("--lr", type=float, help="Initial learning rate")
    parser.add_argument("--rank", type=int, help="LoRA rank")
    parser.add_argument("--identity_loss_weight", type=float, help="ArcFace identity loss weight (0.08-0.10 safe range)")
    parser.add_argument("--mask_min_value", type=float, default=0.10, help="Regional loss background weight")
    parser.add_argument("--save_every", type=int, default=100, help="Checkpoint interval")
    parser.add_argument("--job_name", help="Custom job name")

    args = parser.parse_args()
    create_recipe(
        model_key=args.model,
        dataset_dir=args.dataset_dir,
        trigger=args.trigger,
        class_noun=args.class_noun,
        steps=args.steps,
        lr=args.lr,
        rank=args.rank,
        identity_loss_weight=args.identity_loss_weight,
        mask_min_value=args.mask_min_value,
        save_every=args.save_every,
        sample_every=args.save_every,
        job_name=args.job_name,
        paths=paths,
    )
