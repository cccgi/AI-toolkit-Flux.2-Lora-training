"""
Master Orchestrator: Raw Photos -> Trained ArcFace LoRA (1 command, 4 stages)

Chains the entire pipeline:

  Stage 1 (ingest_dataset)    - quality-gate, smart-crop, upscale, mask, reserve holdouts
  Stage 2 (caption_engine)    - disentangled captioning (identity vs. styling)
  Stage 3 (create_recipe)     - ArcFace-fork training config + smoke config + launcher
  Stage 4 (evaluate_run)      - run manually AFTER training finishes (hours later) —
                                see the printed instructions at the end.

Usage:

    python -m pipeline.run_full_pipeline \\
        --input_dir "/path/to/raw_photos_of_subject" \\
        --subject_name my_subject \\
        --trigger tkn \\
        --class_noun woman \\
        --model flux2_klein_4b
"""
import os
import sys
import json
import argparse

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from pipeline.ingest_dataset import ingest_dataset
from pipeline.caption_engine import process_dataset_captions
from pipeline.create_recipe import create_recipe, load_paths, get_model_presets


def run_full_pipeline(
    input_dir,
    subject_name,
    trigger="tkn",
    class_noun="person",
    model="flux2_klein_4b",
    target_res=1024,
    holdout_ratio=0.15,
    min_blur=75.0,
    no_upscale=False,
    caption_backend="template",
    caption_provider="gemini",
    caption_api_key=None,
    steps=None,
    rank=None,
    identity_loss_weight=None,
    save_every=100,
    mask_min_value=0.10,
    job_name=None,
):
    paths = load_paths()
    datasets_root = paths.get("datasets_root") or os.path.join(_REPO_ROOT, "datasets")
    dataset_dir = os.path.join(datasets_root, subject_name)

    print("#" * 70)
    print("# STAGE 1/4: Smart Ingestion & Dataset Formatter")
    print("#" * 70)
    ingest_dataset(
        input_dir=input_dir,
        output_dir=dataset_dir,
        trigger=trigger,
        class_noun=class_noun,
        target_res=target_res,
        holdout_ratio=holdout_ratio,
        min_blur=min_blur,
        enable_upscale=not no_upscale,
    )

    print()
    print("#" * 70)
    print("# STAGE 2/4: Smart Disentangled Captioning Engine")
    print("#" * 70)
    process_dataset_captions(
        dataset_dir=dataset_dir,
        trigger=trigger,
        class_noun=class_noun,
        backend=caption_backend,
        provider=caption_provider,
        api_key=caption_api_key,
        overwrite=True,  # overwrite the template captions ingest_dataset wrote with
                         # the (potentially richer, if --caption_backend api) Stage 2 ones
    )

    print()
    print("#" * 70)
    print("# STAGE 3/4: 1-Click Recipe & Config Engine (ArcFace-aware)")
    print("#" * 70)
    presets = get_model_presets(paths)
    bat_path, sh_path, config_path, smoke_config_path, run_output_dir = create_recipe(
        model_key=model,
        dataset_dir=dataset_dir,
        trigger=trigger,
        class_noun=class_noun,
        steps=steps,
        rank=rank,
        identity_loss_weight=identity_loss_weight,
        mask_min_value=mask_min_value,
        save_every=save_every,
        sample_every=save_every,
        job_name=job_name,
        paths=paths,
    )

    is_windows = os.name == "nt"
    launcher = bat_path if is_windows else sh_path

    print()
    print("#" * 70)
    print("# PIPELINE READY — Stages 1-3 complete.")
    print("#" * 70)
    print(f"""
Dataset prepared at: {dataset_dir}
  - Training images + masks + captions: {dataset_dir}/training, /masks
  - True holdouts (NEVER used in training, reserved for Stage 4 evaluation):
    {dataset_dir}/holdouts

TRAINING LAUNCHER (Stage 3 output) — run this to actually train:
  {launcher}

This launcher automatically:
  1. Checks the GPU is free (aborts safely if something else is using it)
  2. Runs a 1-step smoke test (aborts safely if the config/model/dataset has
     a real bug, before wasting hours on a broken full run)
  3. Runs the full {steps or presets[model]['default_steps']}-step training
     under ai-toolkit-perceptual (ArcFace identity-anchor + depth-consistency loss)

AFTER TRAINING FINISHES — run Stage 4 to get the forensic scorecard:
  python -m pipeline.evaluate_run \\
    --run_dir "{run_output_dir}" \\
    --holdout_dir "{dataset_dir}/holdouts"

This produces:
  - holdout_evaluation_grid.jpg  (visual audit: holdouts vs. every checkpoint)
  - evaluation_report.md         (ArcFace cosine similarity scorecard, peak
                                   checkpoint auto-detected, drift warnings)
""")
    return launcher, run_output_dir


if __name__ == "__main__":
    paths = load_paths()
    presets = get_model_presets(paths)

    parser = argparse.ArgumentParser(description="Master Orchestrator: Raw Photos -> Trained ArcFace LoRA")
    parser.add_argument("--input_dir", required=True, help="Folder of raw photos of the subject")
    parser.add_argument("--subject_name", required=True, help="Short name for this subject (used as dataset folder name)")
    parser.add_argument("--trigger", default="tkn", help="LoRA trigger word (short, unique, not a real word)")
    parser.add_argument("--class_noun", default="person", help="Class noun (e.g. woman, man, person)")
    parser.add_argument("--model", default="flux2_klein_4b", choices=list(presets.keys()), help="Target model preset")
    parser.add_argument("--target_res", type=int, default=1024, help="Target minimum resolution for ingestion/upscaling")
    parser.add_argument("--holdout_ratio", type=float, default=0.15, help="Fraction of accepted photos reserved as true holdouts")
    parser.add_argument("--min_blur", type=float, default=75.0, help="Min Laplacian variance (blur) threshold")
    parser.add_argument("--no_upscale", action="store_true", help="Disable AI upscaling in ingestion")
    parser.add_argument("--caption_backend", default="template", choices=["template", "api"], help="template = fast/free/offline, api = higher quality via Gemini/OpenAI vision")
    parser.add_argument("--caption_provider", default="gemini", choices=["gemini", "openai"])
    parser.add_argument("--caption_api_key", help="API key for vision captioning (or set GEMINI_API_KEY / OPENAI_API_KEY env var)")
    parser.add_argument("--steps", type=int, help="Override default training steps for the chosen model preset")
    parser.add_argument("--rank", type=int, help="Override default LoRA rank for the chosen model preset")
    parser.add_argument("--identity_loss_weight", type=float, help="Override default ArcFace identity loss weight (safe range 0.08-0.10)")
    parser.add_argument("--save_every", type=int, default=100, help="Checkpoint + sample interval")
    parser.add_argument("--mask_min_value", type=float, default=0.10, help="Regional loss background weight")
    parser.add_argument("--job_name", help="Custom job name (default: auto-generated from subject/model/rank/steps)")

    args = parser.parse_args()
    run_full_pipeline(
        input_dir=args.input_dir,
        subject_name=args.subject_name,
        trigger=args.trigger,
        class_noun=args.class_noun,
        model=args.model,
        target_res=args.target_res,
        holdout_ratio=args.holdout_ratio,
        min_blur=args.min_blur,
        no_upscale=args.no_upscale,
        caption_backend=args.caption_backend,
        caption_provider=args.caption_provider,
        caption_api_key=args.caption_api_key,
        steps=args.steps,
        rank=args.rank,
        identity_loss_weight=args.identity_loss_weight,
        save_every=args.save_every,
        mask_min_value=args.mask_min_value,
        job_name=args.job_name,
    )
