# Flux.2 Klein ArcFace LoRA Training Pipeline

A fully automated, end-to-end LoRA training pipeline for **FLUX.2 Klein (4B/9B)**
that solves the single hardest problem in identity LoRA training: **generic
"beauty attractor" collapse**, where the model drifts toward a generic
good-looking face instead of the actual person's specific bone structure.

The fix is **ArcFace identity-anchor loss + depth-consistency loss** during
training (via the [ai-toolkit-perceptual](https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual)
fork of [Ostris ai-toolkit](https://github.com/ostris/ai-toolkit)), combined
with a disentangled captioning strategy and forensic holdout evaluation that
actually measures identity fidelity mathematically instead of eyeballing it.

Everything here was built and validated across many real training attempts —
see [`docs/LESSONS_LEARNED.md`](docs/LESSONS_LEARNED.md) for the full forensic
history of what didn't work and why, before landing on this pipeline.

**Designed for:** a fresh Windows/Linux machine with an NVIDIA GPU, nothing
installed yet — no Python packages, no venv, no models, no ai-toolkit. Two
setup scripts get you from nothing to a working double-click training pipeline.

---

## What this actually does differently

Standard LoRA training on a diffusion loss alone tends to converge toward
whatever face the base model finds "easiest" / most represented in its
pretraining data — a generic, symmetric, conventionally-attractive face —
rather than the specific person's actual bone structure, especially under
heavy styling/makeup/lighting variation in prompts. More training steps alone
often makes this *worse*, not better (the "attractor" gets stronger as the
diffusion loss converges).

This pipeline adds:
1. **ArcFace identity-anchor loss** — a real face-recognition model
   (ArcFace, same technology used in face-unlock security systems) scores
   how close each training-step's generated face is to the true reference
   identity, and backpropagates that as an additional loss term. This
   directly penalizes drift toward a "different but pretty" face.
2. **Depth-consistency loss** — keeps the 3D facial geometry (not just 2D
   pixel similarity) consistent with the reference, catching cases where a
   generated face looks similar in a flat 2D sense but has drifted structurally.
3. **Identity-disentangled captioning** — captions describe everything
   EXCEPT permanent facial bone structure (clothing, lighting, hairstyle,
   background, expression), forcing the model to bind facial geometry
   entirely to the trigger token rather than learning it's describable/
   variable.
4. **True holdout evaluation** — a portion of your source photos are NEVER
   shown to the trainer, reserved purely to mathematically score final
   identity fidelity (ArcFace cosine similarity) — not just "does this look
   right to me," which is subject to confirmation bias once you've stared at
   generated samples for hours.

---

## Prerequisites

- **Windows or Linux**, NVIDIA GPU (12GB+ VRAM recommended for 4B, 24GB+ for
  9B — see the [rank/resolution/VRAM notes](docs/LESSONS_LEARNED.md)).
- **Python 3.10, 3.11, or 3.12** (NOT 3.13+ — insightface/onnxruntime/
  torchcodec don't ship Windows wheels for 3.13 yet; the installer will find
  a compatible interpreter for you if your system default is newer).
- **git**.
- ~30-50GB free disk space (base model + venvs + your training data/outputs).
- A [HuggingFace account](https://huggingface.co/join) + access token, since
  the FLUX.2 Klein base models are gated (you must accept the license on the
  model page first). Free, takes 2 minutes.

You do **not** need to have ai-toolkit, PyTorch, CUDA toolkit, or any Python
packages pre-installed — the setup scripts below handle everything.

---

## Setup (run once)

### 1. Clone this repo

```bash
git clone https://github.com/cccgi/AI-toolkit-Flux.2-Lora-training.git
cd AI-toolkit-Flux.2-Lora-training
```

### 2. Get a HuggingFace token and accept the model licenses

1. Create a token at https://huggingface.co/settings/tokens (Read access is enough).
2. Accept the gated license on the model page(s) you plan to use (while logged in):
   - https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B (for 4B)
   - https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B (for 9B)
3. Set the token as an environment variable so the downloader can use it:
   ```bash
   # Windows (PowerShell)
   $env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
   # Linux/macOS
   export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
   ```

### 3. Run the installer

```bash
python setup/install.py
```

This will:
- Find (or tell you to install) a compatible Python 3.10-3.12 interpreter.
- Clone `ai-toolkit-perceptual` into a sibling folder next to this repo
  (e.g. `../ai-toolkit-perceptual`), unless you already have it — pass
  `--ai-toolkit-dir "C:\path\to\existing\install"` to point at one.
- Create a dedicated venv for it and install PyTorch (auto-detects your GPU —
  RTX 50-series/Blackwell cards get the `cu128` build, everything else gets
  `cu126`; override with `--gpu cu126`/`--gpu cu128`/`--gpu cpu` if detection
  guesses wrong) plus all training dependencies from `requirements.txt`.
- Create a separate lightweight `.venv` for this repo's own pipeline scripts
  (face detection, captioning, evaluation — CPU-only torch, no CUDA needed
  here since these are small models doing quick inference, not training).
- Write `setup/paths.json` recording every resolved path — nothing else in
  this repo hardcodes a path; it's all read from this file.

This step takes a while (compiling/downloading PyTorch + ~40 packages).

### 4. Download the base models

```bash
python setup/download_models.py --model 4b
```

Downloads (skips anything already present, safe to re-run if interrupted):
- FLUX.2 Klein 4B base model (~8GB) — use `--model 9b` for the 9B model
  instead (~18GB), or `--model both` for both.
- Qwen3-4B (or Qwen3-8B) text encoder, pre-fetched into the shared
  HuggingFace cache (also downloads automatically on first training run if
  you skip this — pre-fetching just avoids a long silent pause on your first
  smoke test).
- ~300MB of small face-detection/identity/upscale models this repo's own
  ingestion pipeline needs (YOLOFace for face detection, ArcFace w600k_r50
  for identity embeddings/masks, 4x-UltraSharp for upscaling small photos) —
  these are NOT the same models as the ArcFace *training loss*, which is
  loaded separately by `ai-toolkit-perceptual` itself at training time from
  its own model cache.

If a download fails with a 401/403, you haven't accepted the model's license
yet (or `HF_TOKEN` isn't set) — the error message links to the exact page.

---

## Training a LoRA (every time after setup)

### Option A: Double-click / guided prompts

- **Windows:** double-click `RUN_FULL_PIPELINE.bat`
- **Linux/macOS:** `./run_full_pipeline.sh`

Answer the prompts (folder of photos, subject name, trigger word, model
preset, captioning method). This runs Stages 1-3 below, then prints the path
to a training launcher script.

### Option B: Command line

```bash
python -m pipeline.run_full_pipeline \
  --input_dir "/path/to/raw_photos_of_subject" \
  --subject_name alex \
  --trigger tkn \
  --class_noun person \
  --model flux2_klein_4b
```

### What happens

```
[Raw Photos]
     │
     ▼ Stage 1: Ingestion & Quality Gating
[Framed Photos + Precision Face Masks + Reserved True Holdouts]
     │
     ▼ Stage 2: Disentangled Captioning
[Synchronized Image-Caption Pairs, Identity Bone Structure Left Uncaptioned]
     │
     ▼ Stage 3: 1-Click Recipe Engine (ArcFace-fork aware)
[Optimized Training Config + Smoke-Test Config + Launcher Script]
     │
     ▼ Training Execution on GPU (you run the launcher script)
[Checkpoints & Multi-Domain Evaluation Samples]
     │
     ▼ Stage 4: Forensic Evaluation (run manually after training finishes)
[ArcFace Cosine Identity Scorecard + Visual Holdout Audit Grid + Peak Checkpoint]
```

**Stage 4 is not auto-chained** — training takes hours, so once it finishes,
run:

```bash
python -m pipeline.evaluate_run \
  --run_dir "<path printed by Stage 3>" \
  --holdout_dir "<dataset>/holdouts"
```

This outputs `holdout_evaluation_grid.jpg` (visual comparison of every saved
checkpoint against the true holdout photos) and `evaluation_report.md` (a
step-by-step ArcFace cosine similarity table with the peak checkpoint
auto-detected and a drift warning if the model started degrading late in
training).

---

## Model presets

| Preset | Rank | Resolution | Steps | ArcFace loss weight | Use case |
|---|---|---|---|---|---|
| `flux2_klein_4b` | 64 | 1024 (native) | 1500 | 0.08 | **Recommended default.** Fast, strong identity fidelity. |
| `flux2_klein_4b_strong` | 96 | 1024 (native) | 1800 | 0.09 | Push identity harder without crossing the over-bake threshold (see below). |
| `flux2_klein_9b` | 64 | 1024 (native) | 2000 | 0.10 | Larger base model. Needs `layer_offloading` (see pitfall below) and significantly more VRAM/time. |

Override any of `--rank`, `--steps`, `--identity_loss_weight` on the CLI to
deviate from a preset. See `pipeline/create_recipe.py`'s module docstring and
[`docs/LESSONS_LEARNED.md`](docs/LESSONS_LEARNED.md) for why these specific
defaults were chosen and what happens if you push them too far.

---

## Repository layout

```
pipeline/
  ingest_dataset.py      Stage 1: quality gate, smart-crop, upscale, mask, holdout split
  caption_engine.py      Stage 2: identity-disentangled captioning (template or API-based)
  create_recipe.py       Stage 3: generates ArcFace training config + launcher scripts
  evaluate_run.py        Stage 4: ArcFace holdout scoring, visual grid, markdown report
  face_engine.py         Shared: YOLOFace detection + ArcFace embeddings (used by 1, 2, 4)
  upscale_engine.py      Shared: ESRGAN upscaling for undersized source photos
  run_full_pipeline.py   Orchestrates Stages 1-3 in one command
setup/
  install.py             Clones ai-toolkit-perceptual, creates venvs, installs torch/deps
  download_models.py     Downloads base models + small face/upscale models
  requirements-pipeline.txt  Lightweight deps for this repo's own scripts (no GPU torch)
  paths.json             Generated by install.py — all resolved paths live here
docs/
  LESSONS_LEARNED.md     Forensic history: what failed, why, and the fixes that worked
RUN_FULL_PIPELINE.bat    Windows double-click entry point
run_full_pipeline.sh     Linux/macOS entry point
```

---

## Troubleshooting

- **"No compatible Python found"** — install Python 3.12 from
  python.org and re-run `setup/install.py`.
- **401/403 downloading base models** — accept the license on the
  HuggingFace model page while logged in, set `HF_TOKEN`, re-run
  `setup/download_models.py` (already-downloaded files are skipped).
- **`CUDA error: no kernel image is available for execution on the device`**
  — you have an RTX 50-series (Blackwell) card and got the wrong torch
  build; re-run `setup/install.py --gpu cu128`.
- **Training on 9B is catastrophically slow (~15+ min/step)** — you have
  `low_vram: true` + `gradient_checkpointing: true` together, which thrashes
  CPU↔GPU memory. The 9B preset already avoids this
  (`low_vram: false` + `layer_offloading: true`) — if you're hand-editing a
  config, don't combine those two settings on 9B.
- **Faces look "close but not quite right" / drift toward a generic look
  under heavy styling** — this is the hardest remaining failure mode even
  with ArcFace loss. See `docs/LESSONS_LEARNED.md` for what helps (more/
  varied training photos, `flux2_klein_4b_strong` preset, lower inference
  CFG + identity-protective negative prompts).

---

## License

This repository's own code (`pipeline/`, `setup/`) is MIT licensed — see
[LICENSE](LICENSE). `ai-toolkit-perceptual` (MIT) and `ai-toolkit` (MIT) are
separate projects you install alongside this one. FLUX.2 Klein base models
are subject to Black Forest Labs' own model license — read it on the
HuggingFace model page before use.
