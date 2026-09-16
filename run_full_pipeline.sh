#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -f "setup/paths.json" ]; then
    echo "========================================================="
    echo "  FIRST-TIME SETUP REQUIRED"
    echo "========================================================="
    echo
    echo "setup/paths.json not found. Run these first:"
    echo "  1) python3 setup/install.py"
    echo "  2) python3 setup/download_models.py"
    echo
    exit 1
fi

echo "========================================================="
echo "  FULL LORA TRAINING PIPELINE (photos -> trained LoRA)"
echo "  Stage 1: Ingest & quality-gate raw photos"
echo "  Stage 2: Disentangled auto-captioning"
echo "  Stage 3: Generate ArcFace training config + launcher"
echo "========================================================="
echo
echo "This prepares everything up to (but does not itself run) training."
echo "At the end you'll get a launcher script to run when ready to"
echo "actually train (that step takes hours on the GPU)."
echo

read -rp "Folder of raw photos of the subject: " INPUT_DIR
if [ -z "$INPUT_DIR" ] || [ ! -d "$INPUT_DIR" ]; then
    echo "[ABORT] Folder not found or not given: $INPUT_DIR"
    exit 1
fi

read -rp "Short subject name (e.g. alex, no spaces): " SUBJECT_NAME
if [ -z "$SUBJECT_NAME" ]; then
    echo "[ABORT] No subject name given."
    exit 1
fi

read -rp "Trigger word - short, unique, not a real word [default: tkn]: " TRIGGER
TRIGGER=${TRIGGER:-tkn}

read -rp "Class noun [default: person]: " CLASS_NOUN
CLASS_NOUN=${CLASS_NOUN:-person}

echo
echo "Model presets:"
echo "  1 = flux2_klein_4b         (rank64, 1024px, 1500 steps  -- recommended default)"
echo "  2 = flux2_klein_4b_strong  (rank96, 1024px, 1800 steps  -- stronger identity push)"
echo "  3 = flux2_klein_9b         (rank64, 1024px, 2000 steps, layer_offloading -- much slower, needs more VRAM)"
read -rp "Choose model preset [1/2/3, default 1]: " MODEL_CHOICE
case "$MODEL_CHOICE" in
    2) MODEL=flux2_klein_4b_strong ;;
    3) MODEL=flux2_klein_9b ;;
    *) MODEL=flux2_klein_4b ;;
esac

echo
echo "Captioning backend:"
echo "  1 = template  (fast, free, fully offline heuristic captions)"
echo "  2 = api       (higher quality, needs GEMINI_API_KEY or OPENAI_API_KEY env var)"
read -rp "Choose captioning backend [1/2, default 1]: " CAP_CHOICE
if [ "$CAP_CHOICE" = "2" ]; then CAP_BACKEND=api; else CAP_BACKEND=template; fi

echo
echo "========================================================="
echo "  Input folder:    $INPUT_DIR"
echo "  Subject name:    $SUBJECT_NAME"
echo "  Trigger:         $TRIGGER"
echo "  Class noun:      $CLASS_NOUN"
echo "  Model preset:    $MODEL"
echo "  Caption backend: $CAP_BACKEND"
echo "========================================================="
echo
read -rp "Press Enter to continue, or Ctrl+C to cancel..."

.venv/bin/python -m pipeline.run_full_pipeline \
  --input_dir "$INPUT_DIR" \
  --subject_name "$SUBJECT_NAME" \
  --trigger "$TRIGGER" \
  --class_noun "$CLASS_NOUN" \
  --model "$MODEL" \
  --caption_backend "$CAP_BACKEND"

echo
echo "[DONE] Pipeline prep complete. See the launcher path printed above --"
echo "       run that when you're ready to start training."
