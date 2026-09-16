@echo off
setlocal enabledelayedexpansion
title Flux.2 Klein ArcFace LoRA Pipeline - Photos to Trained Checkpoint
cd /d "%~dp0"

if not exist "setup\paths.json" (
    echo =========================================================
    echo   FIRST-TIME SETUP REQUIRED
    echo =========================================================
    echo.
    echo setup\paths.json not found. Run these first:
    echo   1^) python setup\install.py
    echo   2^) python setup\download_models.py
    echo.
    pause
    exit /b 1
)

echo =========================================================
echo   FULL LORA TRAINING PIPELINE (photos -^> trained LoRA)
echo   Stage 1: Ingest ^& quality-gate raw photos
echo   Stage 2: Disentangled auto-captioning
echo   Stage 3: Generate ArcFace training config + launcher
echo =========================================================
echo.
echo This prepares everything up to (but does not itself run) training.
echo At the end you'll get a launcher script to run when ready to
echo actually train (that step takes hours on the GPU).
echo.

set /p INPUT_DIR="Folder of raw photos of the subject: "
if "%INPUT_DIR%"=="" (
    echo [ABORT] No input folder given.
    pause
    exit /b 1
)
if not exist "%INPUT_DIR%" (
    echo [ABORT] Folder not found: %INPUT_DIR%
    pause
    exit /b 1
)

set /p SUBJECT_NAME="Short subject name (e.g. alex, no spaces): "
if "%SUBJECT_NAME%"=="" (
    echo [ABORT] No subject name given.
    pause
    exit /b 1
)

set /p TRIGGER="Trigger word - short, unique, not a real word [default: tkn]: "
if "%TRIGGER%"=="" set TRIGGER=tkn

set /p CLASS_NOUN="Class noun [default: person]: "
if "%CLASS_NOUN%"=="" set CLASS_NOUN=person

echo.
echo Model presets:
echo   1 = flux2_klein_4b         (rank64, 1024px, 1500 steps  -- recommended default)
echo   2 = flux2_klein_4b_strong  (rank96, 1024px, 1800 steps  -- stronger identity push)
echo   3 = flux2_klein_9b         (rank64, 1024px, 2000 steps, layer_offloading -- much slower, needs more VRAM)
set /p MODEL_CHOICE="Choose model preset [1/2/3, default 1]: "
if "%MODEL_CHOICE%"=="2" (set MODEL=flux2_klein_4b_strong) else if "%MODEL_CHOICE%"=="3" (set MODEL=flux2_klein_9b) else (set MODEL=flux2_klein_4b)

echo.
echo Captioning backend:
echo   1 = template  (fast, free, fully offline heuristic captions)
echo   2 = api       (higher quality, needs GEMINI_API_KEY or OPENAI_API_KEY env var)
set /p CAP_CHOICE="Choose captioning backend [1/2, default 1]: "
if "%CAP_CHOICE%"=="2" (set CAP_BACKEND=api) else (set CAP_BACKEND=template)

echo.
echo =========================================================
echo   Input folder:    %INPUT_DIR%
echo   Subject name:    %SUBJECT_NAME%
echo   Trigger:         %TRIGGER%
echo   Class noun:      %CLASS_NOUN%
echo   Model preset:    %MODEL%
echo   Caption backend: %CAP_BACKEND%
echo =========================================================
echo.
pause

".venv\Scripts\python.exe" -m pipeline.run_full_pipeline ^
  --input_dir "%INPUT_DIR%" ^
  --subject_name "%SUBJECT_NAME%" ^
  --trigger "%TRIGGER%" ^
  --class_noun "%CLASS_NOUN%" ^
  --model "%MODEL%" ^
  --caption_backend "%CAP_BACKEND%"

if errorlevel 1 (
    echo.
    echo [FAILED] Pipeline stopped with an error. See the output above.
    pause
    exit /b 1
)

echo.
echo [DONE] Pipeline prep complete. See the launcher path printed above --
echo        run that when you're ready to start training.
pause
