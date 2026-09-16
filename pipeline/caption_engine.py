import os
import sys
import glob
import json
import base64
import argparse
import requests
from PIL import Image

SYSTEM_PROMPT_TEMPLATE = """You are an expert image captioner for LoRA identity training in diffusion models.
Your task is to write a single-sentence comma-separated caption for the provided image to train the trigger: "{trigger} {class_noun}".

CRITICAL IDENTITY DISENTANGLEMENT RULES:
1. ALWAYS start the caption with: "{trigger} {class_noun},"
2. EXPLICITLY DESCRIBE all non-identity features in rich detail:
   - Clothing items, colors, collar styles, and fabrics
   - Accessories: wristwatch, necklace, earrings, glasses, hat, rings
   - Hairstyle, hair color, and how hair is styled (e.g., tied back in a neat bun, loose straight dark hair)
   - Setting and background environment (e.g., modern cafe with white marble table, car interior with leather seats)
   - Lighting conditions (e.g., soft overcast daylight, bright direct sunlight, warm ambient indoor lighting)
   - Camera shot type and head angle (e.g., close-up portrait, medium shot, head tilted slightly, looking at camera)
   - Facial expression (e.g., neutral expression, subtle gentle smile, wide candid smile)
3. STRICTLY FORBIDDEN: NEVER describe the subject's permanent facial bone structure or facial anatomy.
   - DO NOT mention nose bridge width, jawline shape, chin roundness, cheek fullness, or eyelid crease anatomy.
   - All facial geometry MUST remain uncaptioned so the diffusion model learns to bind the subject's bone structure directly to the trigger token.
4. Format output as a clean, single comma-separated description with no markdown, quotes, or conversational filler.
"""

def image_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def caption_with_gemini_api(img_path, trigger, class_noun, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    b64_img = image_to_base64(img_path)
    mime = "image/png" if img_path.lower().endswith(".png") else "image/jpeg"
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(trigger=trigger, class_noun=class_noun)

    payload = {
        "contents": [{
            "parts": [
                {"text": system_prompt},
                {
                    "inline_data": {
                        "mime_type": mime,
                        "data": b64_img
                    }
                },
                {"text": f"Caption this image starting with '{trigger} {class_noun},':"}
            ]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 150
        }
    }

    resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API Error {resp.status_code}: {resp.text}")

    data = resp.json()
    caption = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    return clean_caption(caption, trigger, class_noun)

def caption_with_openai_api(img_path, trigger, class_noun, api_key):
    url = "https://api.openai.com/v1/chat/completions"
    b64_img = image_to_base64(img_path)
    mime = "image/png" if img_path.lower().endswith(".png") else "image/jpeg"
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(trigger=trigger, class_noun=class_noun)

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Caption this image starting with '{trigger} {class_noun},':"},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_img}"}}
                ]
            }
        ],
        "max_tokens": 150,
        "temperature": 0.2,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI API Error {resp.status_code}: {resp.text}")

    data = resp.json()
    caption = data["choices"][0]["message"]["content"].strip()
    return clean_caption(caption, trigger, class_noun)

def caption_with_template(img_path, trigger, class_noun, pose=None):
    """
    High-speed heuristic fallback that extracts basic aspect ratio framing
    and formats a compliant template caption.
    """
    try:
        with Image.open(img_path) as im:
            w, h = im.size
        ar = w / float(h)
        if ar < 0.8:
            framing = "medium portrait shot"
        elif ar > 1.2:
            framing = "horizontal close-up portrait"
        else:
            framing = "close-up portrait"
    except Exception:
        framing = "portrait photograph"

    pose_str = f", {pose.replace('_', ' ')}" if pose else ""
    caption = f"{trigger} {class_noun}{pose_str}, {framing}, natural daytime lighting, looking at camera"
    return caption

def clean_caption(caption, trigger, class_noun):
    caption = caption.strip().replace("\n", " ").strip('"').strip("'")
    prefix = f"{trigger} {class_noun},"
    if not caption.lower().startswith(f"{trigger.lower()} {class_noun.lower()}"):
        caption = f"{prefix} {caption}"
    return caption

def process_dataset_captions(
    dataset_dir,
    trigger="tkn",
    class_noun="person",
    backend="template",
    provider="gemini",
    api_key=None,
    overwrite=False,
):
    print(f"=== Starting Smart Disentangled Captioning Engine ===")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Backend: {backend} (Provider: {provider})")
    print(f"Trigger: '{trigger}', Class noun: '{class_noun}'")

    train_dir = os.path.join(dataset_dir, "training") if os.path.exists(os.path.join(dataset_dir, "training")) else dataset_dir
    img_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        img_files.extend(glob.glob(os.path.join(train_dir, ext)))
    img_files = sorted(img_files)

    if not img_files:
        raise FileNotFoundError(f"No training images found in {train_dir}")

    print(f"Found {len(img_files)} images to caption.")

    # Check for API key if API backend chosen
    if backend == "api":
        if not api_key:
            env_var = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
            api_key = os.environ.get(env_var)
            if not api_key:
                print(f"Notice: No API key found for {provider}. Falling back to smart template backend.")
                backend = "template"

    success_count = 0
    for idx, fpath in enumerate(img_files):
        fname = os.path.basename(fpath)
        base, _ = os.path.splitext(fpath)
        txt_path = f"{base}.txt"

        if os.path.exists(txt_path) and not overwrite:
            print(f"[{idx+1}/{len(img_files)}] Skipping existing caption: {fname}")
            success_count += 1
            continue

        try:
            if backend == "api" and provider == "gemini":
                cap = caption_with_gemini_api(fpath, trigger, class_noun, api_key)
            elif backend == "api" and provider == "openai":
                cap = caption_with_openai_api(fpath, trigger, class_noun, api_key)
            else:
                cap = caption_with_template(fpath, trigger, class_noun)

            with open(txt_path, "w", encoding="utf-8") as fp:
                fp.write(cap)

            print(f"[{idx+1}/{len(img_files)}] {fname} -> {cap}")
            success_count += 1
        except Exception as e:
            print(f"Error captioning {fname}: {e}")
            # Fallback to template if API failed
            fallback_cap = caption_with_template(fpath, trigger, class_noun)
            with open(txt_path, "w", encoding="utf-8") as fp:
                fp.write(fallback_cap)
            print(f"  Fallback applied: {fallback_cap}")
            success_count += 1

    print(f"=== Captioning Complete: {success_count}/{len(img_files)} captions written. ===")
    return success_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Disentangled Captioning Engine")
    parser.add_argument("--dataset_dir", required=True, help="Path to prepared dataset directory")
    parser.add_argument("--trigger", default="tkn", help="LoRA trigger word")
    parser.add_argument("--class_noun", default="person", help="Class noun")
    parser.add_argument("--backend", default="template", choices=["template", "api"], help="Captioning backend")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "openai"], help="API provider")
    parser.add_argument("--api_key", help="API key for vision model provider")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt captions")

    args = parser.parse_args()
    process_dataset_captions(
        dataset_dir=args.dataset_dir,
        trigger=args.trigger,
        class_noun=args.class_noun,
        backend=args.backend,
        provider=args.provider,
        api_key=args.api_key,
        overwrite=args.overwrite,
    )
