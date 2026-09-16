import os
import sys
import glob
import re
import json
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline.face_engine import FaceEngine

def parse_samples(samples_dir):
    """
    Parses sample files in directory.
    Returns: dict mapping step -> dict mapping prompt_idx -> filepath
    """
    files = [f for f in os.listdir(samples_dir) if f.endswith(('.jpg', '.png')) and not f.startswith('.')]
    step_data = {}
    for f in files:
        # Match pattern: timestamp__000000350_2.jpg or similar
        m = re.search(r'__(\d{9})_(\d+)\.', f)
        if m:
            step = int(m.group(1))
            pidx = int(m.group(2))
            if step not in step_data:
                step_data[step] = {}
            step_data[step][pidx] = os.path.join(samples_dir, f)
    return step_data

def evaluate_run(run_dir, holdout_dir, output_report=None, grid_output=None):
    print(f"=== Starting Forensic Holdout Evaluation ===")
    print(f"Run directory: {run_dir}")
    print(f"Holdout directory: {holdout_dir}")

    samples_dir = os.path.join(run_dir, "samples")
    if not os.path.exists(samples_dir):
        raise FileNotFoundError(f"Samples directory not found at {samples_dir}")

    face_engine = FaceEngine()

    # 1. Load True Holdout Embeddings & Images
    holdout_files = []
    # Check if holdout_manifest.json exists
    manifest_p = os.path.join(holdout_dir, "holdout_manifest.json")
    if os.path.exists(manifest_p):
        with open(manifest_p, "r") as f:
            manifest = json.load(f)
            for item in manifest:
                holdout_files.append((item["name"], item["path"]))
    else:
        # Fallback: no manifest found, scan for images directly. Assumes every
        # image in the holdout directory is a valid holdout candidate.
        exts = ("*.png", "*.jpg", "*.jpeg")
        for ext in exts:
            for p in sorted(glob.glob(os.path.join(holdout_dir, ext))):
                fname = os.path.basename(p)
                holdout_files.append((fname, p))

    if not holdout_files:
        raise ValueError(f"No holdout images found in {holdout_dir}")

    print(f"Found {len(holdout_files)} true holdouts for evaluation:")
    holdout_embs = []
    holdout_images = []
    for name, p in holdout_files:
        img_bgr = cv2.imread(p)
        if img_bgr is None: continue
        detect = face_engine.detect_face(img_bgr)
        box = detect["box"] if detect else None
        emb = face_engine.extract_embedding(img_bgr, box)
        if emb is not None:
            holdout_embs.append(emb)
            # Center face crop for visual grid
            pil_img = Image.open(p).convert("RGB")
            if box:
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                crop = pil_img.crop((
                    max(0, x1 - int(bw*0.2)),
                    max(0, y1 - int(bh*0.2)),
                    min(pil_img.width, x2 + int(bw*0.2)),
                    min(pil_img.height, y2 + int(bh*0.2))
                ))
            else:
                crop = pil_img
            holdout_images.append((name, crop))
            print(f"  Loaded holdout '{name}': Face detected, embedding norm {np.linalg.norm(emb):.2f}")

    # Compute holdout centroid identity vector (unit normalized)
    centroid_emb = np.mean(holdout_embs, axis=0)
    centroid_emb = centroid_emb / np.linalg.norm(centroid_emb)
    print("Computed canonical holdout centroid identity vector.")

    # 2. Evaluate Checkpoints
    step_data = parse_samples(samples_dir)
    steps = sorted(step_data.keys())
    print(f"Evaluating {len(steps)} checkpoints: {steps}")

    results = [] # list of dicts: step, prompt_scores, mean_sim, best_probe, img_crops
    for s in steps:
        p_dict = step_data[s]
        prompt_sims = {}
        crops_by_prompt = {}

        for pidx, fpath in p_dict.items():
            img_bgr = cv2.imread(fpath)
            if img_bgr is None: continue
            detect = face_engine.detect_face(img_bgr)
            box = detect["box"] if detect else None
            emb = face_engine.extract_embedding(img_bgr, box)

            pil_img = Image.open(fpath).convert("RGB")
            if box:
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                crop = pil_img.crop((
                    max(0, x1 - int(bw*0.2)),
                    max(0, y1 - int(bh*0.2)),
                    min(pil_img.width, x2 + int(bw*0.2)),
                    min(pil_img.height, y2 + int(bh*0.2))
                ))
            else:
                crop = pil_img

            crops_by_prompt[pidx] = crop

            if emb is not None:
                sim = face_engine.compute_similarity(emb, centroid_emb)
                prompt_sims[pidx] = sim

        mean_sim = float(np.mean(list(prompt_sims.values()))) if prompt_sims else 0.0
        results.append({
            "step": s,
            "mean_sim": mean_sim,
            "prompt_sims": prompt_sims,
            "crops": crops_by_prompt,
        })
        print(f"  Step {s:04d}: Mean ArcFace Cosine Similarity = {mean_sim:.4f}")

    # 3. Analyze Trajectory & Find Peak Checkpoint
    scores = [r["mean_sim"] for r in results]
    best_idx = int(np.argmax(scores))
    peak_step = results[best_idx]["step"]
    peak_score = results[best_idx]["mean_sim"]

    # Detect Attractor Drift (sim peaked early and dropped while steps continued)
    last_score = results[-1]["mean_sim"]
    drift_detected = (peak_score - last_score) > 0.04 and best_idx < len(results) - 2

    # 4. Generate Visual Audit Grid
    if not grid_output:
        grid_output = os.path.join(run_dir, "holdout_evaluation_grid.jpg")

    panel_sz = 300
    header_sz = 40
    # Show holdouts in top row, followed by key steps (baseline, peak-100, peak, peak+100, final)
    key_step_indices = sorted(list(set([
        0,
        max(0, best_idx - 2),
        best_idx,
        min(len(results) - 1, best_idx + 2),
        len(results) - 1,
    ])))

    num_cols = max(len(holdout_images), 4)
    num_rows = 1 + len(key_step_indices)

    grid_img = Image.new("RGB", (num_cols * panel_sz, num_rows * (panel_sz + header_sz)), (22, 22, 22))
    draw = ImageDraw.Draw(grid_img)

    # Row 0: True Holdouts
    for c, (hname, hcrop) in enumerate(holdout_images[:num_cols]):
        x = c * panel_sz
        y = 0
        draw.text((x + 10, y + 10), f"HOLDOUT: {hname[:18]}", fill=(255, 215, 0))
        grid_img.paste(hcrop.resize((panel_sz, panel_sz), Image.Resampling.LANCZOS), (x, y + header_sz))

    # Next Rows: Checkpoint samples across prompts
    for r_idx, s_idx in enumerate(key_step_indices):
        res = results[s_idx]
        s_val = res["step"]
        s_sim = res["mean_sim"]
        y = (r_idx + 1) * (panel_sz + header_sz)
        is_peak = (s_val == peak_step)
        color = (0, 255, 128) if is_peak else (255, 255, 255)
        prefix = "★ PEAK " if is_peak else ""

        for c_idx in range(num_cols):
            x = c_idx * panel_sz
            pidx = c_idx # map col to prompt index
            crop = res["crops"].get(pidx)
            psim = res["prompt_sims"].get(pidx, 0.0)
            draw.text((x + 10, y + 10), f"{prefix}Step {s_val} [P{pidx}] (Sim: {psim:.3f})", fill=color)
            if crop:
                grid_img.paste(crop.resize((panel_sz, panel_sz), Image.Resampling.LANCZOS), (x, y + header_sz))

    grid_img.save(grid_output, quality=90)
    print(f"Saved visual holdout audit grid: {grid_output}")

    # 5. Write Markdown Report
    if not output_report:
        output_report = os.path.join(run_dir, "evaluation_report.md")

    report_lines = [
        f"# Forensic Identity Evaluation Report",
        f"",
        f"- **Run Directory:** `{run_dir}`",
        f"- **Holdout Reference Set:** `{holdout_dir}` ({len(holdout_files)} true holdouts)",
        f"- **Evaluation Metric:** ArcFace 512-dim Cosine Similarity vs True Holdout Centroid",
        f"- **Visual Audit Grid:** [{os.path.basename(grid_output)}]({grid_output})",
        f"",
        f"## Executive Summary",
        f"- **Peak Checkpoint:** **Step {peak_step}** (Cosine Similarity: `{peak_score:.4f}`)",
        f"- **Final Step ({results[-1]['step']}):** Cosine Similarity `{last_score:.4f}`",
        f"- **Generic Attractor / Drift Status:** **{'WARNING: DRIFT DETECTED' if drift_detected else 'STABLE / NO ATTRACTOR COLLAPSE'}**",
        f"",
        f"## Step-by-Step ArcFace Holdout Similarity Scorecard",
        f"",
        f"| Step | Mean Holdout Sim | Probe Details | Trajectory Status |",
        f"| :--- | :--- | :--- | :--- |",
    ]

    for r in results:
        s = r["step"]
        ms = r["mean_sim"]
        status = "★ **Peak Likeness**" if s == peak_step else ("Converging" if s < peak_step else "Late Stage")
        p_strs = [f"P{p}: {sim:.3f}" for p, sim in sorted(r["prompt_sims"].items())]
        p_detail = ", ".join(p_strs)
        report_lines.append(f"| Step {s:04d} | **{ms:.4f}** | {p_detail} | {status} |")

    report_lines.extend([
        f"",
        f"## Engineering Recommendations for Re-runs",
        f"- **Optimal Deployment Checkpoint:** Checkpoint at **Step {peak_step}** is the most faithful to true holdout geometry.",
    ])
    if drift_detected:
        report_lines.append(
            f"- **Drift Correction:** The model peaked at Step {peak_step} and drifted thereafter. In your next run, reduce final learning rate decay or set total steps to `{peak_step + 100}`."
        )
    else:
        report_lines.append(
            f"- **Trajectory Health:** Similarity steadily increased or plateaued smoothly without attractor collapse."
        )

    with open(output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"Saved forensic evaluation report: {output_report}")
    return peak_step, peak_score, output_report, grid_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forensic Holdout Evaluation Engine")
    parser.add_argument("--run_dir", required=True, help="Path to training output directory")
    parser.add_argument("--holdout_dir", required=True, help="Path to true holdouts directory")
    parser.add_argument("--output_report", help="Path to output markdown report")
    parser.add_argument("--grid_output", help="Path to output visual grid image")

    args = parser.parse_args()
    evaluate_run(
        run_dir=args.run_dir,
        holdout_dir=args.holdout_dir,
        output_report=args.output_report,
        grid_output=args.grid_output,
    )
