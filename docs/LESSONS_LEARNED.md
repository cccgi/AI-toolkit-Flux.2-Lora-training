# Lessons Learned: Forensic History of ArcFace LoRA Training on FLUX.2 Klein

This documents the actual failure modes encountered across many real training
attempts before arriving at the defaults baked into this pipeline. Read this
before deviating from the presets — most "obvious" tweaks here were already
tried and had a specific, measured downside.

## 1. Plain diffusion-loss training converges to a "generic beauty attractor"

**Symptom:** Train a standard LoRA (no identity loss) for enough steps, and
generated faces start looking conventionally attractive and plausible — but
subtly (or not so subtly) *not the actual person*. This gets WORSE with more
steps, not better, once the diffusion loss has mostly converged: the model
has "learned the outfit/pose/lighting" and starts filling in facial detail
from its generic pretrained prior instead of the specific reference identity.

**Fix:** ArcFace identity-anchor loss (`face_id` config block) directly
penalizes drift from the reference identity's face embedding, independent of
how well the diffusion loss has converged. This is the core reason this
pipeline exists — it's not solvable by "just training longer" or "just using
better captions" alone.

## 2. Downscaling to 768px caps identity precision — train at native 1024

**Symptom:** A 9B run trained at 768px underperformed a 4B run trained at
1024px on identity fidelity, despite 9B being the larger, theoretically more
capable model. Fine facial detail (eye shape, jaw contour) needs resolution
to actually resolve — 768px throws away exactly the detail ArcFace loss needs
to lock onto.

**Fix:** Always train at native 1024 resolution. If VRAM is the concern,
solve it with `layer_offloading` (see #4), not by downscaling.

## 3. LoRA rank must scale with model size, not stay fixed

**Symptom:** A 9B run at rank 32 underperformed a 4B run at rank 64 on
identity strict-match tests (roughly 50% strict likeness vs. 100% on the 4B
run, evaluated against true holdout photos) — despite 9B having far more
raw parameters to work with.

**Why:** Rank 32 on a 9B model is proportionally about half the effective
capacity that rank 64 provides on the 4B model, given 9B's deeper
cross-attention stack. Steering a bigger, deeper network's identity-relevant
layers needs a proportionally larger LoRA, not a fixed one.

**Fix:** This pipeline's 9B preset defaults to rank 64 (same as 4B), not a
scaled-down rank32. If you want to push identity harder specifically on 4B
without hitting the ArcFace over-bake ceiling (#5), go up in rank instead
(the `flux2_klein_4b_strong` preset at rank 96) rather than raising the loss
weight further.

## 4. CRITICAL PITFALL: `low_vram` + `gradient_checkpointing` together is catastrophic on 9B

**Symptom:** A 9B training run that should take ~30 seconds/step instead
takes **16+ minutes per step** — a ~30x slowdown. `nvidia-smi` during this
shows LOW GPU utilization despite the process clearly "running," which is
the signature of CPU↔GPU memory thrashing, not compute-bound work.

**Root cause:** `low_vram: true` combined with `gradient_checkpointing: true`
on a 9B model causes the trainer to constantly move state between CPU and
GPU memory in a way that isn't proper block-swapping — it's thrashing.

**Fix:** Use `low_vram: false` + `layer_offloading: true` instead. This uses
the fork's actual block-swap CPU-offload mechanism, which is fast (smoke-
tested clean at ~10-32 seconds/step for 9B rank64 @ 1024 on a single RTX
3090, depending on dataset bucket diversity — see #8). This pipeline's 9B
preset is already configured this way — **do not** manually set
`gradient_checkpointing: true` alongside `low_vram: true` if hand-editing a
9B config.

Side note: `layer_offloading: true` auto-converts `qtype: qfloat8` to
`float8` internally. This is expected fork behavior, not a bug — you'll see
it reflected in logs and it's fine.

## 5. ArcFace identity loss weight has a narrow safe range (~0.08-0.10)

**Symptom:** Pushing `identity_loss_weight` above ~0.10-0.12 on a small
(~20 image) dataset causes the ArcFace loss to dominate the diffusion loss.
Results: misaligned pupils, jagged/broken eye contours, a "carved"/plastic-
looking skin texture, and much worse prompt flexibility (the model resists
following styling/pose instructions because it's fighting to satisfy the
identity loss above all else).

**Fix:** Stay within 0.08 (4B default) to 0.10 (9B default, since 9B was
observed to resist the identity loss somewhat more than 4B at the same
weight — hence the slightly higher default). If identity still isn't strong
enough, prefer raising rank (capacity) or dataset size/quality before pushing
this weight further.

## 6. `diff_output_preservation` conflicts with the ArcFace pipeline — leave it off

**Symptom:** `diff_output_preservation: true` combined with
`cache_text_embeddings: true` raises a `ValueError` (the two features are
mutually exclusive in the fork). Even with `cache_text_embeddings: false`,
enabling `diff_output_preservation` alongside ArcFace's own reference anchors
caused a `RuntimeError: backward through the graph a second time` — a real
architectural conflict, not a config mistake, when both features try to
independently keep their own computation graphs alive per step.

**Fix:** Leave `diff_output_preservation: false`. ArcFace's own identity
anchoring already does the job this feature was meant for (preventing drift
from a reference), without the conflict.

## 7. Captions must NOT describe the person's permanent facial bone structure

**Symptom (design rationale, not an observed failure):** If captions describe
"a woman with a narrow chin and almond eyes," the model can learn to
associate that description with the styling/pose variables in the prompt
rather than binding that specific bone structure permanently to the trigger
token — undermining the whole point of a per-identity LoRA.

**Fix:** This pipeline's captioning stage (`caption_engine.py`) explicitly
forbids describing nose bridge width, jawline shape, chin roundness, cheek
fullness, or eyelid crease anatomy in captions. It DOES describe everything
else in detail (clothing, accessories, hairstyle, background, lighting,
framing, expression) — the goal is to make every non-identity variable
explicit and disentangled, leaving facial geometry as the one constant thing
bound to the trigger word.

## 8. Mixed aspect-ratio training sets cause bucket-switch slowdowns

**Symptom:** A tiny (4-6 image) test set with several different aspect
ratios took noticeably longer per step than a same-resolution set of similar
size — the trainer has to switch resolution "buckets" between images,
reloading model state each switch, adding overhead disproportionate to actual
compute.

**Practical implication:** This overhead amortizes away on a normal-sized
real training set (20+ images across a handful of buckets, run for 1500+
steps) — it's mainly visible on very small smoke/test runs. Don't panic if a
tiny smoke test with wildly mixed aspect ratios seems slow; it's not
necessarily indicative of your real run's per-step time. If you want a
faster, more representative smoke test, use same-resolution source images.

## 9. Aggressive inference-time settings can partially compensate, but don't fully fix, residual identity drift

**Observation:** Even a well-trained ArcFace LoRA can still drift toward a
"similar but not exact" look under heavy styling prompts (dramatic makeup,
strong lighting, unusual angles) at default inference settings. Lowering
CFG scale (e.g., to 3.0 from a higher default) combined with an identity-
protective negative prompt reduced this drift measurably in side-by-side
tests — but scored only a modest real improvement against true reference
photos (not just against the old baseline), not a full fix.

**Takeaway:** Comparing new settings only against your OLD baseline
overstates improvement — a change can look like a big win relative to a
known-weak baseline while still being mediocre against the actual ground
truth. Always score against true holdout/reference photos directly, not just
against your previous best attempt.

## 10. Always smoke-test before a full run

**Practical rule, not a specific bug:** A 1-step smoke test (this pipeline
generates one automatically alongside every full config) takes ~10-30
seconds and will surface most real bugs — bad paths, OOM, config typos,
architectural conflicts — immediately. Skipping it means discovering the same
bug an hour or more into a run that was going to take many hours anyway.
Every launcher script this pipeline generates runs the smoke test
automatically before the full run, and aborts safely if it fails.
