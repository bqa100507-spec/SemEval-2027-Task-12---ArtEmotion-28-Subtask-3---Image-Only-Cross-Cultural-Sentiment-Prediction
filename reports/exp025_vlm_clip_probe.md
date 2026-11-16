# EXP-025 — VLM–CLIP semantic complementarity probe

Research-only, Built with Qwen. E50+L remains production; EXP-022/023/024 stay closed. No confirmation annotations or Development labels are used.

**Model/provenance:** Qwen/Qwen2.5-VL-3B-Instruct at `66285546d2b821cf421d4f5eb2576359d3770cd3`; its [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/66285546d2b821cf421d4f5eb2576359d3770cd3/LICENSE) restricts use to non-commercial research/evaluation. Competition eligibility is not established. Weight SHA256 and processor/tokenizer Git blob hashes are pinned in configs/exp025.json. Transformers4.51.3 [documents this architecture](https://huggingface.co/docs/transformers/v4.51.3/model_doc/qwen2_5_vl); image-only processing uses the official slow PIL processor without video utilities. CLIP-L text revision/assets reuse EXP-017; original77-token capacity is preserved through complete75-BPE-token chunks, then a normalized token-count-weighted mean of unit projections. No silent truncation.

**Extraction:** one fixed English prompt/image, image pixels only, greedy80-token maximum, no target-language/caption/vote/artist inputs. Thirty aspect-ratio strata with fixed hash sampling make the smoke label-free and reproducible. Per-image JSON records are atomically saved immediately; embeddings add checksummed768D projections. Cache identity includes image SHA, model/processor revision/assets, prompt, decoding, pixel limits, precision and CLIP text revision/policy/runtime. Head scripts only read caches. NF4 and FP16 records are incompatible identities.

**Actual local smoke:** RTX4060 Laptop8GB; initial available nvidia-smi memory5666MiB. Original FP16 weights are7.51GB, so NF4 was explicitly tested with vision/output towers retained in FP16. Initial quantization of the tied output head failed at bitsandbytes quant-state assertion; excluding lm_head fixed execution. The failure log/cache is retained. Windows Hugging Face symlink download failed with WinError1314; explicit local-directory downloads succeeded without administrator access. No historical cache or environment was changed; bitsandbytes0.46.1 was installed into an isolated task add-on directory.

The first prompt generated/encoded30/30 nonempty descriptions but introduced a forbidden nationality term. A single grounding/length prompt revision was tried before any head scores. Its separate30-image run completed in120.96s: mean3.32374s/image, peak allocated3,633,156,096 bytes (3.38GiB), peak reserved4,708,106,240 bytes (4.38GiB), no empty output, no silent CLIP truncation. All30 had complete token coverage and fit one CLIP chunk. Estimated generation for1335 images is4437s (~74min), excluding remaining setup/text encoding; this is an NF4 local extrapolation, not measured L4 performance.

**Quality blocker:** visual contact-sheet review found incorrect scene/figure/sky details, including a demon/table story for an etching and a sunny blue sky for a dark landscape. The revised prompt still emitted nationality wording. Full extraction was therefore not launched. Both prompts' outputs, telemetry and failure logs are preserved under ignored cache/runs directories. No claim is made that unquantized FP16 will solve these errors. The Colab notebook provides a separately pinned FP16 attempt and stops/exports on smoke flags.

**Frozen head comparison:** A visual-only (original normalized CLIP-B/SigLIP2/CLIP-L); B semantic-only; C early concatenation; D late A/B blend. Every raw branch is projected to128D with fitting-painting-only PCA, then L2-normalized. All use the same original vote-weighted, full image-language interaction LR with C0.3. Thus projected input/head capacity is equal; early PCA can still allocate variance differently, an acknowledged limitation.

True and shuffled early/late fusion use the same protocol. Independent fixed-seed derangements within each fitting/query partition preserve semantic distributions without transferring held-out semantics into fitting. Late semantic weights{0,.25,.5,.75,1} are selected only on original inner training validation. P2 folds1/3 are screened first. Either true fusion must beat A and its shuffled fusion on both folds with both painting-paired bootstrap lower bounds above0 to automatically add folds2/4/5. This is an exploratory continuation heuristic on reused P2, not promotion or independent confirmation; VLM-only performance never gates continuation. Macro Log Loss and per-painting effects are reported alongside Contrast.

**Real CV results: NOT AVAILABLE.** The actual head-entry check correctly rejected incomplete30/1335 semantic coverage without invoking the VLM or producing partial-population scores. The direction remains inconclusive; do not infer complementarity or failure from smoke descriptions or synthetic tests.

Commands (Python environment pinned separately):

```text
python scripts/extract_exp025.py --inputs INPUTS --cache SEMANTIC_CACHE --model-cache LOCAL_MODELS --max-seconds 7200 --full
python scripts/run_exp025.py --inputs INPUTS --cache SEMANTIC_CACHE --run-dir runs/exp025_vlm_probe
```

For explicit local NF4 use `--config configs/exp025_nf4.json` on both commands. Default FP16 requires a safe10GiB free-VRAM margin. The existing private TASK-025C input ZIP is reused for Colab; no new data packaging framework is introduced. Persistent caches survive completed-image boundaries; a current interrupted generation may restart.

Validation: **258 passed,0 failed,1 Windows symlink fixture skipped**. Includes complete-token chunking, image-only protocol sampling, cache corruption/identity/reuse, matched head capacity, held-label independence, partition-local derangements, inner-only blend selection, fusion-vs-shuffle continuation and mocked notebook export/download/release. Notebook schema/AST pass; real Colab execution is unverified. Read-only reload verified30 semantic records (max70 CLIP tokens); all499 protected historical/owner files and E50+L production SHA remain unchanged.

Current implementation is ready to attempt an FP16 L4 smoke, **not a verified full extraction/CV result**. Local smoke quality failure remains recorded; the next full-run attempt must pass the smoke checks. Source/PR pins and the notebook link are provided in delivery.

Linux dependency resolution for CPython3.12.10/glibc2.28 succeeded with all55 requested package pins unchanged. This checks availability/resolution, not actual Colab installation/GPU execution. The notebook scientific source is pinned to `c52e7417b2435cdea40d9b1acd9aa392cc8b9c1f`.
