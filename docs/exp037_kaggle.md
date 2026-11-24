# EXP-037 · Progressive joint vision–language adaptation

**CPU validated. Kaggle pretrained preflight and scientific results pending.** EXP-036 remains the incumbent. Replaying its sealed 849 Development predictions gave **0.470338787 Contrast / 0.729708508 bits**. Excluding English, Arabic and Chinese from that same panel gave **0.448751126 / 0.752287512**. The hosted result of **0.4703 (#1065924)** is owner-reported.

## Recipe

- SigLIP2 So400m patch14-384 at `e8e487298228002f3d8a82e0cd5c8ea9c567f57f` provides native patch tokens and a pooled skip connection. Fresh rank-32 LoRA covers Q/V and both MLP projections in its last six blocks. The run uses no EXP-036 adapter weights.
- Qwen3-VL-4B-Instruct at `ebb281ec70b05090aa6165b016eac8ec08e71b17` receives the image and a fixed neutral prompt once per painting. The final decoder states at active native image-token positions supply keys and values. No generation, vocabulary logits, per-language Qwen calls, or offline Qwen feature bank are used. Layer indices 28–35 each receive rank-16 Q/V QLoRA with alpha 32.
- Two residual 256-wide, four-head cross-attention blocks use SigLIP patches as queries with padding-masked Qwen context. The pooled fused features and normalized native SigLIP vector feed shared and language-specific logits. An official language missing from fitting uses the shared term and mean fitted language intercept.
- Count-weighted binomial BCE is combined with 0.2 times count-weighted eight-emotion cross-entropy and 0.1 times cosine loss against the sealed, centered EXP-036 training caption targets. The training population contains 1,571 paintings, 16,019 eligible pairs, 82,072 mapped votes and 16,001 valid caption targets. Both auxiliary heads are omitted from the inference bundle.
- AdamW uses an effective batch of eight paintings, gradient clip 1.0, weight decay 0.01 and a stage-local 5% warmup followed by cosine decay to 10% of the initial rate. Each microbatch uses the vote and masked caption mass of the entire optimizer batch. Each stage starts from the preceding selected checkpoint with a new optimizer and schedule.

| Stage | Epochs | Tuned groups | Initial rates | Trainable parameters |
| --- | ---: | --- | --- | ---: |
| A | 4 | Fusion and three heads | 1e-3 | 2,386,793 |
| B | 6 | A plus visual LoRA | 5e-4 / 1e-4 | 5,366,633 |
| C | 8 | B plus Qwen QLoRA | 3e-4 / 1e-4 / 3e-5 | 6,677,353 |

Parameter counts use the pinned dimensions and 25 fitted languages. The stages answer a progressive-training question. Later stages have additional optimization steps, so their scores alone do not isolate the causal effect of an adapter.

Every epoch is scored on all 849 Development pairs. Within each stage, select by highest full Contrast, then lowest Macro Log Loss, then earliest epoch. Always publish both the standalone and the best incumbent blend over `[0, .10, .20, .25, .35, .50, .75, 1]`. Ties between blends favor lower alpha. The evaluation-like subset is diagnostic. The best overall ZIP can be the original incumbent at alpha zero. Report public Development selection bias when interpreting these scores.

## Run All

1. Upload **`runs/exp037/handoff/exp037_inputs.zip`** as one **private** Kaggle Dataset and attach it to [EXP037_Kaggle.ipynb](../notebooks/EXP037_Kaggle.ipynb). Kaggle can mount the ZIP intact or extract it. Original artwork names, including apostrophes, are stored in a checked mapping to ASCII hash-based filenames. The input includes private training annotations and public Development counts for scoring, so keep it private.
2. Select one CUDA GPU. Prefer L4 or A100 if available. For T4 ×2 the launcher uses the first GPU and FP16 compute with NF4 Qwen weights. Enable **Internet** for the pinned Python 3.12.10 environment and public model assets. Optional `SIGLIP_SNAPSHOT` and `QWEN_SNAPSHOT` values point to the exact local snapshots and do not replace dependency installation.
3. Set `AUTHORIZE_RUN = True` and click **Run All**. The notebook verifies the mounted input and shows live worker output. After real-image forward/backward, isolation and memory checks, A, B and C run automatically. The preflight tries 4, 2, then 1 painting per microbatch with 768 MiB reserved-memory headroom. Training can reduce microbatch size on a pre-update OOM while retaining the effective optimizer batch. The throughput forecast is diagnostic. Session allowance is 10.5 hours including setup, with four minutes reserved for final export.
4. Use **Output** to persist or download `exp037_results.zip` or the newest `exp037_progress.zip`, together with its `.sha256` receipt. Save output before abandoning an interactive session. To resume on a fresh notebook session, attach the original input and the newest saved archive as private datasets. Use the same GPU type and set `RESUME_INPUT` to the archive or its extracted root containing `export_manifest.json`, authorize, and Run All. A resumed run verifies checkpoint hashes and runtime identity. Completed stages are reused. If the runtime is forcibly terminated between exports, work after the last persistent export must be replayed.

Worker `38b1e381127c5f4fdc02dfc9845caeba90d9667e` is embedded in the private input. ZIP SHA-256 `eaf16b9301cb131fbdf61a1b86fb191d3617b2141e39398f0c3b12bcf57ffce2`. Input manifest SHA-256 `026720f406c02236aeaf5bfdc806853816323b66267f81f0e3e2dbbfe1ce7e0e`. Offline readback verified 1,765 files including all 1,656 Train and Development images.

## Expected outputs

| Path under `/kaggle/working/` | Contents |
| --- | --- |
| `exp037_run/{A,B,C}/preflight.json` | Real-image loss and gradient checks, adapter updates, allocated/reserved VRAM, timed samples and stage forecast |
| `exp037_run/{A,B,C}/epoch_*/` | Epoch metrics, training speed, validation, six-decimal predictions and sealed learned tensors |
| `exp037_run/{A,B,C}/selected/` | Selected inference weights and manifest, standalone and ensemble predictions and both stage submission ZIPs |
| `exp037_run/exp037_best_submission.zip` | Exact ZIP selected across the three stages and incumbent blends, with the coefficients and SHA-256 in `selection.json` |
| `exp037_progress.zip`, `exp037_results.zip` | Verified progress transfers and final export, with SHA-256 receipts |
| `exp037_diagnostic.zip` | Launcher errors and worker log if the job stops or fails |

The worker saves sealed optimizer boundaries every 24 updates and portable snapshots every 96 updates, at each epoch and after each selected stage. Boundary state includes the optimizer, scheduler, scaler, RNG, progress position and runtime identity. GPU replay compares each selected checkpoint's regenerated six-decimal Development predictions against its saved CSV before publication.

## Fresh-image inference

Extract a completed result and use its chosen `<stage>/selected` directory with the worker and pinned runtime from the handoff. Provide the official input CSV and an input root containing `images/`.

```sh
python scripts/run_exp037.py infer --bundle /path/to/C/selected \
  --input /path/to/st3_test.csv --images /path/to/input-root --output /path/to/exp037-inference
```

For the frozen blend, generate EXP-036 predictions for those same input keys with [its separate inference pipeline](exp036_evaluation.md), then pass `--incumbent /path/to/exp036-submission.zip`. The blend reuses the saved alpha. Inference uses neither captions nor vote labels. Real fresh-image GPU replay of the EXP-036 candidate was outside the available local runtime for this handoff.

## Competition eligibility

Public organizer material did not establish permission for Qwen3-VL in this ST3 configuration. The organizer FAQ defers LLM guidance, and the reviewed public issue discussions did not supply an explicit ruling for this use. The Codabench rules could not be independently read back during preparation. Qwen's Apache-2.0 license and its online, image-only use are separate facts. Treat all EXP-037 ZIPs as experimental until organizer eligibility is established. Training uses only official Train votes and caption targets. Development counts are scorer-only, and the handoff contains no Evaluation data. [Official organizer README and FAQ](https://github.com/artemotion-28/artemotion-28) · [Organizer discussions](https://github.com/artemotion-28/artemotion-28/issues) · [Competition](https://www.codabench.org/competitions/18468/) · [Pinned Qwen checkpoint](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17)
