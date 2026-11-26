# Qwen semantic pipeline recovery

**Decision — stop this implementation on the available RTX 4060 Laptop GPU.** The best measured recipe was 5.53× faster and produced six repeatable, schema-valid descriptions. It still projected to 2.29 GPU-hours for all 1,571 eligible training paintings before reserve, above the inherited two-hour budget. This is a feasibility result. The predictive value of the semantic features remains unmeasured.

## Where the original runtime went

I replayed two of the six pinned training artworks using the original prompt and generation settings. Both reproduced the archived text exactly. The original six-artwork run averaged 29.05 seconds of generation and 99.33 new tokens per artwork. The instrumented two-artwork replay averaged 27.55 seconds end to end.

| Stage | Mean seconds/artwork on the replay |
| --- | ---: |
| Image read, SHA-256 check and RGB decode | 0.108 |
| CPU image processing and prompt tokenization | 0.154 |
| Host-to-GPU tensor transfer | 0.007 |
| Vision encoder | 1.015 |
| Remaining prompt prefill | 1.188 |
| Autoregressive decoding | 24.557 |
| Other overhead, including generation control and output handling | 0.524 |
| End to end | 27.555 |

CUDA events timed the forwards, including any host launch gaps within those intervals. The vision pass is a subset of prefill. The first next-token probability is computed in prefill, followed by 99 and 86 cached decoder steps on the two replayed artworks. Decoder intervals accounted for about 89% of end-to-end time. The checkpoint loaded with NF4 double quantization and FP16 compute. All 715 parameter and buffer tensors were on CUDA, with no model-level CPU or disk offload, and every decoder forward returned a KV cache. The measurements identify the costly stage. They do not isolate driver scheduling, kernel launch overhead and GPU arithmetic within a decoder step.

## What the compact recipe preserved

The checkpoint, resize policy, greedy decoding, labels policy and five semantic fields stayed fixed. The optimized recipe shortened the fields, capped generation at 72 new tokens, used SDPA in both towers, and left-padded batches for cached decoder inference. The parser rejects missing, extra or duplicate fields, incomplete JSON and unmarked interpretations. It can join an entity list verbatim. It never supplies missing visual claims. The batch comparison measures those changes together. [Attention backend documentation](https://huggingface.co/docs/transformers/v4.57.1/en/attention_interface)

The six-image visual review found that the compact batch-three output kept the woodland path and lone figure, the Derain village and trees, and the Bilibin riders and shield. It omitted the original Sisley output's unsubstantiated windmill and grazing animals. It also lost useful detail, including the Bilibin shield color and Cyrillic lettering, and called the Zadkine gathering a protest without enough visible evidence. Schema validity and repeatability do not measure semantic correctness or ST3 accuracy.

## Measured throughput and feasibility

| Recipe on the same six paintings | Seconds/artwork | Mean output tokens | Schema-valid | Peak allocated / reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| Archived original, batch 1, generation only | 29.05 | 99.33 | 4/6 | 2.89 / 3.02 |
| Compact SDPA, batch 3, end to end | 8.41 | 55.50 | 6/6 | 3.03 / 3.16 |
| Same compact recipe, batch 6, end to end | 5.25 | 55.50 | 6/6 | 3.28 / 3.44 |

The two batch-six runs took 32.01 and 30.99 seconds. The first batch of three replayed exactly. All six descriptions replayed exactly at width six and matched the width-three outputs. All compact outputs reached an EOS token below the fixed 72-token cap. At width six, decoding still took roughly 25 seconds per batch. The GPU had enough memory for the tested widths.

| Scope | Optimized width-six estimate | With 25% reserve and one model load |
| --- | ---: | ---: |
| Five-fold P2 extraction, 1,335 distinct artworks | 1.95 GPU-hours | 2.45 GPU-hours |
| Complete eligible training population, 1,571 distinct artworks | 2.29 GPU-hours | 2.88 GPU-hours |

These estimates multiply the measured batch time by the number of batches rounded up and add the measured 33.33-second model load in the budget column. They exclude one-time checkpoint hash verification and downstream CPU fitting. They are projections from six paintings. Meeting the same full-training budget would require a measured batch-six time of about 21.9 seconds or less, or about 0.274 artworks/second. No other GPU was benchmarked. Expanding the batch search or spending more GPU time was outside this bounded recovery.

Full extraction and grouped CV did not start. Corrected Ridge remains the hosted reference at 0.4655 Contrast, submission #1063064, per the owner. The previous P2 reference replay remains available. B and C have no scores, and no candidate was fitted or packaged. The historical benchmark, E50+L Development predictions, both Ridge ZIPs, Fixed Ridge heads and the Corrected Ridge residual bank all matched their pinned or previously recorded SHA-256 values. Organizer eligibility for generated semantic intermediates remains unresolved. [Prior reference replay and rule review](competition_semantic_vlm.md)

## Reproduction

Use the pinned runtime and verified local checkpoint from the original preflight. The runner uses the original hashed sample and a new run directory. The second command adds only the fixed six-wide check and requires the first receipt.

```powershell
.venv/Scripts/python.exe scripts/run_competition_semantic_recovery.py --run-dir runs/competition_semantic_vlm/recovery_replay
.venv/Scripts/python.exe scripts/run_competition_semantic_recovery.py --run-dir runs/competition_semantic_vlm/recovery_replay --check-batch-six
```

The measured run receipts live in `runs/competition_semantic_vlm/recovery/`. `benchmark.json` has SHA-256 `818441d0fcf8c561815a3f8db971264c2a74728c24cd3caefba2928164c5774c`. `batch_six.json` has SHA-256 `f1677c4c1c88839bc1d4e02b8d64a20403e36121496086f74d05daa739c79b91`. Both contain raw outputs and per-stage timings.
