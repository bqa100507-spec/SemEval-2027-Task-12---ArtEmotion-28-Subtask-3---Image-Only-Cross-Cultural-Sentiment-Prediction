# Direct culture-conditioned Qwen3-VL · Stage 1

**Technical feasibility passed on the RTX 4060 Laptop GPU.** The pretrained forward path and bounded adapter updates worked on real official training images with synthetic vote targets. The measured workload projects to **2.32 GPU-hours for the two-epoch fold-3 pilot**, or **2.90 hours with 25% reserve**. This sprint has no measured predictive score. The current hosted reference is Corrected Ridge, 0.4655 Contrast on submission #1063064, per the owner.

## Model and matched control

The pinned Apache-2.0 checkpoint is Qwen3-VL-4B-Instruct at `ebb281ec70b05090aa6165b016eac8ec08e71b17`. The input contains pixels and the target-language name in one fixed prompt. The native processor, vision encoder, all three DeepStack features, multimodal RoPE and causal masks are retained. The grouped path encodes vision once per painting and uses those tokens for each language row. The readout is layer normalization and a scalar linear projection on the last unmasked Qwen state, followed by a sigmoid. Hidden-state prediction bypasses the vocabulary head, token generation and KV caching. [Pinned checkpoint](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17) · [Frozen recipe](../configs/competition_direct_qwen.json)

- **A** trains a 7,681-parameter head on frozen, native Qwen cultural states. Frozen features are reusable across fits with the same recipe, including folds.
- **B** uses the same head and initialization and adds 327,680 rank-8 LoRA parameters to the `q_proj` and `v_proj` of text layers 32–35. The backbone weights and vision encoder are frozen. Only the final four decoder blocks require adapter backpropagation. Compute uses the verified NF4/FP16 runtime with FP32 head and adapter parameters.
- Both arms use the binomial negative log-likelihood `sum(k * softplus(-z) + (n - k) * softplus(z))` divided by the fitting partition's mean vote mass per painting when using one painting per update. Using the same denominator across steps preserves the relative weight of observed votes, and chunked language backpropagation preserves the unchunked objective. The implementation also supports conventional per-batch vote normalization. B adds `0.01 * mean(||ΔW||²)` across the eight projections to constrain the change from pretrained weights. No ranking loss is used. The fixed future comparison C averages B and Corrected Ridge probabilities 50/50. OOF predictions are used for matched scoring after fitting.

EXP-014 showed a small gain in fusion, and EXP-033 found useful adaptation that still fell short of the strongest ensemble. EXP-029's projector and QLoRA underperformed their matched reference. EXP-030 and EXP-031 learned under supervision but later lost Contrast even while their fitted objectives improved, and EXP-031 clipped every update. This sprint tests native target-language states against Corrected Ridge and isolates selective adaptation with a matched frozen control. The previous semantic run's small gain came from a separate blended system with generated generic descriptions. [EXP-014](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/cd321f142142120c17c495458390deb6574ad856/reports/exp014_clip_large_lora.md) · [EXP-029 recovery](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/3cb19bdaf34c8555bb08ad1b09cfc0605dc96dae/reports/exp029_recovery.md) · [EXP-030 analysis](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/7da299dd4474818a47dab781c4271cf6628eff79/reports/exp030_error_analysis.md) · [EXP-031](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/01bf9bd93b3c7417eb4a76413f3dd15f041a155a/reports/exp031_final_report.md) · [EXP-033](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/pull/40) · [Completed semantic comparison](competition_semantic_evaluation.md)

## What actually ran

The CPU preparation verified the P2 training-pair and Corrected Ridge OOF file hashes, the original fold assignments and painting isolation. Fold 3 has **1,068 fitting paintings / 10,879 pairs** and **267 held-out paintings / 2,721 pairs**. The feasibility run used six fitting paintings straddling low, median and high language-panel size, totaling 62 pairs and 18 decoder chunks. It read only pair keys and image bytes. The six optimizer steps used synthetic counts, and no trained state was exported. Checkpoint and metadata hashes passed.

The first preflight surfaced batch-shape dependence in the FP16 vision path. Encoding duplicate images in the native reference versus encoding each painting once gave maximum hidden difference **0.1875** and minimum cosine **0.99999595**. The follow-up verified **bit-exact** grouped and native results when supplying the same vision tensors, plus **bit-exact** single-image end-to-end native parity. The real padded mask and token IDs matched the official processor, and repeated grouped forwards matched exactly. At zero LoRA initialization both arms had exactly the same hidden states and logits.

Changing the language while holding the pixels fixed changed the hidden state by L2 **192.47** in the 2×2 diagnostic. Changing the artwork with language fixed yielded L2 **34.02**. The crossed difference was **31.81**, establishing that the path permits an image–language interaction. These are forward-path checks and do not measure whether the interaction is culturally predictive.

The synthetic updates produced finite, nonzero adapter and head gradients, changed all 16 adapter matrices and left frozen parameter versions unchanged. The six steps all clipped at the preregistered global norm of 1.0, with preclip norms **1.34–6.30**. Real-vote clipping frequency and gradient balance should be checked in the authorized pilot. Synthetic clipping alone gives no basis for changing the recipe.

| RTX 4060 Laptop measurement | Result |
| --- | ---: |
| A frozen feature extraction | 5.42 pairs/s |
| B QLoRA forward, backward and optimizer | 4.51 pairs/s |
| B inference forward | 5.05 pairs/s |
| A peak allocated / reserved | 2.97 / 3.04 GiB |
| B peak allocated / reserved | 3.72 / 3.86 GiB |
| B margin against CUDA-reported initial free memory | 3.08 GiB |
| Successful technical check, including hash verification and model load | 88.02 s |

## Cost before owner authorization

| Workload | RTX 4060 estimate | With 25% reserve |
| --- | ---: | ---: |
| A feature extraction across P2, once | 41.51 min | 51.88 min |
| B fitting per fold-3 epoch | 39.75 min | 49.68 min |
| B held-out forward per fold-3 epoch | 8.98 min | 11.23 min |
| Matched fold-3 pilot including two B epochs and held-out forwards | **2.32 h** | **2.90 h** |
| All five folds with the same two-epoch schedule and shared A cache | 8.85 h | 11.06 h |

Estimates scale separate per-painting overhead and per-decoder-chunk costs by the exact P2 counts. They include checkpoint load time in the totals and omit CPU-only head fitting and scorer time. The sample had 234–247 image tokens per artwork, so the 25% reserve allows for additional work on variable sizes and device load. Neither the estimate nor the reserve is an epoch-scale throughput measurement. Frozen FP16 readout states across all P2 pairs would need roughly 66 MiB before metadata.

The Colab L4 has 24 GB nominal memory, providing more room than the measured local footprint. No L4 session ran. A conservative planning assumption is the same **2.90-hour** reserved pilot allocation, with actual rate and peak to be filled in by this same bounded runner on the assigned L4. Hardware specifications alone give no dependable runtime multiplier for NF4 QLoRA and this microbatch shape. [NVIDIA L4 specifications](https://www.nvidia.com/en-us/data-center/l4/#specifications)

The next authorization is for **one fold-3 pilot, with a proposed 3-hour ceiling on the locally measured GPU**. Score the fixed two-epoch endpoint and record intermediate validation curves without picking an epoch on the outer fold. Compare A, B, the unchanged Corrected Ridge reference, and the fixed complementarity blend on the same held-out rows. The feasibility checker contains no scientific training stage. No CV, final refit, Development predictions or submission ZIP were created. Organizer guidance on pretrained VLM eligibility remains unresolved. [Organizer FAQ](https://github.com/artemotion-28/artemotion-28#faqs)

## Reproduction

Use the pinned Python and CUDA environment from the semantic recovery. `--cuda-feasibility` is opt-in, uses synthetic counts and has a hard five-minute process limit. Use a distinct directory for another technical run. The CPU-only default checks the data keys and matched protocol.

```powershell
.venv/Scripts/python.exe scripts/run_competition_direct_qwen.py
# With the existing recovery runtime and Transformers overlays on PYTHONPATH
.venv/Scripts/python.exe scripts/run_competition_direct_qwen.py --cuda-feasibility --run-dir runs/competition_direct_qwen/stage1_replay
```

The measured source commit is `d5c917fc501330ebf5f52163fcea36f85dd98db7`. The durable receipt is `runs/competition_direct_qwen/stage1_parity/feasibility.json`, SHA-256 `c84582d199dc2ebb624691ec5a8a58bca931818ff38f64eb06dd57901ec69a34`. The first attempt's FP16 difference is preserved under `runs/competition_direct_qwen/stage1/`. The local timing projection is in `runs/competition_direct_qwen/stage1_parity/cost_projection.json`. E50+L production predictions and the Corrected Ridge archive still match their recorded SHA-256 values.

The clean suite passed 311 tests with three skips on the base repository runtime. The four targeted CPU tests also passed with the pinned Qwen runtime overlay, including the tiny native Qwen multimodal test skipped by the older base Transformers package. The tests cover exact grouped-path parity, last-unmasked-state selection, equivalence to expanded individual votes, microbatch gradient equivalence, zero-init matching and updates confined to the decoder adapters and readout.
