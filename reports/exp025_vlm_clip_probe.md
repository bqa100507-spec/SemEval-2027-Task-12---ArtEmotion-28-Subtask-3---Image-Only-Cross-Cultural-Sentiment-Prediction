# EXP-025 — finish the scientific probe (TASK-027B)

**Status:** implementation/notebook corrected; real FP16 L4 and five-fold results are not available in this execution environment. Local hardware is an RTX4060 Laptop8GB (~5845MiB free), below the pinned FP16 safety margin. No new GPU run, Development gold, confirmation labels, production change or leaderboard upload occurred in TASK-027B.

| Evidence | Actual status |
| --- | --- |
| Prior revised-prompt NF4 smoke |30/30 valid cached descriptions/embeddings;3.32374s/image;3.38GiB allocated /4.38GiB reserved; max70 CLIP tokens, complete coverage |
| FP16 L4 smoke and paired latency/quality comparison |Not run here; notebook measures both precisions on the same30 paintings/prompt, retaining actual generation timings rather than cache-read time |
| Full semantic coverage |NF4 local30/1335; no complete FP16 cache |
| Five-fold Contrast / Macro Log Loss / true-vs-shuffled gains |Not measured; no partial-population or synthetic scores presented as scientific results |
| Scientific decision |Pending real FP16 and five-fold evidence; do not discard semantic fusion based on noisy descriptions alone |

The unchanged checkpoint is Qwen2.5-VL-3B-Instruct at `66285546d2b821cf421d4f5eb2576359d3770cd3`; prompt, greedy decoding, pixel limits, software pins and existing CLIP-L text encoder stay fixed. [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/66285546d2b821cf421d4f5eb2576359d3770cd3/LICENSE): non-commercial research only; competition eligibility remains unconfirmed. Built with Qwen.

**One review, then automatic execution.** The notebook runs/reuses NF4 and FP16 smoke, displays all30 artworks and both descriptions side by side with latency/VRAM and explicit warning reasons, and asks once for APPROVE or REJECT. The criterion is broadly useful visual semantics, not factual perfection. Inaccuracies, nationality/artist wording and uncertain details are warnings, preserved with descriptions and image IDs. Invalid inference, missing/corrupt/nonfinite embeddings and lost tokens remain hard technical failures. Automated word flags do not claim to measure factual hallucination rates.

Approve → automatically extract all1335 FP16 records and evaluate all five original P2 grouped folds. Reject serious systematic scene hallucination → preserve/export smoke and stop; recommend considering direct VLM hidden-state features or a stronger model, without launching either. A checksum-compatible approved smoke is reused on resume; changed smoke requires review. No prompt revisions or repeated approval loops. Review waiting is bounded to10 minutes within the allocation; silence is not approval.

**Scientific comparison unchanged:** A visual-only (original CLIP-B/SigLIP2/CLIP-L), B VLM text embeddings, C early fusion, D late fusion; true-vs-shuffled early/late controls; E50+L fixed external reference. Fold-local128D PCA and the same vote-weighted interaction LR C0.3 control head capacity. Inner training validation alone selects late weights{0,.25,.5,.75,1}. Shuffle mappings stay inside fitting/query partitions. CLIP text chunks cover every token. No E50 blend, VLM fine-tuning or auxiliary loss.

All five folds run regardless of intermediate scores. Report pooled/fold-mean/per-fold Contrast and Macro Log Loss; B−A, C−A, D−A, C−shuffled-C, D−shuffled-D and C/D−E50+L with painting-paired bootstrap intervals, wins/regressions and fold changes. Bootstrap is evidence, not admission. Repeated P2 use remains exploratory; no automatic promotion.

The notebook reuses the existing TASK-025C private ZIP. Source/prompt/precision/text-encoder cache keys remain strict and compatible; head results use a new source identity. Existing70 local cache files and499 protected historical/owner files were checked byte-identical. Tests and the final pinned notebook/PR link are reported with delivery.

Validation:265 CPU tests passed,1 Windows symlink fixture skipped. New tests cover warning-only cache reuse, hard token-loss rejection, exact-smoke approval/rejection, one notebook pause, rejected-run preservation, approved resume without another pause and completion of all five folds despite zero fusion gain. Notebook schema/AST and mocked flow pass; these do not establish real L4 performance or real CV scores.

The corrected notebook pins scientific source `fc3ec336ca99871fa87d1d51c4b1ddfedb208c86`; default authorization remains False. Upload the existing TASK-025C ZIP, choose L4, authorize once and Run all. The only requested pause is the agreed visual smoke assessment.
