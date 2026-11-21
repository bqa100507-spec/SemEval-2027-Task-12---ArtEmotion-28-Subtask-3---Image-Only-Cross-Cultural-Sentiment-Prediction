# Qwen3-VL semantic-cultural preflight

**Decision — retain Corrected Ridge.** Qwen3-VL ran on the local GPU, but two of six outputs failed the frozen schema, and the measured extraction rate exceeded the two-hour budget. The run stopped before full extraction or model fitting. No comparison of the new models' predictive accuracy is available.

## Rules and prior evidence

- The organizer specifies painting pixels and target language as ST3 inputs and prohibits using other sources to get annotations for final evaluation paintings. The FAQ points to the competition platform for rules on external resources. The archived Codabench terms permit pretrained image and language models in general. Explicit permission for VLM-generated intermediate descriptions remains unresolved, so the sprint is research-only. No Development or evaluation gold was used. [ST3 specification](https://github.com/artemotion-28/artemotion-28/blob/cf7cc444cf56617f3b8417497c6930ad172c5c99/README.md#st3-image-only-cross-cultural-sentiment-prediction) · [Organizer FAQ](https://github.com/artemotion-28/artemotion-28#faqs) · [Codabench](https://www.codabench.org/competitions/18468/)
- Qwen3-VL-4B-Instruct has an Apache-2.0 model license. Both safetensors shards (8.876 GB combined) and all pinned tokenizer and processor files verified at revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`. The six-image run used FP16 compute with NF4 weights, a fixed image-size range, greedy generation, and a five-field JSON prompt. The VLM received no target language, votes, or human captions. [Pinned model card](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17) · [Recipe and hashes](../configs/competition_semantic_vlm.json)
- EXP-017 and Track 3 used human-caption supervision. Track 3's cross-annotator teacher cleared its transfer gate on zero of five folds. EXP-029–031 trained supervised Qwen heads or adapters and trailed their matched E50+L references. EXP-025 is the closest precedent. Its owner-reported five-fold late fusion scored 0.446692 Contrast, compared with 0.447073 for visual-only, despite true captions outperforming shuffled ones in early fusion. EXP-028 detected a frozen-Qwen contribution against a shuffled control on fold 3, while its fusion trailed the strongest visual-only arm. This sprint tests the narrower difference of structured Qwen3 descriptions, field-specific features, and a vote-trained language readout against Corrected Ridge. [EXP-017](exp017_text_supervision.md) · [Track 3](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/e38994e/reports/competition_track3_caption.md) · [EXP-025](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/a12491e/reports/exp025_vlm_clip_probe.md) · [EXP-028](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/077b1b7/reports/exp028_final_closure.md) · [EXP-029–031](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/tree/01bf9bd/reports)

## Measured GPU preflight

| Measurement | Result |
| --- | ---: |
| RTX 4060 Laptop CUDA memory, total | 7.996 GiB |
| Peak PyTorch allocated / reserved | 2.89 / 3.02 GiB |
| Model load / total preflight wall time | 30.36 / 262.84 s |
| Generation per artwork, mean (range) | 29.05 s (24.29–31.91 s) |
| Generation rate | 0.03442 artwork/s |
| Valid descriptions / generated | 4 / 6 |
| Same-image repeat | Identical raw output |
| P2 projection for 1,335 artworks | 10.77 GPU-hours before the 25% budget margin |

The Bilibin illustration output identified the rider, horse, red shield, and Cyrillic lettering. The Derain output correctly located the hillside village and framing branches. The Sisley output added a windmill and grazing animals that were not apparent in the inspected image. The Pissarro woods and Zadkine crowd descriptions captured the main composition, but `entities` was a JSON array in both. The preflight kept all six raw descriptions, including the rejected outputs. This sample cannot establish a hallucination rate or whether the semantic features help sentiment prediction.

The public 849-pair Development input contains 85 distinct paintings. At the measured per-artwork rate, they would take roughly 41 minutes of generation. The estimate excludes model loading, visual feature extraction, and scoring.

## Matched comparison

| System | Full P2 Contrast | Full P2 Macro Log Loss | Test-like Contrast | Test-like Macro Log Loss |
| --- | ---: | ---: | ---: | ---: |
| A. Corrected Ridge, replayed OOF | 0.454134 | 0.708500 | 0.463108 | 0.717479 |
| B. Semantic-cultural ridge | — | — | — | — |
| C. Corrected Ridge + visual-semantic head | — | — | — | — |

A was recomputed on 13,600 matched P2 pairs and 11,643 pairs after excluding English, Arabic and Chinese. The frozen OOF CSV and fold assignments verified against the pinned handoff. The candidate runner rejected the incomplete semantic cache (4 of 1,335 paintings). No candidate fold results, paired intervals, or Development ZIP exist. The owner-reported hosted Contrast for Corrected Ridge remains 0.4655 (#1063064).

B fits field-wise TF-IDF, SVD, and a language-conditioned ridge head within each outer training fold. C fits an equivalent head on the concatenated semantic and three frozen visual feature blocks, and uses a fixed 75/25 probability blend of A and that joint head. The fixed blend avoids training on any outer-fold outcomes. The frozen advancement rule requires +0.005 test-like Contrast, +0.002 full-P2 Contrast, at least four filtered fold wins, a positive lower paired interval and trimmed gain, and at most +0.005 bits of Macro Log Loss regression. Reused P2 remains exploratory evidence.

## Reproduction and protected artifacts

The preflight used Python 3.12.10, the project's CUDA 12.6 Torch 2.7.1 environment, and the isolated overlay below. The input path defaults to the existing pinned P2 handoff. Use a new run directory for a fresh preflight. The measured run used generator and runner source at `86426c0ca36437b9ab62a7cac32fa0b189d21c93`.

```powershell
$vlmOverlay = 'runs/competition_semantic_vlm/overlay'
.venv/Scripts/python.exe -m pip install --target $vlmOverlay --no-deps -r requirements-semantic-vlm-overlay.txt
.venv/Scripts/python.exe -m pip install --target $vlmOverlay --no-deps --index-url https://download.pytorch.org/whl/cu126 'torchvision==0.22.1+cu126'
$env:PYTHONPATH = (Resolve-Path $vlmOverlay).Path
.venv/Scripts/python.exe scripts/run_competition_semantic_vlm.py fetch
.venv/Scripts/python.exe scripts/run_competition_semantic_vlm.py preflight --run-dir runs/competition_semantic_vlm/preflight_replay
```

The recorded result is `runs/competition_semantic_vlm/preflight_attempt1/preflight.json`, SHA-256 `f7722128d4cfa50be59a1afa36a0d429bce4272380c9dab8f9fedfd467e8c7f6`. The raw image review used `runs/competition_semantic_vlm/preflight_contact_sheet.png`. E50+L Development predictions, both Ridge Development ZIPs, Fixed Ridge heads, and the Corrected Ridge residual bank all matched their previously recorded hashes. No production outputs were replaced.
