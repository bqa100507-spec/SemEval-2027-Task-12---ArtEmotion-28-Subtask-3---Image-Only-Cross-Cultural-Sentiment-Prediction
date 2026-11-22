# EmotionCLIP ST3 comparison

## Phase 0 — UNCERTAIN

Local P2 evaluation can proceed. Competition use remains pending organizer clarification. The [pinned model card](https://huggingface.co/jiangchengchengNLP/EmotionCLIP/blob/367bd3dbe08afa4e5e021b8b2617cc65a2409194/README.md) identifies OpenAI CLIP ViT-B/32 and EmoSet-118K prefix tuning for the named `EmotionCLIP.pth` release. It lists facial emotion data at project level and describes the named weights as having no extra facial-emotion training. The [earlier card](https://huggingface.co/jiangchengchengNLP/EmotionCLIP/blob/5b79907b94a7a7daeac4fb4fb2eb6761edea584b/README.md) lists the mixed checkpoint separately. The checkpoint's training-row manifest and a complete training-stage record are unavailable. The requested Hugging Face model and the [similarly named CVPR 2023 video model](https://github.com/Xeaver/EmotionCLIP) are separate implementations. No reviewed source documents direct use of ArtEmis, ArtELingo, WikiArt or ArtEmotion for the named weights. OpenAI CLIP's individual pretraining images are also unverified.

| Overlap evidence | Finding |
| --- | --- |
| Exact duplicate bytes or pixels | Unverified. No comparable pretrained image hashes or images were retrieved. |
| Different reproductions of the same artwork | Unverified. A comparison would need artwork-level identifiers or cross-reproduction image matching with manual confirmation. |
| Shared sources | ArtEmotion specifies WikiArt. The [EmoSet paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_EmoSet_A_Large-scale_Visual_Emotion_Dataset_with_Rich_Attributes_ICCV_2023_paper.pdf) documents artistic images and collection from Openverse, Pexels, Pixabay and Rawpixel. Indirect redistribution of WikiArt artworks is plausible, with no specific common source item established. |
| Emotion labels on overlapping artwork | EmoSet-118K uses human-verified emotion labels. No matched artwork or exposure to its emotion label was established. |
| Metadata coverage | The pinned ArtEmotion release exposes 1,572 raw training artwork IDs and 85 Development artwork IDs across 849 requests. The [official EmoSet repository](https://github.com/JingyuanYY/EmoSet) documents generated emotion-prefixed IDs and paths. The [public mirror's schema](https://huggingface.co/datasets/udeman/EmoSet-118K) has no original URLs, artwork titles, artists or hashes. These fields provide no comparable identifier for an artwork-level join. |

An audit using actual EmoSet pixels could compare exact hashes first and then search resized or recompressed reproductions via perceptual or visual similarity. It would still need the specific checkpoint's fitting subset to attribute exposure. This pass did not download EmoSet or inspect external evaluation-painting labels.

The [organizer README at the checked revision](https://github.com/artemotion-28/artemotion-28/blob/516d7641c3c1d099e63aa8d066971c6b07c89384/README.md) specifies image-plus-target-language ST3 inference, demonstrates frozen base CLIP, explicitly prohibits external annotations for evaluation paintings and requires external-resource citations. Its FAQ defers broader external-data guidance. The [Codabench page](https://www.codabench.org/competitions/18468/) served a bot challenge, so the platform rules could not be independently checked. The question for organizers is whether an image-only ST3 run may use this frozen checkpoint when its external supervised corpus contains artwork and the checkpoint's exposure to emotion annotations of evaluation artworks cannot be ruled out.

The model card declares Apache-2.0. The [EmoSet dataset terms](https://vcc.tech/EmoSet) restrict dataset and derivative commercial use, prohibit dataset redistribution and require citation. The [EmoSet code license](https://github.com/JingyuanYY/EmoSet/blob/main/LICENSE.txt) is MIT. Keep those scopes distinct and cite the model publisher, EmoSet and OpenAI CLIP. Commercial deployment would require clarifying how the data terms apply to the downstream weights.

## Extraction and protocol

Used the public, ungated `EmotionCLIP.pth` at `367bd3dbe08afa4e5e021b8b2617cc65a2409194` (SHA-256 `78a3fd5ed67ebbd34ce6af0c79baaf152e8d57763307c3a5240eb5614b2dba68`). The checkpoint stores visual and text attention prefixes for CLIP ViT-B/32. Its publisher implementation yields a 512-dimensional projected image embedding and scaled image-to-text similarities. The nine emotion-level features are the publisher's prompt softmax scores for amusement, anger, awe, contentment, disgust, excitement, fear, sadness and neutral. The checkpoint has no standalone emotion classifier head, and the scores do not estimate cultural vote proportions without a fitted ST3 readout.

The local implementation uses tensor-only loading and reproduces the pinned preprocessing specification (bicubic shorter-side resize to 224, rounded center crop, RGB conversion, and OpenAI CLIP normalization). It preserves the publisher's visual token index 20 and reads the materialized K/V prefixes. It avoids importing the publisher's top-level code, which references a differently named mixed checkpoint and loads pickled functions. A separate CPU check on three real training artworks matched the publisher's raw image embeddings, text embeddings and softmax scores exactly. The nine token sequences and padding matched the [OpenAI tokenizer](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/simple_tokenizer.py).

The RTX 4060 Laptop GPU preflight used 16 real training images and verified bit-exact repeat inference on that batch. Model and prompt initialization took 1.23 s. The full 1,335-artwork cache took 7.77 s after initialization, including image decoding and preprocessing, at 377 MiB peak allocated and 420 MiB peak reserved. The cache passed all coverage, finiteness and normalization checks and was reused on a second invocation. The runtime was Python 3.12.10, PyTorch 2.7.1+cu126, Pillow 11.2.1 and Tokenizers 0.21.1.

The three fitted arms used the same painting-grouped outer folds and count-weighted EXP-032 no-graph ridge head with its frozen regularization. CLIP-B and EmotionCLIP image embeddings are L2-normalized. The nine emotion features retain their softmax scale. In each fold the heads used 1,068 fitting paintings and held out 267. Corrected Ridge and Fixed Ridge came from their sealed OOF reference. The only combination was fixed in advance at 80% Corrected Ridge and 20% EmotionCLIP embedding predictions, followed by six-decimal rounding. No ensemble parameters were learned. The predeclared advancement conditions included at least +0.002 test-like Contrast, four test-like fold wins, a positive paired lower bound, positive outlier sensitivity checks, no full-P2 Contrast regression and no more than +0.005 bits of pooled Log Loss regression.

## Measured results

| Model | Full Contrast | Full Log Loss | Test-like Contrast | Test-like Log Loss |
| --- | ---: | ---: | ---: | ---: |
| Fixed Ridge | 0.451702 | 0.709247 | 0.460043 | 0.718425 |
| Corrected Ridge | 0.454134 | **0.708500** | 0.463108 | **0.717479** |
| CLIP-B, matched readout | 0.440503 | 0.736314 | 0.446918 | 0.747005 |
| A · Emotion prompt scores | 0.349224 | 0.828202 | 0.351629 | 0.842531 |
| B · EmotionCLIP embedding | 0.422760 | 0.762681 | 0.430207 | 0.773583 |
| C · Corrected Ridge + 20% B | 0.454736 | 0.710772 | 0.463745 | 0.719935 |

P2 covers 13,600 pairs and 1,269 Contrast-eligible paintings. Test-like P2 excludes English, Arabic and Chinese and recomputes eligibility (11,643 pairs and 1,252 Contrast-eligible paintings). Repeated P2 evaluation makes this exploratory. Hosted Development #1063064 scored 0.4655 on a separate population.

| Fold | Full C−Ridge Contrast | Test-like C−Ridge Contrast | Test-like C−Ridge Log Loss |
| --- | ---: | ---: | ---: |
| 1 | −0.000682 | +0.000154 | +0.001150 |
| 2 | +0.000860 | +0.001166 | +0.002608 |
| 3 | +0.000978 | +0.000115 | +0.001708 |
| 4 | +0.001996 | +0.002433 | +0.002160 |
| 5 | −0.000199 | −0.000740 | +0.004172 |

Over 10,000 painting-paired bootstrap draws, C improved pooled full P2 Contrast by +0.000603 (95% CI [−0.001286, +0.002454]) and test-like P2 by +0.000637 (95% CI [−0.001500, +0.002769]). Removing the largest 5% of absolute painting effects gave −0.000154 full and +0.000572 test-like. Removing the top 1% of painting wins gave −0.000623 full and −0.000755 test-like. Test-like C improved 634 paintings and regressed on 618, with pooled Log Loss +0.002456 bits. Exact fold metrics for every arm are in the sealed result JSON.

Painting-centered test-like errors against Corrected Ridge had correlations of 0.959 for the EmotionCLIP embedding and 0.921 for the prompt scores. The matched CLIP-B control was 0.985. The emotion-tuned embedding made somewhat different errors and trailed CLIP-B by 0.016711 test-like Contrast (paired 95% CI for B−CLIP-B [−0.025968, −0.007307]). The blend gain was small and sensitive to which paintings were included. The nine prompt scores underperformed both embedding arms.

## Decision and artifacts

**Retain Corrected Ridge.** The one frozen blend was evaluated despite the weaker standalone scores. Its improvement fell below the frozen effect threshold, included zero and disappeared when the largest wins were removed. No full-training candidate or Development ZIP was warranted. Eligibility also remains uncertain independently of performance. The next competitive test should seek a genuinely different signal, such as one CPU-only whole-image branch for color, tonal contrast and coarse composition, evaluated with a fixed blend and the same grouped protocol. Those measurements would preserve image properties that the 224-pixel crop and normalized embedding may suppress. Their usefulness remains untested here.

The CPU comparison took 71.60 s end to end, at 189.5 MiB peak RSS. Source revision `1ccd1dd03e256b853ff46914342ce98d879f76db`. `runs/competition_emotionclip/features/` contains the 3.23 MiB frozen cache and runtime manifest. `runs/competition_emotionclip/results/` contains `oof.csv`, painting-level results, pooled and fold metrics, fit receipts and an artifact seal. OOF SHA-256 is `cb2e2309974405ae9a69148132f1e2cc955cd1db6baa9b47399285e0df426065`.

Independent readback verified the seals, all pooled and fold metrics, incumbent OOF predictions and fixed blend. The released E50+L predictions, Fixed Ridge heads and manifest, Corrected Ridge neighbor bank and both archived Ridge ZIPs matched their recorded hashes. Tests passed in a clean checkout (324 passed, 3 skipped). The working copy's pre-existing Colab notebook edits caused three historical notebook-test failures and were preserved.

```powershell
.\.venv\Scripts\python.exe scripts/run_competition_emotionclip.py fetch
.\.venv\Scripts\python.exe scripts/run_competition_emotionclip.py extract --device cuda
.\.venv\Scripts\python.exe scripts/run_competition_emotionclip.py cv --device cuda
.\.venv\Scripts\python.exe -m pytest -q tests/test_competition_emotionclip.py
```

Here `cv --device cuda` selects the CUDA-produced cache. All fits and scoring run on CPU. For a replay, use a fresh `--run-dir` and copy its verified `features/` directory from the original run.
