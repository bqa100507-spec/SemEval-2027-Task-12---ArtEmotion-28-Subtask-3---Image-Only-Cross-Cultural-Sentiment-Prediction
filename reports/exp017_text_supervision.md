# EXP-017 Human-caption supervision

## Frozen protocol

The base is release `2199868364ca315fed1b141d6e565528314ba5d5`. Production E50+L is the reference. The new protocol ranks image-content groups by SHA-256 of seed, canonical painting ID and image digest, separated by NUL bytes. Seed 17027 reserves ceil(15% of groups), and seed 17028 ranks the remainder for round-robin assignment to five folds. Internal stopping reserves ceil(10% of each training partition's groups) using seed 17029 plus fold ID. These paintings appeared in historical studies, so the confirmation set is an experiment-level holdout. [Frozen configuration](../configs/exp017.json)

Only original official training rows supply captions and votes. The audit passed with exactly 16,019 eligible pairs and 1,571 content-distinct training paintings. It locked 236 paintings for confirmation and 1,335 for CV. Teacher extraction covers the 85% development-training paintings' eligible binary-caption rows. Original captions retain their exact text. Exact-text deduplication saves encoder work while every annotation retains its vote association. Mean targets retain repeated annotation rows and exclude ambiguous labels and empty/invalid text.

CLIP reuses the text tower and trained projection of `openai/clip-vit-large-patch14` at `32bd64288804d66eefd0ccbe215aa642df71cc41`. E5 uses `intfloat/multilingual-e5-base` at `d128750597153bb5987e10b1c3493a34e5a4502a`, with the `passage: ` prefix and attention-masked mean pooling. Both return 768-wide normalized FP32 embeddings and use pinned tokenizer assets and weight hashes. CLIP truncates at 77 tokens and E5 at 512. The original Python environments are used with their existing dependencies.

The pinned organizer FAQ leaves external-model guidance pending. The earlier saved public competition page allows pretrained models in general, but provides no checkpoint-specific E5 authorization. CLIP is the primary existing-resource condition. E5 is **RESEARCH_ONLY**, and its MIT license does not establish competition eligibility. No explicit E5 prohibition was found. The E5 model card lists multilingual retrieval/contrastive training from XLM-R and warns about weaker low-resource coverage. [Organizer FAQ](https://github.com/artemotion-28/artemotion-28/blob/cf7cc444cf56617f3b8417497c6930ad172c5c99/README.md#faqs) · [CLIP license](https://github.com/openai/CLIP/blob/main/LICENSE) · [E5 card and procedure](https://huggingface.co/intfloat/multilingual-e5-base/tree/d128750597153bb5987e10b1c3493a34e5a4502a)

The sentiment branch matches archived EXP-013. It uses concatenated unit CLIP-B, SigLIP 2 Base and CLIP-L features divided by sqrt(3), then a width-64 linear layer, LayerNorm, GELU and dropout 0.20. Sentiment combines a shared readout, a language bias and the scaled language–visual interaction. The auxiliary branch takes the width-64 hidden vector and width-64 language embedding, then applies Linear(128,64), GELU and Linear(64,64). Its normalized output predicts the normalized 64-dimensional unwhitened PCA caption target. PCA fits only on the applicable training subset. Internal stopping paintings supply BCE only. Each selected epoch count is followed by a seeded reset and full-partition refit with a separately fitted PCA.

Every condition allocates the same model, including the unused auxiliary branch in the BCE-only control, to match initialization and dropout RNG state. AdamW uses learning rate 0.001, weight decay 0.01, batch size 256, clipping 1.0, up to 150 epochs and patience 15. Seeds are 2027, 2028 and 2029. The loss is vote-normalized BCE plus 0.1 times MSE averaged over the 64 target coordinates. Auxiliary pairs use inverse valid-pair counts within each painting, normalized within the batch. Caption counts do not multiply auxiliary weights. Degenerate targets have zero auxiliary weight.

M0 refits all three original weighted LR heads inside each new training fold and preserves E50+L's nested six-decimal rounding. M1 is BCE-only, M2 adds CLIP text and M3 adds E5 text. The three seed probabilities are averaged and rounded to six decimals. B1, B2 and B3 are the fixed 50/50 rounded blends with M0. Unknown training languages fall back deterministically to M0. Student inference takes only artwork feature lookups and target language.

The promotion candidates are B2 and B3. Before scoring, each must gain at least +0.003 pooled Contrast over M0 and +0.002 over B1, win at least three folds against M0, and keep fold-mean Macro Log Loss within M0 +0.005 bits. All 45 student seed/fold fits must complete with finite parameters and isolated partitions. The highest qualifying pooled Contrast wins. If neither qualifies, retain M0 and keep the confirmation set sealed.

Only a CV winner can open the confirmation set. It must gain at least +0.001 Contrast over M0, have positive gain over B1 and keep Macro Log Loss within M0 +0.005 bits. A confirmation failure ends the study. Only a confirmed fixed candidate can use the full training split and then score the 849 public development pairs once. E5 remains RESEARCH_ONLY regardless of measured scores.

```powershell
.venv/Scripts/python.exe scripts/run_st3.py caption-audit
.venv/Scripts/python.exe scripts/run_st3.py caption-token-audit --teacher clip
.venv/Scripts/python.exe scripts/run_st3.py caption-token-audit --teacher e5
.venv/Scripts/python.exe scripts/run_st3.py caption-extract --teacher clip
.venv/Scripts/python.exe scripts/run_st3.py caption-extract --teacher e5
.venv/Scripts/python.exe scripts/run_st3.py caption-cv
.venv/Scripts/python.exe scripts/run_st3.py caption-holdout
.venv/Scripts/python.exe scripts/run_st3.py caption-dev
```

All stages share `runs/exp017_text_supervision/`. Use a shared fresh `--run-dir` to repeat the study. Completed CV folds are checkpointed under `cv_work/`. If confirmation passes, run `caption-extract --teacher clip --full-training` or its selected E5 equivalent before `caption-dev`. Progressive text caches are separate from historical image caches.

## Caption audit and teacher diagnostics

There are 165,013 original training annotations and nonempty captions, with 161,883 distinct exact texts. The audit found 3,130 repeated-text rows and 123 repeated combinations of painting, language, emotion and caption. These are descriptive duplicate counts; all released annotation rows retain their vote association. Positive annotations total 105,158, negatives 47,289 and ambiguous annotations 12,566.

| Emotion | Original training annotations |
| --- | ---: |
| amusement | 16,723 |
| awe | 18,998 |
| contentment | 49,702 |
| excitement | 19,735 |
| anger | 3,433 |
| disgust | 10,074 |
| fear | 14,109 |
| sadness | 19,673 |
| other | 1,035 |
| something else | 11,531 |

All 16,019 eligible pairs have valid binary-caption supervision. Caption lengths have median 70 characters, 90th percentile 115, 95th percentile 134, 99th percentile 179 and maximum 1,671. Full painting/language/emotion counts and per-annotation identities are in `audit/annotation_counts.csv` and `audit/annotation_mapping.csv`. The annotation ID is the original CSV row position, and painting identity uses the painting field and image-content hash.

IsiNdebele has 2,062 training annotations, IsiXhosa 1,872 and Kazakh 2,426, but none has an eligible ST3 pair. Among the 25 eligible languages, sparse coverage is greatest for Emakhuwa with 5 pairs, Kinyarwanda with 9, Darija with 17 and Urdu with 18. Text availability is complete; eligibility thresholds account for this sparse pair coverage.

The descriptive tokenizer audit covers all 165,013 training captions across all 28 languages. It performs no embedding extraction or parameter fitting. CLIP truncates 32,367 captions, or 19.61%. Its largest truncation rates are Burmese 97.41%, Kazakh 88.38%, Tamil 87.49%, Kyrgyz 74.93%, Hindi 48.83%, Korean 32.85%, Chinese 29.24% and Thai 25.69%. English has zero truncation. The byte-level tokenizer emitted no unknown tokens. This does not establish semantic coverage. The CLIP card limits its assessed language scope to English. [CLIP model card](https://huggingface.co/openai/clip-vit-large-patch14/blob/32bd64288804d66eefd0ccbe215aa642df71cc41/README.md)

E5's maximum observed length was 279 tokens including the prefix and special tokens, so no caption exceeded its 512-token context. The full audit recorded 302 unknown-token occurrences, including 189 in Yoruba and 68 in Korean. Tokenization counts describe encoding behavior. E5's language metadata does not list Darija, Emakhuwa, Igbo, IsiNdebele, IsiZulu, Kinyarwanda, Setswana or Yoruba as separate supported codes. Darija may share Arabic coverage, and the remaining gaps and the card's low-resource warning warrant caution. The run preserves per-language token lengths, character and whitespace-word ratios and unknown-token counts in `audit/*_all_training_tokens.csv`.

Both teacher extractions covered the same 69,719 eligible binary-caption annotation rows and 69,153 exact texts from the 1,335 development-training paintings. The locked paintings supplied no teacher targets or embeddings. The same caption text can occur in multiple annotations, and its frozen embedding is reused without transferring annotation labels between partitions.

| Teacher | Repeat probe | Embedding duplicates among unique texts | Centroid norm | Mean component variance | Extraction wall time | Peak CUDA allocated / reserved |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CLIP text | Bit-identical FP32 | 21 | 0.833348 | 0.000397827 | 251.11 s | 738 / 854 MiB |
| Multilingual E5 | Bit-identical FP32 | 0 | 0.892506 | 0.000264886 | 214.68 s | 1,915 / 3,050 MiB |

Unit norms were within 1.8×10⁻⁷ of one. No batch reduction was required. Each teacher's NPZ is 230,141,694 bytes and each progressive SQLite cache is 289,742,848 bytes. E5 downloaded 1,112,201,288 bytes of weights with verified SHA-256. CLIP reused its existing 1,710,540,580-byte checkpoint and instantiated only the 123,650,304-parameter text tower and projection. E5 instantiated 278,043,648 parameters. Both ran frozen FP32 on the RTX 4060 Laptop using the unchanged Python 3.12, PyTorch 2.7.1+cu126 and Transformers 4.49.0 environment.

## New-protocol train-only results

These scores use 13,600 pairs from the 1,335 development-training paintings. Contrast includes 1,269 paintings and pooled Macro Log Loss includes 21 languages. Each fold holds out 267 paintings. The historical EXP-010 CV used different paintings and training sizes, so its scores are a separate protocol. The scorer reconstructs the published ST3 specification. Training BCE uses natural logarithms; evaluation Macro Log Loss uses base-2 logarithms.

Each score cell gives Cultural Contrast / Macro Log Loss.

| System | Fold mean | Pooled OOF | Contrast fold wins vs M0 | Painting wins / regressions vs M0 | Pair-order agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 | 0.446971417 / 0.711228815 | 0.447097707 / 0.710554182 | 0/5 | 0 / 0 | 0.697709685 |
| M1 | 0.437666659 / 0.716859251 | 0.437792268 / 0.716045717 | 1/5 | 609 / 660 | 0.692201334 |
| M2 | 0.438889557 / 0.716952090 | 0.439016172 / 0.716105044 | 1/5 | 605 / 664 | 0.693918512 |
| M3 | 0.439038102 / 0.716748977 | 0.439157265 / 0.715933928 | 1/5 | 617 / 652 | 0.694409134 |
| B1 | 0.448448637 / 0.707212585 | 0.448585747 / 0.706474066 | 3/5 | 668 / 601 | 0.699426864 |
| B2 | 0.449609900 / 0.706991442 | 0.449753673 / 0.706239161 | 4/5 | 659 / 610 | 0.699527218 |
| B3 | 0.449015956 / 0.707160757 | 0.449148635 / 0.706424388 | 3/5 | 682 / 587 | 0.699739078 |

Pair-order agreement gives half credit to prediction ties and excludes tied gold pairs.

### Exact per-fold scores

| Fold | M0 | M1 | M2 | M3 | B1 | B2 | B3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.433428997 / 0.666222015 | 0.423966958 / 0.667393244 | 0.424857459 / 0.666962590 | 0.423729793 / 0.667547601 | 0.434594193 / 0.659326267 | 0.435352315 / 0.659091339 | 0.434985722 / 0.659490312 |
| 2 | 0.459193722 / 0.724814258 | 0.445389617 / 0.726876594 | 0.446714229 / 0.725295970 | 0.445932888 / 0.725310027 | 0.458890750 / 0.719070181 | 0.459254116 / 0.718787190 | 0.458551673 / 0.718806307 |
| 3 | 0.433536160 / 0.713191639 | 0.439895562 / 0.727978620 | 0.438877162 / 0.728019551 | 0.442692888 / 0.729219345 | 0.442272274 / 0.713990501 | 0.442060226 / 0.713927892 | 0.443570429 / 0.714070502 |
| 4 | 0.461302362 / 0.739182174 | 0.449276477 / 0.755738595 | 0.451508646 / 0.755137475 | 0.450243885 / 0.755399456 | 0.462735589 / 0.739413284 | 0.466010655 / 0.737918114 | 0.463208696 / 0.739216374 |
| 5 | 0.447395846 / 0.712733987 | 0.429804682 / 0.706309200 | 0.432490286 / 0.709344862 | 0.432591054 / 0.706268457 | 0.443750380 / 0.704262692 | 0.445372190 / 0.705232675 | 0.444763257 / 0.704220290 |

### Incremental text effect and uncertainty

All intervals use 10,000 paired painting resamples, seed 172027, conditional on these repeatedly studied paintings.

| Comparison | Pooled Contrast change | Paired 95% interval |
| --- | ---: | ---: |
| M2 minus M1 | +0.001223903 | [-0.000688629, +0.003080491] |
| M3 minus M1 | +0.001364997 | [+0.000014291, +0.002666535] |
| B2 minus B1 | +0.001167926 | [+0.000269630, +0.002043707] |
| B3 minus B1 | +0.000562889 | [-0.000058026, +0.001168071] |
| B1 minus M0 | +0.001488039 | [-0.001750361, +0.004845444] |
| B2 minus M0 | +0.002655966 | [-0.000759359, +0.006141706] |
| B3 minus M0 | +0.002050928 | [-0.001206895, +0.005425620] |

**NO-GO.** B2 gained +0.002656 pooled Contrast over M0 and +0.001168 over the matched BCE-only B1, below the required +0.003 and +0.002. Its four fold wins and loss guardrail passed. B3 gained +0.002051 over M0 and +0.000563 over B1, also below both Contrast thresholds. Its three fold wins and loss guardrail passed. All 45 seed/fold fits completed with finite weights. The guarded commands retained M0 before any confirmation or public-development scoring.

CLIP supervision has a positive matched-ensemble estimate, with a paired interval above zero, while its standalone increment over M1 remains uncertain. E5 has a small positive standalone increment, with the interval just above zero, and an uncertain ensemble increment over B1. The standalone students all trail M0. Neither text ensemble established the predeclared competition gain. The 236-painting confirmation set remains sealed and there is no EXP-017 candidate development ZIP.

### Painting and language changes

M2 improved Contrast on 688 paintings and regressed on 581 against M1. M3 improved 635 and regressed 634. B2 improved 684 and regressed 585 against B1, while B3 improved 647 and regressed 622. B2’s largest matched gain was Renoir’s `woman-s-head-4` (+0.215511), and its largest regression was Chagall’s `illustration-for-louis-aragon-s-work-one-who-says-things-without-saying-anything-1976-17` (−0.091221). B3’s largest gain was Cézanne’s `portrait-of-joachim-1896` (+0.085633), and its largest regression was Lotto’s `the-trinity-1523` (−0.057245).

CLIP supervision reduced standalone pooled language loss in 9 of 21 eligible languages and reduced the ensemble loss against B1 in 10. Burmese had the largest ensemble reduction (−0.002985 bits) and Thai the largest increase (+0.000887 bits). E5 reduced standalone loss in 11 languages and ensemble loss against B1 in 12, with Tamil improving most (−0.000821 bits) and Setswana regressing most (+0.000605 bits). Against M0, B2 reduced language loss in 20 of 21 languages and B3 in 19. The records describe prediction changes on this dataset.

Per-fold painting and language records are in `cv/paintings.csv` and `cv/languages.csv`. The pooled metric files retain each language’s pair count and inclusion status.

### Caption coverage across all training languages

All listed annotations have a nonempty caption. E5 truncation is zero in every listed language.

| Language | Training annotations | Eligible ST3 pairs | CLIP truncation across all training captions |
| --- | ---: | ---: | ---: |
| Arabic | 5,744 | 535 | 9.56% |
| Burmese | 6,983 | 198 | 97.41% |
| Chinese | 6,579 | 1,113 | 29.24% |
| Darija | 6,024 | 17 | 5.78% |
| Emakhuwa | 1,850 | 5 | 0.00% |
| English | 6,166 | 648 | 0.00% |
| Hausa | 7,084 | 930 | 0.00% |
| Hindi | 6,906 | 908 | 48.83% |
| Igbo | 7,376 | 1,269 | 5.40% |
| Indonesian | 5,732 | 296 | 0.14% |
| IsiNdebele | 2,062 | 0 | 7.47% |
| IsiXhosa | 1,872 | 0 | 2.67% |
| IsiZulu | 7,583 | 756 | 1.89% |
| Kazakh | 2,426 | 0 | 88.38% |
| Kinyarwanda | 5,534 | 9 | 0.34% |
| Korean | 7,290 | 1,053 | 32.85% |
| Kyrgyz | 6,111 | 642 | 74.93% |
| Malay | 7,063 | 720 | 0.00% |
| Setswana | 5,974 | 333 | 0.49% |
| Swahili | 6,947 | 993 | 0.36% |
| Tagalog | 7,924 | 658 | 0.19% |
| Tamil | 6,620 | 662 | 87.49% |
| Thai | 4,986 | 240 | 25.69% |
| Turkish | 7,195 | 572 | 1.56% |
| Urdu | 3,024 | 18 | 1.39% |
| Uzbek | 7,312 | 1,044 | 0.51% |
| Vietnamese | 7,559 | 1,228 | 12.24% |
| Yoruba | 7,087 | 1,172 | 17.27% |

## Fitting diagnostics and cost

The 64 unwhitened PCs retained a mean 89.3199% of CLIP target variance across full-fold fits (range 89.2368–89.3790%), and 84.5928% for E5 (84.5785–84.6139%). The corresponding inner-fit means were 89.3271% and 84.6148%. No projected target was degenerate. PCA means, components, explained variance, target vote composition and partition membership are saved for every inner and refit transformation.

Each student has 145,370 parameters, including the allocated auxiliary branch in the BCE-only control. Mean selected refit epochs were 23.87 for M1, 25.20 for M2 and 24.13 for M3. All selections were between epochs 14 and 40, with the 15-epoch patience applied to internal BCE. Full training/validation BCE and auxiliary curves are under `cv_work/fold_*/curves.csv`.

The complete CV command took 987.31 s. Summed student fitting took 152.01 s for M1, 275.34 s for M2 and 307.06 s for M3, including epoch selection and refit. The 15 CPU LR fits took 51.46 s. Student held-out prediction totaled 1.01 s across all 45 fits. Peak student CUDA allocation was 80.71 MiB, with 92 MiB reserved. The frozen teacher extraction cost is recorded above. Teachers are absent from student inference; the measured inference paths use the existing three frozen image representations.

### Seed variation and loss curves

These are mean ± sample SD of separately scored seed models, rather than the three-seed probability ensemble above. Every seed was retained.

| System | Pooled Contrast across seeds | Pooled Macro Log Loss across seeds |
| --- | ---: | ---: |
| M1 | 0.427712884 ± 0.004392595 | 0.721510386 ± 0.001137320 |
| M2 | 0.429555036 ± 0.003069161 | 0.721698951 ± 0.002804223 |
| M3 | 0.429434002 ± 0.004539053 | 0.721490633 ± 0.001255671 |
| B1 | 0.446183068 ± 0.002561404 | 0.707627281 ± 0.000447512 |
| B2 | 0.447464554 ± 0.001698387 | 0.707424431 ± 0.000896290 |
| B3 | 0.446746222 ± 0.002201099 | 0.707590424 ± 0.000444490 |

The mean first, best and final internal-validation BCE values are shown below, along with the final full-partition refit losses. BCE is in nats and the auxiliary MSE already averages over 64 coordinates. Its coefficient remains 0.1.

| Student | First stopping BCE | Best stopping BCE | Final stopping BCE | Final refit BCE | Final auxiliary MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| M1 | 0.537391718 | 0.506636920 | 0.514342801 | 0.439483809 | 0.000000000 |
| M2 | 0.537390471 | 0.506487189 | 0.515320261 | 0.437615071 | 0.001759956 |
| M3 | 0.537381899 | 0.506364053 | 0.515183645 | 0.438825299 | 0.005237105 |

## Verification and decision

Independent replay re-fitted all 15 fold-specific LR heads and reproduced M0 exactly. It reloaded all 45 saved student checkpoints, reproduced every seed prediction, reconstructed each three-seed average and each rounded ensemble, and recomputed all pooled metrics. It re-fitted all 20 PCA transformations and checked their training membership, means, components and target vote composition. It also checked each selected epoch against the internal BCE curve and confirmed image-content isolation. All 95,200 CV prediction rows passed exact-coverage checks. This additional verification took 526.36 s and is separate from the CV timing.

A fresh 64-caption batch for each teacher reproduced the saved FP32 features and a second forward pass byte for byte, with maximum absolute error zero. Both Python environments passed their dependency checks. The Windows suite passed 123 tests and skipped the existing symlink fixture.

The label-free production regression reproduced all 849 E50+L, E50 and CLIP development rows byte for byte, and validated their archives. The E50+L CSV SHA-256 remains `162875b641407efea3bf28d1d147c2d9a3dd3dc2faed8863f9438bee517199a3`. The original E50+L archive also validated. All 1,133 distinct historical files in the combined EXP-017 and release-016 inventories matched their SHA-256 digests. Production inference code, model pins and bundles retain exact release parity.

Artifacts include `independent_verification.json`, `teacher_replay.json`, `seed_fold_metrics.csv`, `seed_pooled_metrics.csv`, `curve_summary.csv`, `pooled_language_changes.csv`, `historical_validation.json` and the original per-fold fits, checkpoints and curves under `runs/exp017_text_supervision/`. The locked holdout and public-development commands both completed their guards with NO-GO.

Keep E50+L for competition. A focused CLIP-supervision follow-up has some support from its matched-ensemble interval, but this run did not meet the required effect size. The E5 ensemble increment is uncertain and its competition eligibility remains provisional. A broader text-teacher search is not supported by this comparison. Any follow-up should predeclare its protocol and seek confirmation beyond these repeatedly studied paintings before competition promotion.
