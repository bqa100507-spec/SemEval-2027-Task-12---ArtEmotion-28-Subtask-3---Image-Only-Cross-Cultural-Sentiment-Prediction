# EXP-037 final closure

**Complete and independently replayed from artifacts.** The original EXP-037C ensemble scored **0.472818377 Contrast / 0.728087866 bits** on official Development. That is **+0.002479589 / −0.001620642 bits** versus the active EXP-036 incumbent. EXP-037 remains experimental while Qwen eligibility is unresolved. EXP-036 remains active at **0.470338787 / 0.729708508** locally and **0.4703 (#1065924)** as reported by the owner. Corrected Ridge is the primary fallback and E50+L the historical released fallback. [Cultural diagnostic](exp037_cultural_diagnostic.md) · [Evaluation inference](../docs/exp037_evaluation.md) · [EXP-038 recommendation](exp038_design_recommendation.md)

## Reproduced Development scores

Full scoring covers 849 pairs and 85 paintings, with 82 eligible paintings and 17 eligible Log Loss languages. The Evaluation-like filter removes English, Arabic and Chinese, retains 723 pairs and 80 paintings, and recomputes eligibility to 79 paintings and 14 languages. Each cell is Contrast / Macro Log Loss in bits.

| Candidate | Full Development | Evaluation-like |
| --- | ---: | ---: |
| Corrected Ridge | 0.465462 / 0.740305 | 0.441220 / 0.763903 |
| EXP-036 CS-CVA epoch 5 | 0.437632 / 0.742911 | 0.423661 / 0.766657 |
| EXP-036 incumbent | 0.470339 / 0.729709 | 0.448751 / 0.752288 |
| EXP-037A epoch 4 standalone | 0.420828 / 0.756185 | 0.406011 / 0.781415 |
| EXP-037B epoch 6 standalone | 0.440745 / 0.750681 | 0.431162 / 0.773686 |
| EXP-037C epoch 6 standalone | 0.445957 / 0.744758 | 0.437744 / 0.766806 |
| A blend, alpha 0.00 | 0.470339 / 0.729709 | 0.448751 / 0.752288 |
| B blend, alpha 0.10 | 0.470492 / 0.729281 | 0.450027 / 0.751737 |
| **C original blend, alpha 0.20** | **0.472818 / 0.728088** | **0.454041 / 0.750275** |
| **Post-hoc 65% Ridge / 15% CS-CVA / 20% C6** | **0.473143 / 0.729246** | **0.453667 / 0.751549** |

The original blend is `round6(.80 * EXP036 + .20 * C6)`, using the saved, already rounded EXP-036 blend. Flattening to nominal 60/20/20 differs on **181 rows** by up to **1e−6**. The new 65/15/20 archive is an explicitly flat probability composition. It improves full Contrast by 0.000325 over the original and gives back 0.000374 Evaluation-like Contrast and 0.001158 full Log Loss bits. Preserve both archives with their separate identities. [Selection and uncertainty](exp037_cultural_diagnostic.md)

## Artifact and training verification

- The results ZIP has **112 sealed payload members** and an export manifest. All hashes and ZIP CRCs passed. The three downloads were copied byte for byte to ignored `runs/exp037/closure/`. Both `exp037_best_submission.zip` and the separate `exp037c_ensemble_submission.zip` match the embedded best and C ensemble byte for byte. The owner-reported progress ZIP was unavailable, so its hash was not verified.
- Verified all 1,765 pinned input files, their runtime source against Git, and official organizer checkout `cf7cc444cf56617f3b8417497c6930ad172c5c99`. Reconstructed public gold counts from the official Development votes, with exact parity against the handoff. Training has 1,571 distinct paintings, 16,019 eligible pairs and 82,072 binary votes, disjoint from Development by painting and image. Caption targets are the sealed EXP-036 training-only matrix with 16,001 nonzero masked-in rows. Evidence covers the official releases through Development.
- Worker `38b1e381127c5f4fdc02dfc9845caeba90d9667e`, notebook `8ec141a304a5031ba76fac6c2a9541c696bb6a30`. The notebook references the exact handoff archive and manifest. Runtime records show one Tesla T4 with CUDA 12.6, Python 3.12.10, Torch 2.7.1+cu126, Transformers 4.57.6, and bitsandbytes 0.48.2. Both base revisions and processors match the input config pins. The worker verified pretrained asset hashes at load. The result export preserves their fingerprints and receipts.
- Verified every epoch seal and every learned tensor name, shape and finiteness. Found exactly the permitted SigLIP2 last-six-block Q/V and MLP adapter tensors and Qwen final-eight-block Q/V adapter tensors. Trainable parameters were **2,386,793 / 5,366,633 / 6,677,353**. SigLIP2 adapters were unchanged across A, and Qwen adapters were unchanged through all of A and B. The archived preflight pass and pinned checks support base-weight isolation. Base weights themselves, stage-entry snapshots and individual historical A/B optimizer states were omitted by the worker. The pinned stage transition code restores the sealed selected A4 and B6 states, and the saved epoch and selected inference tensors agree exactly. The exact training lineage before each first optimizer update is a recorded protocol check rather than an independently replayed GPU step.
- Log boundaries verify **197 updates per epoch**, totaling **3,546** across 18 epochs. Final C optimizer, scheduler, scaler and RNG were independently read. All Adam step counters and the scheduler show **1,576 C updates**, and the final learned tensors equal C8. The original A4/B6/C6 selections and all original grid results reproduce from saved probabilities. All six stage ZIPs and the best ZIP pass exact key coverage, schema, six-decimal bounds and blend recomposition. The archived run also replayed each selected checkpoint before sealing it.

| Artifact | SHA-256 |
| --- | --- |
| Original `exp037_results.zip` | `852154e257a198f575dde6d47475aa80588aa48348b50792d47e34603bacb194` |
| Original `exp037_best_submission.zip` and C ensemble | `b50861a1414fd863b4293bb6a8b3dd336bd6f2bc9071f8e20559b2774f4e1f33` |
| Post-hoc 65/15/20 ZIP | `9628ec9e8179a21c9594a3878bee9a480a69044aa347da0f3c019192405f0fa0` |
| Local deployment index | `57a9ddfdac7408ded386004498682270a210947513560c73de63e0da49a3c98b` |

The preserved private audit receipt `runs/exp037/closure/audit.json` also records every stage archive hash, preflight receipt, coefficient and sealed checkpoint. The public aggregate reports exclude images, individual captions, labels and prediction CSVs.

## All 18 epochs

Train entries are binary BCE, eight-emotion CE, caption cosine distance and total loss, in that order. The total is BCE + 0.2 CE + 0.1 caption. Times separate the fitting and validation loops. The linked CSV includes all per-epoch pre-clip group gradient norms, clipping fractions, tensor changes, CUDA memory peaks and checkpoint SHA-256 hashes.

| Epoch | BCE / CE / Caption / Total | Full Contrast / LL | Evaluation-like Contrast / LL | Fit / val seconds | Paintings/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 | 0.59720 / 2.14228 / 0.99025 / 1.12468 | 0.383481 / 0.798733 | 0.367379 / 0.826565 | 551.7 / 40.6 | 2.85 |
| A2 | 0.52113 / 1.92968 / 0.90229 / 0.99730 | 0.407467 / 0.779797 | 0.391578 / 0.808459 | 557.1 / 40.6 | 2.82 |
| A3 | 0.49913 / 1.85500 / 0.84212 / 0.95434 | 0.417771 / 0.757324 | 0.400578 / 0.783310 | 557.7 / 40.7 | 2.82 |
| **A4** | 0.48802 / 1.82288 / 0.81414 / 0.93401 | 0.420828 / 0.756185 | 0.406011 / 0.781415 | 554.6 / 40.6 | 2.83 |
| B1 | 0.48580 / 1.80216 / 0.79668 / 0.92590 | 0.435756 / 0.757697 | 0.418375 / 0.782190 | 611.1 / 40.7 | 2.57 |
| B2 | 0.47675 / 1.75777 / 0.76628 / 0.90493 | 0.421587 / 0.756705 | 0.404650 / 0.781297 | 610.6 / 40.8 | 2.57 |
| B3 | 0.46687 / 1.72311 / 0.74576 / 0.88607 | 0.439709 / 0.748716 | 0.428995 / 0.773021 | 611.5 / 40.8 | 2.57 |
| B4 | 0.45853 / 1.69689 / 0.73282 / 0.87119 | 0.432736 / 0.750977 | 0.420950 / 0.774408 | 612.3 / 40.8 | 2.57 |
| B5 | 0.45201 / 1.67876 / 0.72578 / 0.86034 | 0.440128 / 0.750322 | 0.432098 / 0.773484 | 610.8 / 40.8 | 2.57 |
| **B6** | 0.44736 / 1.66799 / 0.72166 / 0.85313 | 0.440745 / 0.750681 | 0.431162 / 0.773686 | 612.0 / 40.8 | 2.57 |
| C1 | 0.45014 / 1.66516 / 0.71998 / 0.85517 | 0.433760 / 0.761929 | 0.426205 / 0.787029 | 736.9 / 41.0 | 2.13 |
| C2 | 0.45047 / 1.65194 / 0.71488 / 0.85235 | 0.445110 / 0.747568 | 0.437477 / 0.770089 | 736.1 / 40.9 | 2.13 |
| C3 | 0.44323 / 1.63119 / 0.70792 / 0.84026 | 0.442336 / 0.747260 | 0.431683 / 0.770110 | 737.1 / 41.1 | 2.13 |
| C4 | 0.43872 / 1.61510 / 0.70207 / 0.83195 | 0.443246 / 0.741795 | 0.434147 / 0.764304 | 736.9 / 41.0 | 2.13 |
| C5 | 0.43683 / 1.60661 / 0.69686 / 0.82784 | 0.442176 / 0.750102 | 0.433342 / 0.773419 | 736.2 / 41.1 | 2.13 |
| **C6** | 0.43238 / 1.59502 / 0.69403 / 0.82079 | 0.445957 / 0.744758 | 0.437744 / 0.766806 | 735.9 / 41.1 | 2.13 |
| C7 | 0.43259 / 1.59122 / 0.69153 / 0.81999 | 0.437206 / 0.749693 | 0.426701 / 0.772285 | 736.5 / 41.4 | 2.13 |
| C8 | 0.43157 / 1.58860 / 0.69042 / 0.81833 | 0.438315 / 0.748626 | 0.428887 / 0.771029 | 735.9 / 41.3 | 2.13 |

[Epoch history and checkpoint hashes](exp037_assets/epoch_history.csv)

The recorded fit loops total **3 h 16 min 21 s**, plus **12 min 16 s** of per-epoch validation. The owner-reported full notebook was about **3 h 43 min**. All stages used microbatch 4 and effective batch 8. Preflight throughputs were 2.16 / 2.51 / 2.11 paintings/s for A/B/C. Peak allocated / reserved during epochs was **3.89 / 4.26**, **4.00 / 4.31**, and **4.12 / 4.35 GiB**. Four of 3,546 updates clipped, including none in C4–C8. Stage C preflight Qwen grad norms were 0.085947 and 0.024719, and the probe moved its adapters by 0.000496 L2. Actual saved Qwen tensor changes were **0.618010** from B6 to C1 and **0.100013**, **0.053369**, and **0.027550** across the C6, C7, and C8 boundaries. The archive supplies these combined-objective gradients. Per-objective backbone gradients were not recorded in routine training.

### Interpretation of the stages

- **Stage A.** Learning improved every standalone epoch. On the same Development panel A4 trailed CS-CVA by 0.016804 Contrast and Ridge by 0.044635. It had worse ordering and centered deviation errors, and adding 10% A4 to the incumbent lowered full Contrast from 0.470339 to 0.469044. Every A checkpoint's best full blend was alpha zero. Cross-experiment differences with CS-CVA and Ridge also include representation, fitting schedule and objectives.
- **Stage B.** B6 gained **+0.019917 full / +0.025151 Evaluation-like** over A4, with 55/82 full painting wins. B6 beat CS-CVA by **+0.003113** full Contrast while costing **+0.007770 bits**. B's centered RMSE improved relative to A and CS-CVA. B's ordering moved little versus A despite the larger Pearson gain.
- **Stage C.** C6 improved on B6 by **+0.005212 full / +0.006582 Evaluation-like**, with **−0.005923** full Log Loss bits. C2 was within 0.000847 full Contrast of C6, and C4 had the lowest standalone full and filtered Log Loss. C7 and C8 each lost both rank agreement and calibration relative to C6 while training loss fell. Conditional full Contrast intervals for C7−C6 and C8−C6 were **[−0.015435, −0.002442]** and **[−0.012812, −0.002812]**. The pattern is consistent with late overfitting on this panel. Checkpoint selection and one run limit causal and generalization claims. Progressive comparisons also include additional optimization and new schedules.

## Caption gradient audit

The preflight logged **0.0** weighted caption gradient at the fusion projections in all three stages. The seeded sample was Van Gogh’s *Vineyards with a View of Auvers* at training painting index 1494. All nine pairs had valid unit-normalized targets and **48** weighted votes. The next three sampled paintings had **55, 36 and 60** valid votes. The denominator and masking in the diagnostic were positive. The training loss uses the effective-batch caption vote mass, as in the pinned worker. Cosine similarities are evaluated in FP32 on caption vectors cast to float. AMP still applies to their producers.

The historical caption-only `caption_language.weight` changes direction between saved epochs, beyond what AdamW weight decay alone can produce. The final C8 Adam first-moment norms are positive for **all five** caption-only tensors, ranging from **0.000454 to 0.002802**. The proxy test used saved A4/B6/C6 heads and four authorized paintings’ unit targets, with deterministic synthetic encoder activations. All 36 standalone FP32, unscaled FP16 and 128-scaled FP16 backwards produced nonzero caption head, language embedding, fusion projection and proxy adapter gradients. For the exact first painting, unscaled FP16 visual/Qwen projection norms were **0.00556/0.01341** (A4), **0.04342/0.10685** (B6), and **0.02379/0.06582** (C6). The native tiny Qwen CPU regression independently tests caption-only connectivity into both enabled adapter families.

The native retest loaded both exact pretrained models and A4/B6/C6 on the RTX 4060. With unscaled FP16 it reproduced the zero projection gradient on the original painting at all three saved checkpoints. Scaling the same caption loss by **128** before backward, then dividing the recorded norms by 128, restored nonzero gradients in both projections for both paintings at all three checkpoints. A second scale of 8,192 gave similar norms. That isolates loss-scale-sensitive FP16 underflow in the diagnostic. The loss value, data, masks, weights and seeded forward were held fixed. Peak native probe memory was 3.77 GiB allocated / 3.85 GiB reserved, with no optimizer steps.

| Original painting, saved checkpoint | Visual projection at 128× | Qwen projection at 128× | Visual LoRA at 128× | Qwen QLoRA at 128× |
| --- | ---: | ---: | ---: | ---: |
| A4 | 0.001339 | 0.000710 | frozen | frozen |
| B6 | 0.000786 | 0.000889 | 0.005567 | frozen |
| C6 | 0.001164 | 0.001373 | 0.010983 | 0.000739 |

The historical component probe omitted loss scaling, whereas training backpropagated the scaled **combined** objective. The sealed C8 GradScaler state recovered scale **128** and growth tracker **1,576**. The exact T4 two-step preflight activations are unavailable, so this is a reproduced failure mode with the same sample and exact pretrained paths at saved checkpoints. It explains why a logged zero is a false negative for connectivity. Historical caption-only parameter movement and the native retest demonstrate caption learning and gradient reachability through enabled adapters. Per-objective gradient magnitudes across the original training steps and the causal performance value of captions remain unmeasured. Future preflight should use the training scale, divide norms by it and audit several positive-mass paintings. [Proxy measurements](exp037_assets/caption_probe.json) · [Native scaled measurements](exp037_assets/caption_native_scaled_probe.json)

## Scope and eligibility

The official README requires artwork-only ST3 prediction with the requested target language. Its FAQ defers LLM/resource guidance. The public issue explicitly asks whether VLMs are allowed and had zero replies when checked. Codabench rules were inaccessible through the available readback. The current status stays **UNCONFIRMED_QWEN_ST3**. The closure used official Train and public Development, and the refreshed deployment path uses pixels and target language. [Organizer rules and FAQ](https://github.com/artemotion-28/artemotion-28#competition-rules-and-terms) · [Public issue](https://github.com/artemotion-28/artemotion-28/issues/1) · [Codabench](https://www.codabench.org/competitions/18468/)
