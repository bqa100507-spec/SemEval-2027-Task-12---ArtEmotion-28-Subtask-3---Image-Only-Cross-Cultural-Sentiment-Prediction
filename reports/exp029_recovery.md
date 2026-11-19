# EXP-029 recovered pilot

**EXP029 = PILOT_RECOVERED / EXPLORATORY.** Production remains E50+L.

The original attempt 20261119T103915Z-2c837e97 completed both arms and failed at scoring. The validation frame already contained the baseline probability, causing a column collision inside the shared scorer. The worker now projects gold onto the pair keys and vote counts before scoring. The shared scorer is unchanged.

CPU recovery verified the original export, the model identity 3ad45229687ce1fb1315843c29955c49dde33832a74ea5b8f2ccbdeeb17e8e26 and both sets of sealed member hashes. The original training revision remains 061f6bf0d3aa4d0b24bfbf1766e9139f307a9d0c. The rebuilt cross-fit training and separate validation baseline probability fingerprints exactly matched the original receipt. Both sealed CSVs match their saved logits at six decimals and cover the same 493 pairs from 48 paintings.

| Arm | Cultural Contrast | Macro Log Loss (bits) | Validation BCE by epoch (nats) | Selected epoch |
| --- | ---: | ---: | --- | ---: |
| E50+L | 0.417755 | 0.781648 | — | — |
| Projector only | 0.415210 | 0.784559 | 0.527746 → 0.529317 | 1 |
| Projector + QLoRA | 0.410691 | 0.786886 | 0.531374 → 0.532725 | 1 |

| Comparison | Δ Contrast | Δ Macro Log Loss (bits) |
| --- | ---: | ---: |
| Projector only vs E50+L | −0.002545 | +0.002911 |
| Projector + QLoRA vs E50+L | −0.007065 | +0.005238 |
| Projector + QLoRA vs projector only | −0.004520 | +0.002327 |

Both arms regressed on the pilot validation set. The failed run computed the shuffle in memory but did not persist its predictions. That comparison is unavailable in CPU recovery. [Private recovered results and ZIP](https://drive.google.com/drive/folders/16jHYau-I474eS6z0S-trX73Ezc0HHH1h)

The [recovery notebook](../notebooks/EXP029_Recovery_Colab.ipynb) uses a single CPU Run all and writes a fresh Drive attempt. It reconstructs the baseline by refitting the original CPU LR heads only. It reuses both trained Qwen checkpoints and leaves their original files untouched. Optional L4 shuffle mode requires explicit opt-in and checks correct-image checkpoint parity before running shuffled inference. The L4 recovery mode has not been executed.
