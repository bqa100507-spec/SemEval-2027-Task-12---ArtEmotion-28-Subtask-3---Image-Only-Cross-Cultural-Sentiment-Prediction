# ST3 competition Track 1 — fixed ridge selected

**Recommendation: FIXED RIDGE** as the January challenger. Keep E50+L as production. On test-like P2, fixed ridge reached **0.460043042 Contrast / 0.718424752 Macro Log Loss**. Learned stacking reached **0.459793881 / 0.717905364**. Its small Log Loss improvement came with lower primary Contrast in both scopes. The selected fixed-ridge bundle has been refitted, saved, and exercised on unseen images with the real encoders on CPU.

## Matched comparison

Before training, the historical replay reproduced all EXP-032 ridge and graph OOF rows and the matched E50+L reference. Full P2 pooled Contrast and Macro Log Loss, and the five fold metrics, matched the archive to `1e-12`. The new fixed-ridge experiment then reproduced all **13,600** archived ridge probabilities exactly.

The two alternatives share every outer ridge head and fitting row. Each outer training partition was split into three inner painting groups. Three sets of held-out component probabilities, all produced inside that outer training partition, fitted one global weight vector. The frozen simplex objective used painting-centered MSE plus `0.10 ×` Macro Log Loss in bits and `0.005 ×` squared distance from `(0.25, 0.25, 0.50)`. The inner stack fit excluded English, Arabic and Chinese and recalculated eligibility. Base heads retained all eligible fitting languages. The source and config were committed before outer scoring. There was no weight grid or alternate stack.

Pooled scores use 1,269 eligible paintings and 21 Macro Log Loss languages on full P2, and 1,252 paintings and 18 Macro Log Loss languages on filtered P2. Each cell is **Contrast / Macro Log Loss in bits**.

| Outer fold | Full fixed | Full learned | Filtered fixed | Filtered learned |
| --- | ---: | ---: | ---: | ---: |
| 1 | .438684 / .665183 | .438147 / .664863 | .445910 / .678168 | .445692 / .677847 |
| 2 | .463904 / .724336 | .462949 / .724213 | .468017 / .733469 | .466722 / .733194 |
| 3 | .435016 / .711310 | .434728 / .710790 | .439185 / .722344 | .439041 / .721621 |
| 4 | .467131 / .736788 | .466997 / .736211 | .482765 / .744125 | .482801 / .743564 |
| 5 | .453135 / .711884 | .453153 / .711295 | .464207 / .717193 | .464616 / .716518 |
| **Pooled** | **.451702 / .709247** | **.451321 / .708815** | **.460043 / .718425** | **.459794 / .717905** |
| E50+L pooled, reference | .447098 / .710554 | — | .455486 / .719860 | — |

| Learned − fixed | Contrast delta [paired 95% interval] | LL delta | Fold wins | Painting wins / ties / losses | After 5% absolute-delta trim |
| --- | --- | ---: | ---: | ---: | ---: |
| Full | −.000381094 [−.000767196, +.000001915] | −.000431948 | 1/5 | 622 / 0 / 647 | −.000262043 |
| Filtered | −.000249161 [−.000687919, +.000197097] | −.000519388 | 2/5 | 615 / 0 / 637 | −.000102515 |

Filtered painting-level delta percentiles (5th/25th/median/75th/95th) were `−.012433, −.003789, −.000106, +.003471, +.012188`. Full P2 percentiles were `−.011050, −.003801, −.000125, +.003140, +.010428`. The bootstrap uses 10,000 painting-paired draws. All P2 figures are exploratory. Fixed ridge cleared its frozen reference gate, gaining **+.004604079** full and **+.004557282** filtered versus E50+L, with five fold wins in each. The filtered interval was **[+.001847802, +.007312673]**. Learned weights missed the **+.002** filtered incremental Contrast gate, and the paired interval included zero.

| Fold | CLIP-B | SigLIP2-Base | CLIP-L |
| --- | ---: | ---: | ---: |
| 1 | .199039 | .269416 | .531545 |
| 2 | .184734 | .261605 | .553661 |
| 3 | .204744 | .239948 | .555308 |
| 4 | .208172 | .254779 | .537049 |
| 5 | .200825 | .233211 | .565964 |
| Mean | .199503 | .251792 | .548705 |
| Population SD | .008034 | .013426 | .012620 |

## Bundle, cost and verification

- The full refit used the authorized **1,571 official `split=train` paintings, 16,019 pairs and 82,072 binary votes**. All three solvers converged. An in-memory repeat refit reproduced every coefficient and language mapping exactly. The single final refit kept the already fixed 25/25/50 weights and original nested rounding. No score was computed on that fit's training paintings. Development labels were not used and the final evaluation labels remain untouched.
- Nested CV and evaluation took **63.83 s** on two CPU threads, with **195.52 MiB** peak process RSS. The three final CPU fits plus dataset and cache validation took **9.57 s**, peaking at **290.87 MiB**. In a CPU microbenchmark on 849 precomputed rows, fixed and learned fusion alone took about **8.0 µs** and **3.5 µs** per call respectively. Both use the same three vision forwards and frozen processors.
- The real CPU smoke used two unseen synthetic image contents and four requested pairs, including an image alias and Kazakh's shared-slope fallback. A deliberate stop after CLIP-B cached its batch published no output. Rerunning recovered both CLIP features and computed the other two branches. The interrupted command took **8.42 s**, the resumed command **22.01 s**, and the all-cache run **9.01 s** with **6.27 ms** of head inference. The resumed and warm CSVs were byte-identical. Independent coefficient/cache replay reproduced all four values, and the official-schema validator accepted the ZIP.
- Production source files were unchanged. All release bundle hashes and the original 849-row E50+L CSV SHA `162875b641407efea3bf28d1d147c2d9a3dd3dc2faed8863f9438bee517199a3` matched. The clean checkout suite passed **284 tests, 2 skipped**. In the active working copy, the same suite reported three failures in two already-edited historical Colab notebooks. Their edits were preserved. The focused challenger, scorer, submission and production-inference tests passed **43 tests, 1 skipped** there.

## Local artifacts and January runbook

| Artifact | Local path or SHA-256 |
| --- | --- |
| CPU experiment | `runs/competition_track1/results/` contains the compact `oof.csv`, `painting_contrast.csv`, `metrics.json` and checksum seal. Per-fold solver and stack receipts are under `fold_1` through `fold_5`. |
| January challenger | `runs/competition_track1/january_bundle/` contains 412,722-byte `heads.npz` and `manifest.json` with encoder pins, preprocessing assets, fit receipts and provenance. |
| Heads SHA-256 | `9105a0623867916c046dfd3ec4e1fa9fb6854c2d69068570c3b169bf47c89e76` |
| Bundle manifest SHA-256 | `1d211fc70c27d5ababaf6b7f05afa75c498854c4201b6039e333891d5016185a` |
| OOF SHA-256 | `d57d56b83cb3fb68caccea2ed5762dd5938a0db024580085f86f72ccfea2930d` |
| Smoke ZIP | `runs/competition_track1/smoke_warm/submission.zip`, four synthetic pairs only. |

The CLI runs on CPU and does not invoke the production prediction mode. Local replay and refit require the already verified historical feature caches. The standard encoder weight cache and the existing SigLIP2 Python overlay are required for cold inference.

```powershell
$py = '.venv/Scripts/python.exe'
& $py scripts/run_competition_track1.py verify-history
& $py scripts/run_competition_track1.py cv

# Refit into a new destination only if regenerating the frozen challenger.
& $py scripts/run_competition_track1.py refit --output runs/competition_track1/reproduced_bundle

# After official release, supply the input CSV and the directory containing images/.
& $py scripts/run_competition_track1.py predict --input data/eval_st3/st3_test.csv --data-dir data/eval_st3 --output runs/competition_track1/january_predictions
& $py scripts/run_st3.py validate-prediction --input data/eval_st3/st3_test.csv runs/competition_track1/january_predictions/submission.zip
```

The official image/key release and separate owner promotion decision are the remaining steps before a real January submission. Use the locked bundle. Validate all pair keys against the official input before any manual upload. The synthetic ZIP is an inference and packaging check, with no competition score. [Production fallback runbook](../docs/competition_release.md#evaluation-day-runbook)
