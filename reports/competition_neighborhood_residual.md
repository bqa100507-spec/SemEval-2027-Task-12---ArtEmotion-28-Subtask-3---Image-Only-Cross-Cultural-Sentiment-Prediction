# ST3 — Bounded image-neighborhood residual correction

**Decision: ADD CORRECTION.** The fixed formula crossed the predeclared P2 threshold. It improved filtered Contrast by **+.003064948**, with a paired 95% interval of **[+.000530183, +.005567321]**, and lowered Macro Log Loss by **.000946075 bits**. This is a recommendation from the bounded exploratory comparison. Fixed ridge remains the unchanged deployable challenger and E50+L remains production.

Each score cell is **Cultural Contrast / Macro Log Loss in bits**.

| Outer fold | Full ridge | Full corrected | Filtered ridge | Filtered corrected |
| --- | ---: | ---: | ---: | ---: |
| 1 | .438684 / .665183 | .443922 / .664179 | .445910 / .678168 | .450869 / .677251 |
| 2 | .463904 / .724336 | .459778 / .724501 | .468017 / .733469 | .466237 / .733380 |
| 3 | .435016 / .711310 | .437806 / .710339 | .439185 / .722344 | .441599 / .721110 |
| 4 | .467131 / .736788 | .469969 / .735697 | .482765 / .744125 | .485596 / .742602 |
| 5 | .453135 / .711884 | .458720 / .710922 | .464207 / .717193 | .471317 / .716054 |
| **Pooled** | **.451702 / .709247** | **.454134 / .708500** | **.460043 / .718425** | **.463108 / .717479** |

| Corrected − ridge | Δ Contrast [paired 95% interval] | Δ Macro LL | Fold wins | Painting wins / losses | After 5% absolute trim | After removing top 1% of wins |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full P2 | +.002432 [+.000067, +.004745] | −.000746 | 4/5 | 648 / 621 | +.001528 | +.000830 |
| Filtered P2 | +.003065 [+.000530, +.005567] | −.000946 | 4/5 | 640 / 612 | +.001799 | +.001120 |

There were no painting ties in either scope. Filtered leave-one-fold-out gains range from **+.002086 to +.004304**. The gain survives dropping any single fold and the top painting wins. It falls below +.002 after either painting trim, and the filtered median paired gain is only **+.000600**. The correction is useful across folds, with a substantial part of its measured advantage coming from larger painting-level wins. The paired intervals describe these observed paintings and do not account for adaptive use of P2.

The completed run took **167.56 s CPU wall time** with two numerical-library threads and **205.12 MiB** peak RSS. The correction itself took **0.62 s** summed across 13,600 OOF pairs with image features and banks in memory. The measured elapsed time includes fitting the inner ridge heads and scoring. Clean-checkout CPU tests passed **288**, with **2 skipped**. Independent re-scoring of the saved OOF matched both pooled Contrast and Log Loss to `1e-12` on both scopes. The OOF CSV SHA-256 is `fc8273e3a05438f2058d41d47e5743102afd6b450cc86beb6c2b049bde3095fd`.

## Duplication audit and frozen method

| Earlier experiment | Overlap and distinction |
| --- | --- |
| EXP-012 | Corrected E50+L logits using encoder disagreement and global/language coefficients. It had no image-neighborhood bank. |
| EXP-013 | Fit a nonlinear parametric image–language head and averaged its probabilities with E50+L. It did not estimate local errors of the incumbent. |
| EXP-024 | Closest precedent. Its centered kNN predicted from neighbors' observed deviations around painting means and from language priors. The current test borrows the similarity geometry, neighborhood size and kernel temperature, then estimates only the leftover cultural error of cross-fitted fixed ridge and adds it to the unchanged ridge probability. |
| EXP-028 | Fused Qwen states and visual embeddings under a matched linear readout. Its experiments change feature inputs or mix different predictors. |
| EXP-029 and EXP-032 | EXP-029 learned a parametric Qwen residual around E50+L. EXP-032 pooled language residual structure to regularize global image slopes. Neither retrieved painting-specific ridge errors at prediction time. |

[EXP-012](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/c9a487747b8b340d5adb3cb17772e890fd400583/reports/exp012_language_residual.md) · [EXP-013](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/b0dfc75d27285eca12abf061b5bd7b0a5667544c/reports/exp013_nonlinear_head.md) · [EXP-024](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/428c90411578fe39416248f78337466b03d10e94/reports/exp024_visual_retrieval.md) · [EXP-028](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/077b1b7bb267c57d34ba1412cf4ee10ce968773a/reports/exp028_final_closure.md) · [EXP-029](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/3cb19bdaf34c8555bb08ad1b09cfc0605dc96dae/reports/exp029_recovery.md) · [EXP-032](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/4d4bb0f0debfc082ccc1e1fb6ce36e00ac1a5583/reports/exp032_multitask_cultural_interaction.md)

Each of the five outer fitting sets supplied its own bank. For a training painting-language pair, subtract the probability from the ridge model that excluded that painting within the outer fit. Center the resulting errors within that painting using vote counts. Paintings with fewer than two eligible languages supply no bank entries. For each outer-held artwork, retrieve the **64** most similar distinct training paintings with cosine similarity over equally weighted CLIP-L and SigLIP2 features. For the requested language, use `exp((cosine−1)/0.10)` times the number of votes as the neighbor weight. Sum weighted centered errors and divide by weighted vote mass plus **20 zero-residual pseudo-votes**. Add the result to ridge, clip to `[0,1]` and round to six decimals. No supported neighbors gives the original probability. The query uses image embeddings and target language, with no query labels or cross-language responses.

The 15 painting-grouped inner ridge fits reproduced Track 1's component prediction hashes in all five outer scopes. Each held-out painting was absent from the fitting, inner cross-fitting and neighbor bank of its outer fold. The fixed ridge outer OOF reference remained exact. The configuration was committed before scoring with one set of smoothing constants and no outer label-based selection. The advance gate required at least **+.002 filtered Contrast**, a positive 95% paired lower bound, at least three filtered fold wins, positive trimmed gain, no full-P2 Contrast regression and at most **+.005 bits** of Macro Log Loss regression in either scope. Filtering excludes English, Arabic and Chinese and recomputes official score eligibility. Both scopes use the same reused P2 artwork and votes. [Frozen recipe](../configs/competition_neighborhood_residual.json) · [Organizer evaluation](https://github.com/artemotion-28/artemotion-28#evaluation)

## Reproduction and preservation

Run `.venv/Scripts/python.exe scripts/run_competition_neighborhood.py` with the existing P2 and Track 1 caches. Completed output is protected against overwrite. The ignored run directory `runs/competition_neighborhood_residual/results/` holds `metrics.json`, `oof.csv`, `painting_deltas.csv` and their checksum seal. The first attempt completed the folds but hit a path-type error while hashing the source file before artifact publication. The retry changed only provenance handling and reran the identical recipe.

The E50+L release bundles and frozen 849-pair predictions, the Track 1 ridge bundle and OOF digests, and the DINOv3 branch at `1ca61aa396c07bda24d1c623f55cdfe4cc84b933` were verified unchanged. No model deployment, encoder extraction, GPU run, merge, promotion, or additional search was performed.
