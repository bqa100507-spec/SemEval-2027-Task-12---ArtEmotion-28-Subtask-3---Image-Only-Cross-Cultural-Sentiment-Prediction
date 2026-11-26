# DINOv2-L language-conditioned region selection

**Decision: retain Corrected Ridge.** Target-language queries changed which regions received attention and lowered training BCE in all five folds. On held-out paintings they reduced Contrast relative to shared queries in every fold. A fixed 20% blend also lowered pooled Contrast and increased Macro Log Loss. The candidate did not advance to a full-training refit or Development inference.

## Matched setup

All arms use the same frozen DINOv2-L checkpoint (`47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`), official 224-pixel center-crop preprocessing and 256 final-layer patch tokens per artwork. The shared cache holds L2-normalized tokens in FP16. Each fold fits its 64-dimensional PCA on fitting paintings only and uses that same projection in all three arms. A pools patches uniformly. B learns a shared 16-dimensional query. C adds a language-dependent query derived from the prediction head's existing language coefficients. B and C each have 1,706 trainable parameters. A has 1,690. The prior initialization, language-conditioned prediction head, count-weighted binary cross-entropy, painting batches and fixed 24-epoch schedule are shared. A uses mean patch features, so its result is a matched pooling control with a different representation from the previous DINOv2-L CLS-token result.

The original P2 folds cover 1,335 paintings and 13,600 pairs, with 1,269 Contrast-eligible paintings. Test-like P2 removes English, Arabic and Chinese, then recomputes eligibility on 11,643 pairs and 1,252 Contrast-eligible paintings. Both populations reuse P2 and are exploratory. Corrected Ridge uses its sealed out-of-fold predictions. The 80/20 blend weight was fixed before scoring, with no fold-specific fitting. Only P2 training images and official training votes were accessed for this comparison.

## Pooled out-of-fold results

| System | Full Contrast | Full Log Loss | Test-like Contrast | Test-like Log Loss |
| --- | ---: | ---: | ---: | ---: |
| Corrected Ridge | 0.454134 | 0.708500 | **0.463108** | **0.717479** |
| A · Global pooling | 0.410345 | 0.798307 | 0.420283 | 0.810654 |
| B · Shared query | 0.410428 | 0.798845 | 0.420221 | 0.811362 |
| C · Language query | 0.379261 | 0.804311 | 0.389363 | 0.817626 |
| Corrected Ridge + 20% C | 0.451558 | 0.716120 | 0.461530 | 0.725593 |

Log Loss is macro-averaged in bits. Corrected Ridge’s hosted Development reference of 0.4655 is on a different population.

| Fold | Full C−B | Full C−Ridge | Full blend−Ridge | Test-like C−B | Test-like C−Ridge | Test-like blend−Ridge |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | −0.027744 | −0.077311 | −0.003669 | −0.031242 | −0.077301 | −0.002914 |
| 2 | −0.035581 | −0.069465 | −0.003097 | −0.035575 | −0.074681 | −0.005272 |
| 3 | −0.037385 | −0.070755 | −0.000761 | −0.031480 | −0.064134 | +0.001819 |
| 4 | −0.037363 | −0.079872 | −0.001494 | −0.034957 | −0.071386 | +0.002170 |
| 5 | −0.017298 | −0.076981 | −0.003901 | −0.020660 | −0.081500 | −0.003736 |

| Comparison | Population | Contrast Δ | Painting-paired 95% CI | After removing largest 5% absolute differences |
| --- | --- | ---: | --- | ---: |
| C−B | Full | −0.031167 | [−0.040129, −0.022254] | −0.023073 |
| C−B | Test-like | −0.030858 | [−0.040738, −0.020747] | −0.023220 |
| C−Ridge | Full | −0.074873 | [−0.086503, −0.063128] | −0.059706 |
| C−Ridge | Test-like | −0.073745 | [−0.086551, −0.060642] | −0.057317 |
| Blend−Ridge | Full | −0.002575 | [−0.005322, +0.000231] | −0.002041 |
| Blend−Ridge | Test-like | −0.001578 | [−0.004655, +0.001534] | −0.001533 |

The blend won two test-like folds and increased pooled test-like Log Loss by 0.008115 bits, exceeding the 0.005-bit guardrail. Dropping its top 1% of test-like painting wins brought its Contrast delta to −0.004112. A and B differed by only 0.000062 in test-like Contrast. This comparison gives little evidence that learning a shared pooling query helps on this token representation.

## Attention and EXP-033

For C, the mean within-artwork total variation distance between language maps was 0.4533. The highest-weight patch changed in 93.6% of the language-pair comparisons. B uses the same map for each language of an artwork. In the audit example from fold 1, El Greco’s *The Virgin of the Immaculate Conception*, the Hindi and Igbo maps had a total variation distance of 0.534. The saved maps show weights over contextual tokens. On the test-like eligible paintings, the correlation of per-painting map divergence with C’s Contrast change from B was −0.062. Test-like C−B was negative even in the lowest map-divergence quartile (−0.0150) and in the highest (−0.0406). This pattern and the lower fitting BCE are consistent with fitting language-specific routing that transfers poorly. The comparison cannot distinguish sparse supervision from limitations of the tokens or the query parameterization as the cause.

EXP-033 pooled spatial SigLIP2 features using learned content queries and gave each target language the same pooled representation, then computed language deviations in the readout. It also tested selective vision adapters. This comparison kept the encoder frozen and moved the target-language signal into region selection itself, with an equal-parameter shared-query control.

## Runtime and reproduction

The first 24 official images took 2.81 seconds to extract after model load. The full 1,335-image bank took 87.19 seconds after load (15.31 artworks/s) on an RTX 4060 Laptop GPU, at 1.34 GiB peak CUDA reserved. Model load took 4.48 seconds. The bank occupies 667.5 MiB and was reused on a second invocation with its checksum verified. All readout fits ran on CPU. The complete CV run took 144.25 seconds at 1.68 GiB peak RSS. B averaged 5.55 seconds of fitting per fold and C averaged 9.63. Cached C readout inference after PCA took 25–44 ms per held-out fold, excluding token extraction and projection. An unseen artwork would incur one encoder pass plus projection, then language-specific readout.

```powershell
.\.venv\Scripts\python.exe scripts/run_competition_region_selection.py extract --device cuda
.\.venv\Scripts\python.exe scripts/run_competition_region_selection.py cv --device cuda
.\.venv\Scripts\python.exe -m pytest -q
```

For `cv`, `--device` identifies the cache to consume. Fitting uses CPU. The runner defaults to the existing pinned local checkpoint, P2 handoff and Corrected Ridge reference. Use a fresh `--run-dir` to rerun a sealed result, with a copy of the verified patch cache under its `patches/` directory.

Source: `9f1de82bc62f919469b7b4893071000d63d94c9c`. Local artifacts are under `runs/competition_region_selection/`. `patches/manifest.json` seals the token bank and input provenance. `results/seal.json` seals the OOF predictions, exact fold metrics, attention audit and example maps. OOF SHA-256 is `11339b66e1ef41b2fbb04f2abb1f21891896ac4a2f49a1e0d027cb2893f2fada`. Metrics SHA-256 is `04baad2fc68380ba551614f01d1b6a33ffbf041dd0c768fc5b7938aaa47f70e7`. Independent readback verified exact scores, incumbent rows and protected production hashes. The CPU suite finished with 321 passed and 3 skipped.
