# Evidence-grounded cultural concepts · CPU pilot

**Decision — retain Corrected Ridge and stop this concept approach.** Matched language associations improved the standalone concept predictor in all five folds. The fixed concept blend with Corrected Ridge lost in all five folds and increased Macro Log Loss. The pilot identifies a language-specific concept signal and fails to show incremental competition value for the registered combination.

## Evidence and bounded design

ArtECulture's Table 4 reports **55.62%** accuracy for matched knowledge, **45.51%** for internal reasoning, and **50.07%** for cross-culture knowledge with Qwen3.6-27B. Its benchmark uses dominant eight-way emotion labels and explanations across English, Chinese and Arabic. ST3 uses distributions of binary votes and within-painting Cultural Contrast. ArtELingo-28 also reports stronger transfer between culturally related languages in its captioning setup. These results motivate testing the mechanism under ST3's different supervision and scoring. [ArtECulture paper](https://arxiv.org/html/2608.03358#S5) · [ArtELingo-28 paper](https://aclanthology.org/2024.emnlp-main.1165/)

- **A** is the sealed Corrected Ridge OOF reference. It already adds a local visual-neighborhood residual correction to Fixed Ridge. Reporting Fixed Ridge separately isolates the contribution of the existing neighborhood branch.
- **B** is a language-independent binomial model with an intercept and observable concept features. **C** uses shared concept effects, a regularized language intercept, and regularized concept-by-language deviations. C's static control suppresses its deviations and keeps its shared effects and language intercept. The wrong-language control permutes supported deviations within each concept, pairing languages in cyclic vote-support order. It preserves the target-language prior, exact support mask, and input image concepts. The wrong language can vary across concepts.
- The predeclared vocabulary has 43 observable categories. It reads entities, scene and visual cues from existing pixel-only Qwen descriptions. It excludes atmosphere, speculative-context fields, negated or hedged fragments, named identities and unspecified associations such as protest. Each fold retains concepts seen in at least 18 fitting paintings. Each interaction needs at least 12 independent fitting paintings and 100 votes, with at least two supported languages per concept. FP64 binomial fitting uses the official positive and total vote counts with ridge penalties 8, 16 and 32 for shared concept, language and interaction terms respectively. Unsupported deviations remain zero. The associations describe observations in the released training population.
- Only outer-training paintings determine vocabulary support, language priors and coefficients. Held-out inputs are restricted to artwork-derived concepts and target-language keys. Corrected Ridge and Fixed Ridge probabilities enter only as sealed held-out references, after the concept fits. The fixed derived comparisons use 75% Corrected Ridge and 25% of the corresponding concept probability. They require no learned fusion, reused training residuals or extra fit. B alone has zero Contrast by construction, and its probability blend with A preserves A's Contrast to floating-point precision. B still tests global sentiment calibration.

## Matched P2 results

| System | Full Contrast | Full Macro Log Loss | Test-like Contrast | Test-like Macro Log Loss |
| --- | ---: | ---: | ---: | ---: |
| Fixed Ridge | 0.451702 | 0.709247 | 0.460043 | 0.718425 |
| A. Corrected Ridge, including neighborhood correction | **0.454134** | **0.708500** | **0.463108** | **0.717479** |
| B. Culture-independent concepts | 0.000000 | 0.891488 | 0.000000 | 0.905837 |
| C. Matched culture-conditioned concepts | 0.366880 | 0.867002 | 0.369905 | 0.883452 |
| C with static priors only | 0.332623 | 0.871665 | 0.332767 | 0.888629 |
| C with wrong-language associations | 0.296960 | 0.876450 | 0.298822 | 0.893398 |
| A + B | 0.454134 | 0.730108 | 0.463108 | 0.739674 |
| A + C, static prior control | 0.446799 | 0.727968 | 0.455024 | 0.738161 |
| A + C, matched | 0.448467 | 0.727528 | 0.457043 | 0.737721 |
| A + C, wrong language | 0.444972 | 0.728325 | 0.453198 | 0.738500 |

The full and test-like populations contain 13,600 and 11,643 pairs and 1,269 and 1,252 Contrast-eligible paintings. Test-like excludes English, Arabic and Chinese. Log Loss is in bits. The 10,000-draw paired bootstrap resamples paintings within the reused P2 evaluation population.

| Contrast comparison | Full Δ | Test-like Δ and 95% paired CI | Test-like fold wins | Test-like Δ after 5% absolute-difference trim |
| --- | ---: | ---: | ---: | ---: |
| A − Fixed Ridge | +0.002432 | +0.003065 [+0.000530, +0.005567] | 4/5 | +0.001799 |
| C − wrong-language C | +0.069920 | +0.071084 [+0.059432, +0.082359] | 5/5 | +0.049914 |
| C − static-prior C | +0.034257 | +0.037138 [+0.029063, +0.045046] | 5/5 | +0.025908 |
| A+C − A+wrong C | +0.003495 | +0.003845 [+0.001204, +0.006492] | 5/5 | +0.002012 |
| A+C − A+static C | +0.001668 | +0.002019 [+0.000220, +0.003812] | 4/5 | +0.001109 |
| A+C − A | −0.005667 | −0.006065 [−0.009423, −0.002746] | 0/5 | −0.004536 |
| A+C − Fixed Ridge | −0.003235 | −0.003000 [−0.005960, −0.000016] | 0/5 | −0.002585 |

The test-like fold deltas for A+C versus A were **−0.008427, −0.005097, −0.005473, −0.003517, and −0.007909**. Removing the strongest 1% of winning paintings moved the pooled test-like delta to −0.008537. The blend also added 0.019028 full-P2 and 0.020242 test-like Log Loss bits. It missed the preregistered +0.002 test-like Contrast gain, fold consistency, paired and outlier checks, and +0.005-bit Log Loss guardrail. The language comparison establishes a distinct signal within the concept model. The stable losses to both Ridge references leave no evidence of a useful deployable increment for this fixed concept blend. These results apply to this recipe and population.

## Grounding and eligibility audit

The lexicon matched 1,191 of 1,335 paintings. After fitting-only filtering, 29–30 concepts and 353–365 concept-language cells survived per fold. Some retained interaction evidence was available for 11,591 of 13,600 held pairs. Three descriptions rejected by the source schema contributed empty concepts. The audit used 12 deterministically selected pixel files with matching organizer image hashes. Of 20 retained concept mentions, 19 appeared visible and one indoor setting was indeterminate. Coverage misses included a visible construction scene and two readers. The Rubens description's identification of Madonna was excluded while the visible child was retained. This small review shows the distinction between selecting observable categories and independently establishing their visual presence. The manual review did not alter the frozen model or outputs.

The pilot used only released ST3 training votes for concept associations. It imported no ArtECulture knowledge tuples, judgments, annotations, images or rationales and no ArtELingo external training files. ArtECulture partly reuses ArtELingo and adds annotations, creating a potential cross-benchmark overlap risk if it were imported. The archived Qwen prompt has pixels only and receives no painting identifier, native caption, target language or vote information. Qwen's Apache-2.0 checkpoint license applies to the model. Public materials leave pretraining overlap and eligibility for generated intermediate descriptions unresolved. The official task forbids obtaining external annotations for evaluation paintings and points to the competition platform for LLM guidance. This run opened no Development, evaluation or confirmation votes. [ArtECulture data construction](https://arxiv.org/html/2608.03358) · [Pinned Qwen model](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/ebb281ec70b05090aa6165b016eac8ec08e71b17) · [ST3 rules](https://github.com/artemotion-28/artemotion-28/blob/cf7cc444cf56617f3b8417497c6930ad172c5c99/README.md) · [Organizer FAQ](https://github.com/artemotion-28/artemotion-28#faqs)

## Execution and artifacts

Run `python scripts/run_competition_cultural_concepts.py` from the project environment. The one frozen CPU run used **62.29 seconds of wall time** and **165.1 MiB peak RSS**, with ten converged fits. The clean suite passed 317 tests with three skips. Source and config were committed before the run at `7ad4fa7b7535c296883088258d00176928b05183`. The sealed OOF reproduced pooled scores exactly and preserved both Ridge references row for row. OOF SHA-256 is `c5f4d3bbeaab3410c9861afb9a4b7bb2c367c461530470f01031db3932405c75`. Metrics SHA-256 is `299262b97758659ab306f98001b305d4c0bbe5035e0a2d4dadaf116dc9b000f3`.

Results and seals are under `runs/competition_cultural_concepts/results/`. The adjacent `grounding_review.json`, `grounding_sheet.png`, and `verification.json` preserve the manual audit and replay. Production files and the Corrected Ridge archive match their previous hashes. No deployment candidate was selected.
