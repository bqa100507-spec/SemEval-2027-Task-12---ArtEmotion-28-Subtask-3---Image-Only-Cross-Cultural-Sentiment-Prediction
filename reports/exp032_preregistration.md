# EXP-032 · Regularized multitask cultural interaction

**Question.** Can learned relationships between language responses improve full-rank image–language slopes beyond ordinary ridge with the same head, feature caches, and overall penalty scale? E50+L remains the frozen benchmark. The five P2 painting folds have been used before, so any gain is exploratory.

This recipe implements direction A approved from commit **01bf9bd93b3c7417eb4a76413f3dd15f041a155a**. The research branch starts from the released E50+L ancestry. [EXP-032 strategy](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/01bf9bd93b3c7417eb4a76413f3dd15f041a155a/reports/exp032_research_strategy.md) · [History audit](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/01bf9bd93b3c7417eb4a76413f3dd15f041a155a/reports/research_history_audit.md) · [EXP-031 closure](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/blob/01bf9bd93b3c7417eb4a76413f3dd15f041a155a/reports/exp031_final_report.md) · [Frozen config](../configs/exp032.json)

## Three systems

1. Frozen matched E50+L OOF probabilities.
2. The new parameterization with isotropic deviation ridge.
3. The same parameterization and ridge budget, with a learned graph allocating that budget across directions.

Each of the last two fits a binary head per cached CLIP-B, SigLIP2-Base and CLIP-L block. Logits are global intercept + language intercept + shared image slope + image times language-specific slope deviation. Language intercepts and deviations each sum to zero over fitting languages. An orthonormal Helmert basis enforces the constraints and preserves Euclidean penalty scale without imposing a low-rank bottleneck. Unseen languages use the shared score. The loss is binomial negative log likelihood in nats divided by the fitting vote count. Heads use coefficients for regularized intercepts and shared slope of 1/(0.3 × fitting votes), then a selected multiplier on deviation ridge. The three component probabilities use the existing nested six-decimal blend, with final weights 0.25, 0.25 and 0.5.

### Why a uniform graph has no effect

For deviation matrix D (languages by feature coordinates), the ordinary penalty is (λ/2) tr(DᵀD). Let P center vectors within the supported languages and act as zero on isolates. Let L be the unnormalized symmetric adjacency Laplacian scaled to trace(P), the supported count minus one. The learned system uses (λ/2) tr[Dᵀ(I + ρ(L−P))D], with ρ ∈ {0.25, 0.50}. Equivalently it adds graph smoothing and subtracts exactly the same isotropic budget on that subspace. A uniform complete graph over the supported languages gives L=P and returns the control exactly. The two penalties have equal trace on the global sum-zero subspace, and their eigenvalues remain at least (1−ρ)λ. Isolates do not enter P or L and retain ordinary ridge. This controls average regularization strength, while preserving differences in direction-specific regularization. Report realized energies, eigenvalues, and coefficient norms, since equal trace does not imply the fitted solutions have equal norms.

## Learn and verify the graph on fitting data

- The three unit visual blocks are concatenated and divided by √3 for an additive binomial nuisance head with shared image slope and language intercept. Three deterministic SHA-256 painting splits give out-of-fold logits inside each fitting scope. Residuals subtract these logits from logit[(k+0.5)/(n+1)]. Within a painting, the shared term cancels in residual differences, but its forecast is retained for the case-mix diagnostic.
- For each co-observed language pair, estimate unbiased, weighted variance of the centered residual difference. Weights use the harmonic mean of the two vote counts, capped at 10. Require Kish effective overlap ≥30, then set confidence to effective overlap divided by effective overlap +50. Multiply confidence by exp(−variance / median supported variance). Unsupported edges have zero weight. Normalize the resulting Laplacian by trace as above.
- For each outer fitting population, assign two disjoint, balanced SHA-256 halves. Recompute the nuisance cross-fit and graph independently inside each half. Require ≥15 commonly supported edges, ≥6 common languages, and Spearman ≥0.25 between their dispersion-based similarities (before overlap weighting). In both directions, the top third of similarity edges selected on one half must have mean paired association ≥0.03 on the other. Association is twice the weighted covariance divided by the sum of both weighted variances after centering within four shared-forecast quantiles on the co-observed paintings.
- Also require normalized Frobenius distances from the uniform-supported and overlap-only penalties of at least 0.10 and 0.08, respectively. The overlap-only diagnostic preserves the graph's eligible edges and confidence weights but drops residual similarities. Report overlap and association rank stability too. These checks provide evidence beyond raw co-occurrence. Co-observed painting selection, annotator noise, and case mix can still explain part of an association. Edge weights describe responses on this sample and carry no causal or population-cultural interpretation.

Compute and preserve the five fitting-only diagnostics before outer scoring. If any scope fails, record NO_GO_GRAPH_STABILITY. Skip outer fitting and scoring in that case. The exact held-out keys are known for verification, but their vote labels never enter their fold's graphs, heads or selection.

## Selection and decision

Use the same two complementary inner halves for both conditions. Pool predictions across the two inner validation halves. Among deviation ridge ratios 0.5, 1 and 2, pick highest Contrast within 0.005 bits of the best inner Macro Log Loss, then lower loss, then smaller ratio. Fix that same ratio for both outer systems. Use the same rule for coupling fractions, with the selected no-graph inner loss as guardrail baseline. If both fractions miss that guardrail, report it and choose the one with lower inner loss (ties use higher Contrast then smaller fraction). Refit on all eligible outer fitting paintings. Convergence is mandatory. Seal both models per fold and verify saved predictions before pooling and scoring.

Exploratory GO requires all of the following **against both** matched comparators. Pooled Contrast gain ≥0.001, a positive lower endpoint of the 95% paired painting percentile interval (10,000 bootstrap samples), positive Contrast in at least three of five folds, and pooled Macro Log Loss regression ≤0.005 bits. Otherwise record NO_GO_OOF. The result cannot promote a model. Bootstrap intervals condition on this P2 population and do not correct for earlier model selection on P2.

## Reproduce on CPU

Use Python 3.12 and the numerical pins in [requirements.txt](../requirements.txt). Point to the already extracted EXP-023 handoff. The only opened paths are the six checksum-pinned inputs in the config, and caches are subset to P2 before any fit. The protocol file has confirmation identifiers for isolation checks and contains no confirmation votes. The historical role string “development” in this handoff means the P2 cross-validation pool.

    python scripts/run_exp032.py --inputs /path/to/exp023_unpacked --run-dir runs/exp032/p2

The same run directory reuses checksum-sealed fitting graphs and complete outer fits when code, config, inputs and runtime agree. The CPU CLI is the experiment entry point. It emits per-fold graph and convergence receipts, a decision, and, when the gate allows outer scoring, coefficients, CSVs and metrics. Graph-stage NO-GO keeps the diagnostics and decision. No images or features are written to Git.
