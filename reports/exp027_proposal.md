# EXP-027 — proposal only: within-painting VLM supervision

**PROPOSED_ONLY. No implementation, GPU authorization, training, new checkpoint or scientific result. Production remains E50+L.** EXP-026 stays EXPLORATORY_COMPLETED; its weights, notebooks and outputs are preserved.

Hypothesis: supplement vote-weighted BCE with a within-painting term to improve image-specific differences across target languages, using the same pinned Qwen backbone, decoder LoRA and sentiment head. The owner's label-free profile analysis suggests relatively stable language offsets; this is a description of predictions, not proof of the failure mechanism. The audited L0 fold1 result0.340691 remains below E50+L0.433429, with different fitting populations. Higher between-painting prediction variance is not itself a Contrast error: within-painting Pearson is invariant to a shared additive shift and positive scaling.

Any future authorized design should:

- Freeze one explicit within-painting objective, normalization, numerical safeguards and lambda-selection plan before reading new held-out outcomes. No objective or lambda is selected in this proposal. Include a genuinely matched BCE-only control with identical data, initialization, seed, optimizer, update budget and checkpoint selection.
- Develop/select only on the chosen outer-training/inner-validation population. Do not reuse examined EXP-026 fold1 scores or its per-language/per-painting errors to tune the new objective. Choose the next held-out fold in advance; P2 folds2–5 are new only relative to this VLM study, not independent confirmation after repeated project-wide CV reuse.
- Initialize from the pinned pretrained model with fresh zero-output LoRA, not an EXP-026 fitted checkpoint: that checkpoint trained on many paintings from folds2–5 and would leak into a newly held-out fold. Keep all future folds painting-disjoint.
- Explicitly handle insufficient eligible languages, constant observed proportions and near-constant predictions. Keep scorer eligibility separate from the differentiable training surrogate. Retain BCE and report Macro Log Loss; do not force lower between-painting variance as a proxy for Contrast.
- Review [EXP-004](exp004_contrast_aware.md), which already tested pairwise/Pearson auxiliary objectives on frozen CLIP linear interactions, with small gains below its promotion gate. Those different architecture/population results do not validate a VLM loss or lambda.

This is a direction for a new owner task, not a command to execute. No five-fold run, confirmation access, Development scoring, submission, blending or production replacement is authorized.
