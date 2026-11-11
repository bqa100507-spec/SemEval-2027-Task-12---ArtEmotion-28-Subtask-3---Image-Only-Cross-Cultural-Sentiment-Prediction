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
.venv/Scripts/python.exe scripts/run_st3.py caption-extract --teacher clip
.venv/Scripts/python.exe scripts/run_st3.py caption-extract --teacher e5
.venv/Scripts/python.exe scripts/run_st3.py caption-cv
.venv/Scripts/python.exe scripts/run_st3.py caption-holdout
.venv/Scripts/python.exe scripts/run_st3.py caption-dev
```

All stages share `runs/exp017_text_supervision/`. Use a shared fresh `--run-dir` to repeat the study. Completed CV folds are checkpointed under `cv_work/`. If confirmation passes, run `caption-extract --teacher clip --full-training` or its selected E5 equivalent before `caption-dev`. Progressive text caches are separate from historical image caches.
