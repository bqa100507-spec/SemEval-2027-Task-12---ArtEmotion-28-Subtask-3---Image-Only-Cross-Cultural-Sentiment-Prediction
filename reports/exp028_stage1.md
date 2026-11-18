# EXP-028 Stage 1 — frozen CLIP–Qwen fusion

**EXP028_STAGE1 = BLOCKED_VERIFICATION.** The CPU implementation and synthetic
comparison are complete. The accessible original Qwen cache page has only
5,257 of the 13,600 required target-language states. The Drive connector
returned its 1,000-item limit without a continuation token, and the alternate
browser session required sign-in. The exact missing count for the entire Drive
folder remains unknown. The real fold-3 entry point stopped at the cache gate,
before indexing vote counts or fitting a head. There are no scientific scores.

## Cache evidence

The audited source is `MyDrive/exp026/features/8f1e57257da52d7212b0/`.
All 999 NPZ payloads listed on the available page were downloaded and audited.
The 500 state files supplied 5,257 eligible language states across 500 images
and 500 neutral states. Neutral states are excluded from the comparison.
The audit verified the original cache identity, the pinned extractor source SHA,
the pretrained Qwen revision, the original prompt fingerprints, unique expected
image/language keys, the NPZ array receipts, dtype/shape and finite values.
It verified all 1,335 original artwork bytes against the P2 image map and checked
the three frozen visual cache file hashes and complete coverage.

| Quantity | Verified Drive page / local copy | Entire Drive folder |
| --- | ---: | --- |
| Vision arrays on unique P2 image SHA keys |499|≥499 verified|
| Target-language decoder states |5,257|≥5,257 verified|
| P2 images without a verified vision array |836|0–836|
| P2 pairs without a verified language state |8,343|0–8,343|
| Corrupt/incompatible entries |0 of999 payloads|Unlisted entries unaudited|
| NPZ payload bytes read and hashed |542,461,320|Unverified|

The source route for historical states is the existing pinned frozen extractor
with adapters disabled. The legacy array seals bind bytes to that extractor
identity and include checkpoint asset pins. They do not independently attest
each GPU forward. The local pretrained checkpoint and processor asset bytes
also passed the original EXP-026 asset verifier on CPU in 8.62 seconds.
No TASK-031 logits, fitted QLoRA states or neutral vectors were used as predictors.

The [cache audit summary](../docs/exp028_stage1_cache_receipt.json) preserves the
byte digest of the complete page receipt and both missing-key set digests.
The private receipt at `runs/task033/verified_drive_page/cache_audit.json`
contains all 999 per-file SHA-256 receipts. `missing_pairs.json` lists the exact
local missing keys. These counts and hashes describe the audited copy.

## Frozen screen

The [fold-3 manifest](../configs/exp028_fold3_manifest.json) binds painting
identities, pair keys and original pixel digests. It records 961 inner fitting
paintings / 9,811 pairs, 107 stopping paintings / 1,068 pairs, 1,068 refit
paintings / 10,879 pairs, and 267 held-out paintings / 2,721 pairs. Both labels
and fitted transforms stay out of the held-out fitting path.

The eight LR arms use the original Model C interaction matrix, exact integer
vote weights, the frozen 0.1/0.3/1.0 C grid, L-BFGS and max_iter 3000. The
implementation fits unwhitened PCA on fitting data only, uses painting-equal
weights on Qwen rows, fixes eigenvector signs by their largest-magnitude
coordinate, and shuffles Qwen features within each language and each partition.
C512 and its two 512-dimension unimodal controls have 13,338 fitted scalars
with 25 languages. All arms select C by inner vote-weighted BCE in nats and
refit on the same full outer training paintings.

The matched reference refits the three original Model C heads with C=0.3 on
each applicable population, with original nested six-decimal 50/50 blends.
The official published-specification Contrast and Macro Log Loss scorer runs
on identical held-out keys once per arm. The report includes per-language
details, paired painting wins/losses and 10,000 painting bootstrap draws.
Real historical six-decimal E50+L parity is a runtime gate before candidate
fitting. Synthetic exact parity passed. Fold 3 is reused exploratory evidence.

## Verification and performance

- The CPU fold entry point exited 2 after verifying the cache and refused a fit.
  The complete local audit took 9.58 seconds.
- A 75-painting synthetic screen completed all 38 LR fits in 1.13 seconds.
  The largest fit took 12 solver iterations. The maximum difference between the
  binary-expanded and count-compressed BCE was 2.22e-16 nats. The maximum RSS
  sampled at a fit boundary was 154,259,456 bytes. These measurements describe
  the tiny fixture. Production-size runtime and peak RSS remain unmeasured.
- Local available RAM was approximately 0.84 GiB at benchmarking. The full
  screen has a 2 GiB free-RAM entry check and a per-fit sparse workspace
  estimate checked against available memory and a 6 GiB process budget.
- CPU tests exercised cache and gate failures, exact keys, deterministic PCA,
  vote weighting, convergence, rounding, capacity and bootstrap.

## Complete the cache audit or fill the actual gap

[EXP028_Cache_Colab.ipynb](../notebooks/EXP028_Cache_Colab.ipynb) defaults
`AUTHORIZE_EXPERIMENT=False`. Run on a CPU runtime to audit the full mounted
original Drive folder with the pinned `exp023_handoff.zip`. This creates a
complete disk enumeration and verified receipt, including missing-key hashes.
It will also audit a compatible partial extraction if one already exists.

If the complete audit shows missing states, a separately authorized L4 run
can set `AUTHORIZE_EXPERIMENT=True` to compute only missing keys, using the
pretrained vision and decoder route, exact original prompts, unchanged runtime
lock, disabled adapters and a parity check. Complete painting boundaries persist
in the source-specific new Drive directory. The notebook exports a verified
diagnostic ZIP and requests release. The pinned worker was exercised on CPU with
the original input ZIP and available payloads. Actual Colab GPU behavior awaits
an authorized run.

Using only the verified page as input, the upper bound is 835 vision forwards
needed for missing decoder work and 8,343 decoder sequences in 1,440 batches
of up to eight, plus two vision forwards and two sequences for parity. The
full mounted audit can reduce this bound to zero. New decoded vector storage
would be about65.2 MiB plus receipts. New vision tokens at 128–256 per image
would take about0.41–0.82 GiB. At the proposal's 2–8 sequences/sec planning rate,
decoder time is about 17–70 minutes plus model loading, vision, Drive I/O and
read-back. There is no new measured GPU throughput. The upper allocation limit
is 2.5 hours including setup, verification and export.

From the checkout, after copying the audited original and any new compatible
additional cache to the CPU workspace, use a new run directory each time:

```text
python scripts/run_exp028_stage1.py --stage audit --inputs <handoff> --workspace <project-root> --qwen-cache <original-cache> --run-dir <new-audit>
python scripts/run_exp028_stage1.py --stage screen --inputs <handoff> --workspace <project-root> --qwen-cache <original-cache> --run-dir <new-fold3-run>
```

Add a second `--qwen-cache <additional-cache>` when using the new shard. Enable
the CPU screen after independent full coverage and the complete test suite.
Use the existing Python environment with `psutil==7.2.2`. A failed parity,
convergence, integrity or resource check blocks publication. Stage 2 remains
outside this implementation. E50+L remains production.
