# EXP-039 final closure — scaling the VLM-embedding axis

**Closed: NO_GO.** No D candidate passed the frozen P2 gate against the on-platform C2. The best ZIP is the byte-identical EXP-038 incumbent `B2_C2_cscva3` (`62093a31…`, owner-reported hosted 0.4737). **The incumbent is unchanged.** The owner decided on 2026-11-28 to close EXP-039 without implementing patch 2 (`runs/exp039/claude_code_patch2_prompt.md`, **not executed**). [Runbook](../docs/exp039_kaggle.md) · [Config](../configs/exp039.json)

Every number below was recomputed on CPU by `scripts/audit_exp039.py` from the archived predictions. There were **no mismatches**: the largest difference from the archived metrics was 1.1e-16, against an audit tolerance of 1e-9. The receipt is `runs/exp039/closure/audit.json`.

## Question

EXP-038's C2 gain came almost entirely from E3, Qwen3-VL-Embedding-2B with one neutral instruction. The question was whether a larger checkpoint (E5, 8B), a pool of neutral instructions (E6) or a higher input resolution (E7, 1024²) improves the image-level deviation estimate, when each is passed through the unchanged C2 recipe.

## Runs

| Run | Worker / overlay manifest | Results ZIP | Outcome |
| --- | --- | --- | --- |
| 1 | `471e400` / `6e2e53ad…` | `bde0d198…` (in `runs/exp039/run1/`) | TECHNICAL_PARTIAL: E5 loader failure. Superseded. |
| **2 (official)** | `8f99b9f` / `1ebcfca8…` | `7aa681d59f6d2109150d573188de413f336cb146be6b4951cc948b8563ce0192` (125,502,969 bytes) | COMPLETE, NO_GO; E5 failed again in Benchmark (device placement). Best ZIP `62093a31…` (8,958 bytes). Progress `6c7f896b…`. |

### Run 2 audit

- **Archive integrity:** all export members match the export manifest. All 6 stage seals match, with the run identity plus `e5_label: E5` and `gpu_count: 2`.
- **Runtime:** Tesla T4 ×2, Python 3.12.10.
- **Validate:** every check passed, including both snapshots and the key audit. The key audit found 2B with 625 checkpoint keys against 626 class keys, `lm_head` tied, nothing ignored. It found 8B with 749 against 750, untied, ignoring only `lm_head.weight`.
- **C2 replay against the EXP-038 run-4 values:** exact, on the same platform.
- **Gate reference:** the on-platform C2, as recorded in `p2/metrics.json`.
- **Recomposition:** all 8 B1/B2 compositions were rebuilt from their components and their members match byte for byte. B2(C2) recomputed on-platform equals the incumbent member byte for byte.
- **Development selection:** none.

### Reproducibility check

Run 2's D2, D3 and D4 gate numbers equal run 1's exactly (every field; deterministic CPU stages over identical E6/E7 features).

## P2 gate (test-like scope, vs on-platform C2 filtered 0.468648)

| Candidate | Filtered Contrast | Δ | 95% CI (20,000 draws, seed 203904) | Fold wins | 5% trim | Full Δ | Filtered LL Δ | Gate |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| D1 (E5, 8B) | — | — | — | — | — | — | — | NOT_AVAILABLE (E5 FAILED_OPTIONAL) |
| D2 (E6, instruction pool) | 0.468120 | −0.000528 | [−0.001711, +0.000690] | 2/5 | −0.000598 | −0.000547 | +0.000903 | FAIL |
| D3 (E7, 1024²) | 0.467953 | −0.000694 | [−0.001283, −0.000092] | 0/5 | −0.000481 | −0.000466 | −0.000267 | FAIL |
| D4 (VLM third: E3+E6+E7) | 0.469866 | +0.001218 | [−0.000528, +0.002973] | 3/5 | +0.001066 | +0.001157 | −0.003505 | FAIL |

### Standalone arms (P2 filtered)

| System | Contrast | Macro LL | Global RMSE |
| --- | ---: | ---: | ---: |
| Fixed Ridge | 0.460043 | 0.718425 | 0.141391 |
| E3 (2B, neutral) | 0.460446 | 0.709892 | 0.137733 |
| E6 (4-instruction pool) | 0.460733 | 0.711171 | 0.139727 |
| E7 (1024²) | 0.458203 | **0.707121** | **0.135121** |

Paired intervals against E3 (filtered): E6 +0.000286 [−0.005404, +0.006014], E7 −0.002244 [−0.004459, −0.000064].

**Conclusion.** For frozen Qwen3-VL-Embedding-2B, both instruction pooling and input resolution are saturated for Cultural Contrast. E6 matches E3. E7 improves global calibration (Log Loss and global RMSE) but slightly hurts the language deviation, and D3 has a negative interval. D4's pooled third has the best Log Loss and a positive but non-significant Contrast gain.

**E5/D1 is untested** because of two technical failures in the optional arm:
- Run 1: the loader was too strict about the untied `lm_head` (fixed by patch 1).
- Run 2: multi-GPU device placement in `last_token`: `hidden` was on `cuda:1` while the index was on `cuda:0` (not fixed).

The owner chose not to pursue 8B. Its expected value is low given that E6 and E7 saturated, and an 8B model split over two GPUs adds risk on Evaluation day.

## Development (reporting only; 82 full / 79 Evaluation-like paintings)

| System | Full Contrast / LL | Evaluation-like Contrast |
| --- | ---: | ---: |
| B2_C2_cscva3 (incumbent, on-platform) | 0.473743 / 0.725075 | 0.456763 |
| B2_D2 / B2_D3 / B2_D4 | 0.471471 / 0.473921 / 0.471606 | 0.456459 / 0.457455 / 0.454612 |
| C2 | 0.473462 / 0.731046 | 0.450837 |
| C2_mix (no neighborhood correction) | 0.475294 / 0.730806 | 0.459284 |
| C2 + C6 without CS-CVA (diagnostic) | 0.476208 | 0.456378 |

**Watch items** (observations only; no action taken):

1. **C2 + C6 without CS-CVA** scores 0.4762 on full Development, but it is not better Evaluation-like (0.4564 against the incumbent's 0.4568).
2. **The neighborhood correction cuts both ways.** It raises C2 on P2 (filtered 0.466105 → 0.468648, +0.0025) but lowers it on Development Evaluation-like (0.459284 → 0.450837).

## Eligibility

Every Qwen-dependent output is `UNCONFIRMED_QWEN_ST3`. MetaCLIP 2 (E2) is CC-BY-NC-4.0. No Evaluation data was used.

## Incident log

Both incidents were caught by optional-arm isolation: C2 and the other candidates completed in both runs. Each incident cost about 50 minutes of GPU session.

1. **Run 1, E5 loader.**
   - *Symptom:* `missing_keys: ['lm_head.weight']`.
   - *Root cause:* the 8B config is untied and its checkpoint ships no head; the loader treated any missing key as fatal.
   - *Fix:* patch 1 accepted exactly that key for untied configs and added a CPU key audit (checklist item 8).
2. **Run 2, E5 device placement.**
   - *Symptom:* `indices should be either on cpu or on the same device as the indexed tensor (cuda:1)` in `last_token`.
   - *Root cause:* with `device_map="auto`, the last hidden state lives on the second GPU while the pooling index is built on the first. No CPU test or mock exercises multi-device placement.
   - *Fix:* none, by owner decision (patch 2 not executed).
   - *Rule:* checklist items 9 and 10.

## Possible successors

None proposed here. The VLM-embedding image axis looks saturated at 2B for Contrast.
