# EXP-040 Evaluation-phase inference (deployment)

Status: **prepared, not exercised on real Evaluation inputs.** It applies only if EXP-040 is GO. For NO_GO the best ZIP is the EXP-038 incumbent: use `scripts/deploy_exp038.py`.

## Steps

1. **Prepare.** `python scripts/deploy_exp040.py prepare --results exp040_results.zip --output deploy40` copies the hash-checked Development bundle:
   - Track 1 heads, including the diagnostic E3L ridge;
   - neighbor banks;
   - training features, including the adapted E3L embeddings of the 1,571 training paintings;
   - `bundle/e3l/`, which holds `weights.pt` (LoRA + head at the deployment epoch), `init_head.npz` and `e3l.json`.

   It also copies `decision.json`, `selection.json` and `run.json`.
2. **Infer.** Run:

   ```
   python scripts/deploy_exp040.py infer --selected-only --bundle deploy40 --input <official input CSV> --images <artwork root> --output out40 --cscva <CS-CVA-3 ZIP> --c6 <C6 ZIP> --e3-snapshot <pinned 2B snapshot>
   ```

   This does the following:
   - extracts the base Fixed Ridge features and E1/E2/E3 with the pinned encoders;
   - runs the E3L forward (E3 formatting, batch 1, FP16 autocast, single GPU) with the saved LoRA and head;
   - applies the selected candidate's mix and neighbor bank;
   - composes B2 = round6(0.80 × round6(0.75 × Fx + 0.25 × CS-CVA-3) + 0.20 × C6);
   - writes ZIPs and `inference_receipt.json`, which records the measured E3L throughput.

   At deployment epoch 0, E3L is the frozen E3, so no LoRA forward is needed.
3. **Inputs used.** Only artwork pixels and the target language. Painting IDs and image paths are used to locate files and join results.

## Measured throughput

On the owner's RTX 4060 Laptop GPU, the E3L forward with LoRA attached in eval mode, batch 1 and FP16 autocast, ran at:
- **3.25 paintings/s** for 267 P2 held-out paintings in the step-0 pass;
- 2.43 paintings/s for 107 paintings in the smoke inner pass, which ran right after training on the same GPU.

Model load takes 26 s. The receipt is `runs/exp040/local_smoke/receipt.json`.

An Evaluation set the size of Development (85 paintings) therefore needs about 30 s of E3L forward after loading. Kaggle T4 throughput will be in `exp040_smoke.json`.

All EXP-040 ZIPs are `UNCONFIRMED_QWEN_ST3`. E2 is MetaCLIP 2 (CC-BY-NC-4.0).
