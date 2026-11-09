# Project state

| Component | Status |
| --- | --- |
| Public dataset | Pinned ST3 checkout and local integrity audit complete |
| Baseline | Frozen CLIP plus annotator-weighted LR implemented |
| Local evaluation | Public-rule scorer and starter-kit Log Loss diagnostic implemented |
| Submission | Development CSV validation, ZIP packaging and independent ZIP validation implemented |
| Tests | Lightweight fixtures and GitHub Actions workflow |

Run `py scripts/setup.py` on Windows, then use the virtual environment for `scripts/run_st3.py baseline`, `scripts/run_st3.py submission`, `scripts/run_st3.py validate runs/baseline_dev/submission.zip`, and `python -m pytest -q`. See the [README](../README.md) for executable paths and standalone audit and evaluation commands.

Pending work includes reviewing the setup branch and uploading the development archive manually if desired. The first follow-up experiment is the organizers' shared-image-weights ablation. Keep the data, image embeddings, `C` and scoring fixed, and remove only the image-language interaction block. Report paired per-painting Contrast changes and per-language Log Loss alongside the aggregate scores.
