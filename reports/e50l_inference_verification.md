# TASK 011 E50+L production inference verification

E50+L is the default prediction mode. The project owner reported hosted Development Contrast 0.4643 for submission `#1046158`. The integration reused the frozen EXP-010 checkpoint, image processor and nested 50/50 probability blend. The original E50 and CLIP heads and modes are still available.

## Exact development parity

The CLIP-L head was reconstructed on the original 16,019 training pairs and 1,571 training paintings with C=0.3, seed 2027 and the original language ordering. It converged in 60 iterations, matching EXP-010. Two preparations produced the same non-pickle head file SHA-256 `4d3a213db3c9ec86e16d56f82566ab5f715fb4905d0e946b8a32920da9a13a42`. EXP-010 persisted the final ensemble CSV, so preparation checked exact composed parity and separately compared the fitted and reloaded CLIP-L component probabilities.

The general command ran each mode from an isolated directory containing the public input CSV and 85 artworks. It used no annotations and extracted no new features. All 849 rows, input ordering and probability formatting matched byte for byte. The E50 and CLIP regressions passed with the CLIP-L bundle path set to a missing directory.

| Mode | Reference | Prediction CSV SHA-256 |
| --- | --- | --- |
| E50+L | EXP-010 | `162875b641407efea3bf28d1d147c2d9a3dd3dc2faed8863f9438bee517199a3` |
| E50 | EXP-007 | `89591c9ccefe47b570933ab2f7765e3927882d0912ec3288914b6c692e50b0d1` |
| CLIP | EXP-007 | `e4c0fc06ed640f10b702b4b2a95451b1cad4665c86f2cbf170ef2c7c3cce9e47` |

Each archive validates against all 849 rows and contains the exact CSV at its root. The authoritative EXP-010 E50+L ZIP SHA-256 remains `38c2eca8ee4c0557bf198bf5df5dbf90571a2851404ded6302f40e42dc138e0d`. The repackaged parity ZIP SHA-256 is `0cf8a108bd76404e72901b1375253efef61ab4cebc5a3ada1b12a1a500e444dd`. The CSV bytes, CRC and compressed size match. Repackaging updated the ZIP member timestamp.

## New-image CUDA smoke

The cold fixture had eight input rows, seven image paths and six unique synthetic RGB image contents. It included two language rows for the same painting and a byte-identical image alias. All three real encoders ran sequentially on the RTX 4060 Laptop GPU and each encoded six images. The validated output also exercised the shared image terms for Kazakh, an official language absent from training.

| Cold worker | Process wall time | Model startup | Peak CUDA allocated | Peak CUDA reserved |
| --- | ---: | ---: | ---: | ---: |
| CLIP-B | 30.39 s | 13.12 s | 625 MiB | 684 MiB |
| SigLIP 2 | 22.46 s | 14.57 s | 416 MiB | 476 MiB |
| CLIP-L | 25.49 s | 16.16 s | 1,226 MiB | 1,244 MiB |

Process wall time includes interpreter startup, checkpoint checks, inference and cache persistence. Model startup covers checkpoint lookup, hashing, load and device transfer. The cold end-to-end command took 78.61 s. A warm rerun with a different CLIP-L batch size took 18.81 s, loaded no encoder weights, reused six features per encoder, and reproduced the CSV byte for byte. Replacing one image at the same path encoded just that image in each worker, reused the other five features per worker and changed only the corresponding prediction.

A second fixture contained 18 fresh artworks. The process tree was forcibly stopped after CLIP-L committed four image features. The output directory stayed unpublished. Rerunning the same configuration reused all 18 CLIP-B features, all 18 SigLIP 2 features, and the four committed CLIP-L features. It encoded the remaining 14 CLIP-L images and published a valid 18-row ZIP in 40.20 s.

Two fresh CLIP-L production embeddings were byte-identical to the EXP-010 extractor at the same batch size. Explicit CPU mode ran all three real encoders on one distinct synthetic image and produced a validated two-row ZIP in 65.83 s.

## Failure checks and preservation

The real smoke rejected missing images, traversal paths, duplicate keys, inconsistent painting-to-image mappings, malformed headers, unrecognized languages and an unavailable CLIP-L bundle. Failed runs published no output. A unit test also confirms that a CLIP-L worker failure does not publish an E50 fallback.

The full Windows suite passed 104 tests and skipped one fixture because the host disallows test symlinks. Both pinned Python environments passed dependency checks. SHA-256 verification covered 545 pre-existing files, including the E50 bundle, all prior prediction artifacts and 33 cache and checkpoint files.

The smoke artifacts are in `runs/e50l_inference_smoke/`. The separate default CLIP-L bundle is in `runs/e50l_inference_models/`. Evaluation-phase artwork and labels were outside verification. [Windows runbook](../docs/e50l_inference.md)
