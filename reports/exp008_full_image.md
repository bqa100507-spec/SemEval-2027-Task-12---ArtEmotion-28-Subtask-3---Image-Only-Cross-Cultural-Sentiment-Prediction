# EXP-008 Full-image CLIP coverage

**Decision: retain EXP-007 E50.** The project owner reported hosted Development Contrast 0.4600 for Codabench submission `#1045921`. Its local score was 0.4600261535860795 Contrast and 0.7442135227739901 Macro Log Loss. Both alternatives failed the train-only gate, so EXP-008 produced no development score or archive. [EXP-007 result](exp007_fusion.md)

## Label-free audit

The pinned CLIP processor uses bicubic resize to a 224-pixel short edge, a 224 × 224 center crop, and the checkpoint RGB mean and standard deviation. For original dimensions `w × h`, the resized long edge is `floor(224 × max(w,h) / min(w,h))`. Retained area is 224 divided by that edge. The audit read dimensions, painting/image identifiers and split membership for 1,658 public artworks. It grouped by orientation, split and the existing style folders without reading annotation labels. Style and geometry were used only for analysis. [Pinned processor](https://huggingface.co/openai/clip-vit-base-patch32/blob/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/preprocessor_config.json)

| Quantity | Min | P5 | P25 | Median | P75 | P95 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Width / height | 0.182 | 0.613 | 0.750 | 0.903 | 1.345 | 1.668 | 6.383 |
| Center area retained | 15.7% | 54.0% | 67.7% | 74.9% | 80.6% | 91.4% | 100% |

Mean retention was 73.9%. Substantial cropping, defined as retention below 80%, affected 1,181 paintings (71.2%).

| Group | Paintings | Mean retained | Substantially cropped |
| --- | ---: | ---: | ---: |
| Train | 1,572 | 73.9% | 1,118 |
| Validation | 86 | 75.4% | 63 |
| Portrait | 826 | 73.6% | 599 |
| Landscape | 735 | 71.4% | 582 |
| Near-square (width/height 0.90 to 1/0.90) | 97 | 96.4% | 0 |

Of the style folders with at least 30 paintings, mean retention ranged from 68.9% in Early Renaissance (41 paintings) to 81.3% in Minimalism (37). The full audit is in `audit/geometry.csv`, `audit/groups.csv` and `audit/audit.json` under the EXP-008 run.

## Frozen comparison

- **E50** uses the saved center-crop CLIP and SigLIP 2 probabilities. **E50-Full** replaces the CLIP input with the full frame, bicubic resized on the long side to 224, then centered on a 224 × 224 square. The padding is RGB (123, 117, 104), the checkpoint channel means rounded to 8-bit values. The processor rescales and normalizes the square with its own parameters. The full frame survives spatially at lower resolution.
- **E50-Dual** L2-normalizes the equal average of the unit center-crop and full-image embeddings. Both alternatives use the original frozen CLIP ViT-B/32 weights. Each alternative fits only its CLIP head with the original annotator-weighted Model C Logistic Regression at C=0.3. All three blends use the same saved SigLIP 2 probabilities at weight 0.5. The exact EXP-002 painting folds were verified.
- Commit `49e84b9` froze the protocol before CV. Promotion required five complete and converged folds, at least +0.001 pooled Contrast, at least four fold wins, a strictly positive lower 95% paired painting-bootstrap bound, and at most +0.01 bits in both fold-mean and pooled Macro Log Loss. Rank passing candidates by pooled Contrast and prefer Full on an exact tie.

Each metric cell below gives Cultural Contrast ↑ / Macro Log Loss ↓.

| Fold | E50 | E50-Full | E50-Dual |
| --- | ---: | ---: | ---: |
| 1 | 0.457082 / 0.725124 | 0.448835 / 0.730938 | 0.452982 / 0.725236 |
| 2 | 0.441425 / 0.727116 | 0.437157 / 0.734903 | 0.441068 / 0.728740 |
| 3 | 0.419136 / 0.722669 | 0.417576 / 0.729526 | 0.419918 / 0.722119 |
| 4 | 0.471097 / 0.726467 | 0.464925 / 0.732384 | 0.469848 / 0.726999 |
| 5 | 0.420799 / 0.731953 | 0.420038 / 0.731081 | 0.422484 / 0.728872 |
| Fold mean | 0.441908 / 0.726666 | 0.437706 / 0.731766 | 0.441260 / 0.726393 |
| Pooled OOF | **0.442055** / 0.725927 | 0.437825 / 0.730894 | 0.441386 / **0.725602** |

| Alternative | Δ pooled Contrast | Paired 95% interval | Fold wins | Painting wins / regressions | Languages with lower loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | −0.004230 | [−0.006679, −0.001700] | 0/5 | 675 / 821 | 2/21 |
| Dual | −0.000670 | [−0.002020, +0.000702] | 2/5 | 717 / 779 | 12/21 |

Intervals use 10,000 paired painting draws (seed 82028), conditional on the fixed out-of-fold predictions. Both alternatives passed the loss guardrails. Dual reduced pooled loss by 0.000326 bits. Uzbek had the largest loss improvement for Full (−0.002313 bits) and Dual (−0.002690). Indonesian had the largest regression for both (+0.017918 and +0.004298 bits). Full's largest painting gain was Botticelli's *St. Jerome in Penitence* (+0.222153), and its largest drop was Sam Francis's *Chinese Wall L* (−0.223172). Dual gained most on Van Gogh's *Sheet with Many Sketches of Figures* (+0.148040) and dropped most on Robert Brackman's *Life About Me* (−0.132553). Complete pooled changes are in `cv/paintings.csv` and `cv/languages.csv`.

## Aspect-ratio subgroups

The counts are Contrast-scored paintings. Changes are relative to E50 within each group. Macro Log Loss applies the scorer's language eligibility separately in each group.

| Group | Scored | E50 Contrast / loss | Δ Contrast Full / Dual | Δ loss Full / Dual |
| --- | ---: | ---: | ---: | ---: |
| Retention <65% | 263 | 0.447602 / 0.750852 | +0.001139 / +0.001671 | +0.010704 / +0.001029 |
| Retention 65–<80% | 798 | 0.446817 / 0.727960 | −0.005236 / −0.001196 | +0.003706 / −0.000671 |
| Retention 80–<95% | 380 | 0.430499 / 0.701794 | −0.006171 / −0.001185 | +0.002856 / −0.001014 |
| Retention ≥95% | 55 | 0.426279 / 0.762681 | −0.001909 / −0.000657 | +0.003876 / −0.000478 |
| Portrait | 749 | 0.416179 / 0.749472 | −0.003682 / −0.000305 | +0.004217 / −0.000471 |
| Landscape | 661 | 0.473359 / 0.700788 | −0.004964 / −0.001140 | +0.005458 / −0.000495 |
| Near-square | 86 | 0.426814 / 0.747127 | −0.003364 / −0.000232 | +0.004139 / −0.000490 |

Below 65% retention, the small Contrast gains came with higher loss. Exploratory paired 95% intervals include zero for Full [−0.006035, +0.008343] and Dual [−0.002310, +0.005601]. This subgroup result does not justify a geometry-based switch.

## Cost, verification and next step

GPU extraction and CPU CV took 78.45 s wall-clock. The RTX 4060 Laptop GPU cache build covered 1,571 training images with eligible pairs in 23.60 s wall-clock. Decode took 4.69 s, letterboxing and normalization 7.69 s, and GPU transfer, encoder and normalized output readback 5.41 s. Computing the cached Dual average took 0.006 s on CPU. Peak GPU allocation was about 703 MiB. Across five folds, the CPU LR fits took 17.75 s for Full and 17.73 s for Dual. Each has the original head dimension.

Full uses one 224-pixel CLIP pass per unseen image, like E50, with different preprocessing. Dual requires an additional CLIP pass plus the embedding average. SigLIP 2 still runs once in each configuration. These are pass counts and extraction timings, with no end-to-end inference benchmark for the alternatives.

All ten CPU heads converged. The 48,057 out-of-fold probabilities passed validation. The original blend replayed exactly for all 16,019 pairs, both new cache arrays have unit norm, and the cached Dual values exactly replayed from the two views. A 53.13 s rerun reused the new cache and reproduced the prediction, painting, language, subgroup and bootstrap files byte for byte. A SHA-256 inventory verified 168 earlier run files and eight earlier feature-cache files unchanged. The EXP-007 archive still validates against all 849 development pairs. All 72 lightweight tests passed locally.

```powershell
.venv/Scripts/python.exe scripts/run_st3.py coverage-audit
.venv/Scripts/python.exe scripts/run_st3.py coverage-cv
.venv/Scripts/python.exe scripts/run_st3.py coverage-dev
```

Prioritize batched, resumable inference for the fixed E50 system with coverage checks on newly released artwork. Keep the submitted blend fixed.
