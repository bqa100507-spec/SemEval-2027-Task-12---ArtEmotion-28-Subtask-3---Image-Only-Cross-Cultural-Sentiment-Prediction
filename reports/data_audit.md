# ST3 public data audit

Verified on 2026-11-09 from the pinned organizer checkout `cf7cc444cf56617f3b8417497c6930ad172c5c99`. The organizer verifier passed with Windows path separators normalized in the verifier subprocess. The independent audit decoded every referenced image, checked the painting-to-path mapping, and matched all 849 development input keys and image paths to eligible validation pairs.

| Split | Paintings | Annotations | Languages | Positive | Negative | Ambiguous | Eligible pairs | Eligible languages |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1,572 | 165,013 | 28 | 105,158 | 47,289 | 12,566 | 16,019 | 25 |
| val | 86 | 8,732 | 28 | 5,602 | 2,455 | 675 | 849 | 22 |
| total | 1,658 | 173,745 | 28 | 110,760 | 49,744 | 13,241 | 16,868 | 25 |

There were zero missing, orphaned or undecodable images, zero unknown labels, and zero train/validation overlaps by painting or image path. Of 86 validation paintings, 85 have at least one eligible pair and appear in the development input.

| Split | Positive share among binary votes | Positive share among eligible-pair votes | Eligible pair share (min / Q1 / median / Q3 / max) |
| --- | ---: | ---: | --- |
| train | 68.98% | 67.86% | 0 / 0.4 / 0.8 / 1 / 1 |
| val | 69.53% | 68.84% | 0 / 0.4 / 0.8 / 1 / 1 |
| total | 69.01% | 67.91% | 0 / 0.4 / 0.8 / 1 / 1 |

## Language coverage

| Language | Train annotations | Val annotations | Train eligible pairs | Val eligible pairs |
| --- | ---: | ---: | ---: | ---: |
| Arabic | 5,744 | 323 | 535 | 29 |
| Burmese | 6,983 | 377 | 198 | 10 |
| Chinese | 6,579 | 356 | 1,113 | 61 |
| Darija | 6,024 | 317 | 17 | 0 |
| Emakhuwa | 1,850 | 90 | 5 | 0 |
| English | 6,166 | 349 | 648 | 36 |
| Hausa | 7,084 | 373 | 930 | 49 |
| Hindi | 6,906 | 347 | 908 | 44 |
| Igbo | 7,376 | 392 | 1,269 | 71 |
| Indonesian | 5,732 | 306 | 296 | 12 |
| IsiNdebele | 2,062 | 103 | 0 | 0 |
| IsiXhosa | 1,872 | 95 | 0 | 0 |
| IsiZulu | 7,583 | 399 | 756 | 42 |
| Kazakh | 2,426 | 117 | 0 | 0 |
| Kinyarwanda | 5,534 | 297 | 9 | 0 |
| Korean | 7,290 | 381 | 1,053 | 55 |
| Kyrgyz | 6,111 | 306 | 642 | 28 |
| Malay | 7,063 | 386 | 720 | 45 |
| Setswana | 5,974 | 298 | 333 | 12 |
| Swahili | 6,947 | 372 | 993 | 55 |
| Tagalog | 7,924 | 414 | 658 | 41 |
| Tamil | 6,620 | 365 | 662 | 37 |
| Thai | 4,986 | 263 | 240 | 14 |
| Turkish | 7,195 | 378 | 572 | 29 |
| Urdu | 3,024 | 165 | 18 | 1 |
| Uzbek | 7,312 | 388 | 1,044 | 49 |
| Vietnamese | 7,559 | 403 | 1,228 | 70 |
| Yoruba | 7,087 | 372 | 1,172 | 59 |

Recheck with `.venv/Scripts/python.exe scripts/run_st3.py audit --official --output runs/data_audit.json` on Windows or substitute `.venv/bin/python` on Linux or WSL.
