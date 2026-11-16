# EXP-025 CLOSED / NO-GO — caption-based semantic fusion

Final five-fold P2 results supplied by the owner from the completed Colab pipeline; not independently replayed in TASK-028.

| Model | Contrast | Macro Log Loss (bits) |
| --- | ---: | ---: |
| A visual-only |0.447073|0.715490|
| B VLM semantic-only |0.396442|0.786286|
| C early fusion |0.428974|0.731578|
| D late fusion |0.446692|0.714754|
| E50+L reference |0.447098|0.710554|

True early fusion beats shuffled early fusion by+0.019216 Contrast but loses to visual-only by−0.018099. **CLOSED / NO-GO for caption-based semantic fusion.** This does not invalidate VLM-centric supervised modeling. Preserve the1335-image FP16 semantic cache and all prior artifacts; do not rerun EXP-025. Earlier notebook/run instructions are archival. Production remains E50+L.
