# EXP-023 resource eligibility review

Checked **2026-11-16**. Status: **ELIGIBILITY_UNRESOLVED**. Implementation and CPU/synthetic verification are authorized; scientific GPU execution, confirmation access and promotion are blocked.

| Official source | Direct evidence and limit |
| --- | --- |
| [Organizer README at current main](https://github.com/artemotion-28/artemotion-28/blob/cf7cc444cf56617f3b8417497c6930ad172c5c99/README.md#st3-image-only-cross-cultural-sentiment-prediction) | `git ls-remote` still resolves main to `cf7cc444cf56617f3b8417497c6930ad172c5c99`. ST3 supplies pixels and target language; captions are not inference inputs. This does not explicitly decide auxiliary training-caption distillation. |
| [FAQ](https://github.com/artemotion-28/artemotion-28/blob/cf7cc444cf56617f3b8417497c6930ad172c5c99/README.md#faqs) | External-resource guidance remains pending: “Guidance will be published with the competition platform details.” No explicit permission covering this procedure was found. |
| [Rules](https://github.com/artemotion-28/artemotion-28/blob/cf7cc444cf56617f3b8417497c6930ad172c5c99/README.md#competition-rules-and-terms) | Annotation research-use permission and publication rules do not explicitly authorize every ST3 training procedure. |
| [All organizer issues](https://api.github.com/repos/artemotion-28/artemotion-28/issues?state=all&per_page=100), [issue #1](https://github.com/artemotion-28/artemotion-28/issues/1) | Live API returned one open issue asking about model/resource limitations and zero comments. No organizer answer was available. No separate announcement resolving captions was found. |
| [Official ST3 competition](https://www.codabench.org/competitions/18468/) | The accessible response contained unrelated text, not usable competition rules. This is a retrieval limitation, not evidence of permission or prohibition. Historical captures are not treated as current authoritative approval. |
| [Pinned CLIP-L model card](https://huggingface.co/openai/clip-vit-large-patch14/blob/32bd64288804d66eefd0ccbe215aa642df71cc41/README.md) | Existing asset pin records MIT and the verified caption cache revision. The web fetch failed. Its existing local checksum/model-card record is distinct from competition eligibility; no new teacher is introduced. |

The local pinned model card was independently read and matches SHA `de4274ec48590ab3aa1c239720ff55e99a37f4165c117ba3ad7af2b7df72687c`. It warns about English-only evaluation and task-specific limitations; these remain research limitations, particularly for multilingual caption targets. The [original OpenAI CLIP software license](https://github.com/openai/CLIP/blob/main/LICENSE) is MIT, verified separately. Neither that software license nor the cached teacher's availability establishes organizer permission for this training procedure.

## Clarification question — prepared, not posted

For ST3, may a system use **only the original official training captions**, encoded by the existing pinned `openai/clip-vit-large-patch14` text tower, as auxiliary training targets for a frozen-image-feature sentiment head, while inference receives **only artwork pixels and target language**? No external text, generated captions, held-out captions, test labels or other-language test responses would be used. Please confirm whether this training procedure and pinned pretrained resource are eligible for the competition.

No message/issue was sent. An owner status string is not a substitute for an organizer reply.

## Enforced evidence boundary

`configs/exp023_eligibility.json` is unresolved by default. A future reviewed update must name an authoritative HTTPS organizer source, its exact byte SHA-256, an exact permission excerpt, and explicit coverage of ST3/original-caption auxiliary targets/the pinned CLIP-L text resource. `verify_eligibility` independently fetches/checks that evidence before any scientific GPU fit. The cloud supervisor rejects unresolved status before environment installation/training. The source/evidence update creates a new source and handoff identity; it cannot be bypassed by editing notebook booleans or changing a checksum to match an unreviewed file.

The frozen scientific settings and decision gates do not change when resource permission is resolved. Until then, the delivered notebook is a tested implementation with a deliberate eligibility block, not an authorized real-cloud run.
