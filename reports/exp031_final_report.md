# EXP-031 · Hoàn tất nghiên cứu và giữ E50+L

**EXP-031 = COMPLETE (6/6) / NO-GO FOR PRODUCTION.** Epoch 3 là checkpoint tốt nhất theo Cultural Contrast trên P2 fold 3. Nó đạt **0.410688 / 0.729746 bits**, so với E50+L **0.433536 / 0.713192**. Giữ branch, checkpoint và bằng chứng để tái lập.

## Giả thuyết và cấu hình đã chạy

EXP-030 tìm thấy tín hiệu Qwen vượt prior ngôn ngữ nhưng phần tương tác dự đoán kém khớp gold hơn E50+L. EXP-031 kiểm tra một gói thay đổi: neo binary logit vào prior fitting, tách mức sentiment của tranh khỏi tương tác ngôn ngữ, giữ giám sát tám cảm xúc và thêm ranking có trọng số theo bằng chứng phiếu. Kỳ vọng là cải thiện thứ tự cùng tranh và giảm logit cực đoan. Đây là so sánh recipe, vì cả tham số hóa, optimizer schedule theo nhóm, learning rate, decay và seed đều thay đổi so với EXP-030. [EXP-030 error analysis](exp030_error_analysis.md) · [Thiết kế và audit EXP-031](exp031_readme.md)

| Thành phần | Cấu hình chạy |
| --- | --- |
| Population | P2 fold 3. Fitting 1,068 tranh / 10,879 cặp. Held-out 267 tranh / 2,721 cặp. Tám count từ annotation ghim, đối chiếu k/n của P2. |
| Backbone | Native frozen BF16 vision và NF4 double-quantized BF16 decoder Qwen2.5-VL-7B-Instruct, revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`. SDPA math deterministic, tối đa 200,704 pixel theo processor, gradient checkpointing. |
| Tham số học | 5,208,361. Decoder Q/V LoRA rank 16, alpha 32, 28 block. Shared scalar, rank-32 interaction factorization, hai conditional softmax bốn cảm xúc. |
| Binary head | `b_l + s(x) + d(x,l)`. `b_l` logit prior Jeffreys từ fitting. Interaction center trên 25 ngôn ngữ, anchor tại trạng thái pretrained của ảnh xám cố định. E50+L chỉ dùng để đối chiếu. |
| Loss | Count-weighted joint CE8 + 0.5 BCE. Ranking `0.15 × mean_painting` của logistic margin theo dấu chênh phiếu. Confidence ramp từ z=1 đến 2.5 theo phương sai Jeffreys, mẫu số là số cặp gold không tie. L2 interaction `0.01 × mean_painting,language(d²)`. |
| Tối ưu | AdamW, head 3e-4 và LoRA 3e-5, decay 0.05, cap norm 1, seed 2031, sáu epoch. Microbatch hai cặp, update xấp xỉ 16 cặp và giữ tranh nguyên nhóm. Free graph conditional sau từng microbatch. |
| Lựa chọn | Contrast cao nhất trên fold 3, tie lấy epoch sớm nhất. Fold dùng để chọn epoch và đã bị tái sử dụng, nên điểm tối đa mang tính exploratory. |

## Xác minh độc lập bằng CPU

- Đọc final report attempt `20261120T210825Z-f167122a` và ZIP gốc. SHA-256 ZIP `20ff395cc9a1a61cb2300dddf44c8db3b61a3dd6c17107a60550f1cd411a8c92`. Cả 17 file trong manifest khớp. Report ngoài ZIP thống nhất với results trong ZIP, trạng thái `COMPLETE`, 6/6 epoch, best epoch 3. Hai attempt cũ hiện không có export trong thư mục Drive đã kiểm tra. Biên nhận attempt cuối xác nhận **resume từ epoch 3**, replay 2,721 xác suất ở tolerance 1e-6. Không đủ log để xác định toàn bộ thời gian hoặc nguyên nhân của từng lần ngắt. [Final report](https://drive.google.com/file/d/1KPJeeqmsigk16F_7cAk_VmLZhOXSXv72/view) · [ZIP](https://drive.google.com/file/d/1hGfcjDEEy6SU4NgfzYEBuMR6kEE51pjw/view)
- Xác minh toàn bộ **18 file thành viên của sáu epoch seal**, gồm sáu `resume.pt`, sáu prediction CSV và sáu epoch JSON. Tất cả seal khớp source `d3d52325f54c41d5779fb88cf568d38a929b39a7`, config, lock và fold manifest trong Git. Handoff và manifest khớp hash. 25 prior tính lại từ fitting khớp chính xác. Checkpoint load trên CPU bằng `weights_only=True`. Mỗi file có 122 tensors trainable FP32 hữu hạn, 122 AdamW states phù hợp, anchor 3,584 chiều và RNG CPU/CUDA. Adam step tăng đúng theo số update tích lũy, tới **5,688**. Đây là kiểm tra bytes, cấu trúc và trạng thái tối ưu, còn biên nhận replay và CUDA parity đến từ run gốc.
- Epoch 3 có `resume.pt` SHA-256 `472f47d5f70ecb6cf76002ec91c6ae05b9f66b19b4c8f1aa8387885d88030b23`, prediction CSV `d1cd3f4de65b8d486330a98db03d09465c31e9673975600155089274f23418a8`. Giữ nguyên thư mục Drive và bản audit đọc lại dưới ignored `runs/exp031/closure_evidence/`. Notebook đã chạy ghim mã thực thi ở trên, bản notebook phát hành ở `5a388612a4f18acd3b14fcba749e24ce0c6d6f4f`. [Best checkpoint](https://drive.google.com/drive/folders/1REycDEo28Mk66tPCAZTJU1AVrzCqhWo5) · [Toàn bộ checkpoint](https://drive.google.com/drive/folders/17W2N6L1CF1tPuZGObbfNaBr3gokwiz8n)
- Tính lại scorer trên đúng fold 3 từ `training/pairs.csv`, `training/protocol.csv` và matched `history/exp022_R0.csv` của handoff. Các file khớp pin EXP-026. Cả sáu bộ dự đoán phủ đúng 2,721 khóa, scorer tái lập Contrast và Macro Log Loss tới 1e-12, BCE theo phiếu tới 2e-7. Có 253 tranh đủ Contrast và 21 ngôn ngữ đủ Log Loss. Pipeline audit dùng `read_protocol(..., labels=True)` → `fold3` → `score(gold[painting,language,k,n], predictions)`. Không dùng Development gold hay nhãn confirmation, không chạy lại model. [Scorer](../src/st3/metrics.py) · [Manifest](../configs/exp028_fold3_manifest.json)

## Learning curves

| Epoch | Contrast ↑ | Macro LL ↓ (bits) | Train total | Train CE8 | Train BCE | Train rank | Held CE8 | Held BCE | Held rank | Norm trước clip | Giây train + val |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.389104 | 0.737103 | 1.962320 | 1.668174 | .542572 | .149498 | 1.585050 | .512644 | .147094 | 16.91 | 6,734.3 |
| 2 | 0.407767 | 0.736292 | 1.830822 | 1.557597 | .503008 | .137179 | 1.572636 | .512622 | .140221 | 12.96 | 6,463.8 |
| **3** | **0.410688** | **0.729746** | 1.791954 | 1.523733 | .493707 | .132968 | 1.559604 | .509563 | .138543 | 11.61 | 6,465.6 |
| 4 | 0.410193 | 0.730515 | 1.754818 | 1.492300 | .482977 | .129503 | 1.572962 | .509292 | .133633 | 10.38 | 6,708.9 |
| 5 | 0.400981 | 0.743214 | 1.717390 | 1.460770 | .471850 | .125939 | 1.571782 | .517151 | .136802 | 9.11 | 6,499.5 |
| 6 | 0.383856 | 0.748567 | 1.682469 | 1.431305 | .461858 | .121053 | 1.594538 | .524416 | .139398 | 8.09 | 6,506.0 |

Train interaction mean-square tăng từ .043495 lên .207679. Held interaction mean-square ở epoch 3/4/6 là .118153/.169981/.164384. Train clip fraction **1.000 ở cả sáu epoch**. 1.762–1.834 cặp fitting/s theo epoch. Tổng thời gian train + validation của sáu epoch là **39,378 s (10.94 giờ)**, trong đó training loop **36,128 s (10.04 giờ)**. Không cộng lại thời gian attempt cuối vào các tổng epoch này. Attempt cuối mất 20,096 s trước export và hoàn tất epoch 4–6. Peak epoch allocated/reserved trên L4 là **6.982/7.883 GiB**. Native-path parity ghi max error 0.0, peak probe 6.983/7.883 GiB. Runtime Python 3.12.10, Torch 2.7.1+cu126, Transformers 4.51.3, bitsandbytes 0.46.1, CUDA 12.6, driver 580.82.07. Đây là thời gian chiếm dụng theo giai đoạn, không phải thời gian kernel hoạt động.

| Cùng fold 3 | Contrast | Macro LL | Spearman theo tranh | Concordance theo tranh |
| --- | ---: | ---: | ---: | ---: |
| E50+L | .433536 | .713192 | .387925 | 69.36% |
| EXP-030 epoch 3 | .419801 | .733054 | .373590 | 68.98% |
| EXP-031 epoch 3 | .410688 | .729746 | .378254 | 69.18% |
| EXP-031 epoch 6 | .383856 | .748567 | .345876 | 67.31% |

| Paired Contrast | Δ | CI 95% | Thắng / thua |
| --- | ---: | ---: | ---: |
| EXP-031/3 − E50+L | −.022848 | [−.042224, −.004327] | 108 / 145 |
| EXP-031/3 − EXP-030/3 | −.009113 | [−.029380, +.010538] | 118 / 135 |
| EXP-031/4 − EXP-031/3 | −.000495 | [−.009613, +.008809] | 133 / 120 |
| EXP-031/6 − EXP-031/3 | −.026832 | [−.044050, −.010057] | 122 / 131 |

Bootstrap percentile 10,000 lần, seed 203104, lấy lại nguyên tranh trên 253 tranh chung. Spearman dùng average rank. Concordance bỏ gold ties, cho 0.5 nếu dự đoán tie. Các khoảng điều kiện trên checkpoint và fold đã chọn, không hiệu chỉnh adaptive reuse, epoch selection hay khác seed.

## Kết luận

**Đã đo.** Epoch 3 hơn EXP-030 .003308 bits về Macro Log Loss và nhỉnh hơn nhẹ về rank, nhưng thấp hơn .009113 Pearson và không vượt E50+L ở cả hai metric chính. Epoch 4 giảm surrogate rank ngay cả trên held-out, từ .138543 xuống .133633, trong khi Pearson gần như đi ngang và Spearman giảm. Cuối run, loss fitting tiếp tục giảm nhưng ranking held-out, Contrast và xác suất đều xấu hơn. Tỷ trọng dự đoán ≤.1 hoặc ≥.9 tăng từ 17.76% ở epoch 3 lên 36.19% ở epoch 6 theo trọng số macro-language. Squared error trên tỷ lệ phiếu tăng từ .05382 lên .05723. Dấu hiệu này phù hợp với overfitting muộn và thay đổi phân bố độ tin cậy.

**Còn nghi vấn.** Rank surrogate thưởng dấu và biên độ logit dưới confidence, còn scorer Pearson đo độ thẳng hàng của xác suất thực với tỷ lệ thực nghiệm. Shared logits triệt tiêu trong ranking nhưng sigmoid ảnh hưởng đến spacing. Anchor, regularizer và clipping đều khả thi về gradient. Tuy vậy, clip ở mọi step và norm giảm không cho biết nhóm tham số nào làm tăng norm hay nếu bỏ clip model có tốt hơn. AdamW dùng moment, nên không diễn giải hệ số clip như hệ số giảm learning rate trực tiếp. Chưa có đối chứng giữ nguyên các yếu tố khác để tách lỗi optimizer, prior, rank, representation và kích thước dữ liệu. Hai recipe 7B có chi phí cao và cùng hụt production, đủ để dừng đầu tư hướng này trong vòng kế tiếp. [Audit lịch sử và postmortem](research_history_audit.md) · [Chiến lược trước EXP-032](exp032_research_strategy.md)
