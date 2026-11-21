# EXP-032 · Kết quả multitask cultural interaction

**COMPLETE 5/5 · NO_GO_OOF.** Đồ thị học từ fitting tăng pooled Cultural Contrast từ 0.451702 lên 0.452266 so với đối chứng ridge cùng cấu hình. Mức tăng **+0.000564** có khoảng bootstrap theo tranh dương trên P2, nhưng thấp hơn ngưỡng thực dụng **+0.001** chốt trước. Cả hai head mới đều vượt frozen E50+L trên P2. Phần lớn mức tăng đến từ đối chứng ridge. Giữ E50+L là production.

## Câu hỏi và đối chứng

Thí nghiệm kiểm tra xem quan hệ giữa phản ứng của các nhóm ngôn ngữ có giúp ước lượng độ dốc image × language tốt hơn hay không. Hai phương án fit dùng cùng ba cache CLIP-B, SigLIP2-Base và CLIP-L, cùng weights blend 0.25 / 0.25 / 0.50 và cùng quy tắc làm tròn sáu chữ số. Mỗi logit có intercept toàn cục, intercept ngôn ngữ, độ dốc ảnh chung và độ dốc sai lệch theo ngôn ngữ. Hai thành phần riêng theo ngôn ngữ có tổng bằng 0 trong cơ sở Helmert. Loss là binomial negative log likelihood theo số phiếu và chuẩn hóa theo tổng số phiếu fitting.

Đối chứng phạt \(\lambda\operatorname{tr}(D^TD)/2\). Candidate dùng precision \(\lambda[I+\rho(L-P)]\), trong đó \(L\) là Laplacian được scale để có cùng trace với phép chiếu \(P\) lên sai lệch của các ngôn ngữ có cạnh. Uniform complete graph cho \(L=P\) và tái lập đúng đối chứng. Node cô lập giữ nguyên ridge. Chia lại budget phạt theo hướng giúp tách tác động của cấu trúc cạnh khỏi việc tăng ridge nói chung. Tất cả năm fold chọn cùng tỷ lệ ridge 0.5 và mức coupling \(\rho=0.5\), chỉ dựa trên hai inner splits phía fitting. Cả năm lựa chọn đáp ứng guardrail Log Loss nội bộ. [Giao thức chốt trước](exp032_preregistration.md) · [Config](../configs/exp032.json)

## Kết quả trên cùng năm fold P2

Pooled OOF gồm **1.335 tranh, 13.600 cặp**, trong đó **1.269 tranh** đủ điều kiện Contrast và **21 ngôn ngữ** đủ điều kiện Macro Log Loss. Điểm Log Loss dưới đây tính theo bit.

| Hệ thống | Pooled Contrast ↑ | Macro Log Loss ↓ | Spearman theo tranh ↑ | Pair concordance ↑ |
| --- | ---: | ---: | ---: | ---: |
| Frozen E50+L | 0.447098 | 0.710554 | 0.401507 | 69.9704% |
| Đối chứng ridge | 0.451702 | 0.709247 | 0.401646 | 70.0043% |
| Learned graph | **0.452266** | **0.709072** | **0.402282** | **70.0682%** |

| Fold | E50+L Contrast / LL | Ridge Contrast / LL | Graph Contrast / LL | Graph − E50+L ΔContrast / ΔLL | Graph − ridge ΔContrast / ΔLL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | .433429 / .666222 | .438684 / .665183 | .438424 / .665021 | +.004995 / −.001201 | −.000260 / −.000162 |
| 2 | .459194 / .724814 | .463904 / .724336 | .464354 / .724119 | +.005160 / −.000695 | +.000450 / −.000217 |
| 3 | .433536 / .713192 | .435016 / .711310 | .435771 / .711019 | +.002235 / −.002173 | +.000755 / −.000291 |
| 4 | .461302 / .739182 | .467131 / .736788 | .468537 / .736555 | +.007235 / −.002627 | +.001406 / −.000234 |
| 5 | .447396 / .712734 | .453135 / .711884 | .453578 / .711870 | +.006182 / −.000864 | +.000444 / −.000013 |

| So sánh theo tranh | Δ pooled Contrast | Khoảng bootstrap 95% | Fold thắng | Δ Macro Log Loss |
| --- | ---: | ---: | ---: | ---: |
| Graph − E50+L | +.005168 | [+.002533, +.007712] | 5/5 | −.001482 |
| Graph − ridge | **+.000564** | **[+.000160, +.000982]** | 4/5 | −.000175 |
| Ridge − E50+L | +.004604 | [+.002061, +.007062] | 5/5 | −.001308 |

Bootstrap percentile lấy lại nguyên tranh 10.000 lần với seed 203204. Graph thắng 682 tranh, thua 587 so với ridge. Khoảng cho graph − ridge nằm trên 0 nhưng cả cận trên cũng thấp hơn +0.001. P2 và các fold đã được dùng nhiều lần. Các khoảng này giữ nguyên tập tranh và giao thức, chưa bao gồm sự bất định do chọn mô hình qua nhiều EXP hoặc thay đổi population.

## Stability, coupling và độ tin cậy

| Fold | Cạnh đủ support ở cả hai nửa | Spearman similarity giữa hai nửa | Association top 1/3, kiểm chéo hai chiều | Khoảng cách tới penalty theo overlap | Khoảng cách tới uniform |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 190 | .469 | .358 / .417 | .135 | .184 |
| 2 | 190 | .570 | .379 / .443 | .130 | .181 |
| 3 | 193 | .643 | .361 / .395 | .142 | .187 |
| 4 | 193 | .549 | .370 / .377 | .135 | .183 |
| 5 | 189 | .538 | .390 / .383 | .139 | .184 |

Cả năm fitting scope vượt gate trước khi score outer. Có 21 ngôn ngữ có cạnh ở mỗi fold. Darija, Emakhuwa, Kinyarwanda và Urdu thiếu đồng quan sát để ghép cạnh, nên giữ ridge. Rank correlation của overlap giữa hai nửa là .982–.987. Trọng số cạnh toàn scope vẫn tương quan .501–.515 với overlap hiệu dụng. Hai phép kiểm bổ sung cho thấy cấu trúc còn lại sau mức overlap đơn thuần. Learned penalty cách overlap-only .130–.142, và association của các cạnh chọn từ nửa kia đạt .358–.443 khi đã center theo quartile điểm ảnh chung. Điều này ủng hộ tính lặp lại của quan hệ phản ứng trong P2. Noise phiếu và cách tranh được phân cho các nhóm vẫn có thể góp phần tạo quan hệ đó.

Kiểm toán hệ số xác nhận trace khớp với ridge và tổng deviation theo ngôn ngữ bằng 0. Precision nhỏ nhất theo fold bằng .752–.775 lần ridge. Tỷ lệ năng lượng graph \(\operatorname{tr}(D^TLD)/\operatorname{tr}(D^TPD)\), trung bình ba encoder, giảm **.981–.991 → .953–.965**. Chuẩn L2 deviation trung bình lại tăng nhẹ ở cả năm fold. Tăng ridge chung không tái tạo được precision theo hướng này. Có bằng chứng graph thực sự đổi cách phân bổ hệ số. Lợi ích trên score vẫn nhỏ.

| Nhóm theo số tranh fitting của ngôn ngữ | Cặp OOF | E50+L Contrast / LL | Ridge Contrast / LL | Graph Contrast / LL |
| --- | ---: | ---: | ---: | ---: |
| 1–49 | 46 | — / — | — / — | — / — |
| 50–199 | 526 | .058266 / .702389 | .020766 / .699882 | .013497 / .699458 |
| ≥200 | 13,028 | .455128 / .711308 | .460004 / .710324 | .460592 / .710200 |

Scorer tính lại eligibility trong từng slice. Nhóm 1–49 có 0 tranh và 0 ngôn ngữ đủ điều kiện, nhóm 50–199 có 3 tranh và 3 ngôn ngữ, nhóm ≥200 có 1.266 tranh và 19 ngôn ngữ. Do đó chưa có bằng chứng đáng tin rằng graph cải thiện Contrast cho nhóm ít support. Trên 21 ngôn ngữ đủ điều kiện ở toàn P2, macro ECE 10 bin và squared error trên tỷ lệ phiếu lần lượt là **.029853 / .052000** (E50+L), **.028202 / .051684** (ridge), **.027554 / .051639** (graph). Tỷ trọng dự đoán ≤.1 hoặc ≥.9 lần lượt là 14.58%, 15.22%, 15.26%. Candidate có calibration chẩn đoán tốt hơn nhẹ dù dự đoán cực đoan hơn một chút. Chênh lệch LL phù hợp với xu hướng này.

## Thực thi và xác minh

- Chạy CPU trên Windows với i7-13620H, giới hạn hai luồng, Python 3.12.10, NumPy 2.2.6, SciPy 1.15.3 và pandas 2.2.3. Tổng wall **604.71 giây**, peak RSS **227.21 MiB**. Fitting-only graph dùng **31.78 s**, solver inner **355.77 s**, solver outer **183.02 s**. Phần còn lại gồm kiểm đầu vào, nội suy dự đoán, chọn cấu hình, lưu/kiểm kết quả và scorer.
- Cả **225 solver receipts** hội tụ, gồm 45 nuisance, 150 inner và 30 outer. Iteration 89–336, gradient infinity norm cuối lớn nhất **3.145e−7**. CPU replay độc lập đọc hệ số lưu, tái lập đúng cả 27.200 dự đoán của hai hệ thống fit, và chấm lại pooled và từng fold khớp tới 1e−12. E50+L OOF khớp đúng từng khóa và xác suất với input lịch sử.
- Mã chạy **61ee4be9d629533cc5b03ef6bd8f925ac04dcc1e**, config SHA-256 `d244b95fa3521632f67df5370b28a278b64539e556c1bf0a1206c60acb91059b`. Attempt đầu dừng ở graph fold 1 do vector case-mix chưa được lọc về cùng các tranh có cả hai ngôn ngữ. Mã sửa dùng chung mask và có regression fixture với cạnh đủ support nhưng vẫn thiếu tranh. Attempt hoàn tất dùng thư mục mới với cùng config khoa học. Giữ log và manifest của cả hai attempt.
- Artifact đã đối chiếu đủ thành viên và checksum. `runs/exp032/exp032_p2_artifacts.zip` có SHA-256 `d5c5cad843d05ec4fde5b7adaf976107f8944e7c8da515fcd113baa543e4291f`. Gồm OOF CSV, năm graph receipts, coefficients, seals, diagnostics và replay. `metrics.json` SHA-256 `7f776b696ab2e5d53574dda18bcd8779df6ab36ea1c807308063dba273a6cc6d`. [Research PR #39](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/pull/39)

## Quyết định

Kết thúc thử nghiệm graph tại đây theo gate đã chốt. So với ridge, mức tăng +0.000564 có dấu dương nhất quán ở bốn fold nhưng không đủ biên độ thực dụng. Đối chứng ridge đã mang lại +0.004604 trên cùng P2 và giải thích khoảng 89% mức tăng quan sát được của candidate. So sánh đó đánh giá gói tham số hóa và mức phạt. Nó chưa tách riêng đóng góp của mỗi intercept hoặc slope. Có thể giữ đối chứng làm mốc nghiên cứu cho hướng dự phòng SigLIP2 NaFlex spatial tokens khi được duyệt riêng. Kết quả hiện tại chỉ dùng P2, giữ nguyên production và trạng thái PR #37.
