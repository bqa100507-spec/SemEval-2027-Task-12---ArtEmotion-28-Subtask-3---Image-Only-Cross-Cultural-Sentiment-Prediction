# EXP-030 · Phân tích lỗi Qwen2.5-VL-7B trên P2 fold 3

**Epoch 3 đạt đỉnh Contrast quan sát được ở 0.419801, thấp hơn E50+L 0.013735.** Chênh lệch đến từ cả thứ tự và độ lớn tương đối của chênh lệch sentiment giữa các ngôn ngữ. Epoch 4 xếp hạng kém hơn dù BCE held-out hơi tốt hơn. Epoch 5 vẫn xếp hạng kém hơn epoch 3 và cho xác suất cực đoan hơn, với Log Loss cao hơn. Các khoảng bất định của ba chênh lệch Contrast này đều chứa 0.

## Dữ liệu và cách tính

Kiểm tra cả năm seal cùng source `38fb289158798e03fa34d9792c699d63372cdf0a`, config, lock và SHA-256 của từng prediction CSV và epoch JSON. Tái lập đúng 2.721 khóa held-out từ handoff ghim và khớp lại cả hai điểm chính với biên nhận mỗi epoch trong sai số tuyệt đối `1e-12`. Contrast và các chỉ số xếp hạng dùng cùng 253/267 tranh đủ điều kiện. Macro Log Loss dùng 21 ngôn ngữ có ít nhất 20 cặp. Không tải hay chạy checkpoint trọng số. Chỉ dùng P2 fold 3 và votes của phần fitting để tính prior. [Artifacts Drive](https://drive.google.com/drive/folders/1NOL45hyMA-_MXs8MWyStQajs6CedBvIp) · [Quy tắc chấm cục bộ](../docs/competition.md)

Pearson và Spearman được tính theo tranh rồi lấy trung bình đều. Spearman dùng hạng trung bình cho giá trị bằng nhau. Concordance so thứ tự trên 9.249 cặp ngôn ngữ có tỷ lệ phiếu khác nhau, cho 0.5 điểm nếu dự đoán bằng nhau, rồi lấy trung bình đều theo tranh. BCE trong bảng tính theo số phiếu, đơn vị nats. Macro Log Loss tính theo ngôn ngữ, đơn vị bits.

| Mô hình | Pearson | Spearman | Concordance | Macro Log Loss | BCE held-out | Loss fitting |
|---|---:|---:|---:|---:|---:|---:|
| Prior ngôn ngữ | 0.332541 | 0.304699 | 65.13% | 0.885453 | — | — |
| E50+L | 0.433536 | 0.387925 | 69.36% | 0.713192 | — | — |
| Qwen 1 | 0.348525 | 0.331927 | 66.82% | 0.780688 | 0.541579 | 1.967737 |
| Qwen 2 | 0.376081 | 0.344539 | 67.21% | 0.740999 | 0.514882 | 1.837122 |
| Qwen 3 | 0.419801 | 0.373590 | 68.98% | 0.733054 | 0.510588 | 1.788469 |
| Qwen 4 | 0.408526 | 0.364938 | 68.41% | 0.733676 | 0.510156 | 1.737421 |
| Qwen 5 | 0.409218 | 0.360110 | 68.17% | 0.749023 | 0.521552 | 1.696709 |

## Epoch 3 thua ở đâu?

So với E50+L, Qwen epoch 3 có Pearson thấp hơn trên 136 tranh, trong đó **94** tranh cũng giảm concordance, còn **42** tranh có concordance bằng hoặc cao hơn. Nhóm thứ nhất có lỗi thứ tự đo được. Nhóm thứ hai cho thấy đếm số cặp đúng chưa giải thích hết Pearson, vốn còn nhạy với độ lớn tương đối của chênh lệch và vị trí của từng cặp sai. Qwen có 38 tranh Pearson âm, E50+L có 24. Năm tranh thua đậm nhất đóng góp −0.010123 vào trung bình chung, còn năm tranh Qwen thắng đậm bù +0.009363.

| Tranh | Pearson E50+L → Qwen 3 | Quan sát |
|---|---:|---|
| Surikov, *Alupka, Ai-Petri* | 0.572 → −0.052 | Qwen dự đoán Kyrgyz 0.912 dù gold là 0/5, xếp cao hơn Yoruba 0.742 dù gold là 5/5. E50+L cũng đánh giá Kyrgyz quá cao ở 0.727, nhưng xếp thấp hơn Yoruba 0.779. |
| Anne Truitt, *Untitled* (1967) | 0.802 → 0.228 | Concordance giảm từ 77.8% xuống 55.6%. |
| Gyzis, *Portrait of Artemis Gyzi* | 0.277 → −0.199 | Concordance giảm từ 59.4% xuống 37.5%. |
| Modersohn-Becker, *Portrait of a Woman with Pitcher* | −0.480 → 0.248 | Qwen tăng concordance từ 28.6% lên 66.7% dù hạ quá mạnh xác suất tích cực ở vài ngôn ngữ, như Tamil 0.399 với gold 6/6. |

| Chênh lệch Contrast | Δ | CI 95% theo tranh | Thắng / thua |
|---|---:|---:|---:|
| Qwen 3 − E50+L | −0.013735 | [−0.033822, +0.007307] | 117 / 136 |
| Qwen 4 − Qwen 3 | −0.011275 | [−0.032386, +0.009701] | 110 / 143 |
| Qwen 5 − Qwen 3 | −0.010583 | [−0.028554, +0.007866] | 117 / 136 |
| Qwen 3 − prior | +0.087261 | [+0.060439, +0.114548] | 167 / 86 |

Bootstrap percentile ghép cặp 10.000 lượt, seed 203003, mỗi lượt lấy lại 253 tranh với hoàn lại. Đây là bất định mô tả trên fold đã tái sử dụng và đã chọn epoch, không hiệu chỉnh cho việc chọn epoch hay phản ánh biến thiên giữa các lần train.

## Prior và tương tác ảnh–ngôn ngữ

Prior mỗi ngôn ngữ là `(Σk + 0.5)/(Σn + 1)`, ước lượng từ **10.879 cặp fitting**. Để đo phần phi cộng tính, phân rã **logit dự đoán** trên các khóa quan sát thành hiệu ứng tranh + hiệu ứng ngôn ngữ + phần dư. Phép chiếu này không dùng gold held-out. Để so hai mô hình với cùng mốc, thay hiệu ứng ngôn ngữ trong logit bằng logit prior train, bù một hằng số chung để giữ trung bình logit. Đây là phép chẩn đoán đầu ra trên tập khóa hiện có, không phải kết quả một mô hình được huấn luyện lại.

| Mô hình | Tương quan hiệu ứng ngôn ngữ với prior train¹ | % biến thiên logit nội tranh ở phần dư | Contrast cộng tính | Contrast với cùng prior: bỏ → giữ phần dư | Pearson nội tranh sau trừ prior² |
|---|---:|---:|---:|---:|---:|
| E50+L | .991 | 31.4% | .3433 | .3338 → .4270 | .3486 |
| Qwen 3 | .945 | 37.0% | .3339 | .3328 → .4146 | .3190 |
| Qwen 4 | .914 | 40.5% | .3310 | .3336 → .4088 | .3078 |
| Qwen 5 | .904 | 39.3% | .3168 | .3337 → .4037 | .3041 |

¹ Tính trên các khóa quan sát nên ngôn ngữ xuất hiện nhiều có trọng số cao hơn.

² Trừ cùng một prior xác suất khỏi tỷ lệ phiếu và dự đoán, sau đó tính Pearson theo tranh.

Qwen epoch 3 vượt rõ prior đơn thuần trên fold này. Khi giữ cùng prior, phần dư vẫn nâng Contrast của cả hai mô hình, nhưng E50+L giữ lợi thế quan sát được. Qwen có phần dư lớn hơn và kém khớp với biến thiên sentiment sau khi trừ prior. Riêng epoch 4, phần dư còn lớn hơn trong khi độ khớp giảm. Dữ liệu ủng hộ vấn đề **chất lượng điều chỉnh theo tranh**, thay vì thiếu biên độ tương tác. Phân rã không xác định được Qwen có thật sự dựa vào vùng ảnh cụ thể hoặc prompt theo cơ chế nào.

## Epoch 4–5: thứ tự và xác suất

| Mô hình | Bias xác suất¹ | ECE 10 bins¹ | Sai số bình phương¹ | Tỷ trọng `p ≤ .1` hoặc `p ≥ .9`¹ |
|---|---:|---:|---:|---:|
| E50+L | −.0047 | .0124 | .0491 | 15.7% |
| Qwen 3 | +.0087 | .0171 | .0543 | 21.5% |
| Qwen 4 | −.0253 | .0395 | .0543 | 23.5% |
| Qwen 5 | −.0173 | .0595 | .0579 | 29.9% |

¹ Bình quân đều giữa 21 ngôn ngữ đủ chuẩn, rồi đều giữa các cặp trong từng ngôn ngữ. Bias = dự đoán − tỷ lệ phiếu. ECE dùng 10 khoảng đều và tỷ lệ phiếu làm tần suất quan sát. Đây là chẩn đoán phụ thuộc cách chia bin và nhiễu phiếu, không phải phép tách nhân quả của Log Loss.

- **Epoch 4:** Concordance giảm 0.56 điểm phần trăm và Spearman giảm .00865, trong khi BCE tốt hơn .00043 nats và Macro Log Loss chỉ tăng .00062 bits. Có sai thứ tự mới dù hai điểm xác suất tổng hợp gần như giữ nguyên. Ví dụ *Columbine in Green*, gold Swahili 5/5 và Korean 2/5. Qwen 3 xếp .830 trên .685, Qwen 4 đảo thành .694 dưới .751.
- **Epoch 5:** Concordance giảm 0.81 điểm phần trăm so với epoch 3. Macro Log Loss tăng .01597 bits và xấu đi ở 14/21 ngôn ngữ. Trong nhóm `p < .1`, dự đoán trung bình .052 nhưng tỷ lệ tích cực quan sát .160. Trong nhóm `p ≥ .9`, hai giá trị là .949 và .923. Ở *Saint Anthony Abbot and Saint Bernardino da Siena*, Tagalog 4/5 và Kyrgyz 0/5 được dự đoán gần ngang nhau ở epoch 3 (.745 và .748), rồi cách xa hơn theo hướng sai ở epoch 5 (.678 và .815).
- Loss fitting tiếp tục giảm từ 1.78847 xuống 1.69671, trong khi BCE held-out tăng ở epoch 5. Đường cong phù hợp với overfitting muộn trên fold này. Một phép hiệu chỉnh đơn điệu chung có thể đổi Log Loss hoặc Pearson, nhưng giữ nguyên Spearman và thứ tự cặp nếu không tạo tie.

## Mức độ kết luận và một recipe EXP-031

**Đã xác minh:** Epoch 3 là tốt nhất trong năm epoch lưu theo Contrast. Qwen vượt prior train nhưng kém E50+L theo cả ba chỉ số tương quan và xếp hạng quan sát. Epoch 4 đã mất thứ tự và epoch 5 còn tăng xác suất cực đoan, với sai lệch bin và Log Loss lớn hơn. **Được dữ liệu ủng hộ:** Phần tương tác của Qwen có ích nhưng kém đồng hướng với thay đổi sentiment thực tế hơn E50+L. Các cập nhật muộn có thể khuếch đại cả chênh lệch sai lẫn độ tự tin sai. **Chưa xác định:** Mức chênh lệch có tái lập trên tranh độc lập không, và liệu nguyên nhân nằm ở decoder, cách dùng prompt, objective hay nhiễu phiếu. Không thể đo ảnh hưởng nhân quả của pixel hoặc kết luận về phân bố tám cảm xúc từ CSV chỉ chứa `p_positive`.

**Đề xuất duy nhất:** Huấn luyện Qwen 7B native vision trên toàn phần fitting với **đầu dự đoán phân tầng có neo prior**. Binary logit gồm mức sentiment chung của tranh, logit prior ngôn ngữ tính trên fitting, và một tương tác ảnh–ngôn ngữ low-rank được regularize. Học phân bố bốn cảm xúc có điều kiện trong mỗi cực, rồi nhân với khối lượng sentiment tương ứng để khôi phục tám xác suất. Giữ count-weighted CE + BCE, frozen vision và rank-16 QLoRA. Gom theo tranh, giới hạn độ lớn của nhánh tương tác bằng phạt L2 và giữ nó khởi tạo gần 0. Chốt hệ số phạt và cách chọn checkpoint trên split nội bộ theo tranh trước khi xem kết quả đánh giá. Động cơ là giữ prior có ích và hạn chế chỉnh sai quá mạnh, đồng thời vẫn học được các ngoại lệ theo tranh. Việc thay prior hậu nghiệm không cứu Contrast của epoch 3, nên recipe này là giả thuyết huấn luyện có căn cứ để kiểm tra, chưa có bằng chứng sẽ vượt E50+L.
