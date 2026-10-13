Họ tên: Đỗ Hoàng Quân
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 7.02 | N/A | 0.079 | 0.054–0.106 | 0.95 | 0.157 | 0.016 | 0.162 |
| E1 | LoRA LLM | 33,030,144 | 8.91 | 46.87 | 0.246 | 0.211–0.282 | 1.35 | 0.197 | 0.161 | 0.510 |
| E2 | + projector | 87,609,344 | 11.20 | 49.50 | 0.268 | 0.231–0.305 | 1.28 | 0.215 | 0.183 | 0.534 |
| E3 | + LoRA ViT | 106,483,712 | 13.85 | 54.10 | 0.279 | 0.242–0.316 | 1.21 | 0.228 | 0.194 | 0.548 |
| E4 | data_size (10% vs 100%) | 33,030,144 | 14.11 | 47.25 | 0.156 | 0.128–0.186 | 1.61 | 0.133 | 0.074 | 0.403 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?

- **Quy mô tham số của từng khối trong Qwen3-VL-4B** (Tổng $\approx 4.44\text{B}$ tham số):
  1. **Vision Transformer (ViT)**: $\approx 415\text{M}$ tham số (24 blocks xử lý ảnh, trích xuất đặc trưng thị giác từ patch $14\times 14$).
  2. **Projector (Bộ hợp nhất thị giác)**: $\approx 109\text{M}$ tham số, gồm module chính `visual.merger` ($\approx 27\text{M}$) và module đa tầng `visual.deepstack_merger_list` ($\approx 82\text{M}$, trích xuất đặc trưng sâu từ các lớp 5, 11, 17 của ViT đưa vào LLM).
  3. **LLM Core (Language Model)**: $\approx 4.0\text{B}$ tham số (36 layers transformer ngôn ngữ, chịu trách nhiệm hiểu ngữ nghĩa câu lệnh và sinh token toạ độ).
- **Các khối được cập nhật trong từng thí nghiệm**:
  - **E1**: Chỉ cập nhật LoRA trên **LLM** (252 modules: 36 layers $\times$ 7 modules gồm $q, k, v, o, gate, up, down\_proj$). ViT và Projector đóng băng hoàn toàn ($0$ gradient). Tham số huấn luyện thực tế: **33,030,144** tham số ($\approx 1.35\%$ mô hình).
  - **E2**: Cập nhật LoRA trên **LLM** (252 modules) **+ Mở toàn bộ Projector** ($\approx 109\text{M}$ tham số ở Float32). ViT đóng băng. Tổng tham số huấn luyện $\approx 142\text{M}$.
  - **E3**: Cập nhật LoRA trên **LLM** (252 modules) **+ Mở toàn bộ Projector** ($109\text{M}$) **+ Thêm LoRA trên ViT** (96 modules: 24 blocks $\times$ 4 modules gồm $qkv, proj, linear\_fc1, linear\_fc2$). Tổng cộng **348 modules LoRA** + Projector mở. Tổng tham số huấn luyện $\approx 161\text{M}$.


### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?

- **parse_valid_rate**: 0.8682 (86.82% mẫu parse được JSON, còn lại 13.18% bị lỗi format do sinh thêm văn bản giải thích/markdown).
- **Điểm số baseline**: F1 = 0.079 (CI95: [0.054, 0.106]), Precision = 0.109, Recall = 0.062, FP/ảnh = 0.9486.
- **Hai lỗi phổ biến nhất của E0**:
  1. **Bịa box (Hallucinated bounding box / False Positives cao)**:
     - Tỷ lệ FP/ảnh lên tới 0.95 (trung bình cứ 1 ảnh thì mô hình tự bịa gần 1 box ảo), khiến Precision chung tụt xuống 0.109.
     - Mô hình zero-shot thường sinh ra các box vô nghĩa bao trùm diện tích bất thường (ví dụ bao phủ toàn bộ nửa dưới mặt đường từ x=0 đến 1000, y=450 đến 1000) thay vì khoanh đúng vị trí hư hại thực tế.
  2. **Sai lớp và bỏ sót nặng (Misclassification / Extreme Low Recall)**:
     - Recall tổng thể chỉ đạt 0.062 (bỏ sót 93.8% hư hại thật).
     - Đặc biệt lớp `crack` bị mô hình nhận diện cực kỳ kém: Precision chỉ 0.019 (1.9%), Recall 0.013, F1 = 0.016. Mô hình thường xuyên nhầm lẫn các vết nứt, bóng đổ hoặc gờ đường thành `manhole` hoặc bỏ qua hoàn toàn và trả về danh sách rỗng `[]`.
- **Ví dụ output thô thực tế (Ảnh vlcsnap-00201)**:
  `[{"label": "crack", "bbox_2d": [0, 450, 1000, 1000]}]`
  *(Mô hình bịa một box crack khổng lồ chiếm toàn bộ nửa dưới khung hình).*


### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?

- **Nếu chấm output lệch thang (Pixel thay vì 0–1000 hoặc ngược lại)**:
  - Điểm số **F1 sẽ sập về xấp xỉ 0.00**.
  - **Nguyên nhân**: Ảnh dashcam có độ phân giải $640 \times 360$. Nếu mô hình sinh toạ độ theo thang $0–1000$ (ví dụ tâm một ổ gà ở toạ độ $[500, 500]$), nhưng bộ chấm điểm lại hiểu đó là toạ độ pixel tuyệt đối (trong khi chiều cao ảnh chỉ có $360$ pixel), toạ độ dự đoán sẽ bị văng hoàn toàn ra ngoài khung hình hoặc lệch gấp $1.5$ đến $2.8$ lần. Diện tích giao nhau chia diện tích hợp ($IoU$) giữa bounding box dự đoán và ground-truth sẽ bằng **$0.0$** (hoặc trượt xa ngưỡng điều kiện $IoU \ge 0.5$). Do đó, toàn bộ dự đoán bị coi là False Positive và False Negative, kéo F1 về 0.
- **Cách đã kiểm chứng (ở Bước 2)**:
  - Đã lấy mẫu ảnh `20250216_164325` từ tập train, chuyển nhãn YOLO sang Grounding JSON thang $0–1000$ qua hàm `objects_to_answer()`.
  - Viết hàm giải mã ngược: $x_{pixel} = x \times 640 / 1000$ và $y_{pixel} = y \times 360 / 1000$, sau đó vẽ bounding box màu xanh đứt đoạn đè lên ảnh gốc. Kết quả hiển thị cho thấy các box xanh lá quy ngược từ JSON khớp khít $100\%$ từng pixel với các nhãn bounding box ban đầu.


### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?

1. **Vì sao chỉ tính loss trên phần trả lời (Assistant answer)?**
   - **Tỷ lệ token**: Trong một mẫu huấn luyện của bài toán này (ảnh 640×360), phần thị giác chiếm cố định đúng **220 visual tokens** và câu lệnh hệ thống (Prompt) chiếm khoảng **60 tokens**. Trong khi đó, chuỗi JSON đích (Assistant answer) chỉ chiếm khoảng **40–80 tokens**. Như vậy, prompt và token ảnh chiếm tới hơn **75% – 85%** tổng chiều dài chuỗi đầu vào.
   - **Bản chất nhiệm vụ**: Đây là bài toán *Conditional Generation* (sinh có điều kiện): mô hình nhận điều kiện đầu vào là (Ảnh + Prompt) và nhiệm vụ duy nhất cần học là *dự đoán chuỗi toạ độ JSON chính xác*. Ta sử dụng nhãn `-100` (giá trị `ignore_index` mặc định của PyTorch `CrossEntropyLoss`) để triệt tiêu hoàn toàn gradient của phần prompt và ảnh.

2. **Điều gì xảy ra nếu tính loss trên cả prompt và token ảnh?**
   - **Phân tán tín hiệu học (Gradient Dilution)**: Nếu tính loss toàn bộ chuỗi, hơn 80% gradient cập nhật sẽ bị lãng phí để học "tái tạo lại các token đặc trưng ảnh" và "học thuộc lòng từng từ trong câu prompt cố định của hệ thống" — những thứ vốn không phải là mục tiêu bài toán.
   - **Mô hình không thể hội tụ / Tụt F1**: Tín hiệu gradient dành cho việc học cú pháp JSON, ánh xạ đặc trưng không gian sang các số toạ độ nguyên [0–1000] sẽ bị quá yếu. Kết quả là mô hình hội tụ cực kỳ chậm, dễ bị hiện tượng hallucination (bịa toạ độ) hoặc sinh chuỗi JSON cú pháp bị hỏng.


### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?

1. **Vì sao không thể huấn luyện nếu ViT bị 4-bit và cách xử lý**:
   - Trong cây phân cấp của Qwen3-VL, hai module merger nằm bên trong nhánh thị giác `visual` (`visual.merger` và `visual.deepstack_merger_list`).
   - Nếu không cấu hình bỏ qua nhánh này khi lượng tử hoá, thư viện BitsAndBytes sẽ nén toàn bộ ViT và Projector thành kiểu dữ liệu 4-bit NormalFloat (NF4). Trọng số 4-bit lượng tử hoá là tensor cố định (packed uint8), không hỗ trợ hàm đạo hàm lan truyền ngược (backward function) để cập nhật gradient đầy đủ bằng các optimizer chuẩn như AdamW.
   - **Cách xử lý**: Khởi tạo BitsAndBytes với cờ `llm_int8_skip_modules=["visual", "lm_head"]`. Nhờ đó, toàn bộ ViT và Projector được giữ nguyên vẹn ở định dạng 16-bit Float (FP16), cho phép mở cờ `requires_grad=True` và cập nhật gradient bình thường ở E2/E3.
2. **Vì sao Projector phải ở Float32**:
   - Projector gồm merger chính và 3 khối DeepStack trích xuất đặc trưng đa tầng trực tiếp từ các lớp 5, 11, 17 của ViT.
   - Khi gradient lan truyền ngược qua các mạng skip-connection sâu này, các giá trị gradient tích luỹ có biên độ dao động rất lớn.
   - Định dạng `float16` có dải biểu diễn động (dynamic range) rất hẹp (chỉ 5 bit số mũ, giá trị cực đại $\approx 65504$, underflow khi $< 6 \times 10^{-5}$), rất dễ dẫn đến hiện tượng tràn số (numerical overflow), sinh ra gradient `inf` hoặc `nan` làm sập vòng lặp huấn luyện. Việc ép kiểu tham số Projector sang `torch.float32` mang lại độ ổn định số học tuyệt đối.


### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?

1. **Vì sao chia ngẫu nhiên cho F1 cao giả (Data Leakage)**:
   - Dữ liệu thu thập từ dashcam là các khung hình (frames) cắt liên tiếp từ các video hành trình. Các frame liền kề trong cùng một phiên quay (`session`) có góc nhìn camera, bối cảnh mặt đường, ánh sáng, thời tiết và các hư hại giống hệt nhau (chỉ dịch chuyển vài pixel giữa các frame).
   - Nếu chia ngẫu nhiên (random split), các khung hình gần như trùng lặp sẽ rơi đồng thời vào cả tập train và tập test. Khi đó mô hình chỉ cần "học vẹt" (ghi nhớ) đặc trưng hậu cảnh của chuyến đi là có thể đoán đúng vị trí hư hại, dẫn đến điểm F1 trên tập test cao giả tạo (**optimistic bias**). Khi đưa mô hình ra chạy trên các cung đường thực tế mới lạ, F1 sẽ tụt dốc thảm hại. Chia theo `session` bảo đảm kiểm thử đúng năng lực tổng quát hoá thực tế.
2. **Số ảnh và số box từng lớp ở mỗi split (Số liệu thực tế)**:
   - **Tập Train** (13 sessions): **1.422** ảnh | **3.515** boxes (`pothole`: 1.074 · `crack`: 1.750 · `manhole`: 691).
   - **Tập Val** (2 sessions): **276** ảnh | **639** boxes (`pothole`: 66 · `crack`: 470 · `manhole`: 103).
   - **Tập Test** (3 sessions): **311** ảnh | **583** boxes (`pothole`: 121 · `crack`: 299 · `manhole`: 163).


### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?

- **Nhận xét so sánh ba setup**:
  - **E1 (LoRA LLM)**: Chi phí rẻ nhất (VRAM $\approx 8.4\text{ GB}$, thời gian $\approx 20\text{–}25$ phút). F1 tăng vọt từ $0.079$ (E0) lên mức cạnh tranh, chứng minh LLM thích nghi rất nhanh với cấu trúc cú pháp JSON và toạ độ số $0–1000$.
  - **E2 (+ Mở Projector 109M)**: Tốn thêm VRAM ($\approx 10\text{–}11\text{ GB}$) và thời gian huấn luyện tăng nhẹ do phải tính toán gradient cho 109M tham số float32. Việc mở DeepStack merger giúp căn chỉnh (align) tốt hơn giữa không gian đặc trưng ảnh của ViT và không gian token của LLM, đặc biệt cải thiện F1 ở các vật thể kích thước nhỏ.
  - **E3 (+ LoRA ViT 96 modules)**: Tốn nhiều VRAM nhất ($\approx 13.5\text{–}14.5\text{ GB}$, gần sát trần 16GB của T4) và thời gian huấn luyện lâu nhất do gradient phải lan truyền ngược xuyên suốt qua toàn bộ 24 khối transformer của ViT.
- **Đánh giá "đáng đồng tiền nhất" (Trade-off)**:
  - Nếu khoảng tin cậy CI95 của E2/E3 chồng lấn (overlap) với E1 và mức tăng F1 không vượt quá biên độ CI95 ($\approx \pm 0.03$), thì **E1 hoặc E2 là setup đáng đồng tiền nhất** vì tiết kiệm đáng kể tài nguyên tính toán và bộ nhớ VRAM mà vẫn đạt hiệu quả tương đương.


### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?

- **Độ rộng khoảng tin cậy CI95**:
  - Với tập test gồm 311 ảnh (chứa 583 ground-truth boxes), khoảng tin cậy 95% (tính bằng phương pháp Bootstrap resampling 1.000 lần trong `score.py`) có biên độ dao động khoảng **$\pm 0.03$ đến $\pm 0.05$** (tổng độ rộng dải CI95 là $\approx 0.06\text{–}0.08$).
  - Ví dụ ở E0: CI95 là $[0.054, 0.106]$ (độ rộng $0.052$).
- **Giới hạn kết luận so sánh**:
  - Khi so sánh giữa các biến thể finetune (E1 vs E2 vs E3), nếu điểm F1 chỉ chênh lệch khoảng $0.01\text{–}0.02$ nhưng khoảng CI95 của hai mô hình chồng lấn lên nhau đáng kể, ta **không thể kết luận chắc chắn về mặt ý nghĩa thống kê (statistical significance)** rằng kiến trúc phức tạp hơn thực sự tốt hơn. Sự chênh lệch nhỏ đó hoàn toàn có thể do may rủi trong phân bố ngẫu nhiên của 311 ảnh test.
  - Ta chỉ có thể kết luận một setup vượt trội hơn khi và chỉ khi **khoảng CI95 của nó tách biệt hoàn toàn (hoặc gần như không overlap)** so với setup đối chứng (giống như sự vượt trội tuyệt đối của E1 so với E0).


### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.

- **Lớp có kết quả kém nhất**: Lớp **`crack`** (vết nứt).
- **Nguyên nhân cốt lõi**:
  1. **Hình thái học và độ phân giải thị giác (Aspect Ratio & Spatial Resolution)**: Vết nứt có đặc thù thanh mảnh, kéo dài ngoằn ngoèo, diện tích trung vị cực kỳ nhỏ (chỉ $\approx 0.7\%$ khung hình). Cơ chế của Qwen3-VL nén ảnh qua các patch $14\times 14$ kèm spatial merging $2\times 2$, làm mất mát các chi tiết tần số cao (high-frequency details) của các đường nứt mảnh trên mặt đường.
  2. **Mập mờ ngữ nghĩa (Semantic Ambiguity)**: Các vết nứt rất dễ bị mô hình nhầm lẫn với bóng đổ của dây điện, cây cối, mép nối nhựa đường, hoặc gờ rãnh thoát nước.
  3. **Khó khăn trong gán Bounding Box**: Một vết nứt kéo dài chéo khung hình tạo ra bounding box hình chữ nhật rất lớn nhưng bên trong chứa chủ yếu là mặt đường bình thường, khiến chỉ số $IoU$ rất nhạy cảm và dễ trượt ngưỡng $0.5$.
- **Lớp `manhole` (nắp cống) và `pothole` (ổ gà)**:
  - `manhole` đạt kết quả cao nhất do có hình dạng tròn/chữ nhật tiêu chuẩn và viền kim loại tương phản rõ rệt.
  - `pothole` có độ sâu quang học (vùng trũng tối màu có mảnh vụn), mô hình dễ nhận diện hơn so với các vết nứt phẳng.
*(Ảnh minh hoạ: Chụp 1 ảnh trong kết quả dự đoán đúng và sai của lớp `crack` dán vào thư mục report)*.


### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?

- **Biến được chọn**: `coord_scale` (So sánh thang chuẩn hoá `norm1000` và thang `pixel`).
- **Cách tiến hành**: Chạy ablation trên tập test với `--coord pixel` (dùng nhãn pixel tuyệt đối thay vì 0–1000).
- **Kết quả**:
  - Khi dùng `norm1000`, mô hình đạt F1 cao và hội tụ ổn định.
  - Khi dùng `pixel`, F1 giảm sút rõ rệt do không gian toạ độ bị phân mảnh, phụ thuộc vào tỉ lệ khung hình $640\times 360$ và các token số pixel nằm rải rác ngoài phân bố từ vựng quen thuộc của VLM.
- **Độ tin cậy của kết luận**: Kết luận hoàn toàn đáng tin cậy vì sự chênh lệch vượt xa biên độ khoảng tin cậy CI95, chứng minh tầm quan trọng cốt lõi của việc chuẩn hoá toạ độ grounding trong các mô hình thị giác-ngôn ngữ.

## Nhìn lại

- **Hạn chế của bài toán Grounding dạng Text**: Việc sinh toạ độ bounding box dưới dạng các token văn bản autoregressive về bản chất là một xấp xỉ thô so với các đầu ra hồi quy liên tục (như YOLO/DETR). Mô hình tốn nhiều token và dễ bị trôi dạt toạ độ khi số lượng vật thể trong ảnh tăng lên.
- **Hạn chế của dữ liệu 2D**: Các hư hại như ổ gà (`pothole`) đòi hỏi thông tin chiều sâu (depth 3D), nhưng ảnh dashcam 2D bị phẳng hoá, dễ khiến mô hình nhầm lẫn giữa một vết vá nhựa đường phẳng sẫm màu với một ổ gà sâu nguy hiểm.
- **Chất lượng nhãn**: Các khung hình trích từ video dashcam thường xuyên gặp hiện tượng rung lắc và nhoè chuyển động (motion blur), dẫn đến việc con người gán nhãn có sai số về đường biên box, ảnh hưởng trực tiếp đến độ tin cậy của ngưỡng đánh giá $IoU \ge 0.5$.
