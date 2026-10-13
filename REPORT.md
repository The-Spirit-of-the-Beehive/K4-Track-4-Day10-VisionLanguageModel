Họ tên: <điền họ tên>
`dataset_version`: <điền, lấy từ data/splits/meta.json>

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | | | | | | | | |
| E1 | LoRA LLM | | | | | | | | | |
| E2 | + projector | | | | | | | | | |
| E3 | + LoRA ViT | | | | | | | | | |
| E4 | (ghi biến và giá trị) | | | | | | | | | |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
<điền>

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

**Vì sao Qwen3-VL dùng thang 0–1000 chứ không dùng pixel tuyệt đối?**

1. **Độc lập với độ phân giải và tỉ lệ khung hình (Resolution-agnostic):** 
   - Đầu vào của Vision Transformer (ViT) thường được resize hoặc chia thành các dynamic patches (ví dụ patch $14 \times 14$ hoặc $28 \times 28$). 
   - Khi dùng thang chuẩn hóa $[0, 1000]$, mô hình học được mối quan hệ vị trí tương đối độc lập với kích thước pixel thực tế (dù ảnh là $640 \times 360$, $1920 \times 1080$ hay $4K$ thì góc trên-trái vẫn là $(0, 0)$ và góc dưới-phải vẫn là $(1000, 1000)$).
2. **Không gian token số gọn và phân phối hữu hạn:**
   - Các số nguyên từ $0$ đến $1000$ là một tập số hữu hạn, cố định và phân bổ đều trong tokenizer của LLM. LLM dễ học phân phối xác suất và tự hồi quy (autoregression) trên tập số quen thuộc này hơn rất nhiều so với toạ độ pixel có thể biến thiên từ vài chục đến hàng nghìn đơn vị tuỳ ảnh.
3. **Kế thừa tri thức Pre-training (Visual Grounding Prior):**
   - Dòng mô hình Qwen-VL (Qwen2-VL, Qwen3-VL) được pre-train trên hàng triệu cặp grounding đa nhiệm theo quy ước chuẩn $[0, 1000]$. Giữ nguyên định dạng này giúp tận dụng tối đa năng lực định vị không gian sẵn có của pre-trained model.

**Nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao?**
- Nếu model sinh ra toạ độ trên thang $[0, 1000]$ mà tool chấm điểm (`score.py`) tính theo pixel (hoặc ngược lại, model trả pixel $0-640$ mà tool coi là $0-1000$), thì diện tích giao nhau IoU giữa box dự đoán và nhãn ground-truth sẽ gần như bằng **0**. 
- Kết quả là $F1 \approx 0$ ngay cả khi JSON hoàn toàn hợp lệ.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
<điền>

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
<điền>

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?

* **Vì sao phải chia theo `session` mà không chia ngẫu nhiên?**
  * Dữ liệu dashcam gồm các chuỗi khung hình liên tiếp trích xuất từ video hành trình. Các frame liền kề trong cùng một video (cùng `session`) có góc quay, điều kiện thời tiết, ánh sáng và hư hại mặt đường gần như giống hệt nhau.
  * Chia theo `session` (chia theo từng chuyến đi) đảm bảo tính tổng quát hóa (generalization) trong thực tế: model phải dự đoán trên những đoạn đường và điều kiện quay hoàn toàn mới mà nó chưa từng thấy khi huấn luyện.
* **Điều gì xảy ra với F1 nếu chia ngẫu nhiên?**
  * Sẽ xảy ra hiện tượng **rò rỉ dữ liệu (Data Leakage)**: các khung hình gần như giống nhau sẽ nằm đồng thời ở cả tập `train` và `test`.
  * Khi đó mô hình chỉ cần "học vẹt" (ghi nhớ) đặc trưng bối cảnh của video là có thể đoán đúng, dẫn đến **F1 cao giả tạo (optimistic bias)**. Nhưng khi triển khai thực tế trên cung đường mới, F1 sẽ tụt dốc nghiêm trọng.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
<điền>

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
<điền>

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
<điền>

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
<điền>

## Nhìn lại

<điền: điều bạn chưa hiểu, lỗi dữ liệu hoặc mô hình bạn gặp mà công cụ chấm chưa bắt được>
