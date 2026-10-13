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
<điền>

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
<điền>

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
