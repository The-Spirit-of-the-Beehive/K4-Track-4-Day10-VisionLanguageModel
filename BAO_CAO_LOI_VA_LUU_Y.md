---

### PHẦN 4: NỘI DUNG FILE `BAO_CAO_LOI_VA_LUU_Y.md`

Tạo file `BAO_CAO_LOI_VA_LUU_Y.md` ở thư mục gốc và dán nội dung sau:

```markdown
# Báo cáo Phân tích Lỗi Kỹ thuật và Bài học Kinh nghiệm (Lab Day 10)

Trong quá trình triển khai finetune mô hình Vision-Language Model lớn (Qwen3-VL-4B) trên hạ tầng GPU hạn chế (Google Colab Tesla T4 16 GB), nhóm nghiên cứu đã đối mặt và xử lý thành công 8 bẫy kỹ thuật cốt lõi.

---

## 1. Bẫy Subshell của Jupyter Notebook (`!cd` vs `%cd`)
- **Triệu chứng**: Chạy `!cd path/to/repo` sau đó gọi `pip install -r requirements-colab.txt` bị lỗi:
  `ERROR: Could not open requirements file: [Errno 2] No such file or directory`.
- **Nguyên nhân**: Lệnh bắt đầu bằng dấu chấm than `!cd` thực thi trong một subshell tiến trình con tạm thời và đóng lại ngay lập tức. Sau khi lệnh kết thúc, thư mục làm việc của notebook vẫn giữ nguyên ở `/content`.
- **Giải pháp**: Phải sử dụng IPython Magic Command `%cd` để thay đổi vĩnh viễn thư mục làm việc của kernel Python đang chạy.

---

## 2. Nghẽn cổ chai I/O khi đọc ảnh trên Google Drive FUSE
- **Triệu chứng**: Quá trình huấn luyện E1 chạy cực kỳ chậm (mất tới ~47 phút cho 300 bước, chỉ ~16–17 giây/bước dù GPU utilization không cao).
- **Nguyên nhân**: Khi repo nằm trên Google Drive (`/content/drive/MyDrive/...`), mỗi lần DataLoader mở file `.jpg`, hệ thống phải gửi request qua lớp mạng FUSE ảo của Google Drive, gây hiện tượng nghẽn I/O và GPU bị đói dữ liệu (GPU starvation).
- **Giải pháp**: Sao chép toàn bộ thư mục dữ liệu sang ổ đĩa SSD NVMe nội bộ của Colab (`!cp -r data /content/data_local`) và chuyển hướng DataLoader đọc từ `/content/data_local`. Tốc độ nạp ảnh tăng gấp 5–10 lần.

---

## 3. Rò rỉ dữ liệu (Data Leakage) khi chia tập Dashcam
- **Vấn đề**: Dữ liệu là các khung hình cắt từ video hành trình liên tục. Các frame kế tiếp nhau trong cùng một video chỉ dịch chuyển vài pixel.
- **Hậu quả nếu chia ngẫu nhiên (Random Split)**: Các khung hình gần như giống nhau sẽ nằm ở cả tập train và test. Mô hình chỉ cần "học vẹt" bối cảnh khung cảnh là có thể đoán đúng, tạo ra điểm F1 cao giả tạo (optimistic bias), nhưng mất hoàn toàn khả năng tổng quát trên cung đường mới.
- **Giải pháp**: Băm mã SHA256 và gom nhóm toàn bộ ảnh theo phiên quay (`session`). Cả một video chỉ được nằm trọn vẹn trong một split duy nhất.

---

## 4. Lệch thang toạ độ Grounding (Norm1000 vs Pixel)
- **Vấn đề**: Qwen3-VL sử dụng quy ước toạ độ chuẩn hoá $[0, 1000]$ độc lập với kích thước ảnh ($640 \times 360$).
- **Hậu quả**: Nếu mô hình sinh toạ độ theo thang 1000 mà bộ chấm điểm hiểu theo pixel (hoặc ngược lại), toạ độ dự đoán bị lệch gấp $1.5$ đến $2.8$ lần, diện tích giao cắt $IoU = 0.0$, kéo điểm F1 sập về đúng $0.00$.
- **Giải pháp**: Chuẩn hoá chặt chẽ hàm `yolo_to_box()` kẹp toạ độ trong $[0, 1000]$ và sắp xếp bounding box theo thứ tự tất định (từ trên xuống dưới, từ trái sang phải: ưu tiên $y_1$, sau đó đến $x_1$).

---

## 5. Rủi ro Pha loãng Gradient (Gradient Dilution) khi không Mask Loss
- **Vấn đề**: Trong một mẫu huấn luyện, 220 visual tokens và ~60 prompt tokens chiếm tới hơn 80% chiều dài chuỗi đầu vào.
- **Hậu quả nếu không Mask**: Mô hình sẽ lãng phí phần lớn gradient để học tái tạo lại bức ảnh hoặc học thuộc lòng câu prompt hệ thống. Tín hiệu học cho chuỗi toạ độ JSON đích bị phân tán nghiêm trọng, khiến mô hình hội tụ rất chậm hoặc sinh toạ độ ảo giác.
- **Giải pháp**: Gán nhãn `-100` (`ignore_index` của CrossEntropyLoss) cho toàn bộ phần ảnh và prompt, chỉ cho phép tính loss trên câu trả lời của Assistant.

---

## 6. Cạm bẫy lưu trữ trọng số Projector trong PEFT
- **Vấn đề**: Lệnh `model.save_pretrained()` của thư viện PEFT chỉ lưu các tham số LoRA Adapter (vài chục MB).
- **Hậu quả**: Khối Projector (109M tham số) được mở `requires_grad=True` là tham số gốc của mô hình, **không nằm trong adapter**. Nếu không lưu riêng, lúc suy luận bạn sẽ nạp lại Projector gốc chưa huấn luyện, khiến kết quả E2/E3 bị kéo tụt về giống hệt E1!
- **Giải pháp**: Trích xuất riêng state_dict của Projector và lưu thành file `*_projector.pt`, sau đó nạp lại bằng `copy_()` khi suy luận.

---

## 7. Xung đột kiểu dữ liệu AMP giữa GradScaler và AdamW
- **Triệu chứng**:
  * Lỗi 1: `ValueError: Attempting to unscale FP16 gradients`.
  * Lỗi 2: `RuntimeError: expected dtype c10::Half for end but got dtype float` trong kernel `_multi_tensor_adam`.
- **Nguyên nhân**: Khối ViT chạy ở FP16 làm phát sinh gradient FP16 cho Projector. Khi can thiệp ép kiểu gradient thủ công, bộ đệm động lượng (`exp_avg`) của AdamW (vốn được tạo theo kiểu của tham số LoRA là Half) bị lệch kiểu với gradient (Float), làm gãy phép tính nội suy `lerp`.
- **Giải pháp triệt để**: Không sửa gradient ở vòng lặp, mà ép **toàn bộ tham số trainable (cả LoRA và Projector) sang `torch.float32` ngay từ hàm `apply_setup`**. Khi đó, gradient và momentum buffers đều đồng nhất ở FP32, loại bỏ 100% lỗi runtime.

---

## 8. Tràn bộ nhớ GPU (CUDA Out of Memory) trên card T4
- **Triệu chứng**: `OutOfMemoryError: CUDA out of memory` khi bắt đầu backward ở E2.
- **Nguyên nhân**: Projector ở E2 tích luỹ feature maps từ 3 tầng DeepStack (lớp 5, 11, 17 của ViT). Khi chạy `batch_size=2`, activation maps vượt quá dung lượng khả dụng của card T4 (14.56 GB). Ngoài ra, các biến toàn cục từ cell E0/E1 vẫn chiếm dụng ~7 GB VRAM.
- **Giải pháp**:
  1. Dọn dẹp triệt để namespace toàn cục bằng `del` và gọi `gc.collect()`, `torch.cuda.empty_cache()`.
  2. Bật cờ chống phân mảnh: `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"`.
  3. Giảm `batch_size = 1` và tăng `accum_steps = 8` để giữ nguyên effective batch size = 8, hạ mức tiêu thụ VRAM đỉnh xuống mức an toàn (~11.2 GB).