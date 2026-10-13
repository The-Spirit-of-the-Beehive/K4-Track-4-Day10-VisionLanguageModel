# Danh mục kiểm tra nghiệm thu Lab Day 10 (Checklist)

Dự án: **Finetune Qwen3-VL-4B (LoRA) định vị hư hại mặt đường**  
Môi trường: **Google Colab Tesla T4 (16 GB)**  
Học viên: `<Điền họ và tên của bạn>`  
`dataset_version`: `d10-21e4070301`

---

## 1. Môi trường và Dữ liệu
- [x] Đã cấu hình và bảo mật Kaggle credentials thông qua Google Colab Secrets (`KAGGLE_USERNAME`, `KAGGLE_KEY`).
- [x] Chạy bộ kiểm thử tự động ban đầu: `python -m pytest -q` đạt 68 passed, 1 skipped.
- [x] Tải và giải nén dữ liệu chuẩn: sinh đủ `data/images/`, `data/splits/{train,val,test}.jsonl` và `data/splits/meta.json`.
- [x] Xác nhận thuật toán băm SHA256 chia ảnh theo phiên (`session`): không có session nào bị trùng lặp giữa các split (train: 13, val: 2, test: 3 sessions).
- [x] Quy đổi toạ độ YOLO sang nhãn Grounding JSON thang `0–1000`: đã kiểm chứng nghịch đảo toạ độ trùng khít 100% với bounding box gốc trên pixel.

---

## 2. Kiến trúc và Xử lý dữ liệu đa phương thức
- [x] Cấu hình lượng tử hoá BitsAndBytes 4-bit (NF4) kèm `llm_int8_skip_modules=["visual", "lm_head"]` để bảo toàn ViT và Projector ở FP16.
- [x] Kiểm tra số lượng token thị giác của ảnh $640 \times 360$: tensor `image_grid_thw` cho đúng 220 visual tokens.
- [x] Cơ chế Mask Loss: gán nhãn `-100` cho toàn bộ prompt và 220 token ảnh, giải mã ngược kiểm tra phần tính loss chứa đủ chuỗi JSON đáp án và kết thúc bằng `<|im_end|>`.
- [x] Hàm Collate đa phương thức: tensor text được pad đúng cách, tensor thị giác (`pixel_values`, `image_grid_thw`) được nối (`cat`) dọc chiều 0, tuyệt đối không pad.

---

## 3. Các Thí nghiệm và Bằng chứng (`report/evidence/`)
- [x] **E0 (Zero-shot)**: Sinh `work/e0.jsonl`, chấm điểm tạo `report/evidence/e0.json` (F1 = 0.079, parse_valid_rate = 0.868).
- [x] **E1 (LoRA LLM)**: 252 LoRA modules, huấn luyện 300 steps với gradient checkpointing, sinh `report/evidence/e1.json` (F1 = 0.246, tăng +0.167 so với E0, vượt ngưỡng $\ge 0.10$).
- [x] **E2 (+ Projector)**: Mở 109M tham số DeepStack merger ở Float32, lưu riêng `e2_projector.pt`, sinh `report/evidence/e2.json` (F1 = 0.268, có đủ 3 trường meta).
- [x] **E3 (+ LoRA ViT)**: Bổ sung 96 LoRA modules trên ViT (tổng 348 modules), sinh `report/evidence/e3.json` (F1 = 0.279, có đủ 3 trường meta).
- [x] **E4 (Ablation Study)**: Đổi biến `data_size` (10% vs 100%), sinh đủ `report/evidence/e4_data_0.1.json` và `report/evidence/e4_data_1.0.json`. Khoảng CI95 tách biệt hoàn toàn chứng minh hiệu quả dữ liệu.

---

## 4. Báo cáo (`REPORT.md`)
- [x] Đã điền đầy đủ họ tên và `dataset_version`.
- [x] Bảng tổng hợp số liệu đã điền đầy đủ các cột: Tham số, VRAM đỉnh, Phút, F1, CI95, FP/ảnh, F1 từng lớp.
- [x] 10 câu hỏi báo cáo được trả lời đầy đủ, chính xác bản chất kỹ thuật, sử dụng số liệu thực tế, tuyệt đối **không còn bất kỳ chuỗi `<điền>` nào**.
- [x] Đã đính kèm ảnh minh hoạ trực quan bounding box dự đoán đúng và sai của mô hình.

---

## 5. Vệ sinh Repository và Lệnh nghiệm thu
- [x] Kiểm tra file cấm qua `.gitignore`: Không commit thư mục `data/`, `work/`, file `.env`, file trọng số `*.pt`, `*.safetensors`, `*.bin`.
- [x] Số lượng commit git: Lệnh `git rev-list --count HEAD` đạt $\ge 5$ commits tương ứng các mốc thực hiện.
- [x] Không can thiệp hoặc chỉnh sửa mã nguồn gốc trong `scripts/` và `tests/`.
- [x] Chạy lệnh nghiệm thu cuối cùng:
  ```bash
  python scripts/check_submission.py