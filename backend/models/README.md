# CaroNet checkpoint

`caro_policy.pt` chứa **state_dict PyTorch** của `CaroNet`. API tải với `weights_only=True`, `map_location="cpu"`.

`training.json` ghi chỉ số từ lần huấn luyện thực tế, bao gồm kích thước tập train/validation, seed, thời gian, loss theo epoch và độ khớp top-1 với teacher. `best_epoch` và `best_val_loss` xác định model tốt nhất; `checkpoint_sha256` và mục `dataset` giúp kiểm tra đúng checkpoint và dữ liệu. Không thay các chỉ số này bằng dữ liệu giả.

Huấn luyện: `python -m backend.train --samples 6000 --epochs 12` tại thư mục gốc. Checkpoint có validation **policy** loss thấp nhất trong lần chạy được giữ lại, không nhất thiết là epoch cuối. Trainer chạy từ đầu bằng seed cố định, không tự tiếp tục checkpoint cũ hay so sánh các lần chạy trước.

Dữ liệu được xuất vào `backend/data/training/` với file NPZ, CSV train/validation và manifest. Chạy `python -m backend.train --export-only` để xuất mà không đổi model, hoặc `python -m backend.train --dataset backend/data/training/caro_dataset.npz --epochs 12` để học lại từ file đã lưu. Xem [hướng dẫn dữ liệu](../data/README.md).

Dataset là các thế cờ chiến thuật tổng hợp trên 9×9, mục tiêu 3–6 quân, có các mẫu nối/chặn chuỗi và biến đổi xoay/lật. Một số thế cờ tổng hợp không phải trạng thái hợp lệ từ self-play. Teacher trong `rules.py` dùng điểm đoạn mở. Value head học nhãn tương đối từ cùng teacher và không phải xác suất thắng; phiên bản AI hiện tại dùng policy, heuristic và tìm kiếm, không dùng value để chọn nước.

Đây là mô hình học tập nhỏ. Không có tuyên bố sức mạnh thi đấu chuyên nghiệp hoặc bảo đảm thắng. Chế độ khó trên 3×3 bổ sung minimax chính xác.
