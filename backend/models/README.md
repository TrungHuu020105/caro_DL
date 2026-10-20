# CaroNet checkpoint

`caro_policy.pt` chứa **state_dict PyTorch** của `CaroNet`. API tải với `weights_only=True`, `map_location="cpu"`.

`training.json` ghi chỉ số từ lần huấn luyện thực tế, bao gồm kích thước tập train/validation, seed, thời gian, loss theo epoch và độ khớp top-1 với teacher. Không thay các chỉ số này bằng dữ liệu giả.

Huấn luyện: `python -m backend.train --samples 6000 --epochs 12` tại thư mục gốc. Checkpoint có validation loss thấp nhất được giữ lại. Trainer chạy từ đầu bằng seed cố định, không tự tiếp tục checkpoint cũ.

Dataset là các thế cờ chiến thuật tổng hợp trên 9×9, mục tiêu 3–6 quân, có các mẫu nối/chặn chuỗi và biến đổi xoay/lật. Một số thế cờ tổng hợp không phải trạng thái hợp lệ từ self-play. Teacher trong `rules.py` dùng điểm đoạn mở. Value head học nhãn tương đối từ cùng teacher và không phải xác suất thắng; phiên bản AI hiện tại dùng policy, heuristic và tìm kiếm, không dùng value để chọn nước.

Đây là mô hình học tập nhỏ. Không có tuyên bố sức mạnh thi đấu chuyên nghiệp hoặc bảo đảm thắng. Chế độ khó trên 3×3 bổ sung minimax chính xác.
