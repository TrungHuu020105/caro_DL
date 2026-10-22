# CaroNet v2 và thực nghiệm

- caro_policy.pt: PyTorch state_dict, 8 kênh, **34.250 tham số**, nạp CPU bằng weights_only=True.
- training.json: train/validation/test, seed/history, checkpoint tốt nhất, chỉ số theo luật/kích thước, value MSE cùng baseline 0, checksum dữ liệu/model.
- benchmark.json: giải đấu đổi bên, khai cuộc, thắng/hòa/thua, thời gian TB/P95 và toàn bộ nước đi.

Conv 3×3/24 kênh → 3 khối residual → policy và outcome value. Input có loại luật. Value masked pooling; policy mask ô đã đánh và ô đệm.

~~~powershell
.\.venv\Scripts\python.exe -m backend.train --samples 6000 --epochs 18
.\.venv\Scripts\python.exe -m backend.benchmark --pairs 1 --sizes 5 9 15 --rules freestyle exact blocked --budget-ms 80
~~~

Chạy tại thư mục gốc bằng Python 3.10. Chọn checkpoint theo validation **policy CE + 0,5 × outcome value MSE** thấp nhất. Train/validation cùng định nghĩa loss; chỉ dùng test sau chọn checkpoint. Trainer bắt đầu lại từ seed, không tiếp tục optimizer. Dùng --model-dir để giữ thí nghiệm riêng và khởi động lại backend sau train.

CNN thuần argmax policy, không có tactical override. Chiến thuật thuần không chạy CNN. Engine lai kiểm tra thắng/chặn, kết hợp điểm; mức khó alpha-beta có value trong đánh giá lá. Riêng 3×3 minimax chính xác.

Không suy ra tỷ lệ thắng từ top-1. Value chưa hiệu chỉnh thành xác suất thắng, có thể chỉ hơn baseline 0 rất ít. Engine lai không bảo đảm mạnh hơn heuristic ở mọi ngân sách. Benchmark là số đo thực tế, không phải Elo hoặc dữ liệu minh họa.

Checkpoint v1 có 5 kênh không tương thích; API báo cần huấn luyện lại và chuyển sang heuristic. Không sửa giả chỉ số JSON.
