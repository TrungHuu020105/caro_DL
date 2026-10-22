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

## Học bổ sung từ lịch sử người chơi

Vòng học trong learning.py bắt đầu từ trọng số hiện tại, learning rate 0,0002, mặc định 3 epoch. Lịch sử được kiểm tra, phân tích lại policy và lấy kết quả làm value; trộn tối đa 256 mẫu người chơi với tối đa 1.024 mẫu train nền. Validation và test được tách theo ván/đối xứng; nhãn policy không sao chép mọi nước người chơi.

Ứng viên phải qua kiểm định và giải đấu 16 ván đổi bên với model hiện tại, ở cả CNN thuần và engine lai. Ngưỡng tổng 55% điểm, mỗi biến thể ít nhất 50%; đồng thời giữ loss kiểm định trong sai số quy định. Đây là cổng thực nghiệm hữu hạn, không phải bảo đảm sức mạnh trong mọi ván.

Nếu được chấp nhận, cặp checkpoint/metadata mới được lưu, model cũ sao lưu ở learning/versions/<sha256>/ và backend nạp engine mới. Lượt suy nghĩ đang chạy tiếp tục dùng engine cũ. Báo cáo và trọng số ứng viên ở learning/runs/<run-id>/; ứng viên bị loại không ghi đè model đang dùng. Test nền chỉ đánh giá sau khi cổng chấp nhận, không dùng để lựa chọn.

Chi tiết và kết quả một lần chạy thật: [HOC_TU_NGUOI_CHOI.md](../HOC_TU_NGUOI_CHOI.md).
