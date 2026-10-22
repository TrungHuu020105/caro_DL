# Dataset CaroNet v2

Dữ liệu từ các ván rollout **hợp lệ**, đủ ba luật, kích thước 3–25. Không đặt motif đè quân hoặc lấy trạng thái sau khi thắng. Metadata NPZ giữ đầy đủ quỹ đạo và kết quả từng ván.

## Tệp và mảng

- training/caro_dataset.npz: nguồn trainer nạp trực tiếp, không dùng pickle.
- training/train.csv, validation.csv, test.csv: bàn thật/target để xem bằng Excel.
- training/manifest.json: cấu trúc, seed, số mẫu/ván, quy tắc tách tập và SHA-256.

Prefix NPZ: train_, val_, test_. Mỗi tập chứa:

| Mảng | Nội dung |
|---|---|
| boards | int8 (N,25,25), bàn thật ở góc trên trái, đệm bằng 0 |
| sizes | int8 (N,), kích thước thật |
| players | int8 (N,), X=1, O=−1 |
| win_lengths | int8 (N,), mục tiêu K |
| rules | int8 (N,), 0=tự do, 1=chính xác, 2=caro |
| game_ids | int64 (N,), ván nguồn thuộc duy nhất một tập |
| inputs | float32 (N,8,25,25), mask phân biệt bàn và ô đệm |
| policy | float32 (N,625), ô đã đánh/đệm có xác suất 0 |
| value | float32 (N,), kết quả ván × quân đang đi: −1/0/+1 |

Trainer nhóm batch theo kích thước rồi cắt đúng n×n trước forward. CSV cắt về bàn thật: policy NPZ dùng row*25+col, policy JSON CSV dùng row*size+col.

CSV có sample_id, game_id, size, rule, player_to_move, win_length, teacher_move, board_json, policy_json, value_target. BOM UTF-8 giúp Excel đọc tiếng Việt.

## Chống rò rỉ và kiểm tra

Gán ván vào từng tập trước chọn mẫu/augmentation. Fingerprint vị trí chuẩn hóa theo phía đang đi và 8 phép xoay/lật, kèm kích thước/K/luật. Toàn bộ quỹ đạo ván cũng được fingerprint theo đối xứng, không chỉ dựa vào ID.

load_dataset kiểm tra hash tensor, shape/dtype, số quân/lượt, chưa kết thúc, policy/mask, trùng vị trí/quỹ đạo, nhãn outcome. Nó chơi lại quỹ đạo để xác nhận kết quả và nguồn gốc từng mẫu. Test được giữ riêng khỏi chọn checkpoint.

## Xem hoặc sinh lại

~~~python
import numpy as np
with np.load('backend/data/training/caro_dataset.npz', allow_pickle=False) as data:
    n = int(data['train_sizes'][0])
    print(data['train_boards'][0, :n, :n])
    print(data['train_rules'][0], data['train_value'][0])
~~~

Tại thư mục gốc, Python 3.10:

~~~powershell
.\.venv\Scripts\python.exe -m backend.train --export-only --samples 6000 --seed 42
.\.venv\Scripts\python.exe -m backend.train --dataset backend/data/training/caro_dataset.npz --epochs 18
~~~

Hash đọc theo chunk để tương thích Python 3.10. Seed tái tạo cùng tensor với cùng phiên bản generator/thư viện. Timestamp/ZIP metadata có thể đổi hash toàn file NPZ; đối chiếu thêm fingerprint_sha256 của tensor.

Đây là rollout chiến thuật ngẫu nhiên hóa, không phải dữ liệu chuyên gia hoặc reinforcement learning. Policy bắt chước teacher; value phản ánh kết quả theo chính sách rollout.

## Dữ liệu từ người chơi

SQLite có human_games giữ quỹ đạo ẩn danh của ván hợp lệ và human_position_split giữ phân công train/validation cố định của từng thế cờ canonical. Không thu tên, token hoặc mã phòng vào dataset học. Ván đầu hàng/hết giờ/AI đấu AI không cung cấp nhãn kết quả cho vòng này.

learning_data.py dựng lại tiền tố, tạo policy bằng teacher có kiểm tra phòng thủ, và value bằng kết quả ván. Loại các vị trí trùng qua xoay/lật với cả train/validation/test nền; loại quỹ đạo giống dataset nền. Ledger vị trí ngăn một thế đã dùng train trở thành validation khi xuất hiện trong ván mới.

Mỗi lần học xuất experience.npz ở backend/models/learning/runs/<run-id>/, gồm human_train_*, human_validation_*, chỉ số replay train nền và metadata không có thông tin định danh. Các trường tensor giữ quy ước canvas 25×25. Base dataset không bị ghi đè bởi vòng học người chơi. Artifact này khác NPZ nền: nó chỉ chứa phần người chơi và chỉ số replay, không nạp qua load_dataset của trainer nền.
