# Dữ liệu Caro Lab

Thư mục `training/` chứa dữ liệu huấn luyện đã xuất:

| File | Nội dung |
| --- | --- |
| `training/caro_dataset.npz` | Toàn bộ tensor và bàn cờ gốc của tập train/validation; dùng trực tiếp để huấn luyện lại |
| `training/train.csv` | 6.000 mẫu train, mở bằng Excel hoặc đọc bằng pandas/csv |
| `training/validation.csv` | 1.200 mẫu validation, sinh bằng seed riêng |
| `training/manifest.json` | Seed, schema, kích thước, fingerprint dữ liệu và SHA-256 của các file |

`caro.sqlite3` ở thư mục này là **lịch sử ván chơi**, không phải dữ liệu train.

## Mở dữ liệu NPZ

Chạy Python tại thư mục gốc `LabCaro`:

```python
import numpy as np

with np.load("backend/data/training/caro_dataset.npz", allow_pickle=False) as data:
    print(data.files)
    print(data["train_inputs"].shape)  # (6000, 5, 9, 9)
    print(data["train_boards"][0])    # Bàn cờ đầu tiên: X=1, O=-1, trống=0
    print(data["train_players"][0])   # Quân đang đi: 1 hoặc -1
    print(data["train_win_lengths"][0])
    print(data["train_policy"][0].reshape(9, 9))
    print(data["train_value"][0])
```

Các khóa tương ứng của tập validation bắt đầu bằng `val_`. Cả hai tập có:

- `boards`: `(N, 9, 9)`, int8, bàn cờ trước nước đi.
- `players`: `(N,)`, int8, quân đang đi.
- `win_lengths`: `(N,)`, int8, mục tiêu 3–6 quân.
- `inputs`: `(N, 5, 9, 9)`, float32, tensor đầu vào của CNN.
- `policy`: `(N, 81)`, float32, phân phối nhãn teacher trên các ô; chỉ số `row*9 + col`, tính từ 0.
- `value`: `(N,)`, float32, nhãn điểm thế trận từ -1 đến 1.

CSV có các cột `sample_id`, `player_to_move`, `win_length`, `teacher_move`, `board_json`, `policy_json`, `value_target`. `teacher_move` là tọa độ dạng A1–I9 của phần tử policy lớn nhất. Các nhãn policy mềm được giữ nguyên trong cột JSON, không chỉ lưu một nước đi.

Đây là dữ liệu chiến thuật **tổng hợp bằng code**, có các motif nối/chặn và xoay/lật, không phải lịch sử tự đấu hợp lệ hay dữ liệu thu thập từ người chơi. Value không phải xác suất thắng.

## Xuất lại / dùng lại

Chỉ xuất dữ liệu, không thay model hiện có:

```powershell
python -m backend.train --export-only --samples 6000 --seed 42
```

Huấn luyện bằng đúng dữ liệu đã xuất:

```powershell
python -m backend.train --dataset backend/data/training/caro_dataset.npz --epochs 12
```

Huấn luyện mặc định tự sinh và xuất dữ liệu trước khi học. `--data-dir` chọn thư mục xuất khác; `--model-dir` chọn thư mục lưu model khác. `--samples` chỉ dùng khi sinh dữ liệu mới; với `--dataset`, số mẫu lấy từ file. `--seed` điều khiển khởi tạo mô hình/thứ tự batch; seed đã sinh dữ liệu có trong manifest.
