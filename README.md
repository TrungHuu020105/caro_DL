# Caro Lab

Game caro tiếng Việt với **FastAPI + React + Vite + PyTorch**, tổ chức trong hai thư mục chính `backend/` và `frontend/`.

## Tính năng

- **4 chế độ:** đấu AI, hai người cùng máy, chơi online bằng mã phòng, xem AI đấu AI (có tạm dừng).
- **Chỉ chơi sau khi nhấn Bắt đầu:** mở hoặc tải lại trang đều về bàn cờ chờ; AI và đồng hồ không tự chạy. Nút thiết lập lại cũng đưa bàn cờ về trạng thái chờ.
- **Mọi kích thước từ 3×3 đến 25×25**; chọn nhanh 3, 4, 5, 6, 8, 10, 12, 15, 19 hoặc nhập tùy chỉnh.
- Mặc định cần 3 quân trên bàn 3×3, 4 quân trên bàn 4×4, 5 quân từ bàn 5×5. Có nút chọn nhanh **Số quân để thắng** từ 3–6 quân ngay bên dưới kích thước bàn; số lớn hơn kích thước bàn sẽ bị vô hiệu hóa.
- **3 luật:** tự do (ít nhất K quân), chính xác (đúng K quân), caro (đúng K và không bị quân đối thủ chặn cả hai đầu). Mép bàn không tính là quân đối thủ.
- **3 độ khó**, chọn X/O, giới hạn lượt 15/30/60 giây hoặc không giới hạn.
- Gợi ý AI, hoàn tác, đánh số nước đi, đánh dấu quân cuối và chuỗi thắng, âm thanh, sáng/tối, toàn màn hình.
- Giao diện tông **tím/lavender**, có cả chế độ sáng và tối.
- Online đồng bộ qua WebSocket, kiểm tra quyền đi quân và tự kết nối lại. Tải lại trang sẽ chờ bạn nhấn **Kết nối lại phòng**; không tự tạo ván mới.
- Kết quả lưu SQLite; lịch sử riêng trên trình duyệt, thống kê và xem lại từng nước.
- **AI Lab:** kiến trúc, số tham số, biểu đồ loss và các chỉ số thật của lần huấn luyện.
- Giao diện responsive, thao tác bàn cờ bằng các phím mũi tên và Enter/Space.

## Chạy trên máy

Cần **Python 3.11+** và **Node.js 20.19+ hoặc 22.12+**. Checkpoint đã được huấn luyện và đi kèm dự án; không cần API key hoặc GPU.

Mở terminal tại thư mục `LabCaro`. Cài backend:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

Terminal 1 — backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Terminal 2 — frontend:

```powershell
cd frontend
npm install
npm run dev
```

Mở **http://localhost:5173**. Swagger API: **http://localhost:8000/docs**.

Nếu Python hiện tại đã có các thư viện cần thiết, có thể dùng `python -m uvicorn ...` trực tiếp. Trên macOS/Linux dùng `.venv/bin/python` thay đường dẫn Windows.

### Bắt đầu chơi

1. Chọn **Đấu với AI**, **Hai người**, **Chơi online** hoặc **AI đấu AI**.
2. Chọn kích thước bàn, **Số quân để thắng** và độ khó; mở phần nâng cao nếu muốn đổi luật hoặc thời gian.
3. Nhấn **Bắt đầu**. Trước bước này bàn cờ khóa, chưa tạo ván trên máy chủ, chưa chạy AI hoặc đồng hồ.
4. Để chơi cùng bạn, chọn **Hai người** và thay phiên đặt X/O trên cùng máy.

### Chơi online bằng mã phòng

1. Chủ phòng chọn **Chơi online**, thiết lập bàn và số quân thắng, rồi nhấn **Bắt đầu · Tạo phòng**.
2. Chia sẻ mã phòng 6 ký tự. Chủ phòng chơi X.
3. Người còn lại mở game trên thiết bị hoặc phiên trình duyệt riêng, chọn **Chơi online → Tham gia bằng mã phòng**, nhập tên/mã và nhấn **Vào phòng**. Khách chơi O.
4. Ván cờ và đồng hồ bắt đầu khi đủ hai người. Phòng chờ chưa tính giờ.
5. Sau khi tải lại cùng tab, chọn **Kết nối lại phòng** để trở về đúng ván cũ; phiên phòng được giữ trong sessionStorage của tab. Đi lại và gợi ý tắt ở chế độ online.

Trong cùng LAN, người còn lại mở `http://<IP-máy-chạy-game>:5173` khi Vite chạy với `--host 0.0.0.0`. Backend cần chạy như lệnh ở trên; Vite proxy cả `/api` và `/ws`. Chơi từ các mạng khác nhau cần triển khai máy chủ có HTTPS/WSS hoặc tunnel phù hợp.

## Deep learning thực sự được dùng như thế nào?

`backend/app/network.py` định nghĩa **CaroNet**, mạng CNN residual **33.602 tham số** (API tính trực tiếp từ mô hình):

1. Bàn cờ mã hóa thành 5 kênh: quân mình, quân đối thủ, ô trống, mục tiêu K/6 và mặt nạ bàn.
2. Conv 3×3, 24 kênh → 3 khối residual, mỗi khối 2 Conv 3×3.
3. **Policy head** tạo phân phối trên các ô trống; ô đã có quân bị mask.
4. **Value head** dùng adaptive average pooling và MLP để ước lượng thế trận.

Mô hình học có giám sát từ **6.000 thế cờ tổng hợp**, kiểm định trên **1.200 mẫu riêng**; nhãn được sinh bằng bộ chấm điểm các đoạn có thể thắng. Huấn luyện 12 epoch bằng AdamW, soft-target cross entropy + 0,25 × value MSE. Dữ liệu có xoay, lật và các mẫu tấn công/phòng thủ. Checkpoint tốt nhất theo **validation policy loss thấp nhất trong lần huấn luyện** được lưu ở `backend/models/caro_policy.pt`; không tự ghi đè bằng model epoch cuối nếu model đó kém hơn. `backend/models/training.json` lưu `best_epoch`, `best_val_loss`, checksum checkpoint và tham chiếu bộ dữ liệu.

Dữ liệu thực tế được xuất trong **`backend/data/training/`**: `caro_dataset.npz` để nạp lại huấn luyện, `train.csv` và `validation.csv` để mở xem, `manifest.json` mô tả cấu trúc và checksum. Xem [hướng dẫn dữ liệu](backend/data/README.md).

Khi chơi, AI thực hiện suy luận CNN rồi kết hợp chiến thuật:

| Mức | Cách chọn nước |
| --- | --- |
| Dễ | Policy CNN chiếm trọng số lớn; lấy ngẫu nhiên trong các ứng viên tốt; vẫn ưu tiên thắng ngay |
| Vừa | Kết hợp CNN và điểm chiến thuật; kiểm tra nước thắng/chặn theo luật đang chọn |
| Khó | Thêm tìm kiếm phản hồi đối thủ trên tập ứng viên giới hạn; riêng 3×3 dùng minimax chính xác |

**Giới hạn:** đây là mô hình giáo dục, chưa phải AlphaZero. Dữ liệu chiến thuật tổng hợp có thể chứa thế cờ không hình thành từ một ván hợp lệ, không phải dữ liệu self-play. Huấn luyện trên bàn 9×9 với K từ 3 đến 6; kích thước khác được tổng quát hóa nhờ CNN. Value học từ nhãn heuristic, **không phải xác suất thắng đã hiệu chỉnh**, và hiện không dùng để quyết định nước đi. Chỉ số top-1 đo khớp với teacher, không phải tỷ lệ thắng. Nếu thiếu checkpoint, API báo rõ đang dùng heuristic dự phòng.

### Huấn luyện lại

Chạy tại thư mục gốc:

```powershell
.\.venv\Scripts\python.exe -m backend.train --samples 6000 --epochs 12
```

Lệnh mặc định tự sinh và xuất dữ liệu trước khi học. Để chỉ xuất dữ liệu mà giữ nguyên model:

```powershell
python -m backend.train --export-only --samples 6000 --seed 42
```

Để huấn luyện lại bằng đúng bộ dữ liệu đã lưu:

```powershell
python -m backend.train --dataset backend/data/training/caro_dataset.npz --epochs 12
```

Có thể tăng dữ liệu và epoch, ví dụ `--samples 20000 --epochs 30`. Tham số bổ sung: `--batch-size 128 --seed 42`, `--data-dir` để chọn nơi xuất dữ liệu, `--model-dir` để chạy thí nghiệm ở thư mục model riêng. Khi dùng `--dataset`, số mẫu lấy từ file. Trainer chạy mới từ đầu; checkpoint tốt nhất được chọn trong từng lần chạy, không tiếp tục optimizer cũ và không so sánh với các lần chạy trước. Sau khi huấn luyện, khởi động lại backend để tải checkpoint mới. Giao diện chỉ hiển thị kết quả, không tự chạy các tác vụ huấn luyện dài.

## Kiểm tra

```powershell
python -m unittest discover -s backend/tests -v
cd frontend
npm run build
```

Kiểm thử trình duyệt tùy chọn tại thư mục gốc `LabCaro`, khi cả hai server đang chạy:

```powershell
python -m pip install playwright
python -m playwright install chromium
python frontend/tests/e2e.py
```

Script kiểm tra trạng thái chờ, nút Bắt đầu, chọn nhanh số quân thắng, chơi AI/gợi ý/đi lại, thắng và replay, AI Lab, tông tím sáng/tối, bàn 25×25, AI đấu AI, phòng online giữa hai trình duyệt, kết nối lại phòng và bố cục di động. Ảnh kiểm tra được lưu trong `frontend/test-results/`. Định dạng mã frontend bằng `npm run format`.

Kiểm thử bao gồm: mọi kích thước 3–25, bốn hướng thắng, luật chính xác/chặn đầu, hòa, hoàn tác, timeout, tải checkpoint, suy luận nhiều kích thước, AI thắng/chặn, duyệt toàn bộ nhánh người chơi để kiểm tra AI khó 3×3 không thua, xác thực API, revision, quyền đi quân và lưu lịch sử.

## Cấu trúc

```text
backend/
  app/
    main.py       # REST API, WebSocket, clock, vòng đời ứng dụng
    game.py       # Trạng thái, cấu hình và vòng đời ván cờ
    rules.py      # Luật thắng và teacher chiến thuật
    network.py    # PyTorch residual policy/value CNN
    ai.py         # CNN inference, chiến thuật, minimax và tìm kiếm
    storage.py    # SQLite cho ván đã kết thúc
  models/         # Checkpoint và thông số huấn luyện thật
  data/training/  # Dataset NPZ, train/validation CSV và manifest
  tests/          # unittest và FastAPI TestClient
  dataset.py      # Sinh, xuất và nạp dữ liệu huấn luyện
  train.py        # Sinh dữ liệu và huấn luyện tái lập
  requirements.txt
frontend/
  src/
    App.tsx       # Bàn cờ chờ, phòng online, session và AI
    Board.tsx     # Bàn cờ, điều khiển bàn phím
    Settings.tsx  # Chế độ, thiết lập và hộp thoại
    Pages.tsx     # Hướng dẫn, AI Lab, thống kê và replay
    api.ts        # REST client, token, local storage
    types.ts      # TypeScript API types
    styles.css    # Responsive và sáng/tối
  public/
  vite.config.ts
  package.json
```

## Lưu trữ và triển khai

- Ván đang chơi ở RAM, hết hạn sau 12 giờ; khởi động lại backend sẽ xóa ván đang chơi. Ván đã hoàn thành ở `backend/data/caro.sqlite3`. Mở lại trang luôn trở về trạng thái chờ để bạn chủ động bắt đầu ván mới.
- Cần chạy **một worker FastAPI** vì trạng thái ván cờ ở RAM. Triển khai nhiều worker cần dùng kho trạng thái chung.
- Token ngẫu nhiên của ván truyền trong header; WebSocket xác thực bằng frame đầu tiên, không đưa token vào URL. Chưa có hệ thống tài khoản/xếp hạng.
- Với frontend tách origin, thiết lập `VITE_API_URL` trước lúc build và bổ sung origin được phép trong `main.py`. Mặc định dùng reverse proxy cùng origin.
- `npm run build` tạo `frontend/dist`; `npm run preview` dùng kiểm tra build tại cổng 4173. Khi triển khai thật, reverse proxy `/api` và `/ws` đến backend, hỗ trợ WebSocket và bật HTTPS.
- Tài liệu chính thức tham khảo: [FastAPI](https://fastapi.tiangolo.com/), [PyTorch AdaptiveAvgPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html), [Vite](https://vite.dev/guide/).
