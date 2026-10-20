# Caro Lab

Game caro tiếng Việt với **FastAPI + React + Vite + PyTorch**, tổ chức trong hai thư mục chính `backend/` và `frontend/`.

## Tính năng

- **4 chế độ:** đấu AI, hai người cùng máy, online qua mã phòng 6 ký tự, xem AI đấu AI (có tạm dừng).
- **Mọi kích thước từ 3×3 đến 25×25**; chọn nhanh 3, 4, 5, 6, 8, 10, 12, 15, 19 hoặc nhập tùy chỉnh.
- Mặc định cần 3 quân trên bàn 3×3, 4 quân trên bàn 4×4, 5 quân từ bàn 5×5. Có thể chọn mục tiêu 3–6 quân, không vượt kích thước bàn.
- **3 luật:** tự do (ít nhất K quân), chính xác (đúng K quân), caro (đúng K và không bị quân đối thủ chặn cả hai đầu). Mép bàn không tính là quân đối thủ.
- **3 độ khó**, chọn X/O, giới hạn lượt 15/30/60 giây hoặc không giới hạn.
- Gợi ý AI, hoàn tác, đánh số nước đi, đánh dấu quân cuối và chuỗi thắng, âm thanh, sáng/tối, toàn màn hình.
- Phòng online có kiểm tra token, quyền đi quân, revision chống nước đi cũ, WebSocket và tự kết nối lại.
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

### Chơi online cùng bạn

1. Chọn **Chơi online → Tạo phòng mới**.
2. Sao chép mã phòng 6 ký tự.
3. Người thứ hai mở game trên máy khác hoặc cửa sổ riêng tư, chọn **Chơi online → Tham gia bằng mã phòng**.
4. Chủ phòng chơi X; khách chơi O. Khi đủ hai người, ván tự bắt đầu.

Trong cùng mạng LAN, người thứ hai mở `http://<IP-máy-chạy-game>:5173`; cho phép cổng 5173 qua firewall nếu cần. Vite đã proxy `/api` và `/ws` đến FastAPI nên không cần đổi địa chỉ trong mã nguồn. Chơi qua Internet cần triển khai máy chủ với HTTPS/WSS hoặc một tunnel phù hợp.

## Deep learning thực sự được dùng như thế nào?

`backend/app/network.py` định nghĩa **CaroNet**, mạng CNN residual **33.602 tham số** (API tính trực tiếp từ mô hình):

1. Bàn cờ mã hóa thành 5 kênh: quân mình, quân đối thủ, ô trống, mục tiêu K/6 và mặt nạ bàn.
2. Conv 3×3, 24 kênh → 3 khối residual, mỗi khối 2 Conv 3×3.
3. **Policy head** tạo phân phối trên các ô trống; ô đã có quân bị mask.
4. **Value head** dùng adaptive average pooling và MLP để ước lượng thế trận.

Mô hình học có giám sát từ **6.000 thế cờ tổng hợp**, kiểm định trên **1.200 mẫu riêng**; nhãn được sinh bằng bộ chấm điểm các đoạn có thể thắng. Huấn luyện 12 epoch bằng AdamW, soft-target cross entropy + 0,25 × value MSE. Dữ liệu có xoay, lật và các mẫu tấn công/phòng thủ. Checkpoint tốt nhất theo validation policy loss được lưu ở `backend/models/caro_policy.pt`, kết quả thật ở `backend/models/training.json`.

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

Có thể tăng dữ liệu và epoch, ví dụ `--samples 20000 --epochs 30`. Tham số bổ sung: `--batch-size 128 --seed 42`. Sau khi huấn luyện, khởi động lại backend để tải checkpoint mới. Giao diện chỉ hiển thị kết quả, không tự chạy các tác vụ huấn luyện dài.

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

Script kiểm tra chơi AI/gợi ý/đi lại, thắng và replay, AI Lab, sáng/tối, bàn 25×25, AI đấu AI, đồng bộ online giữa hai phiên trình duyệt và bố cục di động. Ảnh kiểm tra được lưu trong `frontend/test-results/`. Định dạng mã frontend bằng `npm run format`.

Kiểm thử bao gồm: mọi kích thước 3–25, bốn hướng thắng, luật chính xác/chặn đầu, hòa, hoàn tác, timeout, tải checkpoint, suy luận nhiều kích thước, AI thắng/chặn, duyệt toàn bộ nhánh người chơi để kiểm tra AI khó 3×3 không thua, xác thực API, revision, quyền đi quân, phòng WebSocket và lưu lịch sử.

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
  tests/          # unittest và FastAPI TestClient
  train.py        # Sinh dữ liệu và huấn luyện tái lập
  requirements.txt
frontend/
  src/
    App.tsx       # Luồng chơi, session, AI, WebSocket
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

- Ván đang chơi/phòng chờ ở RAM, hết hạn sau 12 giờ; **khởi động lại backend sẽ xóa phòng đang chơi**. Ván đã hoàn thành ở `backend/data/caro.sqlite3`.
- Cần chạy **một worker FastAPI** vì phòng và socket ở RAM. Triển khai nhiều worker cần chuyển session/pubsub sang Redis và dùng cơ sở dữ liệu dùng chung.
- Token ngẫu nhiên riêng cho mỗi người, truyền trong header hoặc frame xác thực WebSocket đầu tiên; không đưa token vào URL. Đây là phòng chơi bằng mã mời, chưa có hệ thống tài khoản/xếp hạng.
- Với frontend tách origin, thiết lập `VITE_API_URL` trước lúc build và bổ sung origin được phép trong `main.py`. Mặc định dùng reverse proxy cùng origin.
- `npm run build` tạo `frontend/dist`; `npm run preview` dùng kiểm tra build tại cổng 4173. Khi triển khai thật, reverse proxy `/api` và `/ws` đến backend, bật WebSocket và HTTPS.
- Tài liệu chính thức tham khảo: [FastAPI WebSocket](https://fastapi.tiangolo.com/advanced/websockets/), [PyTorch AdaptiveAvgPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html), [Vite](https://vite.dev/guide/).
