# Caro Lab

Game caro tiếng Việt dùng **FastAPI + React + Vite + PyTorch**, chạy trên máy, không cần API key/GPU. Backend và trainer tương thích **Python 3.10**.

## Tính năng

- Bốn chế độ: đấu AI, hai người cùng máy, online bằng mã phòng, xem AI đấu AI.
- Bàn 3×3–25×25, mục tiêu 3–6 quân; luật tự do, chính xác, caro không bị đối thủ chặn cả hai đầu. Mép bàn không tính là quân chặn.
- Ba engine tại **Tùy chỉnh nâng cao → Engine AI**: CNN thuần, chiến thuật thuần, kết hợp CNN/chiến thuật/tìm kiếm.
- Engine kết hợp có ba mức khó; mức khó dùng alpha-beta theo thời gian, riêng 3×3 dùng minimax chính xác. CNN thuần luôn chọn policy argmax; chiến thuật thuần không gọi CNN.
- Chỉ tạo ván sau **Bắt đầu**. Gợi ý, đi lại, đánh số nước, âm thanh, sáng/tối, toàn màn hình, bàn phím, responsive.
- AI đấu AI có **tạm dừng ở backend**, dừng cả nước đi và đồng hồ.
- **Lưu ván đang chơi vào SQLite sau mỗi thay đổi**; giữ bàn, lượt, mã phòng, quyền truy cập và trạng thái pause qua lần khởi động lại backend.
- Khi tải lại trang, nhấn **Tiếp tục ván đã lưu** hoặc **Kết nối lại phòng**; không tự chạy ván.
- Lịch sử/replay có **Phân tích cơ hội sau ván** và **Thử phương án khác** để nhận phản hồi AI trong nhánh riêng.
- Phân tích nước AI hiển thị điểm CNN, chiến thuật, kết hợp, tìm kiếm và số nút. Heatmap ghi rõ thuộc thế cờ trước nước AI vừa đánh.
- **AI Lab**: loss, test giữ riêng, chỉ số theo kích thước/luật, baseline value; giải đấu engine chạy nền, tiến độ và tải JSON.
- **Học từ người chơi**: thu các ván hợp lệ, mặc định tự thử học sau 5 ván mới; trộn dữ liệu cũ, đánh giá ứng viên và chỉ áp dụng khi qua kiểm tra. AI Lab có nút học ngay, bật/tắt tự học và báo cáo.

## Chạy với Python 3.10

Giữ nguyên .venv Python 3.10 đã có. Nếu cần tạo mới, tại thư mục gốc:

~~~powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
~~~

Terminal backend:

~~~powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
~~~

Terminal frontend, cần Node.js 20.19+ hoặc 22.12+:

~~~powershell
cd frontend
npm install
npm run dev
~~~

Mở **http://localhost:5173**; API docs **http://localhost:8000/docs**. macOS/Linux dùng .venv/bin/python.

## Online và phục hồi

Chủ phòng chọn **Chơi online → Bắt đầu · Tạo phòng**, chơi X. Khách mở game ở thiết bị/phiên riêng, chọn **Tham gia bằng mã phòng**, nhập mã và tên, chơi O. Bàn và đồng hồ bắt đầu khi đủ hai người.

Trong LAN mở http://IP-máy-chạy-game:5173; Vite proxy /api và /ws tới backend. Chơi qua Internet cần triển khai HTTPS/WSS. WebSocket xác thực bằng frame đầu, REST kiểm tra token/lượt/revision.

Sau khi tải lại trang hoặc backend khởi động lại, nhấn **Kết nối lại phòng** trong cùng tab; các chế độ khác dùng **Tiếp tục ván đã lưu**. SessionStorage giữ token theo tab: đóng tab/xóa dữ liệu trình duyệt có thể làm mất quyền tiếp tục, dù ván vẫn trong SQLite.

## AI và dữ liệu

**CaroNet v2: 34.250 tham số**, 8 kênh → Conv 3×3/24 kênh → 3 khối residual → policy và outcome value. Input: quân mình, đối thủ, ô trống, K/6, mặt nạ bàn, ba kênh loại luật. Policy mask ô đã đánh/ô đệm; value dùng masked pooling.

Dữ liệu từ rollout chiến thuật ngẫu nhiên hóa: X/O thay phiên, chỉ đi ô trống, kết thúc khi thắng/đầy bàn. Ván thuộc riêng train/validation/test; loại trùng vị trí và quỹ đạo theo xoay/lật trước augmentation.

- Policy target: teacher có xét luật, ưu tiên thắng rồi chặn.
- Value target: kết quả ván × quân đang đi, −1/0/+1.
- Train/validation cùng loss: **policy CE + 0,5 × outcome value MSE**.
- Chọn checkpoint bằng validation; test chỉ đánh giá sau chọn checkpoint.
- Metadata/checksum, kết quả theo luật/kích thước và baseline value=0 ở backend/models/training.json.

Huấn luyện nền học có giám sát từ rollout; học bổ sung dùng cả lịch sử đấu người. Policy được phân tích lại, value lấy kết quả ván. Chưa phải reinforcement learning/AlphaZero. Top-1 đo khớp teacher, không phải tỷ lệ thắng; value chưa hiệu chỉnh thành xác suất thắng. Xem [thuyết minh](backend/THUYET_MINH_TRAIN_AI.md), [học từ người chơi](backend/HOC_TU_NGUOI_CHOI.md), [dữ liệu](backend/data/README.md), [checkpoint](backend/models/README.md).

## Để AI học từ những ván bạn chơi

1. Hoàn thành các ván **Đấu với AI**, **Hai người** hoặc **Chơi online** bằng thắng trên bàn hoặc hòa.
2. Mở **AI Lab → Học từ các ván với người chơi** để xem số ván mới, bật/tắt tự học hoặc bấm **Học từ lịch sử ngay**.
3. Mặc định tự thử học sau 5 ván mới, 3 epoch. Cần ít nhất 8 vị trí train sau lọc trùng. Các ván đầu hàng, hết giờ, AI đấu AI và quỹ đạo trùng được bỏ qua.
4. Ứng viên được kiểm định và đấu model cũ. Chỉ khi đạt yêu cầu mới nạp cho các lượt AI tiếp theo; nếu chưa đạt, giữ model cũ và báo lý do.

Học chạy nền, vẫn chơi cờ được. Khi dùng giao diện không cần khởi động lại backend sau khi model được chấp nhận. Nếu muốn chạy riêng bằng terminal, dừng backend trước, rồi chạy và mở lại backend sau đó:

~~~powershell
.\.venv\Scripts\python.exe -m backend.learn --epochs 3
~~~

Đây là fine-tuning từ trọng số hiện tại. Lệnh backend.train bên dưới vẫn dùng để huấn luyện nền mới từ đầu.

## Huấn luyện và benchmark

~~~powershell
# Sinh, xuất và huấn luyện trên CPU
.\.venv\Scripts\python.exe -m backend.train --samples 6000 --epochs 18

# Chỉ xuất dữ liệu, giữ model
.\.venv\Scripts\python.exe -m backend.train --export-only --samples 6000 --seed 42

# Dùng dataset đã lưu
.\.venv\Scripts\python.exe -m backend.train --dataset backend/data/training/caro_dataset.npz --epochs 18

# Ba engine, ba luật, cùng khai cuộc và đổi bên
.\.venv\Scripts\python.exe -m backend.benchmark --pairs 1 --sizes 5 9 15 --rules freestyle exact blocked --budget-ms 80
~~~

Trainer bắt đầu lại từ seed, giữ checkpoint có validation loss thấp nhất trong lần chạy; không tiếp tục optimizer cũ. Dùng --model-dir/--data-dir để tách thí nghiệm. Khởi động lại backend sau train. Dataset/checkpoint v1 có 5 kênh không tương thích v2 và cần sinh/huấn luyện lại.

Benchmark lưu ở backend/models/benchmark.json: thắng/hòa/thua, điểm (thắng + 0,5 × hòa)/số ván, thời gian TB/P95, seed/checksum và nước đi. Không suy ra Elo từ lượt chạy nhỏ. Cùng ngân sách tối đa không có nghĩa cùng thời gian tính toán thực tế. AI Lab báo nếu kết quả thuộc checkpoint cũ.

## Kiểm thử

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v
cd frontend
npm run build
npm run format:check
~~~

Kiểm thử trình duyệt khi hai server đang chạy, tại thư mục gốc:

~~~powershell
.\.venv\Scripts\python.exe -m pip install playwright
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe frontend/tests/e2e.py
.\.venv\Scripts\python.exe frontend/tests/improvements_e2e.py
~~~

Kiểm thử riêng vòng học người chơi: chạy frontend/tests/human_learning_e2e.py với backend dùng database, model và thư mục học thử riêng. Script tạo ván thử và có thể thay model thử; không dùng trực tiếp với dữ liệu/model chính.

Kiểm tra gồm nguồn gốc rollout, tách tập/đối xứng, hash Python 3.10, outcome labels, chọn checkpoint, engine độc lập, value trong tìm kiếm, đe dọa kép, pause/timeout, restart, replay sandbox và giải đấu đổi bên. Ảnh lưu ở frontend/test-results/.

Để kiểm thử server mà không chạm lịch sử thật, đặt CARO_DATABASE_PATH và CARO_BENCHMARK_PATH tới file riêng trước khi chạy backend.

## Lưu trữ và triển khai

- SQLite mặc định backend/data/caro.sqlite3: live_games giữ snapshot riêng tư; games giữ lịch sử không có token.
- Snapshot hết hạn sau 12 giờ không hoạt động; lịch sử hoàn thành vẫn còn. Đồng hồ tính thời gian thực khi server tắt, trừ ván đã pause.
- Chạy **một worker FastAPI**: lock/cache trong một process. SQLite giúp phục hồi, chưa đồng bộ cache nhiều worker.
- Chưa có tài khoản/Elo. Lịch sử truy cập qua ID ngẫu nhiên mà trình duyệt giữ, không liệt kê công khai mọi ván.
- Build tạo frontend/dist; triển khai reverse proxy /api và /ws, WebSocket và HTTPS. Khác origin cần VITE_API_URL trước build và CORS phù hợp.
- Thí nghiệm và học bổ sung không chạy cùng lúc. Writer lock ngăn trainer CLI và vòng học cùng ghi checkpoint. Huấn luyện nền chạy CLI; học bổ sung chạy tự động hoặc từ AI Lab.
- Học bổ sung lưu trạng thái, ứng viên, dataset và báo cáo ở backend/models/learning/; phiên bản model cũ ở learning/versions/. Các artifact phát sinh và SQLite không đưa vào Git.
- Các biến dành cho chạy thử cô lập: CARO_DATABASE_PATH, CARO_MODEL_DIR, CARO_LEARNING_DIR, CARO_DATASET_PATH và CARO_BENCHMARK_PATH.
