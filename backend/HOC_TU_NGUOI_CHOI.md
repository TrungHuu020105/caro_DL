# Để AI học từ những ván đấu với người chơi

Chức năng này đã được nối vào game: chơi xong ván hợp lệ → thu kinh nghiệm → học model ứng viên → kiểm định và đấu model hiện tại → áp dụng nếu đạt. Tất cả chạy trên CPU, tương thích `.venv` Python 3.10.

## Cách sử dụng

1. Chơi **Đấu với AI**, **Hai người** hoặc **Chơi online** đến khi có người thắng bằng chuỗi quân hoặc hòa đầy bàn.
2. Mở **AI Lab → Học từ các ván với người chơi**. Xem ván đã thu, ván mới chờ xử lý và số vị trí trước lọc trùng.
3. Mặc định **Tự học khi đủ ván mới** được bật: sau 5 ván hợp lệ mới, backend thử học 3 epoch. Ngưỡng có thể đổi từ 3–50 ván, epoch từ 1–8.
4. Có thể bấm **Học từ lịch sử ngay** khi có ván mới, không phải đợi đủ ngưỡng tự động. Vẫn cần ít nhất 8 vị trí train mới sau các bước lọc.
5. Theo dõi tiến độ epoch, đấu kiểm định và kết luận. Tải báo cáo JSON nếu cần nộp kết quả thực nghiệm.

Nếu báo **Giữ model hiện tại**, việc huấn luyện ứng viên đã chạy nhưng chưa chứng minh tốt hơn model đang chơi. Model gốc giữ nguyên; tiếp tục chơi để bổ sung kinh nghiệm cho đợt sau. Các ván đã xử lý vẫn có thể được replay trong lần học sau, nhưng không tự kích hoạt học lặp vô hạn.

Học chạy nền; có thể chuyển tab chơi cờ. Giải đấu benchmark và học bổ sung không chạy đồng thời. Khi ứng viên được chấp nhận, backend tự nạp engine mới cho các lượt AI sau, không cần restart. Lượt AI đang suy nghĩ tiếp tục dùng đối tượng engine cũ đến hết nước đi.

## Ván nào được thu?

- Có người tham gia: mode ai, local hoặc online.
- Ít nhất 5 nước, kết thúc đúng luật bằng line/full.
- Chơi lại toàn bộ quỹ đạo phải hợp lệ: X/O thay phiên, không đè quân, không đi sau thắng, winner/board/status phải khớp.
- Các quỹ đạo giống nhau qua xoay/lật chỉ tính một ván.

Bỏ qua AI đấu AI, ván dang dở, đầu hàng, hết giờ và dữ liệu không hợp lệ. Không dùng kết quả thua do mất mạng/hết giờ làm nhãn đánh giá thế cờ. Khi backend khởi động, lịch sử cũ cũng được kiểm tra và thu vào nguồn học.

Dataset học không chứa tên, token hoặc mã phòng. Nó lưu tọa độ quân, luật, kích thước, bên đang đi, kết quả và fingerprint ván.

## Mạng học điều gì?

**Policy:** dựng lại các thế cờ trước từng nước từ nước thứ ba trở đi. Teacher kiểm tra nước thắng, các phòng thủ hợp lệ và điểm chiến thuật theo luật để tạo target. Không sao chép máy móc nước người chơi đã đánh. Ví dụ người chơi bỏ lỡ thắng ở C1, target vẫn ưu tiên C1.

**Value:** nhãn `winner × player_to_move`, thuộc −1/0/+1. Nó phản ánh kết quả thực của ván nguồn theo phía đang đi. Đây chưa phải xác suất thắng đã calibration.

Vì policy còn do teacher phân tích lại, đây là **học có giám sát từ trải nghiệm người chơi**, chưa phải reinforcement learning/AlphaZero. Không có tuyên bố AI lập tức giỏi hơn sau mỗi ván.

## Tách tập và chống quên

- Fingerprint toàn ván quyết định train/validation cố định, khoảng 20% ván dành validation.
- `human_position_split` giữ phân công của mỗi vị trí canonical. Một vị trí từng train không chuyển thành validation khi xuất hiện trong ván mới.
- Loại vị trí trùng qua xoay/lật với cả train, validation và test của dataset nền. Loại cả quỹ đạo trùng ván nền.
- Mỗi lần chọn tối đa 256 vị trí train người chơi, 128 vị trí validation người chơi; ôn lại tối đa 1.024 vị trí train nền, mục tiêu tối đa bốn mẫu nền cho một mẫu người chơi.
- Không sửa dataset nền. Test nền không dùng chọn epoch hoặc quyết định áp dụng model.

Nguồn tối đa 500 ván trong một snapshot, ưu tiên các ván chưa xử lý. Ván mới đến trong khi học được giữ cho lần sau. Số vị trí trước lọc có thể lớn hơn số được dùng: trùng lặp, ranh giới validation và giới hạn lấy mẫu đều làm giảm số này.

## Fine-tuning và cổng áp dụng

Khởi tạo CNN từ checkpoint đang dùng, optimizer AdamW mới, learning rate 0,0002, weight decay 0,0001, clip gradient 1,5. Loss giữ `policy CE + 0,5 × value MSE`. Augmentation xoay/lật đồng thời input/policy. Chọn epoch tốt nhất theo validation nền cộng validation người chơi.

Ứng viên sau đó phải đạt cả:

| Kiểm tra | Ngưỡng |
|---|---|
| Loss tổng, policy CE và value MSE trên validation nền | Không vượt baseline × 1,02 + 0,01 |
| Loss validation người chơi, nếu có | Không vượt baseline × 1,10 + 0,02 |
| Tổng điểm đấu model cũ | Ít nhất 55% |
| Điểm riêng ở CNN thuần và engine lai | Mỗi loại ít nhất 50% |

Giải đấu có 16 ván: hai cấu hình bàn/luật, hai biến thể CNN/lai, hai cặp đổi bên cho mỗi biến thể/cấu hình. Cùng khai cuộc cho hai màu, hard, ngân sách tìm kiếm 40 ms/nước. Điểm `(thắng + 0,5 × hòa)/số ván`. Seed arena được ghi trong báo cáo, phụ thuộc checkpoint và dữ liệu đợt học để thay đổi khi có kinh nghiệm mới.

Đây là cổng đánh giá với mẫu hữu hạn, chưa phải chứng minh thống kê rằng ứng viên luôn mạnh hơn. Giới hạn thời gian mềm khiến kết quả có thể đổi theo tải máy.

Nếu đạt, đánh giá test nền để báo cáo sau cổng, sao lưu cặp model/metadata cũ rồi lưu cặp mới và kiểm chứng lại checksum. Nếu lưu lỗi, phục hồi cả hai file. Nếu không đạt, giữ nguyên model đang dùng và lưu ứng viên riêng.

## Một lần chạy đã kiểm tra trên lịch sử lưu sẵn

Lần `20261022T154003-74a2ec04` dùng 35 ván hợp lệ đã lưu:

| Thành phần | Kết quả |
|---|---:|
| Vị trí train người chơi | 256 |
| Vị trí validation người chơi | 128 |
| Vị trí ôn lại train nền | 1.024 |
| Epoch | 3 |
| Validation loss epoch 1 → 3 | 4,03101 → 4,02232 |
| Đấu model cũ | 6 thắng, 3 hòa, 7 thua |
| Điểm | 46,88% |
| Quyết định | Giữ model cũ |

Trọng số ứng viên thực sự thay đổi, nhưng loss giảm chưa đủ để áp dụng khi sức chơi qua cổng không đạt. Không sửa ngưỡng hoặc kết quả để ép ứng viên này vào game. Đây là bản ghi của lần kiểm tra trên, không phải cam kết cho những lần sau.

## File và API

- `backend/app/experience.py`: kiểm tra, ẩn thông tin người chơi và fingerprint quỹ đạo.
- `backend/app/storage.py`: bảng `human_games` và `human_position_split` trong SQLite.
- `backend/learning_data.py`: dựng trạng thái, tạo target, lọc trùng và trộn replay.
- `backend/learning.py`: fine-tune, đánh giá, cổng áp dụng và nạp model.
- `backend/model_store.py`: writer lock liên tiến trình, tự giải phóng khi process đóng.
- `backend/models/learning/state.json`: thiết lập, tiến độ và báo cáo gần nhất.
- `backend/models/learning/runs/<run-id>/`: `experience.npz`, `caro_policy.pt`, `training.json`, `report.json`.
- `backend/models/learning/versions/<sha256>/`: model và metadata cũ trước khi áp dụng bản mới.

Artifact sinh ra và SQLite được bỏ qua trong Git. Dataset nền/checkpoint hiện có vẫn được quản lý như trước. Có thể đổi vị trí chạy thử bằng `CARO_DATABASE_PATH`, `CARO_MODEL_DIR`, `CARO_LEARNING_DIR`, `CARO_DATASET_PATH`.

API: `GET /api/learning`, `POST /api/learning/settings`, `POST /api/learning/run`. Các tham số có giới hạn và chỉ một lần học cùng lúc. Nếu backend tắt giữa chừng, ván chưa xử lý được giữ lại để chạy lại; model được áp dụng có thông tin run để đối chiếu khi khởi động.

## Chạy bằng terminal nếu cần

Ưu tiên giao diện AI Lab để nạp model tự động. Với CLI, dừng backend trước và chạy tại thư mục gốc, giữ Python 3.10:

```powershell
.\.venv\Scripts\python.exe -m backend.learn --epochs 3
```

Sau đó mở lại backend để nạp checkpoint nếu có bản mới. Lệnh này cần có ván hợp lệ mới kể từ lần xử lý trước. Khác với `backend.train`, nó bắt đầu từ trọng số hiện tại và dùng lịch sử người chơi.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v
```

Kiểm thử gồm thu thập/loại trùng, nguồn gốc nhãn, chống rò rỉ giữa các lần học, ngưỡng tự động, chạy API, fine-tuning thật, giữ model khi bị loại, sao lưu/nạp model khi đạt và rollback khi lỗi ghi. Kiểm thử trình duyệt `frontend/tests/human_learning_e2e.py` phải chạy trên database/model thử riêng vì có tạo ván và có thể áp dụng model thử.
