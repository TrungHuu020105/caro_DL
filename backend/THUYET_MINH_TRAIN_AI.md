# Thuyết minh huấn luyện và đánh giá AI CaroNet v2

Tài liệu phản ánh mã nguồn và các artifact hiện tại. Số liệu dưới đây đọc từ models/training.json, models/benchmark.json và data/training/manifest.json. Có thể đối chiếu checksum, nạp lại checkpoint, chơi lại dataset và chạy benchmark bằng Python 3.10.

## 1. Bài toán và vai trò deep learning

Cho bàn n×n, phía đang đi, số quân thắng K và loại luật, cần xếp hạng ô trống và đánh giá kết quả kỳ vọng. Game hỗ trợ n từ 3 đến 25, K từ 3 đến min(n,6), ba luật:

- Tự do: ít nhất K quân liên tiếp.
- Chính xác: đúng K, quá K không thắng.
- Caro: đúng K, không bị quân đối thủ chặn cả hai đầu; mép bàn không tính là quân chặn.

Mạng CNN thực sự được train bằng gradient và checkpoint thực sự được nạp khi chơi. Có ba engine để tách ảnh hưởng các thành phần: CNN thuần, heuristic thuần, engine lai. Game không sử dụng API AI bên ngoài.

Huấn luyện nền dùng **supervised policy distillation và outcome regression từ rollout**; vòng học bổ sung dùng các ván đấu người và replay train nền. Không gọi đây là AlphaZero hoặc reinforcement learning: policy vẫn được teacher phân tích lại, cập nhật bằng supervised loss.

## 2. Pipeline

~~~text
Rollout chiến thuật ngẫu nhiên hóa, đi hợp lệ và dừng khi kết thúc
    → gán toàn bộ ván cho train / validation / test
    → loại trùng quỹ đạo và vị trí theo xoay/lật
    → policy target từ teacher, value target từ kết quả ván
    → xuất NPZ, CSV và manifest có checksum
    → train CNN, augmentation trong batch
    → chọn checkpoint theo validation loss
    → đánh giá một lần trên test sau chọn checkpoint
    → chạy giải đấu CNN / heuristic / hybrid
~~~

Mã chính: dataset.py, train.py, app/network.py, app/rules.py, app/ai.py và benchmark.py.

## 3. Tạo dữ liệu hợp lệ

Mỗi rollout khởi đầu bàn trống, X đi trước và hai phía thay phiên. Chỉ đặt vào ô trống. Sau mỗi nước gọi đúng bộ luật đang chọn; dừng ngay khi thắng hoặc khi hết ô.

Chính sách sinh ván ưu tiên nước thắng/chặn đã phát hiện; ở thế khác dùng điểm chiến thuật và ngẫu nhiên hóa để tạo đa dạng. Teacher là heuristic giáo dục, không phải bộ giải tối ưu hoặc dữ liệu chuyên gia. Một số lựa chọn phòng thủ của engine chạy thật được kiểm tra kỹ hơn teacher sinh nhãn.

Các thế cờ được lấy từ tiền tố của những ván đã hoàn tất. Không thêm motif bằng cách ghi đè quân. Do đó số quân, lượt và lịch sử nguồn có thể kiểm chứng.

| Tập | Số mẫu | Số ván nguồn |
|---|---:|---:|
| Train | 6000 | 622 |
| Validation | 1200 | 131 |
| Test | 1200 | 135 |

Lần chạy này bao phủ **tất cả kích thước 3–25** và ba loại luật. Mục tiêu K được chọn phù hợp kích thước. Số mẫu giữa các kích thước/luật không hoàn toàn bằng nhau; AI Lab có chỉ số theo nhóm để nhận biết khác biệt.

### Tránh rò rỉ

Mỗi ván chỉ thuộc một tập. Không chỉ dựa vào ID: quỹ đạo nước đi cũng có canonical hash qua 8 phép xoay/lật. Vị trí được chuẩn hóa theo góc nhìn người đang đi và xoay/lật, kèm n/K/luật. Một tập fingerprint chung ngăn lặp vị trí trong hoặc giữa các tập.

Sau khi tách tập mới xoay/lật mẫu lưu và augmentation trong batch. Hai thế đối xứng không được đưa sang hai tập khác nhau.

load_dataset kiểm tra toàn bộ tensor, lượt, ô đệm và policy. Nó chơi lại quỹ đạo, xác nhận kết thúc đúng, không đi sau thắng, nhãn outcome và mỗi mẫu đúng là một tiền tố của ván nguồn. Toàn bộ dataset hiện tại đã qua kiểm tra này: không trùng vị trí đối xứng/quỹ đạo giữa các tập, không sai lượt hoặc trạng thái kết thúc dùng làm đầu vào.

### File dữ liệu

- data/training/caro_dataset.npz giữ toàn bộ tensor và quỹ đạo nguồn.
- train.csv, validation.csv, test.csv cho phép xem bàn và target.
- manifest.json ghi seed, cấu trúc mảng và checksum.

NPZ dùng canvas 25×25 để lưu mảng đồng nhất. sizes giữ kích thước thật; mask phân biệt ô đệm. CSV cắt về đúng bàn thật. Trainer cũng nhóm batch theo kích thước và cắt n×n trước suy luận.

## 4. Mã hóa 8 kênh

| Kênh | Nội dung |
|---|---|
| 0 | Quân của phía đang đi |
| 1 | Quân đối thủ |
| 2 | Ô trống trên bàn thật |
| 3 | Hằng số K/6 trên bàn thật |
| 4 | Mặt nạ bàn thật |
| 5–7 | One-hot luật tự do, chính xác, caro |

Ngoài bàn thật tất cả kênh bằng 0. Mạng phân biệt được cùng một thế cờ nhưng khác K hoặc khác luật, thay vì chỉ sửa luật ở bước kiểm tra thắng.

## 5. Kiến trúc CaroNet

Tổng số tham số: **34.250**.

~~~text
Input: (batch, 8, n, n)
  Conv 3×3, 24 kênh + ReLU
  3 residual blocks, mỗi block 2 Conv 3×3, 24 kênh
  ├─ Policy: Conv 1×1 24→16 + ReLU → Conv 1×1 16→1 → flatten
  └─ Value: masked average pooling → Linear 24→32 + ReLU → Linear 32→1 + tanh
~~~

Policy logits tại ô đã có quân/ô đệm được mask bằng số âm lớn trước softmax. Value pooling chỉ lấy trung bình trong mask. Không cần lớp fully connected phụ thuộc kích thước bàn cho policy.

Mạng có thể suy luận các kích thước khác nhau nhờ tích chập. Phiên bản này đồng thời có dữ liệu của nhiều kích thước, và có đánh giá theo từng kích thước; không chỉ dựa vào giả định tổng quát hóa từ 9×9.

## 6. Nhãn policy và value

Teacher tạo điểm các đoạn có thể nối, xét K và luật. Nếu có thắng ngay, policy tập trung vào ô thắng; nếu cần chặn, tập trung vào ô đe dọa. Các thế khác dùng phân phối mềm từ điểm chiến thuật. Đây là ưu tiên chọn nước, không phải xác suất thắng theo ô.

Value target:

~~~text
z = winner × player_to_move
winner: X=+1, O=−1, hòa=0
z ∈ {−1, 0, +1}
~~~

Ví dụ X thắng: mẫu ở lượt X có nhãn +1; mẫu ở lượt O có nhãn −1. Nhãn có nguồn gốc từ kết quả ván, không phải chênh lệch điểm tấn công/phòng thủ gần 0 của phiên bản cũ.

Value vẫn phụ thuộc chất lượng/chính sách rollout. Cùng một thế có thể có nhiều kết quả nếu người chơi lựa chọn khác. Không diễn giải trực tiếp value hoặc (value+1)/2 thành xác suất thắng đã hiệu chỉnh.

## 7. Huấn luyện và chọn checkpoint

| Tham số | Giá trị |
|---|---|
| Python | 3.10, chạy trong .venv |
| Seed | 42 |
| Optimizer | AdamW |
| Learning rate | 0,0015 |
| Weight decay | 0,0001 |
| Batch size tối đa | 128, nhóm theo kích thước |
| Epoch | 18 |
| Gradient clipping | norm tối đa 2 |
| Value loss weight | 0,5 |
| Thiết bị | CPU, tối đa 4 luồng PyTorch |

Loss:

~~~text
L_policy = mean[-sum(q * log_softmax(policy_logits))]
L_value  = mean[(predicted_value - z)^2]
L_total  = L_policy + 0.5 * L_value
~~~

Train và validation cùng định nghĩa total loss. Khi train, xoay/lật đồng thời tensor input và policy; value không đổi dưới đối xứng. Khi validation/test không augmentation ngẫu nhiên.

Trainer nền sau mỗi epoch đánh giá validation, lưu checkpoint nếu total validation loss tốt hơn. Test không tham gia chọn epoch hoặc hyperparameter. Sau chọn xong, nạp lại checkpoint tốt nhất rồi đánh giá test. Trainer nền bắt đầu từ seed. Vòng học người chơi ở mục 14 bắt đầu từ trọng số hiện tại với optimizer mới.

| Chỉ số lần chạy hiện tại | Kết quả |
|---|---:|
| Epoch tốt nhất | 18 |
| Validation total loss | 4.04132 |
| Validation policy top-1 | 44.75% |
| Test total loss | 4.10984 |
| Test policy CE | 3.65439 |
| Test policy top-1 | 45.50% |
| Test outcome value MSE | 0.91089 |
| Test MSE nếu luôn dự đoán value=0 | 0.91583 |
| Thời gian sinh/xuất dữ liệu và train/evaluate | 113.3 giây |

Top-1 là tỷ lệ argmax policy trùng argmax teacher. Khi teacher cho nhiều ô ngang nhau, top-1 chỉ đối chiếu một argmax; đây không phải tỷ lệ thắng hay điểm tối ưu của ván.

**Value chỉ hơn baseline 0 một ít trong lần chạy này.** Việc đổi nhãn và đưa value vào engine không tự chứng minh đầu value đã mạnh. Baseline và chỉ số nhóm được hiển thị để đánh giá đúng mức cải thiện.

Checkpoint SHA-256: 7f8744b20394b94309fb42932d8b02be8a717a5d1a46818d3529bc5dd372a956

Dataset tensor fingerprint: 2378e08f42f522919598182d6515795317a12f255291c53502d86a52a42be3d6

SHA-256 đọc theo chunk, không sử dụng hashlib.file_digest, tương thích Python 3.10. Seed cố định tái lập tensor với cùng generator/thư viện; timestamp trong metadata có thể làm hash cả file NPZ khác dù tensor giống nhau.

## 8. AI chọn nước khi chơi

### CNN thuần

Chỉ lấy policy argmax trên ô hợp lệ, không có bộ thắng/chặn hoặc minimax che kết quả CNN. Mức khó không thay đổi định nghĩa engine thuần này.

### Chiến thuật thuần

Không chạy mạng neural. Dùng điểm đoạn mở, kiểm tra thắng theo đúng luật, tìm phòng thủ loại bỏ các nước thắng ngay. Với luật caro, có thể chặn ở một đầu chung để vô hiệu hóa nhiều ô thắng; không kết luận thua chỉ vì đếm thấy hai ô thắng.

### Engine lai

Chạy CNN, chuẩn hóa điểm CNN/chiến thuật và kết hợp theo độ khó. Ưu tiên thắng ngay, sau đó phòng thủ khi cần. Mức khó trên 3×3 dùng minimax chính xác.

Trên bàn lớn hơn, mức khó dùng alpha-beta negamax với iterative deepening đến tối đa 4 ply, tối đa 10 ứng viên gốc và 6 ứng viên ở nhánh thường; nhánh buộc phòng thủ được xét riêng. Nếu hết ngân sách giữ kết quả vòng độ sâu đã hoàn thành. Tập ứng viên hữu hạn vẫn có thể bỏ lỡ nước tốt.

Đánh giá lá thực sự gọi policy/value CNN. Value được phối hợp với đánh giá tiềm năng đoạn cờ, thay vì chỉ trả value để trang trí. API ghi số nút, độ sâu hoàn thành, số lần dùng value, điểm thành phần và nguồn quyết định.

Ngân sách mặc định khi chơi khó là 250 ms cho phần tìm kiếm. Ngân sách kiểm tra tại biên nút; bước chuẩn bị/suy luận đang chạy không bị ngắt giữa chừng nên elapsed có thể vượt ngân sách. Đây không phải bảo đảm thời gian thực cứng.

## 9. Benchmark sức chơi

Chạy cùng thiết lập cho ba engine, với khai cuộc từ seed, mỗi cặp đổi bên X/O nhưng giữ nguyên khai cuộc. Lần đo:

- 54 ván trên 5×5, 9×9, 15×15; đủ ba luật, K=5.
- Seed 2026, một cặp đổi bên cho mỗi đối đầu/cấu hình.
- Ngân sách tìm kiếm tối đa 80 ms/nước, độ khó hard; đo cả độ trễ thực tế.
- Một engine đấu hai engine còn lại nên mỗi engine có 36 ván; tổng các cột số ván của engine đếm mỗi ván hai lần.
- Điểm = (thắng + 0,5 × hòa)/số ván.
- P95 là phân vị 95% thời gian mỗi nước.

| Engine | Ván | Thắng | Hòa | Thua | Điểm | TB/nước | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN thuần | 36 | 0 | 12 | 24 | 16.67% | 2.21 ms | 3.18 ms |
| Chiến thuật thuần | 36 | 17 | 17 | 2 | 70.83% | 0.74 ms | 1.4 ms |
| Kết hợp + alpha-beta | 36 | 13 | 19 | 4 | 62.50% | 72.05 ms | 84.6 ms |

Trong lượt đo này, CNN thuần còn yếu và engine lai chưa vượt heuristic về điểm. Tìm kiếm lai dùng nhiều thời gian hơn. Đây là kết quả cần trình bày, không đổi tên top-1 thành win rate hoặc chọn riêng trận thuận lợi để quảng bá.

54 ván là mẫu nhỏ; chưa suy ra Elo, sức mạnh chuyên nghiệp hoặc kết luận tổng quát. Thời gian ngân sách khiến số nút và nước đi có thể thay đổi theo tải máy, dù seed khai cuộc giữ nguyên. Bản ghi từng ván, luật, opening, nước đi, checksum và kết quả nằm trong benchmark.json.

AI Lab cho chọn engine, kích thước, luật, số cặp, seed và ngân sách; chạy nền, xem tiến độ và tải JSON. Chỉ một tác vụ cùng lúc. Nếu checksum model đổi, giao diện báo kết quả cũ.

## 10. Phân tích sau ván và giải thích

Replay kiểm tra trước từng nước: có bỏ lỡ thắng ngay không, có bỏ lỡ phòng thủ hợp lệ không, hay mọi phòng thủ một nước đều không cứu được thế cờ. Với luật caro, thử cả nước chặn đầu chung. Không gán nhãn “sai lầm chiến lược” chỉ từ value yếu.

Người chơi chọn mốc và **Thử phương án khác**. Backend dựng lại tiền tố, kiểm tra nước thử, cho AI đáp lại trong nhánh riêng. Ván gốc, lịch sử và thống kê không bị thay đổi.

Bảng điểm trong khi chơi hiển thị riêng policy, CNN, chiến thuật, kết hợp và tìm kiếm. Có trường hợp nước chọn không đứng đầu policy vì ưu tiên thắng/chặn hoặc tìm kiếm. Heatmap ghi rõ thế cờ nguồn, không gọi điểm policy là xác suất thắng.

## 11. Lưu ván, đồng hồ và tương thích

SQLite lưu snapshot sau mỗi thay đổi, gồm token riêng tư, bàn, revision, mã phòng, đồng hồ và pause. Lịch sử công khai theo capability ID không chứa token. Server khởi động lại dựng lại các ván và lock; frontend tiếp tục bằng nút chủ động trong cùng tab.

Pause AI đấu AI lưu paused_at. Khi resume, cộng thời gian pause vào mốc bắt đầu lượt và tổng thời gian pause, nên giữ nguyên thời gian còn lại. Không xử timeout hoặc đặt quân trong lúc pause. Hết giờ không thể được “cứu” bằng undo.

Vẫn chỉ triển khai một worker FastAPI; persistence chưa đồng bộ cache/lock nhiều process. Phiên tab cần giữ token; chưa có tài khoản/Elo.

Tất cả trainer/dataset/API chạy bằng .venv Python 3.10. NumPy trên Python 3.10 được giới hạn dưới 2.3; SHA-256 không dùng API chỉ có từ Python 3.11. Checkpoint v1 không tương thích 8 kênh: API báo rõ và fallback heuristic. Nếu metadata không khớp checkpoint, ẩn số liệu cũ.

## 12. Tái lập và kiểm tra khi bảo vệ

Tại thư mục gốc:

~~~powershell
.\.venv\Scripts\python.exe -m backend.train --samples 6000 --epochs 18 --seed 42
.\.venv\Scripts\python.exe -m backend.benchmark --pairs 1 --sizes 5 9 15 --rules freestyle exact blocked --budget-ms 80
.\.venv\Scripts\python.exe -m unittest discover -s backend/tests -v
~~~

Frontend:

~~~powershell
cd frontend
npm run build
npm run format:check
~~~

Sau khi hai server đang chạy, tại thư mục gốc, Python đã cài Playwright/Chromium:

~~~powershell
python frontend/tests/e2e.py
python frontend/tests/improvements_e2e.py
~~~

Các kiểm thử backend bao gồm luật/kích thước, minimax không thua 3×3, dữ liệu/đối xứng/nguồn gốc, checkpoint, engine độc lập, value, pause/restart, replay, benchmark và toàn bộ vòng học người chơi: thu thập, phân tích lại policy, chống rò rỉ giữa các lần học, cổng chấp nhận/từ chối, nạp model, sao lưu và phục hồi khi lỗi ghi file.

Hai bộ kiểm thử trình duyệt bao gồm chế độ chờ/Bắt đầu, online/kết nối lại, di động, biểu đồ, engine/điểm thành phần, tác vụ thí nghiệm, tải JSON, phân tích replay và giữ đồng hồ qua tải lại trang.

## 13. Giới hạn cần trình bày trung thực

- Dữ liệu nền từ rollout và dữ liệu bổ sung từ lịch sử người chơi; policy vẫn dựa teacher, chưa có chuyên gia hoặc cập nhật policy theo reward.
- Value hơn baseline ít, chưa được calibration; không hứa tìm kiếm dùng value luôn mạnh hơn.
- Phân bố kích thước/luật/giai đoạn chưa cân bằng hoàn toàn, mỗi nhóm test có số mẫu hữu hạn.
- Teacher top-1 có giới hạn khi nhiều nước ngang nhau và không thay thế benchmark.
- Search giới hạn ứng viên/độ sâu và ngân sách mềm; có thể bỏ lỡ chiến thuật xa.
- Benchmark nhỏ và phụ thuộc tải máy; chưa có Elo/kiểm định sức mạnh rộng.
- Persistence phục hồi một process; chưa phải hệ thống nhiều worker hoặc tài khoản đám mây.

Hướng nghiên cứu tiếp theo có thể là dữ liệu expert, cân bằng giai đoạn, teacher tìm kiếm mạnh hơn, đánh giá value theo giai đoạn, nhiều seed/giải đấu lớn hơn hoặc RL tự chơi thực sự. Các hướng này cần thực nghiệm riêng, không trình bày như tính năng đã hoàn thành.

## 14. Học từ các ván đấu với người thật

Vòng mới nằm trong learning.py, learning_data.py, app/experience.py và bảng human_games/human_position_split trong SQLite. Ván AI, local và online được thu khi kết thúc bằng chuỗi thắng hoặc đầy bàn; bỏ qua spectate, đầu hàng, timeout, quỹ đạo lặp và dữ liệu không hợp lệ. Hệ thống quét lại lịch sử cũ khi khởi động.

Tách cả ván theo hash cố định: khoảng 20% thuộc validation. Ledger phân công vị trí canonical cố định ngăn một vị trí từng train trở thành validation ở lần sau. Mọi vị trí trùng train/validation/test nền bị loại khỏi phần người chơi.

Policy được tái phân tích theo luật, không học máy móc nước sai của người chơi; value là kết quả ván theo phía đang đi. Lấy tối đa 256 mẫu train người chơi, 128 mẫu validation người chơi và tối đa 1.024 mẫu train nền để ôn lại. Không ghi đè dataset nền, không dùng test để chọn ứng viên.

Fine-tuning từ trọng số hiện tại bằng AdamW, learning rate 0,0002, mặc định 3 epoch. Chọn epoch theo validation nền cộng validation người chơi. Sau đó so sánh loss với incumbent, và đấu 16 ván đổi bên ở cả CNN/lai. Cổng yêu cầu tổng điểm ít nhất 55%, mỗi biến thể ít nhất 50%, loss nền không tăng quá 2% + 0,01, loss validation người không tăng quá 10% + 0,02.

Nếu đạt, sao lưu model cũ, lưu checkpoint/metadata mới và nạp engine mới cho lượt AI tiếp theo. Các lượt suy nghĩ đang chạy giữ đối tượng engine cũ, không bị thay trọng số giữa tìm kiếm. Nếu không đạt, giữ model hiện tại; lưu ứng viên và báo cáo để xem lại. File lock ngăn trainer CLI ghi trùng. Dữ liệu mới đến trong lúc train được để cho đợt sau.

Mặc định tự thử sau 5 ván mới, tối thiểu 8 vị trí train sau lọc; có thể tắt tự học, đổi ngưỡng/epoch hoặc chạy ngay trong AI Lab. Đây là học bổ sung theo lô từ tương tác người chơi, không phải cập nhật gradient ngay sau từng cú nhấp hay bảo đảm thắng nhiều hơn sau mỗi ván.

Một lần chạy đã được kiểm tra trên 35 ván lưu sẵn: 256 mẫu train người chơi, 128 mẫu kiểm định và 1.024 mẫu nền; 3 epoch. Validation loss lần lượt 4,03101 → 4,02475 → 4,02232. Đấu model cũ đạt 6 thắng/3 hòa/7 thua, 46,88% điểm. Vì không đạt ngưỡng, model gốc được giữ nguyên. Chi tiết ở [HOC_TU_NGUOI_CHOI.md](HOC_TU_NGUOI_CHOI.md).
