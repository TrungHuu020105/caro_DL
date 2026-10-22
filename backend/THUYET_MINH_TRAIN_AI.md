**THUYẾT MINH QUÁ TRÌNH HUẤN LUYỆN AI CARO LAB**

Tài liệu này mô tả đúng phiên bản mã nguồn và checkpoint hiện có. Các kết quả thực nghiệm lấy từ `models/training.json` và được đối chiếu bằng cách nạp lại `models/caro_policy.pt`. Các thống kê trùng lặp được tính trên tensor đầu vào trong dataset đã xuất, không phải số liệu ước đoán.

1. **Mô tả phương pháp bằng một câu**

   Caro Lab huấn luyện mạng CNN residual tên CaroNet bằng học có giám sát, sử dụng các thế cờ tổng hợp và nhãn do một bộ chấm điểm chiến thuật tạo ra. Khi chơi, hệ thống kết hợp policy của CNN với luật thắng/chặn và thuật toán tìm kiếm.

   Có thể gọi cách huấn luyện này là **học bắt chước bộ đánh giá chiến thuật**, hoặc **supervised distillation from a tactical teacher**. Teacher trong dự án là thuật toán viết tay; student là CNN được cập nhật trọng số bằng backpropagation.

   Dự án hiện chưa huấn luyện bằng reinforcement learning, chưa có vòng lặp AlphaZero/MCTS và chưa tự học thêm từ người chơi. Chế độ AI đấu AI là chế độ chơi, không phải một tiến trình huấn luyện tự động.

   ```text
   Sinh thế cờ bằng NumPy
          ↓
   Teacher chiến thuật tạo policy/value target
          ↓
   Mã hóa bàn cờ thành tensor 5 kênh
          ↓
   Xuất train/validation ra NPZ và CSV
          ↓
   Huấn luyện CaroNet bằng PyTorch
          ↓
   Đánh giá validation sau từng epoch
          ↓
   Lưu checkpoint có validation policy loss thấp nhất
          ↓
   FastAPI nạp checkpoint để suy luận khi chơi
   ```

2. **Các file cần mở khi trình bày**

   | File | Vai trò |
   | --- | --- |
   | [dataset.py](dataset.py) | Sinh thế cờ, tạo nhãn, xuất và nạp dataset |
   | [app/rules.py](app/rules.py) | Chấm điểm chiến thuật và kiểm tra chuỗi thắng |
   | [app/network.py](app/network.py) | Kiến trúc CNN, residual block và hàm `encode()` |
   | [train.py](train.py) | Loss, optimizer, vòng lặp train/validation, lưu best model |
   | [app/ai.py](app/ai.py) | Suy luận CNN và kết hợp chiến thuật/tìm kiếm |
   | [data/training/train.csv](data/training/train.csv) | 6.000 mẫu train ở dạng có thể mở xem |
   | [data/training/validation.csv](data/training/validation.csv) | 1.200 mẫu validation |
   | [data/training/caro_dataset.npz](data/training/caro_dataset.npz) | Các tensor và bàn cờ gốc dùng để huấn luyện |
   | [data/training/manifest.json](data/training/manifest.json) | Schema, seed, kích thước và checksum của dữ liệu |
   | [models/caro_policy.pt](models/caro_policy.pt) | Trọng số của model được chọn |
   | [models/training.json](models/training.json) | Kết quả theo epoch, best epoch và liên kết với dataset |

   `data/caro.sqlite3` lưu kết quả các ván đã chơi. File này không được trainer dùng làm dữ liệu huấn luyện trong phiên bản hiện tại.

3. **Bài toán mà mạng neural cần học**

   Một mẫu đầu vào gồm trạng thái bàn cờ, quân đang đến lượt và số quân liên tiếp cần có để thắng. Mạng học đồng thời hai đầu ra:

   - **Policy:** phân phối điểm ưu tiên trên các ô trống. Với bàn 9×9, đầu ra có 81 phần tử.
   - **Value:** một số trong khoảng [-1, 1], được huấn luyện để khớp điểm thế trận do teacher tạo.

   Policy trả lời “nên ưu tiên đi ở đâu?”. Value là một nhiệm vụ phụ đánh giá thế trận. Policy ở đây không phải phân phối xác suất thắng của từng nước đi; value cũng chưa phải xác suất thắng đã được hiệu chỉnh.

   Mạng dùng góc nhìn của người đang đi. Vì vậy, ở kênh đầu vào, “quân mình” có thể là X hoặc O, tùy lượt.

4. **Dữ liệu được tạo ra như thế nào?**

   Hàm `generate_split()` trong `dataset.py` sinh từng mẫu theo các bước sau:

   a. Tạo bàn 9×9 gồm toàn ô trống. Quy ước dữ liệu gốc: X = 1, O = -1, ô trống = 0.

   b. Chọn mục tiêu thắng K theo phân phối:

   | K | Xác suất sinh | Số mẫu train thực tế | Số mẫu validation thực tế |
   | --- | ---: | ---: | ---: |
   | 3 quân | 20% | 1.179 | 250 |
   | 4 quân | 20% | 1.185 | 244 |
   | 5 quân | 50% | 3.023 | 598 |
   | 6 quân | 10% | 613 | 108 |
   | Tổng | 100% | 6.000 | 1.200 |

   Đây là xác suất lấy mẫu, nên số lượng thực tế không nhất thiết bằng đúng tỷ lệ nhân với tổng mẫu.

   c. Chọn ngẫu nhiên từ 0 đến 32 vị trí khác nhau và gán X/O xen kẽ. Người đến lượt được xác định từ số quân đã gán ban đầu.

   d. Với xác suất 65%, thử chèn một motif chiến thuật: chọn một trong bốn hướng ngang, dọc, chéo xuống hoặc chéo lên; nếu đoạn dài K nằm trong bàn, đặt K-1 quân cùng phía và để trống một vị trí trong đoạn. Motif có thể thuộc phía đang đi hoặc đối thủ, giúp tạo ví dụ nối chuỗi và chặn chuỗi.

   **65% là xác suất thử chèn, không phải cam kết 65% mẫu cuối có motif**, vì đoạn được chọn có thể vượt biên và bị bỏ qua.

   e. Xoay bàn ngẫu nhiên 0°, 90°, 180° hoặc 270°; sau đó lật trái–phải với xác suất 50%. Nhãn được tính sau biến đổi, nên khớp với bàn cờ cuối cùng.

   Đây là augmentation lúc tạo dataset. Một mẫu được biến đổi rồi lưu lại; chương trình không tăng 6.000 mẫu thành 48.000 mẫu, và không xoay/lật lại mỗi epoch.

   f. Chạy teacher trên bàn đã tạo để lấy policy target và value target; mã hóa đầu vào và lưu mẫu.

   Có hai tập được sinh riêng: train dùng seed 42, validation dùng seed 43. Validation có `max(100, samples // 5)` mẫu. Với 6.000 mẫu train, tổng dữ liệu là 7.200 mẫu, tỷ lệ train:validation = 5:1, tương đương khoảng 83,33%:16,67%, không phải 80%:20% của tổng.

   Các mẫu là thế cờ tổng hợp, không phải 6.000 ván hoàn chỉnh. Việc chèn motif có thể làm thay đổi cân bằng số quân hoặc tạo trạng thái đã có chuỗi thắng. Generator hiện chưa bảo đảm mọi mẫu là một trạng thái hợp lệ xuất phát từ một ván chơi thực sự.

5. **Teacher tạo nhãn policy ra sao?**

   Hàm `segments()` liệt kê tất cả các đoạn dài K nằm hoàn toàn trong bàn, theo bốn hướng. Với mỗi đoạn, `tactical_scores()` đếm số quân mình và số quân đối thủ.

   Nếu đoạn không có quân đối thủ, teacher cộng điểm tấn công cho các ô trống của đoạn đó. Nếu đoạn không có quân mình, teacher cộng điểm phòng thủ. Đoạn có cả hai bên không đóng góp các điểm này. “Đoạn mở” ở đây nghĩa là đoạn không chứa quân cản bên trong, không có nghĩa thuật toán đã đánh giá đầy đủ hai đầu mở theo mọi luật caro.

   Dãy trọng số cơ sở là:

   ```python
   [0, 2, 18, 180, 20000, 2000000, 200000000]
   ```

   Sau đó `weights[K - 1] = 1e7`: đoạn có K-1 quân cùng phía được ưu tiên rất cao vì đi vào ô trống còn lại có thể hoàn thành chuỗi. Điểm phòng thủ nhân 0,96 so với điểm tấn công cùng cấu hình. Một ô thuộc nhiều đoạn có lợi sẽ nhận tổng các đóng góp.

   Teacher còn cộng một điểm ưu tiên trung tâm:

   ```text
   center_bonus(r,c) = 1 / (1 + |r - center| + |c - center|)
   ```

   Ô đã có quân được đánh dấu điểm -1 để loại khỏi các nước đi hợp lệ.

   Ví dụ minh họa với K = 5, một hàng có cấu hình:

   ```text
   Cột:  A B C D E F G H I
         O X X X X . . . .
   ```

   Khi X đến lượt, ô F hoàn thành năm quân X liên tiếp nên được teacher ưu tiên cao. Đây là ví dụ giải thích nguyên lý, không phải trích một dòng cụ thể từ CSV.

   Điểm teacher được đổi thành phân phối nhãn mềm. Với ô trống i có điểm s_i:

   ```text
   z_i = 1,1 × log(1 + max(s_i, 0))
   q_i = exp(z_i - max(z)) / Σ_j exp(z_j - max(z))
   ```

   Ô đã có quân nhận logit -1e9 và xác suất gần bằng 0. Trừ logit lớn nhất giúp tính softmax ổn định hơn. `q` có tổng bằng 1 và có thể phân bổ trọng số cho nhiều ô, thay vì chỉ có một nhãn one-hot.

   Cột `teacher_move` trong CSV là tọa độ của `argmax(q)` để đọc nhanh. Trainer dùng toàn bộ `policy_json`/tensor `policy`, không chỉ dùng cột này.

6. **Nhãn value được tạo ra như thế nào?**

   Teacher được chạy từ hai góc nhìn. Gọi A là điểm ô tốt nhất từ góc nhìn người đang đi, D là điểm ô tốt nhất từ góc nhìn đối thủ:

   ```text
   v_target = tanh((log(1 + A) - log(1 + D)) / 4)
   ```

   Tanh giới hạn giá trị đầu ra trong [-1, 1]. Dấu dương/âm biểu diễn sự chênh lệch giữa hai điểm heuristic. Đây không phải nhãn lấy từ kết quả thắng/thua thật.

   Trong dataset hiện có, value target thực tế chỉ trải trong khoảng khoảng **-0,010205 đến +0,010205**. Nguyên nhân là cả hai phía đều được chấm bằng tổng tấn công và phòng thủ có hệ số gần nhau, nên hai điểm lớn nhất thường rất gần nhau. Vì vậy chưa nên trình bày value head như một bộ tiên đoán thắng/thua mạnh. Cần cải thiện nhãn này nếu muốn dùng value cho tìm kiếm sau này.

7. **Đầu vào 5 kênh của CNN**

   Hàm `encode()` tạo tensor `(5, N, N)` kiểu float32:

   | Kênh | Giá trị tại mỗi ô | Ý nghĩa |
   | --- | --- | --- |
   | 0 | 1 nếu có quân của người đang đi, còn lại 0 | Quân mình |
   | 1 | 1 nếu có quân của đối thủ, còn lại 0 | Quân đối thủ |
   | 2 | 1 nếu ô trống, còn lại 0 | Ô được phép đặt quân |
   | 3 | K/6 tại mọi ô | Điều kiện số quân để thắng |
   | 4 | 1 tại mọi ô trong bàn | Kênh mặt nạ bàn |

   Với bàn 9×9 và batch 128, tensor đầu vào có kích thước `(128, 5, 9, 9)`. Ba chiều cuối là số kênh, chiều cao và chiều rộng; chiều đầu là số mẫu trong batch.

   Kênh 4 hiện luôn bằng 1 trên bàn cờ. Trainer không ghép các bàn kích thước khác nhau bằng padding trong cùng batch. Loại luật `freestyle/exact/blocked` cũng chưa được đưa thành kênh đầu vào.

8. **Kiến trúc CaroNet và số tham số**

   ```text
   Input: B × 5 × N × N
          ↓
   Conv 3×3: 5 → 24, padding=1, stride=1 + ReLU
          ↓
   3 residual blocks, giữ nguyên 24 kênh
          ↓
   Feature map: B × 24 × N × N
          ├─ Policy: Conv 1×1 24→16 + ReLU → Conv 1×1 16→1
          │          → flatten → mask ô đã có quân
          │          → B × N² logits
          └─ Value: AdaptiveAvgPool2d(1) → flatten 24 chiều
                     → Linear 24→32 + ReLU → Linear 32→1 + Tanh
                     → B giá trị
   ```

   Mỗi residual block thực hiện:

   ```text
   output = ReLU(x + Conv2(ReLU(Conv1(x))))
   ```

   Nhánh cộng tắt giữ lại x và giúp việc tối ưu các lớp sâu thuận lợi hơn; đây là ý tưởng residual learning. CaroNet là kiến trúc nhỏ tự định nghĩa, không phải ResNet-18 và không dùng trọng số ImageNet. Tham khảo cơ sở của kết nối residual trong [bài báo của He và cộng sự](https://arxiv.org/abs/1512.03385).

   Conv 3×3 học quan hệ giữa các ô lân cận; padding=1 và stride=1 giữ kích thước không gian. Các khái niệm này được mô tả trong [tài liệu Conv2d của PyTorch](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html).

   | Thành phần | Số tham số |
   | --- | ---: |
   | Stem: Conv 5→24, kernel 3×3 | 1.104 |
   | Ba residual block, tổng sáu Conv 24→24 | 31.248 |
   | Policy head | 417 |
   | Value head | 833 |
   | Tổng | **33.602** |

   Ví dụ, stem có `24 × 5 × 3 × 3 + 24 = 1.104` tham số, gồm trọng số và bias.

   Policy head chỉ dùng tích chập nên không cố định số ô đầu ra là 81: bàn N×N tạo N² logits. Value head dùng [AdaptiveAvgPool2d](https://docs.pytorch.org/docs/2.9/generated/torch.nn.AdaptiveAvgPool2d.html) để đưa feature map về 24×1×1 trước hai lớp Linear. Nhờ vậy cùng bộ trọng số nhận được nhiều kích thước bàn cờ.

   Tuy nhiên, chạy được nhiều kích thước là đặc tính kiến trúc; chất lượng chơi trên 15×15 hoặc 25×25 cần được đánh giá riêng vì dữ liệu train hiện chỉ có 9×9.

   Trong `forward()`, logit của ô đã có quân được thay bằng -1e9. Khi softmax, các ô này gần như không được chọn.

9. **Hàm mất mát và quá trình cập nhật trọng số**

   Gọi q là policy target của teacher, p là policy dự đoán của mạng, v là value target và v_hat là value dự đoán:

   ```text
   L_policy = trung bình theo batch của [-Σ_i q_i × log(p_i)]
   L_value  = trung bình theo batch của [(v_hat - v)²]
   L_total  = L_policy + 0,25 × L_value
   ```

   Policy loss là cross entropy với nhãn xác suất mềm. PyTorch hỗ trợ loại target này; trong dự án, công thức được viết trực tiếp bằng `F.log_softmax()` rồi nhân với q. Xem [CrossEntropyLoss của PyTorch](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html).

   Không nên kỳ vọng policy loss phải giảm về 0: khi target q là phân phối mềm, ngay cả p = q thì cross entropy vẫn bằng entropy của q, có thể lớn hơn 0.

   Một batch đi qua các bước:

   ```python
   logits, predicted_value = model(batch_inputs)
   loss = policy_loss + 0.25 * value_loss
   optimizer.zero_grad()
   loss.backward()
   torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
   optimizer.step()
   ```

   Forward tạo dự đoán; loss đo sai khác so với teacher; backward tính gradient; optimizer dùng gradient để cập nhật các trọng số/bias. Clip gradient giới hạn chuẩn gradient ở ngưỡng 2, không phải cắt từng tham số vào khoảng [-2, 2].

   | Thông số | Giá trị hiện tại |
   | --- | --- |
   | Optimizer | AdamW |
   | Learning rate | 0,0015 |
   | Weight decay | 0,0001 |
   | Batch size mặc định | 128 |
   | Epoch | 12 |
   | Hệ số value loss | 0,25 |
   | Gradient clipping max norm | 2 |
   | Seed khởi tạo/thứ tự batch | 42 |
   | Thiết bị trong trainer hiện tại | CPU |

   AdamW là optimizer có weight decay tách khỏi bước tích lũy gradient; định nghĩa tham khảo ở [tài liệu AdamW](https://docs.pytorch.org/docs/2.9/generated/torch.optim.AdamW.html). Weight decay là một biện pháp điều chuẩn, không phải bằng chứng mô hình đã hết overfitting.

   Với 6.000 mẫu và batch 128, mỗi epoch có 47 batch: 46 batch đủ 128 mẫu và một batch 112 mẫu. Sau 12 epoch có 564 bước cập nhật optimizer. Mỗi epoch dùng `torch.randperm()` để xáo thứ tự mẫu.

   Code hiện chưa dùng learning-rate scheduler, early stopping, Dropout hay BatchNorm. Không nên đưa các kỹ thuật đó vào phần “đã triển khai”.

10. **Validation và cách lưu model tốt nhất**

   Cuối mỗi epoch, model chuyển sang `eval()` và chạy validation trong `torch.inference_mode()`. Validation không cập nhật trọng số.

   Hai chỉ số được tính là:

   - `val_loss`: trung bình policy cross entropy trên 1.200 mẫu validation.
   - `top1_accuracy`: tỷ lệ mẫu mà `argmax(policy_mạng)` bằng `argmax(policy_teacher)`.

   Cần phân biệt: `loss` của tập train gồm policy CE cộng 0,25 × value MSE; `val_loss` hiện chỉ có policy CE. Vì định nghĩa khác nhau, không nên diễn giải chênh lệch hai đường giống một thí nghiệm train/validation sử dụng cùng hoàn toàn một loss.

   Điều kiện chọn checkpoint là:

   ```python
   if val_loss < best_val_loss:
       best_val_loss = val_loss
       best_epoch = epoch + 1
       torch.save(model.state_dict(), temporary_file)
       temporary_file.replace(checkpoint_path)
   ```

   File [models/caro_policy.pt](models/caro_policy.pt) luôn là checkpoint có validation policy loss thấp nhất **trong lần chạy đó**. Nếu epoch cuối kém hơn, trainer giữ model tốt hơn đã lưu trước đó.

   File này lưu state_dict của mạng, không lưu optimizer state để resume. Một lần gọi trainer mới khởi tạo mạng mới từ seed; không tự so sánh với model của lần chạy trước và không tự huấn luyện tiếp checkpoint cũ.

11. **Kết quả thực tế đang có**

   | Epoch | Train loss: CE + 0,25 MSE | Validation policy CE | Teacher top-1 |
   | ---: | ---: | ---: | ---: |
   | 1 | 3,88980 | 3,77506 | 16,17% |
   | 2 | 3,64976 | 3,53235 | 24,75% |
   | 3 | 3,35606 | 3,25774 | 35,00% |
   | 4 | 3,11120 | 3,12173 | 36,00% |
   | 5 | 2,95777 | 2,96798 | 41,42% |
   | 6 | 2,83544 | 2,87467 | 45,00% |
   | 7 | 2,74194 | 2,78426 | 47,50% |
   | 8 | 2,68847 | 2,76776 | 47,92% |
   | 9 | 2,64571 | 2,72221 | 48,00% |
   | 10 | 2,61191 | 2,69941 | 46,00% |
   | 11 | 2,59089 | 2,69249 | 51,33% |
   | 12 | 2,57109 | **2,67932** | 51,00% |

   Best epoch là **12**, do có validation policy loss nhỏ nhất. Epoch 11 có top-1 cao hơn một chút nhưng đó không phải tiêu chí chọn checkpoint.

   Khi nạp lại checkpoint và chạy đúng 1.200 mẫu validation, kết quả kiểm tra là loss **2,679322255...**, khớp giá trị làm tròn 2,67932 trong JSON; top-1 đúng **612/1.200 = 51%**.

   Diễn giải phù hợp: mô hình đã học cách xấp xỉ teacher trên phân phối thế cờ tổng hợp. Không được đổi “51% khớp teacher” thành “AI thắng 51% số ván”. Khi teacher có nhiều nước đồng hạng, phép đo argmax còn phụ thuộc ô nào được chọn để phá hòa.

   Run hiện tại ghi thời gian 24,9 giây trong metadata. Khoảng này bao gồm chuẩn bị/nạp dữ liệu và vòng train/validation, không phải một benchmark chuẩn hóa giữa các máy.

12. **Từ model đã train đến AI đang chơi**

   `AIEngine` trong `app/ai.py` nạp checkpoint lên CPU, đặt model ở `eval()`, dùng `torch.inference_mode()` để suy luận. CNN trả policy cùng value; hệ thống sau đó chọn nước theo engine lai.

   Đầu tiên engine thử các ô hợp lệ theo bộ luật đang chọn để tìm nước thắng ngay. Nếu có, engine đi vào nước thắng. Ở mức vừa/khó, engine cũng ưu tiên chặn nước thắng ngay của đối thủ.

   Nếu chưa rơi vào các trường hợp đó, engine kết hợp điểm neural với điểm chiến thuật. Policy được chia cho xác suất lớn nhất, heuristic được log rồi chuẩn hóa theo điểm lớn nhất; hai thang điểm này được trộn với hệ số:

   | Mức | Hệ số neural | Hệ số heuristic | Lựa chọn bổ sung |
   | --- | ---: | ---: | --- |
   | Dễ | 0,90 | 0,10 | Lấy ngẫu nhiên có trọng số trong tối đa 6 ứng viên tốt |
   | Vừa | 0,55 | 0,45 | Chọn ứng viên có điểm kết hợp cao nhất |
   | Khó | 0,35 | 0,65 | Thử tối đa 12 ứng viên và đánh giá nguy cơ đối thủ phản công |

   Những hệ số này là tham số do lập trình viên đặt trong engine sau huấn luyện; chúng không phải tỷ lệ phần trăm sức mạnh do CNN đóng góp, cũng không được optimizer tự học.

   Với bàn lớn, phần tìm kiếm ở mức khó đặt thử một quân của mình, tính heuristic tốt nhất của đối thủ và kiểm tra đối thủ có thể thắng ngay hay không. Nó chưa phải MCTS hoặc minimax sâu trên toàn bộ cây nước đi.

   Riêng bàn **3×3 ở mức khó**, khi không xử lý thắng/chặn ngay, engine dùng minimax để giải thế cờ; policy CNN dùng để phá hòa giữa các lựa chọn ngang nhau. Việc chế độ này không thua trong kiểm thử không chứng minh riêng CNN đã học chơi hoàn hảo.

   **Value head hiện có được train và được trả ra kết quả phân tích, nhưng không được dùng trong công thức chọn nước.** Không nên nói engine đang dùng value neural để đánh giá các nhánh tìm kiếm.

   Khi thiếu checkpoint, chương trình báo đang dùng chiến thuật dự phòng; khi có checkpoint, CNN thực sự được gọi trong `predict()`.

13. **Giới hạn của thực nghiệm và dữ liệu**

   Đây là những điểm cần trình bày rõ nếu giảng viên hỏi chất lượng đánh giá:

   a. **Dataset chưa lọc trùng giữa train và validation.** Hai tập sinh bằng seed khác nhau, nhưng điều đó không bảo đảm khác nhau hoàn toàn. So sánh byte của tensor đầu vào `(5, 9, 9)` trên dữ liệu hiện tại cho kết quả:

   | Kiểm tra | Kết quả |
   | --- | ---: |
   | Tensor đầu vào khác nhau trong 6.000 mẫu train | 5.879 |
   | Tensor đầu vào khác nhau trong 1.200 mẫu validation | 1.175 |
   | Số tensor khác nhau xuất hiện ở cả hai tập | 13 |
   | Số dòng validation có đầu vào đã xuất hiện trong train | 37/1.200, khoảng 3,08% |

   Vì vậy nên gọi đây là validation sinh bằng seed riêng, chưa gọi là một tập hoàn toàn không trùng dữ liệu train. Chưa có test set thứ ba được giữ riêng cho đánh giá cuối.

   b. **Teacher là heuristic.** Chất lượng nhãn phụ thuộc quy tắc chấm điểm; mô hình không được học trực tiếp từ chuyên gia hoặc kết quả thắng/thua thật.

   c. **Dữ liệu có thể không phải thế cờ hợp lệ của một ván thực.** Generator chưa dừng khi xuất hiện thắng và có thể thay quân khi chèn motif.

   d. **Value target tập trung gần 0**, và value chưa tham gia quyết định nước đi. Cần thiết kế nhãn và đánh giá value riêng nếu muốn dùng nó cho tìm kiếm.

   e. **Train trên 9×9, suy luận nhiều kích thước.** Việc output có đúng shape ở 25×25 không tương đương đã chứng minh chơi mạnh ở 25×25.

   f. **Luật exact/blocked không nằm trong input hoặc teacher.** Việc kiểm tra đúng luật của các nước thắng/chặn do engine và bộ luật xử lý, không phải mạng đã được train riêng cho từng luật.

   g. **Chưa có benchmark thắng/thua giữa CNN thuần, heuristic thuần và bản lai.** Chưa có cơ sở để công bố tỷ lệ thắng, Elo hoặc khẳng định bản lai mạnh hơn một mức cụ thể.

   h. **Không học online từ các ván mới.** Trọng số trong bộ nhớ khi chơi không đổi bởi một ván thắng/thua; muốn thay model phải chạy trainer và nạp checkpoint mới.

   Những giới hạn này mô tả phạm vi của một mô hình học tập, đồng thời gợi ý thí nghiệm tiếp theo; không làm thay đổi việc CNN đã được huấn luyện và được sử dụng thật trong hệ thống.

14. **Cách chạy để giảng viên kiểm tra lại**

   Chạy tại thư mục gốc `LabCaro` bằng Python có các thư viện cần thiết.

   Chỉ sinh và xuất dữ liệu, giữ nguyên checkpoint hiện có:

   ```powershell
   python -m backend.train --export-only --samples 6000 --seed 42
   ```

   Train lại từ đúng dữ liệu đã lưu, nhưng lưu model sang thư mục thí nghiệm để dễ đối chiếu với bản đang có:

   ```powershell
   python -m backend.train --dataset backend/data/training/caro_dataset.npz --epochs 12 --model-dir backend/models/demo
   ```

   Nếu bỏ `--model-dir`, trainer ghi kết quả vào `backend/models/`. Muốn tự sinh dataset mới rồi train ngay:

   ```powershell
   python -m backend.train --samples 6000 --epochs 12 --batch-size 128 --seed 42
   ```

   Khi trình bày CSV, chỉ vào `board_json`, `player_to_move`, `win_length`, `policy_json` và `value_target`. NPZ mới là file trainer nạp trực tiếp. Mỗi file NPZ còn chứa bàn cờ gốc, quân đang đi và các tensor của cả hai tập.

   Seed giúp tái lập cách sinh dữ liệu và thứ tự train; không nên bảo đảm kết quả bit-identical trên mọi phiên bản thư viện và phần cứng. Manifest có checksum file và fingerprint nội dung mảng để xác nhận bộ dữ liệu; training.json có checksum checkpoint và tham chiếu dataset của lần train.

15. **Đoạn nói mẫu khoảng hai phút**

   “Trong đồ án này, em xây dựng AI caro theo hướng kết hợp mạng neural với thuật toán chiến thuật. Phần deep learning là CaroNet, một CNN nhỏ có ba residual block và tổng cộng 33.602 tham số, được cài đặt bằng PyTorch.

   Em sinh 6.000 thế cờ train và 1.200 thế cờ validation trên bàn 9×9. Mỗi mẫu có trạng thái bàn cờ, bên đến lượt và mục tiêu từ ba đến sáu quân liên tiếp. Bộ chấm điểm chiến thuật tạo nhãn mềm cho các ô có thể đi và một nhãn điểm thế trận. Dữ liệu còn được xoay và lật để đa dạng hóa hướng xuất hiện của các mẫu chiến thuật.

   Đầu vào mạng gồm năm kênh. Đầu policy dự đoán mức ưu tiên cho từng ô; đầu value học điểm thế trận phụ. Em tối ưu tổng cross entropy của policy và 0,25 lần MSE của value bằng AdamW, learning rate 0,0015, batch size 128, trong 12 epoch.

   Sau mỗi epoch, em đánh giá validation và lưu model có policy loss thấp nhất. Checkpoint hiện tại ở epoch 12, validation loss 2,67932. Policy argmax trùng teacher ở 612 trên 1.200 mẫu, tương đương 51%; đây không phải tỷ lệ thắng.

   Khi chơi, engine kết hợp policy với kiểm tra thắng, chặn và tìm kiếm. Value head hiện chưa dùng để chọn nước. Hạn chế của phiên bản này là dữ liệu tổng hợp, còn một số mẫu trùng giữa hai tập và chưa có benchmark thắng/thua riêng cho CNN. Hướng phát triển của em là làm sạch và tách dữ liệu tốt hơn, đánh giá CNN riêng, sau đó cải thiện nhãn bằng các ván chơi hợp lệ.”

16. **Những câu hỏi giảng viên có thể đặt ra**

   **“Đây có phải reinforcement learning không?”** — Chưa. Có target do teacher tạo sẵn và dùng loss có giám sát; chưa tối ưu reward từ các ván tự đấu.

   **“Nếu đã có teacher thì CNN học để làm gì?”** — CNN học xấp xỉ một quy tắc đánh giá từ dữ liệu và học đặc trưng không gian thông qua gradient. Đây là bước xây dựng và kiểm chứng pipeline deep learning. Muốn chứng minh lợi ích về sức mạnh hoặc tốc độ so với teacher cần làm benchmark; phiên bản hiện tại chưa có benchmark đó.

   **“51% có nghĩa AI thắng 51% không?”** — Không. Đó là top-1 agreement với teacher trên validation. Cần tổ chức các ván đấu riêng để đo win rate.

   **“Tại sao epoch 12 được lưu dù epoch 11 accuracy cao hơn?”** — Tiêu chí là validation policy cross entropy nhỏ nhất, không phải top-1 lớn nhất. Cross entropy xét cả phân phối nhãn mềm, còn top-1 chỉ xét ô cao nhất.

   **“Mạng có tự tạo được một nước đi không?”** — Có. Bản thân policy cho điểm các ô trống. Tuy nhiên sản phẩm hiện tại dùng engine lai, nên nước đi cuối còn có thể do bộ kiểm tra thắng/chặn hoặc tìm kiếm quyết định.

   **“CNN có được huấn luyện cả 3×3 đến 25×25 không?”** — Chưa. Dữ liệu train hiện chỉ có 9×9; kiến trúc hỗ trợ đầu vào kích thước khác nhờ tích chập và adaptive pooling. Cần đánh giá từng kích thước riêng.

   **“Value có phải xác suất thắng không?”** — Không. Đó là đầu hồi quy điểm heuristic, nhãn hiện gần 0 và chưa được engine dùng để chọn nước.

   **“Model có thay đổi sau mỗi ván chơi không?”** — Không. Training chạy offline bằng `train.py`. Server nạp checkpoint để inference; lịch sử chơi và dataset chưa nối thành một vòng tự học.

   **“Validation có thật sự tách khỏi train không?”** — Được sinh bằng seed riêng, nhưng chưa lọc trùng: hiện có 37 dòng validation trùng tensor đầu vào với train. Đó là hạn chế đã kiểm tra và cần khắc phục trước khi đánh giá kết quả cuối trên test set.

17. **Các bước cải thiện hợp lý tiếp theo**

   Ưu tiên làm sạch và đánh giá trước khi chỉ tăng epoch. Sinh thế cờ bằng các nước đi hợp lệ, dừng khi kết thúc ván; loại trùng rồi tách train/validation/test, có thể nhóm cả các thế cờ đối xứng để giảm rò rỉ dữ liệu. Giữ test set ngoài quá trình chọn hyperparameter và checkpoint.

   Sau đó so sánh ba hệ thống: policy CNN thuần, heuristic thuần và engine lai, với cùng tập ván, luật, bên đi trước và giới hạn suy nghĩ. Báo cáo thắng/hòa/thua cùng thời gian suy luận; không suy ra win rate từ teacher accuracy.

   Có thể cải thiện teacher bằng tìm kiếm sâu hơn hoặc dữ liệu chuyên gia; thu các ván hợp lệ với kết quả cuối để tạo nhãn value theo góc nhìn người đang đi. Chỉ khi có cơ chế học từ kết quả tự chơi và cập nhật policy/value tương ứng mới trình bày là phát triển thêm hướng reinforcement learning/self-play.

   Nếu dùng value cho tìm kiếm, cần sửa engine để thực sự gọi và sử dụng nó, đồng thời đánh giá nhãn/value loss riêng. Nếu muốn học khác biệt giữa các luật hoặc tổng quát nhiều kích thước tốt hơn, cần đưa thông tin luật vào input và bổ sung dữ liệu/thí nghiệm theo kích thước.
