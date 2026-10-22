import { useState } from "react";
import {
  ArrowLeft,
  BrainCircuit,
  CheckCircle2,
  Code2,
  Cpu,
  Database,
  Flag,
  Grid3X3,
  History,
  Lightbulb,
  MoveUpRight,
  Trophy,
  Users,
} from "lucide-react";
import Board, { Mark } from "./Board";
import { MODE_NAMES } from "./Settings";
import type { Game, GameReview, ModelInfo } from "./types";
import Experiments, { RULE_NAMES } from "./Experiments";
import HumanLearning from "./HumanLearning";
import { request } from "./api";

export function Guide() {
  return (
    <div className="page-content">
      <div className="page-intro">
        <span className="eyebrow">CHƠI DỄ DÀNG · NGHĨ SÂU HƠN</span>
        <h1>
          Luật đơn giản.
          <br />
          <em>Chiến thuật vô hạn.</em>
        </h1>
        <p>
          Chỉ cần một phút để bắt đầu. Mỗi ván cờ là một cơ hội để nhìn xa thêm
          một nước.
        </p>
      </div>
      <div className="guide-grid">
        {[
          {
            icon: Grid3X3,
            title: "Chọn bàn cờ của bạn",
            text: "Chọn bàn từ 3×3 đến 25×25 và chọn nhanh Số quân để thắng ngay bên dưới kích thước bàn. Nhấn Bắt đầu để mở bàn cờ; AI và đồng hồ chỉ chạy từ lúc đó.",
          },
          {
            icon: Users,
            title: "X trước, O sau",
            text: "Hai bên lần lượt đặt một quân vào ô trống. Bạn có thể chơi cùng một người tại máy, đấu AI, xem hai AI tự chơi hoặc mời bạn online bằng mã phòng. X luôn đi trước, O đi sau.",
          },
          {
            icon: Trophy,
            title: "Nối thành hàng để thắng",
            text: "Tạo một chuỗi ngang, dọc hoặc chéo đủ số quân đã chọn. Bàn cờ đầy mà không có chuỗi thắng sẽ là hòa.",
          },
          {
            icon: Flag,
            title: "Ba luật, ba cách chơi",
            text: "Tự do: đủ hoặc hơn số quân. Chính xác: chuỗi phải đúng độ dài. Caro: chuỗi đúng độ dài và không bị quân đối thủ chặn cả hai đầu; mép bàn không tính là quân chặn.",
          },
          {
            icon: Lightbulb,
            title: "Một chút trợ giúp",
            text: "Gợi ý đánh dấu nước đi AI đề xuất. Đi lại hoàn tác một lượt ở chế độ hai người hoặc lượt của bạn cùng phản hồi của AI. Hai tính năng này tắt khi chơi online. Khi xem AI đấu AI, bạn có thể tạm dừng và tiếp tục.",
          },
          {
            icon: BrainCircuit,
            title: "AI học và suy nghĩ thế nào?",
            text: "CaroNet học policy từ teacher và value từ kết quả ván. AI Lab có phần học bổ sung từ lịch sử người chơi, tự kiểm định trước khi áp dụng model mới. Bạn có thể chọn CNN thuần, chiến thuật thuần hoặc engine kết hợp; riêng mức khó 3×3 dùng minimax chính xác.",
          },
        ].map(({ icon: Icon, title, text }, i) => (
          <article className="panel guide-card" key={title}>
            <div className="guide-card-top">
              <span className="feature-icon">
                <Icon size={23} />
              </span>
              <span className="eyebrow">0{i + 1}</span>
            </div>
            <h2>{title}</h2>
            <p>{text}</p>
          </article>
        ))}
      </div>
      <div className="guide-tip">
        <Lightbulb size={25} />
        <div>
          <h3>Góc nhỏ chiến thuật</h3>
          <p>
            Giữ quyền chủ động ở trung tâm, tìm nước tạo hai mối đe dọa và luôn
            kiểm tra đối thủ có thể thắng ngay ở lượt kế tiếp hay không.
          </p>
        </div>
      </div>
    </div>
  );
}

export function AILab({
  model,
  onModelChanged,
}: {
  model: ModelInfo | null;
  onModelChanged?: () => void;
}) {
  const history = model?.training.history || [];
  const allValues = history.flatMap((h) => [h.loss, h.val_loss]);
  const low = Math.min(...allValues, 0),
    high = Math.max(...allValues, 1);
  const points = (key: "loss" | "val_loss") =>
    history
      .map(
        (h, i) =>
          `${45 + (i / Math.max(history.length - 1, 1)) * 645},${215 - ((h[key] - low) / (high - low)) * 175}`,
      )
      .join(" ");
  return (
    <div className="page-content">
      <div className="page-intro">
        <span className="eyebrow">BÊN TRONG MỖI NƯỚC ĐI</span>
        <h1>
          Một chút khoa học.
          <br />
          <em>Rất nhiều tò mò.</em>
        </h1>
        <p>
          Khám phá mô hình thực sự đứng sau đối thủ của bạn. Các chỉ số dưới đây
          được đọc trực tiếp từ checkpoint và lần huấn luyện.
        </p>
      </div>
      <div className="metric-grid">
        <Metric
          label="Kiến trúc"
          value="Residual CNN"
          detail="3 khối residual · policy + value"
          icon={BrainCircuit}
        />
        <Metric
          label="Tham số mô hình"
          value={model ? model.parameters.toLocaleString("vi-VN") : "—"}
          detail="Mạng tích chập PyTorch"
          icon={Cpu}
        />
        <Metric
          label="Mẫu huấn luyện"
          value={model?.training.samples?.toLocaleString("vi-VN") || "—"}
          detail={`${model?.training.validation_samples?.toLocaleString("vi-VN") || "—"} validation · ${model?.training.test_samples?.toLocaleString("vi-VN") || "—"} test giữ riêng`}
          icon={Database}
        />
        <Metric
          label="Trạng thái"
          value={model?.loaded ? "Đã sẵn sàng" : "Chưa tải"}
          detail={
            model?.loaded
              ? "Checkpoint được tải trên CPU"
              : "AI chiến thuật dự phòng"
          }
          icon={CheckCircle2}
        />
      </div>
      {model?.load_error && (
        <p className="form-error" role="alert">
          {model.load_error}
        </p>
      )}
      <HumanLearning onModelChanged={onModelChanged} />
      <div className="lab-columns">
        <section className="panel training-chart">
          <div className="panel-heading">
            <h2>Quá trình học của CaroNet</h2>
            <span className="small-badge">
              {model?.training.epochs || 0} epochs
            </span>
          </div>
          <p className="muted">
            Loss nhỏ hơn cho biết mô hình khớp dữ liệu huấn luyện tốt hơn.
          </p>
          {history.length ? (
            <>
              <svg
                viewBox="0 0 720 260"
                role="img"
                aria-label="Biểu đồ loss huấn luyện và kiểm định"
              >
                <defs>
                  <linearGradient id="chart-fill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#8b5cf6" stopOpacity=".18" />
                    <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0" />
                  </linearGradient>
                </defs>
                {[0, 1, 2, 3, 4].map((i) => (
                  <g key={i}>
                    <line
                      x1="45"
                      y1={40 + i * 43.75}
                      x2="690"
                      y2={40 + i * 43.75}
                      stroke="currentColor"
                      opacity=".08"
                    />
                    <text
                      x="12"
                      y={44 + i * 43.75}
                      fill="currentColor"
                      opacity=".5"
                      fontSize="11"
                    >
                      {(high - ((high - low) * i) / 4).toFixed(1)}
                    </text>
                  </g>
                ))}
                <polygon
                  points={`45,215 ${points("loss")} 690,215`}
                  fill="url(#chart-fill)"
                />
                <polyline
                  points={points("loss")}
                  fill="none"
                  stroke="#8b5cf6"
                  strokeWidth="3"
                  strokeLinejoin="round"
                />
                <polyline
                  points={points("val_loss")}
                  fill="none"
                  stroke="#c4904c"
                  strokeWidth="2.5"
                  strokeDasharray="6 4"
                />
                {history
                  .filter((_, i) => i % 2 === 0 || i === history.length - 1)
                  .map((h) => (
                    <text
                      key={h.epoch}
                      x={
                        45 +
                        ((h.epoch - 1) / Math.max(history.length - 1, 1)) * 645
                      }
                      y="244"
                      textAnchor="middle"
                      fill="currentColor"
                      opacity=".5"
                      fontSize="11"
                    >
                      {h.epoch}
                    </text>
                  ))}
              </svg>
              <div className="chart-legend">
                <span>
                  <i />
                  Training loss
                </span>
                <span>
                  <i />
                  Validation loss
                </span>
                <small>Epoch</small>
              </div>
            </>
          ) : (
            <div className="empty-state">
              <BrainCircuit size={34} />
              <p>
                Chưa có dữ liệu huấn luyện.
                <br />
                Chạy lệnh bên dưới để tạo checkpoint.
              </p>
            </div>
          )}
        </section>
        <section className="panel pipeline">
          <h2>Từ bàn cờ đến nước đi</h2>
          {[
            [
              "01",
              "Đọc bàn cờ",
              "8 kênh: quân mình, đối thủ, ô trống, mục tiêu, mặt nạ bàn và ba loại luật.",
            ],
            [
              "02",
              "Suy luận bằng CNN",
              "Policy xếp hạng ô trống; value học kết quả cuối của các ván hợp lệ.",
            ],
            [
              "03",
              "Kiểm tra chiến thuật",
              "Engine lai ưu tiên thắng/chặn; mức khó tìm kiếm alpha-beta bằng policy và value.",
            ],
            [
              "04",
              "Chọn nước hợp lệ",
              "Bộ luật trên máy chủ xác nhận lượt đi và kết quả.",
            ],
          ].map(([num, title, text]) => (
            <div className="pipeline-step" key={num}>
              <span>{num}</span>
              <div>
                <h3>{title}</h3>
                <p>{text}</p>
              </div>
            </div>
          ))}
        </section>
      </div>
      <section className="panel training-command">
        <div>
          <Code2 size={25} />
          <h2>Huấn luyện nền bằng terminal</h2>
        </div>
        <p>
          Lệnh này tạo model nền từ dữ liệu mô phỏng. Để học tiếp từ các ván bạn
          đã chơi, dùng phần Học từ các ván với người chơi phía trên. Nếu chạy
          lệnh tạo model nền, hãy khởi động lại backend sau khi hoàn thành.
        </p>
        <pre>
          <code>
            .\.venv\Scripts\python.exe -m backend.train --samples 6000 --epochs
            18
          </code>
        </pre>
        <p className="muted">
          Huấn luyện từ các ván hợp lệ trên bàn 3×3 đến 25×25, đủ ba luật. Các
          ván thuộc riêng từng tập; loại trùng cả khi xoay/lật thế cờ trước khi
          augmentation. Value học kết quả thắng/hòa/thua theo phía đang đi. Đây
          là học có giám sát từ rollout, chưa phải reinforcement learning; value
          chưa được hiệu chỉnh thành xác suất thắng.
        </p>
        {history.length > 0 && (
          <p className="training-details">
            Loss kiểm định tốt nhất:{" "}
            <strong>{model?.training.best_val_loss}</strong> · Thời gian huấn
            luyện: <strong>{model?.training.seconds}s</strong> · Độ khớp top-1
            với teacher ở epoch cuối:{" "}
            <strong>{(history.at(-1)!.top1_accuracy * 100).toFixed(1)}%</strong>
          </p>
        )}
      </section>
      {model?.training.test_metrics && (
        <section className="panel evaluation-panel">
          <h2>Đánh giá trên tập test giữ riêng</h2>
          <p>
            Checkpoint chọn ở epoch {model.training.best_epoch} theo validation.
            Test chỉ được đánh giá sau khi chọn xong checkpoint.
          </p>
          <div className="evaluation-values">
            <span>
              Policy top-1{" "}
              <strong>
                {(model.training.test_metrics.top1_accuracy * 100).toFixed(1)}%
              </strong>
            </span>
            <span>
              Policy CE{" "}
              <strong>{model.training.test_metrics.policy_loss}</strong>
            </span>
            <span>
              Outcome value MSE{" "}
              <strong>{model.training.test_metrics.value_mse}</strong>
            </span>
            <span>
              Trùng qua xoay/lật giữa các tập{" "}
              <strong>
                {model.training.dataset?.audit
                  ?.duplicate_or_symmetric_positions_between_splits ?? "—"}
              </strong>
            </span>
          </div>
          <p className="muted">
            Top-1 đo mức khớp nước teacher, không đo tỷ lệ thắng. Train và
            validation dùng cùng loss: policy CE + 0,5 × value MSE.
          </p>
          {model.training.test_metrics.value_zero_baseline_mse != null && (
            <p className="muted">
              MSE nếu luôn dự đoán value = 0:{" "}
              {model.training.test_metrics.value_zero_baseline_mse}. So sánh với
              value MSE để xem đầu value học được nhiều hơn dự đoán trung tính
              hay chưa.
            </p>
          )}
          <details>
            <summary>Test theo luật và kích thước</summary>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Nhóm</th>
                    <th>Mẫu</th>
                    <th>Policy top-1</th>
                    <th>Value MSE</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ...Object.entries(model.training.test_metrics.by_rule).map(
                      ([name, value]) => [RULE_NAMES[name], value] as const,
                    ),
                    ...Object.entries(model.training.test_metrics.by_size).map(
                      ([name, value]) => [`${name}×${name}`, value] as const,
                    ),
                  ].map(([name, value]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td>{value.samples}</td>
                      <td>{(value.top1_accuracy * 100).toFixed(1)}%</td>
                      <td>{value.value_mse}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </section>
      )}
      <Experiments loaded={!!model?.loaded} />
    </div>
  );
}

function Metric({
  label,
  value,
  detail,
  icon: Icon,
}: {
  label: string;
  value: string;
  detail: string;
  icon: typeof Cpu;
}) {
  return (
    <div className="panel metric">
      <div>
        <span>{label}</span>
        <Icon size={18} />
      </div>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

export function Statistics({
  history,
  onReplay,
}: {
  history: Game[];
  onReplay: (game: Game) => void;
}) {
  const xWins = history.filter((g) => g.winner === 1).length;
  const oWins = history.filter((g) => g.winner === -1).length;
  const draws = history.filter((g) => g.status === "draw").length;
  return (
    <div className="page-content">
      <div className="page-intro">
        <span className="eyebrow">MỖI VÁN CỜ ĐỀU CÓ CÂU CHUYỆN</span>
        <h1>
          Nhìn lại nước cũ.
          <br />
          <em>Tìm ý tưởng mới.</em>
        </h1>
        <p>
          Lịch sử và kết quả các ván đã hoàn thành trên trình duyệt này. Chọn
          một ván để xem lại từng nước.
        </p>
      </div>
      <div className="metric-grid">
        <Metric
          label="Ván đã hoàn thành"
          value={String(history.length)}
          detail="Lưu tối đa 50 ván gần nhất"
          icon={History}
        />
        <Metric
          label="Quân X thắng"
          value={String(xWins)}
          detail="Người đi trước"
          icon={Trophy}
        />
        <Metric
          label="Quân O thắng"
          value={String(oWins)}
          detail="Người đi sau"
          icon={Trophy}
        />
        <Metric
          label="Ván hòa"
          value={String(draws)}
          detail="Cùng một thế cờ cân bằng"
          icon={Flag}
        />
      </div>
      <section className="panel history-panel">
        <div className="panel-heading">
          <h2>Những ván cờ gần đây</h2>
          <span className="small-badge">{history.length} ván</span>
        </div>
        {history.length ? (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Thời gian</th>
                  <th>Chế độ</th>
                  <th>Bàn cờ</th>
                  <th>Kết quả</th>
                  <th>Nước đi</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {history.map((g) => (
                  <tr key={g.id}>
                    <td>
                      {new Date(g.created_at * 1000).toLocaleString("vi-VN", {
                        day: "2-digit",
                        month: "2-digit",
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </td>
                    <td>{MODE_NAMES[g.config.mode] || "Ván cũ"}</td>
                    <td>
                      {g.config.size}×{g.config.size}
                    </td>
                    <td>
                      <span
                        className={`result-chip ${g.winner === 1 ? "x-result" : ""}`}
                      >
                        {g.status === "draw"
                          ? "Hòa"
                          : `${g.winner === 1 ? "X" : "O"} thắng`}
                      </span>
                    </td>
                    <td>{g.moves.length}</td>
                    <td>
                      <button
                        className="text-button"
                        onClick={() => onReplay(g)}
                      >
                        Xem lại <MoveUpRight size={14} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state">
            <History size={40} strokeWidth={1.3} />
            <h3>Câu chuyện của bạn bắt đầu từ một nước cờ</h3>
            <p>Hoàn thành một ván để xem thống kê và lịch sử tại đây.</p>
          </div>
        )}
      </section>
    </div>
  );
}

export function Replay({ game, onClose }: { game: Game; onClose: () => void }) {
  const [step, setStep] = useState(game.moves.length);
  const [review, setReview] = useState<GameReview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [practice, setPractice] = useState(false);
  const [trial, setTrial] = useState<Game | null>(null);
  const moves = game.moves.slice(0, step);
  const board = Array.from({ length: game.config.size }, () =>
    Array(game.config.size).fill(0),
  );
  moves.forEach((m) => {
    board[m.row][m.col] = m.player;
  });
  function jump(next: number) {
    setStep(next);
    setTrial(null);
    setPractice(false);
    setError("");
  }
  async function analyze() {
    setBusy(true);
    setError("");
    try {
      setReview(
        await request<GameReview>(`/history/${game.id}/review`, {
          method: "POST",
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function tryMove(row: number, col: number) {
    if (busy) return;
    setBusy(true);
    setError("");
    try {
      const branch =
        trial?.moves.slice(step).map((m) => ({ row: m.row, col: m.col })) || [];
      setTrial(
        await request<Game>(`/history/${game.id}/variation`, {
          method: "POST",
          body: JSON.stringify({
            step,
            moves: [...branch, { row, col }],
            reply: true,
          }),
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const terminalPrefix =
    step === game.moves.length &&
    (game.ended_reason === "line" || game.ended_reason === "full");
  const opportunities = review?.events.find(
    (event) => event.number === step + 1,
  );
  const shown = trial || {
    ...game,
    moves,
    board,
    analysis: null,
    current_player: (step % 2 === 0 ? 1 : -1) as 1 | -1,
    winning_line: step === game.moves.length ? game.winning_line : [],
  };
  return (
    <>
      <div className="replay-summary">
        <span>
          {MODE_NAMES[game.config.mode] || "Ván cũ"} · {game.config.size}×
          {game.config.size}
        </span>
        <span>
          {game.winner ? (
            <>
              <Mark player={game.winner} small /> thắng
            </>
          ) : (
            "Ván hòa"
          )}
        </span>
      </div>
      <Board
        game={shown}
        canPlay={
          practice &&
          !busy &&
          !terminalPrefix &&
          (!trial || trial.status === "playing")
        }
        hint={null}
        onMove={(r, c) => void tryMove(r, c)}
        showNumbers
        compact
      />
      <div className="replay-controls">
        <button
          className="icon-button"
          onClick={() => jump(Math.max(0, step - 1))}
          disabled={!step || busy}
          aria-label="Nước trước"
        >
          <ArrowLeft size={19} />
        </button>
        <input
          aria-label="Chọn nước đi"
          type="range"
          min={0}
          max={game.moves.length}
          value={step}
          onChange={(e) => jump(Number(e.target.value))}
          disabled={busy}
        />
        <span>
          {step}/{game.moves.length}
        </span>
        <button
          className="icon-button"
          onClick={() => jump(Math.min(game.moves.length, step + 1))}
          disabled={step === game.moves.length || busy}
          aria-label="Nước tiếp"
        >
          <MoveUpRight size={19} />
        </button>
      </div>
      <div className="review-actions">
        <button
          className="button secondary"
          disabled={busy}
          onClick={() => void analyze()}
        >
          {busy ? "Đang phân tích…" : "Phân tích cơ hội sau ván"}
        </button>
        <button
          className="button secondary"
          disabled={busy || terminalPrefix}
          onClick={() => {
            setPractice(!practice);
            setTrial(null);
            setError("");
          }}
        >
          {practice ? "Trở về ván gốc" : "Thử phương án khác"}
        </button>
      </div>
      {practice && (
        <div className="practice-note" role="status">
          <p>
            Chọn một ô trống để thử. AI sẽ đáp lại; các nước thử được giữ riêng
            trong cửa sổ này.
          </p>
          {trial && (
            <p>
              {trial.status === "won"
                ? `${trial.winner === 1 ? "X" : "O"} thắng trong phương án thử.`
                : trial.status === "draw"
                  ? "Phương án thử hòa."
                  : `Đến lượt ${trial.current_player === 1 ? "X" : "O"}.`}
            </p>
          )}
          {opportunities?.alternatives.map(([r, c]) => (
            <button
              key={`${r}-${c}`}
              className="text-button"
              disabled={busy || !!trial}
              onClick={() => void tryMove(r, c)}
            >
              Thử {String.fromCharCode(65 + c)}
              {r + 1}
            </button>
          ))}
        </div>
      )}
      {error && (
        <p className="form-error" role="alert">
          {error}
        </p>
      )}
      {review && (
        <div className="game-review">
          <h3>Cơ hội chiến thuật</h3>
          <p>{review.scope}</p>
          <div className="review-summary">
            {Object.entries(review.summary).map(([player, counts]) => (
              <span key={player}>
                {player === "1" ? "X" : "O"}: bỏ lỡ {counts.missed_win} nước
                thắng, {counts.missed_block} nước chặn · {counts.double_threat}{" "}
                thế bị đe dọa kép
              </span>
            ))}
          </div>
          {review.events.length ? (
            <ol>
              {review.events.map((event) => (
                <li key={event.number}>
                  <button
                    className="text-button"
                    disabled={busy}
                    onClick={() => jump(event.number - 1)}
                  >
                    Trước nước {event.number} · {event.player === 1 ? "X" : "O"}
                  </button>
                  <p>
                    {event.explanation}
                    {event.alternatives.length > 0 &&
                      ` Ô đề xuất: ${event.alternatives.map(([r, c]) => `${String.fromCharCode(65 + c)}${r + 1}`).join(", ")}.`}
                  </p>
                </li>
              ))}
            </ol>
          ) : (
            <p>
              Không tìm thấy cơ hội thắng/chặn ngay bị bỏ lỡ trong các nước đã
              chơi.
            </p>
          )}
        </div>
      )}
      <button className="button secondary" onClick={onClose}>
        Đóng xem lại
      </button>
    </>
  );
}
