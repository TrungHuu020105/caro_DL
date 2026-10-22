import { useEffect, useRef, useState } from "react";
import { request } from "./api";
import type { HumanLearningState } from "./types";

const DECISIONS: Record<string, string> = {
  promoted: "Đã áp dụng model mới",
  rejected: "Giữ model hiện tại",
  insufficient_data: "Cần thêm ván để học",
  failed: "Lần học chưa hoàn thành",
  interrupted: "Lần học bị gián đoạn",
};

export default function HumanLearning({
  onModelChanged,
}: {
  onModelChanged?: () => void;
}) {
  const [state, setState] = useState<HumanLearningState | null>(null);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [threshold, setThreshold] = useState(5);
  const [epochs, setEpochs] = useState(3);
  const mutating = useRef(false);
  const generation = useRef(0);
  useEffect(() => {
    if (state?.active_checkpoint_sha256) onModelChanged?.();
  }, [state?.active_checkpoint_sha256, onModelChanged]);
  useEffect(() => {
    let active = true;
    const refresh = async () => {
      const requestedGeneration = generation.current;
      try {
        const next = await request<HumanLearningState>("/learning");
        if (
          active &&
          !mutating.current &&
          requestedGeneration === generation.current
        ) {
          setState(next);
          setError("");
        }
      } catch (e) {
        if (active) setError((e as Error).message);
      }
    };
    void refresh();
    const timer = setInterval(() => {
      void refresh();
    }, 2000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);
  useEffect(() => {
    if (state?.settings) {
      setThreshold(state.settings.min_new_games);
      setEpochs(state.settings.epochs);
    }
  }, [state?.settings.min_new_games, state?.settings.epochs]);
  async function configure(changes: {
    enabled?: boolean;
    min_new_games?: number;
    epochs?: number;
  }) {
    const previous = state;
    generation.current += 1;
    mutating.current = true;
    setState((old) =>
      old ? { ...old, settings: { ...old.settings, ...changes } } : old,
    );
    setSaving(true);
    setError("");
    try {
      setState(
        await request<HumanLearningState>("/learning/settings", {
          method: "POST",
          body: JSON.stringify(changes),
        }),
      );
    } catch (e) {
      setState(previous);
      setError((e as Error).message);
    } finally {
      mutating.current = false;
      generation.current += 1;
      setSaving(false);
    }
  }
  async function learn() {
    generation.current += 1;
    mutating.current = true;
    setSaving(true);
    setError("");
    try {
      setState(
        await request<HumanLearningState>("/learning/run", {
          method: "POST",
          body: JSON.stringify({ epochs }),
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      mutating.current = false;
      generation.current += 1;
      setSaving(false);
    }
  }
  function download() {
    if (!state?.report) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(state.report, null, 2)], {
        type: "application/json",
      }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "caro-human-learning.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const busy = saving || state?.status === "running";
  const report = state?.report;
  return (
    <section className="panel human-learning-panel">
      <div className="panel-heading">
        <h2>Học từ các ván với người chơi</h2>
        <span className="small-badge">Học bổ sung</span>
      </div>
      <p>
        Các ván đấu AI, hai người và online kết thúc bằng thắng trên bàn hoặc
        hòa sẽ đóng góp dữ liệu. AI học lại từ thế cờ, kết quả ván và nước đi
        được phân tích lại.
      </p>
      <div className="evaluation-values learning-stats">
        <span>
          Ván hợp lệ đã thu<strong>{state?.stats.total_games ?? "—"}</strong>
        </span>
        <span>
          Ván mới chờ xử lý<strong>{state?.stats.pending_games ?? "—"}</strong>
        </span>
        <span>
          Vị trí trước lọc trùng
          <strong>{state?.stats.total_positions ?? "—"}</strong>
        </span>
        <span>
          Ván giữ riêng kiểm định
          <strong>{state?.stats.validation_games ?? "—"}</strong>
        </span>
      </div>
      <div className="human-learning-settings">
        <label className="experiment-toggle">
          <input
            type="checkbox"
            checked={state?.settings.enabled ?? true}
            disabled={!state || !!busy}
            onChange={(e) => void configure({ enabled: e.target.checked })}
          />{" "}
          Tự học khi đủ ván mới
        </label>
        <label className="field-label">
          Số ván mới để tự học
          <input
            type="number"
            min={3}
            max={50}
            value={threshold}
            disabled={!!busy}
            onChange={(e) =>
              setThreshold(
                Math.max(3, Math.min(50, Number(e.target.value) || 5)),
              )
            }
          />
        </label>
        <label className="field-label">
          Epoch mỗi lần học
          <input
            type="number"
            min={1}
            max={8}
            value={epochs}
            disabled={!!busy}
            onChange={(e) =>
              setEpochs(Math.max(1, Math.min(8, Number(e.target.value) || 3)))
            }
          />
        </label>
        <button
          className="button secondary"
          disabled={!state || !!busy}
          onClick={() => void configure({ min_new_games: threshold, epochs })}
        >
          Lưu thiết lập học
        </button>
      </div>
      <div className="review-actions">
        <button
          className="button primary"
          disabled={!state || !!busy || !state.stats.pending_games}
          onClick={() => void learn()}
        >
          {state?.status === "running"
            ? "Đang học và đánh giá…"
            : "Học từ lịch sử ngay"}
        </button>
      </div>
      <p className="learning-status" role="status">
        {state?.message || "Đang đọc lịch sử người chơi…"}
      </p>
      {state?.status === "running" && state.total > 0 && (
        <div className="experiment-progress">
          <progress max={state.total} value={state.completed} />{" "}
          {state.completed}/{state.total} ·{" "}
          {state.phase === "training"
            ? "epoch"
            : state.phase === "arena"
              ? "ván đánh giá"
              : "bước"}
          . Bạn vẫn có thể chơi cờ.
        </div>
      )}
      {(error || state?.error) && (
        <p className="form-error" role="alert">
          {error || state?.error}
        </p>
      )}
      <p className="muted">
        Mặc định tự thử học sau 5 ván mới. Cần ít nhất{" "}
        {state?.min_positions || 8} vị trí train mới sau lọc trùng. Ván AI đấu
        AI, đầu hàng, hết giờ và ván trùng được loại khỏi nguồn học. Model mới
        chỉ được áp dụng khi qua kiểm định và đấu đạt ít nhất 55% điểm trước
        model cũ.
      </p>
      {report && (
        <div className="human-learning-report">
          <div className="panel-heading">
            <h3>
              {state?.status === "running"
                ? "Kết quả lần học trước"
                : DECISIONS[report.decision] || "Kết quả lần học"}
            </h3>
            <button className="text-button" onClick={download}>
              Tải báo cáo học JSON
            </button>
          </div>
          <p className="muted">
            {new Date(report.finished_at).toLocaleString("vi-VN")} ·{" "}
            {report.seconds}s · Lần {report.run_id.slice(-8)}
          </p>
          {report.data && (
            <p>
              Đã chọn {report.data.selected_train_positions} vị trí người chơi
              để train, {report.data.selected_validation_positions} vị trí giữ
              riêng; trộn với {report.replay_positions || 0} vị trí cũ.
            </p>
          )}
          {report.history && (
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Epoch</th>
                    <th>Train loss</th>
                    <th>Validation loss</th>
                    <th>Khớp teacher</th>
                  </tr>
                </thead>
                <tbody>
                  {report.history.map((row) => (
                    <tr key={row.epoch}>
                      <td>{row.epoch}</td>
                      <td>{row.loss}</td>
                      <td>{row.val_loss}</td>
                      <td>{(row.top1_accuracy * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {report.arena && (
            <p className="arena-summary">
              Đấu model cũ: {report.arena.wins} thắng / {report.arena.draws} hòa
              / {report.arena.losses} thua ({report.arena.games} ván) · Điểm{" "}
              {(report.arena.score_rate * 100).toFixed(1)}%.
            </p>
          )}
          {report.gate?.reasons.length ? (
            <ul>
              {report.gate.reasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          ) : null}
          <p className="muted">
            {report.decision === "promoted"
              ? "Model mới đã được nạp; các lượt AI tiếp theo sử dụng phiên bản mới."
              : report.decision === "rejected"
                ? "Ứng viên đã được lưu cùng báo cáo; AI tiếp tục dùng model cũ. Các ván mới về sau sẽ đóng góp cho lần học tiếp theo."
                : report.message}
          </p>
          {report.gate && <p className="muted">{report.gate.limitation}</p>}
        </div>
      )}
    </section>
  );
}
