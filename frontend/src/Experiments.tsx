import { useEffect, useState } from "react";
import { request } from "./api";
import type { EngineVariant, ExperimentState } from "./types";

export const ENGINE_NAMES: Record<EngineVariant, string> = {
  cnn: "CNN thuần",
  heuristic: "Chiến thuật thuần",
  hybrid: "Kết hợp + tìm kiếm",
};
export const RULE_NAMES: Record<string, string> = {
  freestyle: "Tự do",
  exact: "Chính xác",
  blocked: "Caro",
};

export default function Experiments({ loaded }: { loaded: boolean }) {
  const [state, setState] = useState<ExperimentState | null>(null);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [sizes, setSizes] = useState("5,9,15");
  const [rule, setRule] = useState("freestyle");
  const [pairs, setPairs] = useState(1);
  const [budget, setBudget] = useState(80);
  const [seed, setSeed] = useState(2026);
  const [left, setLeft] = useState<EngineVariant>("cnn");
  const [right, setRight] = useState<EngineVariant>("hybrid");
  const [all, setAll] = useState(true);
  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const next = await request<ExperimentState>("/experiments");
        if (active) setState(next);
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
  const running = submitting || state?.status === "running";
  const variants = all ? ["cnn", "heuristic", "hybrid"] : [left, right];
  async function start() {
    setSubmitting(true);
    setError("");
    try {
      setState(
        await request<ExperimentState>("/experiments", {
          method: "POST",
          body: JSON.stringify({
            sizes: sizes.split(",").map(Number),
            rules: rule === "all" ? ["freestyle", "exact", "blocked"] : [rule],
            variants,
            pairs,
            budget_ms: budget,
            seed,
          }),
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
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
    link.download = "caro-benchmark.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const report = state?.report;
  return (
    <section className="panel experiment-panel">
      <div className="panel-heading">
        <h2>Phòng thí nghiệm sức chơi</h2>
        <span className="small-badge">Đấu cùng điều kiện</span>
      </div>
      <p>
        So sánh các engine bằng những cặp ván chung khai cuộc và đổi bên X/O.
        Mỗi engine có cùng giới hạn thời gian tìm kiếm; thời gian thực tế được
        ghi riêng.
      </p>
      <fieldset disabled={!!running} className="experiment-fields">
        <label className="field-label">
          Bàn thí nghiệm
          <select value={sizes} onChange={(e) => setSizes(e.target.value)}>
            <option value="3">3×3</option>
            <option value="5">5×5</option>
            <option value="9">9×9</option>
            <option value="15">15×15</option>
            <option value="25">25×25</option>
            <option value="5,9,15">5×5, 9×9 và 15×15</option>
          </select>
        </label>
        <label className="field-label">
          Luật thí nghiệm
          <select value={rule} onChange={(e) => setRule(e.target.value)}>
            {Object.entries(RULE_NAMES).map(([id, name]) => (
              <option key={id} value={id}>
                {name}
              </option>
            ))}
            <option value="all">Cả ba luật</option>
          </select>
        </label>
        <label className="field-label">
          Số cặp mỗi đối đầu
          <input
            type="number"
            min={1}
            max={3}
            value={pairs}
            onChange={(e) =>
              setPairs(Math.max(1, Math.min(3, Number(e.target.value) || 1)))
            }
          />
        </label>
        <label className="field-label">
          Ngân sách tìm kiếm
          <select
            value={budget}
            onChange={(e) => setBudget(Number(e.target.value))}
          >
            <option value={40}>40 ms/nước</option>
            <option value={80}>80 ms/nước</option>
            <option value={150}>150 ms/nước</option>
          </select>
        </label>
        <label className="field-label">
          Seed khai cuộc
          <input
            type="number"
            min={0}
            max={4294967295}
            value={seed}
            onChange={(e) =>
              setSeed(
                Math.max(
                  0,
                  Math.min(4294967295, Math.floor(Number(e.target.value) || 0)),
                ),
              )
            }
          />
        </label>
        <label className="experiment-toggle">
          <input
            type="checkbox"
            checked={all}
            onChange={(e) => setAll(e.target.checked)}
          />{" "}
          So sánh cả ba engine
        </label>
        {!all && (
          <>
            {[left, right].map((value, i) => (
              <label className="field-label" key={i}>
                Engine {i === 0 ? "A" : "B"}
                <select
                  value={value}
                  onChange={(e) =>
                    (i === 0 ? setLeft : setRight)(
                      e.target.value as EngineVariant,
                    )
                  }
                >
                  {Object.entries(ENGINE_NAMES).map(([id, name]) => (
                    <option key={id} value={id}>
                      {name}
                    </option>
                  ))}
                </select>
              </label>
            ))}
          </>
        )}
      </fieldset>
      <button
        className="button primary"
        onClick={() => void start()}
        disabled={
          !!running ||
          (!all && left === right) ||
          (!loaded && variants.includes("cnn"))
        }
      >
        {running ? "Thí nghiệm đang chạy…" : "Chạy thí nghiệm"}
      </button>
      {state?.status === "running" && (
        <div className="experiment-progress" role="status">
          <progress max={state.total} value={state.completed} />{" "}
          {state.completed}/{state.total} ván. Bạn có thể chuyển sang chơi cờ.
        </div>
      )}
      {(error || state?.error) && (
        <p className="form-error" role="alert">
          {error || state?.error}
        </p>
      )}
      {report && (
        <div className="benchmark-report">
          <div className="panel-heading">
            <h3>Kết quả {report.total_games} ván</h3>
            <button className="text-button" onClick={download}>
              Tải kết quả JSON
            </button>
          </div>
          <p className="muted">
            {new Date(report.created_at).toLocaleString("vi-VN")} · Seed{" "}
            {report.seed} · {report.pairs} cặp/đối đầu · {report.budget_ms} ms ·
            Model {report.checkpoint_sha256?.slice(0, 10)}
          </p>
          {state?.stale && (
            <p className="form-error">
              Kết quả thuộc checkpoint cũ. Hãy chạy lại với model đang tải.
            </p>
          )}
          <div className="table-scroll">
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>Engine</th>
                  <th>Ván</th>
                  <th>Thắng</th>
                  <th>Hòa</th>
                  <th>Thua</th>
                  <th>Điểm</th>
                  <th>TB/nước</th>
                  <th>P95</th>
                </tr>
              </thead>
              <tbody>
                {report.summary.map((row) => (
                  <tr key={row.engine}>
                    <td>{ENGINE_NAMES[row.engine]}</td>
                    <td>{row.games}</td>
                    <td>{row.wins}</td>
                    <td>{row.draws}</td>
                    <td>{row.losses}</td>
                    <td>{(row.score_rate * 100).toFixed(1)}%</td>
                    <td>{row.mean_ms} ms</td>
                    <td>{row.p95_ms} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted">
            Điểm = (thắng + 0,5 × hòa)/số ván. Mẫu nhỏ chỉ phản ánh lượt chạy
            này; chưa đủ để suy ra Elo hoặc kết luận engine nào luôn mạnh hơn.
            P95 là thời gian mà 95% nước đi không vượt quá.
          </p>
          <details>
            <summary>Kết quả theo kích thước và luật</summary>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Bàn</th>
                    <th>Luật</th>
                    <th>Engine</th>
                    <th>Thắng / Hòa / Thua</th>
                  </tr>
                </thead>
                <tbody>
                  {report.by_configuration.flatMap((group) =>
                    group.summary.map((row) => (
                      <tr key={`${group.size}-${group.rule}-${row.engine}`}>
                        <td>
                          {group.size}×{group.size}
                        </td>
                        <td>{RULE_NAMES[group.rule]}</td>
                        <td>{ENGINE_NAMES[row.engine]}</td>
                        <td>
                          {row.wins} / {row.draws} / {row.losses}
                        </td>
                      </tr>
                    )),
                  )}
                </tbody>
              </table>
            </div>
          </details>
        </div>
      )}
    </section>
  );
}
