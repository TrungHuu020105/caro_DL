import { useEffect, useRef, type ReactNode } from "react";
import {
  Bot,
  Users,
  Play,
  ArrowUpRight,
  SlidersHorizontal,
  X,
  Check,
  Sparkles,
} from "lucide-react";
import type { GameConfig, Mode } from "./types";

export const MODES: {
  id: Mode;
  title: string;
  subtitle: string;
  icon: typeof Bot;
}[] = [
  {
    id: "ai",
    title: "Đấu với AI",
    subtitle: "Thử thách trí tuệ nhân tạo",
    icon: Bot,
  },
  {
    id: "local",
    title: "Hai người",
    subtitle: "Chung máy, chung niềm vui",
    icon: Users,
  },
  {
    id: "spectate",
    title: "AI đấu AI",
    subtitle: "Quan sát & học chiến thuật",
    icon: Play,
  },
];
export const MODE_NAMES = Object.fromEntries(
  MODES.map((m) => [m.id, m.title]),
) as Record<Mode, string>;

export function Dialog({
  title,
  children,
  onClose,
  wide = false,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current;
    dialog?.showModal();
    return () => dialog?.close();
  }, []);
  return (
    <dialog
      ref={ref}
      className={`dialog ${wide ? "dialog-wide" : ""}`}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      onClick={(e) => {
        if (e.target === ref.current) onClose();
      }}
    >
      <div className="dialog-heading">
        <h2>{title}</h2>
        <button className="icon-button" onClick={onClose} aria-label="Đóng">
          <X size={20} />
        </button>
      </div>
      {children}
    </dialog>
  );
}

export function AdvancedSettings({
  config,
  setConfig,
}: {
  config: GameConfig;
  setConfig: (c: GameConfig) => void;
}) {
  return (
    <div className="advanced-settings">
      <label className="field-label">
        Tên của bạn
        <input
          maxLength={24}
          value={config.player_name}
          onChange={(e) =>
            setConfig({ ...config, player_name: e.target.value })
          }
          placeholder="Nhập tên của bạn"
        />
      </label>
      <div className="settings-two">
        <label className="field-label">
          Kích thước tùy chỉnh
          <input
            type="number"
            min={3}
            max={25}
            value={config.size}
            onChange={(e) => {
              const size = Math.max(
                3,
                Math.min(25, Number(e.target.value) || 3),
              );
              setConfig({
                ...config,
                size,
                win_length: Math.min(config.win_length || 5, size),
              });
            }}
          />
        </label>
        <label className="field-label">
          Số quân để thắng
          <select
            value={config.win_length || Math.min(config.size, 5)}
            onChange={(e) =>
              setConfig({ ...config, win_length: Number(e.target.value) })
            }
          >
            {Array.from({ length: Math.min(config.size, 6) - 2 }, (_, i) => (
              <option key={i} value={i + 3}>
                {i + 3} quân liên tiếp
              </option>
            ))}
          </select>
        </label>
      </div>
      <label className="field-label">
        Luật thắng
        <select
          value={config.rule}
          onChange={(e) =>
            setConfig({ ...config, rule: e.target.value as GameConfig["rule"] })
          }
        >
          <option value="freestyle">Tự do — đủ hoặc hơn số quân</option>
          <option value="exact">Chính xác — đúng số quân</option>
          <option value="blocked">
            Caro — đúng số quân, không chặn hai đầu
          </option>
        </select>
      </label>
      <label className="field-label">
        Thời gian mỗi lượt
        <select
          value={config.time_limit}
          onChange={(e) =>
            setConfig({ ...config, time_limit: Number(e.target.value) })
          }
        >
          <option value={0}>Không giới hạn</option>
          <option value={15}>15 giây</option>
          <option value={30}>30 giây</option>
          <option value={60}>60 giây</option>
        </select>
      </label>
      <label className="field-label">
        Quân của bạn khi đấu AI
        <select
          value={config.human_player}
          onChange={(e) =>
            setConfig({
              ...config,
              human_player: Number(e.target.value) as 1 | -1,
            })
          }
        >
          <option value={1}>X — bạn đi trước</option>
          <option value={-1}>O — AI đi trước</option>
        </select>
      </label>
      <div className="soft-note">
        <Sparkles size={18} />
        <span>
          X luôn đi trước. Bạn cũng có thể chọn nhanh số quân để thắng ngay tại
          bảng thiết lập bên cạnh bàn cờ.
        </span>
      </div>
    </div>
  );
}

export default function Settings({
  config,
  setConfig,
  busy,
  onStart,
  onAdvanced,
  started,
}: {
  config: GameConfig;
  setConfig: (c: GameConfig) => void;
  busy: boolean;
  onStart: () => void;
  onAdvanced: () => void;
  started: boolean;
}) {
  return (
    <aside className="settings-column">
      <section className="panel settings-panel">
        <div className="panel-heading">
          <h2>Thiết lập ván cờ</h2>
          <SlidersHorizontal size={16} />
        </div>
        <div className="settings-section settings-modes">
          <span className="eyebrow section-label">01 / CHẾ ĐỘ CHƠI</span>
          <div className="mode-options">
            {MODES.map(({ id, title, subtitle, icon: Icon }) => (
              <button
                key={id}
                className={`mode-option ${config.mode === id ? "selected" : ""}`}
                onClick={() => setConfig({ ...config, mode: id })}
                aria-pressed={config.mode === id}
              >
                <span className="mode-icon">
                  <Icon size={19} strokeWidth={1.8} />
                </span>
                <span>
                  <strong>{title}</strong>
                  <small>{subtitle}</small>
                </span>
                {config.mode === id && (
                  <span className="selection-check">
                    <Check size={11} strokeWidth={3} />
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
        <div className="settings-section settings-sizes">
          <span className="eyebrow section-label">02 / KÍCH THƯỚC BÀN CỜ</span>
          <div className="size-options">
            {[3, 4, 5, 6, 8, 10, 12, 15, 19].map((size) => (
              <button
                key={size}
                className={config.size === size ? "selected" : ""}
                aria-pressed={config.size === size}
                onClick={() =>
                  setConfig({ ...config, size, win_length: Math.min(size, 5) })
                }
              >
                {size}×{size}
                {size === 15 && <span className="size-dot" />}
              </button>
            ))}
          </div>
          {![3, 4, 5, 6, 8, 10, 12, 15, 19].includes(config.size) && (
            <p className="custom-size">
              Đang chọn bàn {config.size}×{config.size}
            </p>
          )}
        </div>
        <div className="settings-section settings-win-length">
          <span className="eyebrow section-label" id="win-length-label">
            03 / SỐ QUÂN ĐỂ THẮNG
          </span>
          <div
            className="win-options"
            role="group"
            aria-labelledby="win-length-label"
          >
            {[3, 4, 5, 6].map((length) => (
              <button
                key={length}
                disabled={length > config.size}
                aria-pressed={config.win_length === length}
                className={config.win_length === length ? "selected" : ""}
                onClick={() => setConfig({ ...config, win_length: length })}
              >
                {length} quân
              </button>
            ))}
          </div>
          <p className="win-length-note">
            Nối {config.win_length} quân liên tiếp để thắng.
          </p>
        </div>
        {(config.mode === "ai" || config.mode === "spectate") && (
          <div className="settings-section settings-difficulty">
            <span className="eyebrow section-label">04 / ĐỘ KHÓ AI</span>
            <div className="difficulty-options">
              {(
                [
                  { id: "easy", label: "Dễ" },
                  { id: "medium", label: "Vừa" },
                  { id: "hard", label: "Khó" },
                ] as const
              ).map((d) => (
                <button
                  key={d.id}
                  className={config.difficulty === d.id ? "selected" : ""}
                  aria-pressed={config.difficulty === d.id}
                  onClick={() => setConfig({ ...config, difficulty: d.id })}
                >
                  <span className={`difficulty-bars bars-${d.id}`}>
                    <i />
                    <i />
                    <i />
                  </span>
                  {d.label}
                </button>
              ))}
            </div>
          </div>
        )}
        <button className="advanced-button" onClick={onAdvanced}>
          <SlidersHorizontal size={14} /> Tùy chỉnh nâng cao{" "}
          <ArrowUpRight size={14} />
        </button>
        <button
          className="button primary start-button"
          disabled={busy}
          onClick={onStart}
        >
          {started ? "Bắt đầu ván mới" : "Bắt đầu"}
          <ArrowUpRight size={17} />
        </button>
        <p className="settings-footnote">
          Chọn thiết lập rồi nhấn Bắt đầu để chơi.
        </p>
      </section>
      <div className="little-note">
        <span className="note-symbol">✳</span>
        <p>
          Thắng thua là một phần.
          <br />
          <strong>Nước đi hay mới là tất cả.</strong>
        </p>
      </div>
    </aside>
  );
}
