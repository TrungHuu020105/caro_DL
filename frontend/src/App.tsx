import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  BrainCircuit,
  ChartNoAxesCombined,
  Check,
  ChevronRight,
  CircleHelp,
  Clock3,
  Flag,
  Gamepad2,
  Hash,
  History,
  Lightbulb,
  LoaderCircle,
  Maximize2,
  Moon,
  Pause,
  Play,
  RotateCcw,
  Sparkles,
  Sun,
  Trophy,
  Undo2,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";
import Board, { Mark } from "./Board";
import Settings, { AdvancedSettings, Dialog, MODE_NAMES } from "./Settings";
import { AILab, Guide, Replay, Statistics } from "./Pages";
import { gameRequest, readSaved, request, save } from "./api";
import type {
  Analysis,
  Game,
  GameConfig,
  ModelInfo,
  Player,
  Session,
} from "./types";

const DEFAULT_CONFIG: GameConfig = {
  size: 15,
  win_length: 5,
  mode: "ai",
  difficulty: "medium",
  rule: "freestyle",
  human_player: 1,
  time_limit: 0,
  player_name: "Bạn",
};
type Tab = "play" | "guide" | "stats" | "lab";
type Modal = "settings" | "resign" | null;
const TABS = [
  { id: "play", label: "Chơi cờ", icon: Gamepad2 },
  { id: "guide", label: "Hướng dẫn", icon: BookOpen },
  { id: "stats", label: "Thống kê", icon: ChartNoAxesCombined },
  { id: "lab", label: "AI Lab", icon: BrainCircuit },
] as const;

export default function App() {
  const [tab, setTab] = useState<Tab>("play");
  const [config, setConfig] = useState<GameConfig>(() => {
    const saved = readSaved<GameConfig>("caro.config", DEFAULT_CONFIG);
    const size =
      Number.isInteger(saved.size) && saved.size >= 3 && saved.size <= 25
        ? saved.size
        : 15;
    return {
      ...DEFAULT_CONFIG,
      ...saved,
      size,
      mode: ["ai", "local", "spectate"].includes(saved.mode)
        ? saved.mode
        : "ai",
      win_length: Math.max(3, Math.min(saved.win_length || 5, size, 6)),
    };
  });
  const [game, setGame] = useState<Game | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [serverReady, setServerReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const [initializing, setInitializing] = useState(true);
  const [error, setError] = useState("");
  const [toast, setToast] = useState("");
  const [modal, setModal] = useState<Modal>(null);
  const [hint, setHint] = useState<Analysis | null>(null);
  const [showNumbers, setShowNumbers] = useState(false);
  const [heatmap, setHeatmap] = useState(false);
  const [paused, setPaused] = useState(false);
  const [aiFailed, setAiFailed] = useState(false);
  const [sound, setSound] = useState(() => readSaved("caro.sound", true));
  const [dark, setDark] = useState(() => readSaved("caro.dark", false));
  const [now, setNow] = useState(Date.now());
  const [history, setHistory] = useState<Game[]>(() =>
    readSaved("caro.history", []),
  );
  const [replay, setReplay] = useState<Game | null>(null);
  const activeSession = useRef<Session | null>(null);
  const currentGame = useRef<Game | null>(null);
  currentGame.current = game;
  const actionLock = useRef(false);
  const mounted = useRef(false);
  const audio = useRef<AudioContext | null>(null);
  const lastSound = useRef("");
  const arenaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    save("caro.config", config);
  }, [config]);
  useEffect(() => {
    setHint(null);
  }, [game?.id, game?.revision]);
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    save("caro.dark", dark);
  }, [dark]);
  useEffect(() => {
    save("caro.sound", sound);
  }, [sound]);
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(timer);
  }, []);
  useEffect(() => {
    if (toast) {
      const timer = setTimeout(() => setToast(""), 3500);
      return () => clearTimeout(timer);
    }
  }, [toast]);

  const acceptSession = useCallback((data: { game: Game; token: string }) => {
    const next = { id: data.game.id, token: data.token };
    activeSession.current = next;
    setSession(next);
    setGame(data.game);
    setHint(null);
    setPaused(false);
    setAiFailed(false);
    setError("");
    setTab("play");
    const ids = readSaved<string[]>("caro.ids", []);
    save(
      "caro.ids",
      [next.id, ...ids.filter((id) => id !== next.id)].slice(0, 100),
    );
  }, []);

  const refresh = useCallback(async (target: Session) => {
    const state = await gameRequest<Game>(target);
    if (activeSession.current?.id === state.id)
      setGame((old) => (!old || state.revision >= old.revision ? state : old));
    return state;
  }, []);

  const startGame = useCallback(
    async (settings: GameConfig) => {
      if (actionLock.current) return;
      actionLock.current = true;
      setBusy(true);
      setError("");
      try {
        const data = await request<{ game: Game; token: string }>("/games", {
          method: "POST",
          body: JSON.stringify(settings),
        });
        acceptSession(data);
        setServerReady(true);
        // Refresh model details when the backend has just become available.
        void request<ModelInfo>("/model")
          .then(setModel)
          .catch(() => {});
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setBusy(false);
        actionLock.current = false;
      }
    },
    [acceptSession],
  );

  const checkServer = useCallback(async () => {
    try {
      const info = await request<ModelInfo>("/model");
      setModel(info);
      setServerReady(true);
      setError("");
    } catch (e) {
      setServerReady(false);
      setError((e as Error).message);
    } finally {
      setInitializing(false);
    }
  }, []);

  useEffect(() => {
    if (mounted.current) return;
    mounted.current = true;
    void checkServer();
  }, [checkServer]);

  useEffect(() => {
    const timer = setInterval(async () => {
      try {
        await request("/health");
        setServerReady(true);
        if (!model) setModel(await request<ModelInfo>("/model"));
      } catch {
        setServerReady(false);
      }
    }, 15000);
    return () => clearInterval(timer);
  }, [model]);

  const act = useCallback(
    async (action: string, body?: unknown) => {
      const target = activeSession.current;
      if (!target || actionLock.current) return;
      actionLock.current = true;
      setBusy(true);
      setError("");
      setHint(null);
      try {
        const state = await gameRequest<Game>(target, action, body);
        if (activeSession.current?.id === target.id)
          setGame((old) =>
            !old || state.revision >= old.revision ? state : old,
          );
      } catch (e) {
        setError((e as Error).message);
        if (action === "ai") setAiFailed(true);
        try {
          await refresh(target);
        } catch {
          /* Keep last valid board visible. */
        }
      } finally {
        actionLock.current = false;
        setBusy(false);
      }
    },
    [refresh],
  );

  const aiTurn =
    !!game &&
    game.status === "playing" &&
    (game.config.mode === "spectate" ||
      (game.config.mode === "ai" &&
        game.current_player !== game.config.human_player));
  useEffect(() => {
    if (!aiTurn || paused || aiFailed || initializing) return;
    const timer = setTimeout(
      () => {
        void act("ai");
      },
      game?.config.mode === "spectate" ? 850 : 380,
    );
    return () => clearTimeout(timer);
  }, [game?.id, game?.revision, aiTurn, paused, aiFailed, initializing, act]);

  useEffect(() => {
    if (
      !game ||
      game.status !== "playing" ||
      !game.config.time_limit ||
      now / 1000 < game.turn_started_at + game.config.time_limit ||
      !session
    )
      return;
    void refresh(session).catch(() => {});
  }, [
    now,
    game?.turn_started_at,
    game?.status,
    session,
    refresh,
    game?.config.time_limit,
  ]);

  useEffect(() => {
    if (!game || !["won", "draw"].includes(game.status)) return;
    setHint(null);
    setHistory((old) => {
      if (old.some((g) => g.id === game.id)) return old;
      const updated = [game, ...old].slice(0, 50);
      save("caro.history", updated);
      return updated;
    });
  }, [game?.id, game?.status]);

  useEffect(() => {
    if (tab !== "stats") return;
    const ids = readSaved<string[]>("caro.ids", []);
    void request<Game[]>("/history", {
      method: "POST",
      body: JSON.stringify({ ids }),
    })
      .then((remote) => {
        setHistory((old) => {
          const byId = new Map([...old, ...remote].map((g) => [g.id, g]));
          const merged = [...byId.values()]
            .sort((a, b) => b.created_at - a.created_at)
            .slice(0, 50);
          save("caro.history", merged);
          return merged;
        });
      })
      .catch(() => {});
  }, [tab]);

  function unlockAudio() {
    if (!sound) return;
    try {
      audio.current ??= new AudioContext();
      void audio.current.resume();
    } catch {
      /* Audio is optional. */
    }
  }
  useEffect(() => {
    if (!game?.moves.length) return;
    const key = `${game.id}:${game.revision}`;
    if (lastSound.current === key) return;
    lastSound.current = key;
    if (!sound || !audio.current || audio.current.state !== "running") return;
    const ctx = audio.current,
      osc = ctx.createOscillator(),
      gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(
      game.moves.at(-1)?.player === 1 ? 520 : 390,
      ctx.currentTime,
    );
    gain.gain.setValueAtTime(0.08, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.12);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.12);
  }, [game?.revision, game?.id, sound]);

  const canPlay =
    !!game &&
    game.status === "playing" &&
    !busy &&
    !aiTurn &&
    (game.config.mode === "local" || game.your_player === game.current_player);
  const finished = game?.status === "won" || game?.status === "draw";
  const actual = game?.config || config;
  const preview = useMemo(
    () => ({
      config,
      board: Array.from({ length: config.size }, () =>
        Array<number>(config.size).fill(0),
      ),
      moves: [],
      analysis: null,
      current_player: 1 as Player,
      winning_line: [],
    }),
    [config],
  );
  const moveCount = game?.moves.length || 0;
  const elapsed = game
    ? Math.max(0, Math.floor((game.ended_at || now / 1000) - game.created_at))
    : 0;
  const remaining = game?.config.time_limit
    ? Math.max(
        0,
        Math.ceil(game.turn_started_at + game.config.time_limit - now / 1000),
      )
    : null;
  const ruleText =
    actual.rule === "freestyle"
      ? "Tự do"
      : actual.rule === "exact"
        ? "Chính xác"
        : "Không chặn 2 đầu";

  function playerName(player: Player) {
    if (actual.mode === "ai")
      return player === actual.human_player ? actual.player_name : "Caro AI";
    if (actual.mode === "spectate")
      return player === 1 ? "Caro AI · X" : "Caro AI · O";
    return player === 1 ? actual.player_name : "Người chơi 2";
  }

  async function getHint() {
    if (!session || busy) return;
    setBusy(true);
    actionLock.current = true;
    setError("");
    try {
      const result = await gameRequest<Analysis>(session, "hint");
      if (
        activeSession.current?.id === session.id &&
        currentGame.current?.revision === result.revision
      )
        setHint(result);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
      actionLock.current = false;
    }
  }

  function prepareGame(settings: GameConfig) {
    if (actionLock.current) return;
    activeSession.current = null;
    setSession(null);
    setGame(null);
    setConfig(settings);
    setHint(null);
    setPaused(false);
    setAiFailed(false);
    setError("");
  }

  async function fullscreen() {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await arenaRef.current?.requestFullscreen();
    } catch {
      setToast("Trình duyệt chưa hỗ trợ mở toàn màn hình.");
    }
  }

  const statusText = !game
    ? "Chưa bắt đầu · Chọn thiết lập để chơi"
    : finished
      ? game.status === "draw"
        ? "Một ván cờ cân tài cân sức!"
        : `${playerName(game.winner!)} ${game.winner === actual.human_player && actual.mode === "ai" ? "thắng rồi!" : "chiến thắng!"}`
      : aiTurn
        ? paused
          ? "AI đang tạm dừng"
          : aiFailed
            ? "AI cần kết nối lại"
            : "AI đang suy nghĩ…"
        : `Đến lượt ${playerName(game.current_player)}`;

  return (
    <div className="app-shell">
      <header className="topbar">
        <button
          className="brand"
          onClick={() => setTab("play")}
          aria-label="Caro Lab, về trang chơi"
        >
          <span className="brand-symbol">
            <span />
            <span />
          </span>
          <span>
            caro<span className="brand-light">lab</span>
            <span className="brand-period">.</span>
          </span>
        </button>
        <nav aria-label="Điều hướng chính">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={tab === id ? "active" : ""}
              onClick={() => setTab(id)}
            >
              <Icon size={17} strokeWidth={1.8} />
              <span>{label}</span>
              {id === "lab" && <span className="nav-new">NEW</span>}
            </button>
          ))}
        </nav>
        <div className="topbar-right">
          <span className={`server-status ${serverReady ? "connected" : ""}`}>
            <i />
            {serverReady ? "Máy chủ sẵn sàng" : "Chưa kết nối"}
          </span>
          <span className="topbar-divider" />
          <button
            className="icon-button theme-toggle"
            onClick={() => setDark(!dark)}
            aria-label={dark ? "Chuyển giao diện sáng" : "Chuyển giao diện tối"}
          >
            {dark ? <Sun size={19} /> : <Moon size={19} />}
          </button>
          <button
            className="avatar"
            onClick={() => setModal("settings")}
            aria-label="Thiết lập người chơi"
          >
            {config.player_name.charAt(0).toUpperCase() || "B"}
          </button>
        </div>
      </header>
      <main>
        {tab === "play" ? (
          <>
            <section className="hero">
              <div className="hero-copy">
                <span className="eyebrow hero-eyebrow">
                  <span /> TRÒ CHƠI QUEN THUỘC. TRẢI NGHIỆM MỚI.
                </span>
                <h1>
                  Một nước cờ.
                  <br className="mobile-break" /> <em>Vạn ý tưởng.</em>
                  <span className="hero-star">✳</span>
                </h1>
                <p>
                  Một chút chiến thuật, một chút bất ngờ. Thách đấu bạn bè
                  <br className="desktop-break" /> hoặc thử sức với AI — bàn cờ
                  đang chờ bạn.
                </p>
                <div className="hero-tags">
                  <span>
                    <BrainCircuit size={14} /> AI Deep Learning
                  </span>
                  <i />
                  <span>
                    <Gamepad2 size={14} /> Hai người cùng máy
                  </span>
                  <i />
                  <span>3×3 → 25×25</span>
                </div>
              </div>
              <div className="hero-art" aria-hidden="true">
                <span className="art-orbit orbit-one" />
                <span className="art-orbit orbit-two" />
                <div className="mini-board">
                  {[1, 0, -1, 0, 1, 0, -1, 0, 1].map((p, i) => (
                    <span key={i}>{p ? <Mark player={p} /> : null}</span>
                  ))}
                  <div className="mini-win-line" />
                </div>
                <span className="art-spark spark-one">✦</span>
                <span className="art-spark spark-two">+</span>
                <span className="art-note">your next move ↗</span>
                <span className="art-dot" />
              </div>
            </section>
            {error && !modal && (
              <div className="error-banner" role="alert">
                <CircleHelp size={18} />
                <span>{error}</span>
                <button
                  onClick={() => {
                    setError("");
                    if (!game) void checkServer();
                    if (aiFailed) setAiFailed(false);
                  }}
                >
                  Thử lại
                </button>
                <button
                  className="icon-button"
                  aria-label="Đóng thông báo"
                  onClick={() => setError("")}
                >
                  <X size={16} />
                </button>
              </div>
            )}
            <div className="game-layout">
              <Settings
                config={config}
                setConfig={setConfig}
                busy={busy || initializing}
                onStart={() => {
                  unlockAudio();
                  void startGame(config);
                }}
                onAdvanced={() => setModal("settings")}
                started={!!game}
              />
              <section className="arena panel" ref={arenaRef}>
                <div className="arena-heading">
                  <div>
                    <span className="live-pill">
                      <i />
                      {finished
                        ? "ĐÃ KẾT THÚC"
                        : game
                          ? "ĐANG CHƠI"
                          : "SẴN SÀNG"}
                    </span>
                    <h2>
                      {actual.size} × {actual.size}
                      <span>/</span>
                      <small>{MODE_NAMES[actual.mode]}</small>
                    </h2>
                  </div>
                  <div className="arena-heading-actions">
                    <button
                      className={`icon-button ${sound ? "" : "toggled"}`}
                      onClick={() => {
                        setSound(!sound);
                        if (!sound) {
                          try {
                            audio.current ??= new AudioContext();
                            void audio.current.resume();
                          } catch {}
                        }
                      }}
                      aria-label={sound ? "Tắt âm thanh" : "Bật âm thanh"}
                      title={sound ? "Tắt âm thanh" : "Bật âm thanh"}
                    >
                      {sound ? <Volume2 size={18} /> : <VolumeX size={18} />}
                    </button>
                    <button
                      className="icon-button"
                      onClick={() => void fullscreen()}
                      title="Toàn màn hình"
                      aria-label="Toàn màn hình"
                    >
                      <Maximize2 size={17} />
                    </button>
                  </div>
                </div>
                <div className="players-strip">
                  {([1, -1] as Player[]).map((p) => (
                    <div
                      key={p}
                      className={`player-card ${game?.current_player === p && !finished ? "active" : ""}`}
                    >
                      <span
                        className={`player-symbol player-symbol-${p === 1 ? "x" : "o"}`}
                      >
                        <Mark player={p} />
                      </span>
                      <div>
                        <strong>
                          {playerName(p)}
                          {(actual.mode === "ai" &&
                            p !== actual.human_player) ||
                          actual.mode === "spectate" ? (
                            <span className="ai-label">AI</span>
                          ) : null}
                        </strong>
                        <small>
                          {p === 1 ? "Quân X · Đi trước" : "Quân O · Đi sau"}
                        </small>
                      </div>
                      {game?.current_player === p && !finished && (
                        <span className="player-turn-dot" />
                      )}
                    </div>
                  ))}
                  <span className="players-vs">vs</span>
                </div>
                <div
                  className={`turn-banner ${finished ? "finished-banner" : ""}`}
                  aria-live="polite"
                >
                  {finished ? (
                    <Trophy size={16} />
                  ) : aiTurn && !paused ? (
                    <LoaderCircle
                      className={aiFailed ? "" : "spin"}
                      size={15}
                    />
                  ) : (
                    <span className="turn-indicator" />
                  )}
                  <strong>{statusText}</strong>
                  {remaining !== null && !finished ? (
                    <span className={remaining <= 5 ? "timer urgent" : "timer"}>
                      <Clock3 size={13} />
                      {remaining}s
                    </span>
                  ) : (
                    !finished && (
                      <span className="turn-subtitle">
                        {actual.win_length} quân liên tiếp để thắng
                      </span>
                    )
                  )}
                  {finished && (
                    <button
                      className="text-button"
                      disabled={busy}
                      onClick={() => prepareGame(actual)}
                    >
                      Chơi tiếp <ArrowRight size={14} />
                    </button>
                  )}
                </div>
                <div className={`board-stage ${game ? "" : "board-ready"}`}>
                  <Board
                    game={game || preview}
                    canPlay={canPlay}
                    hint={hint}
                    onMove={(r, c) => {
                      unlockAudio();
                      void act("move", {
                        row: r,
                        col: c,
                        revision: game!.revision,
                      });
                    }}
                    showNumbers={showNumbers}
                    heatmap={heatmap}
                  />
                  {!game && (
                    <div className="ready-overlay">
                      <span className="ready-icon">
                        <Gamepad2 size={27} strokeWidth={1.5} />
                      </span>
                      <span className="eyebrow">BÀN CỜ CỦA BẠN</span>
                      <h3>Sẵn sàng cho nước đầu tiên?</h3>
                      <p>
                        {config.size}×{config.size} · Nối {config.win_length}{" "}
                        quân · {MODE_NAMES[config.mode]}
                      </p>
                      <button
                        className="button primary"
                        disabled={busy || initializing}
                        onClick={() => {
                          unlockAudio();
                          void startGame(config);
                        }}
                      >
                        {busy ? (
                          <LoaderCircle className="spin" size={16} />
                        ) : (
                          <Play size={16} />
                        )}{" "}
                        Bắt đầu
                      </button>
                      <small>
                        Bạn chọn luật chơi. Nước đầu tiên luôn chờ bạn.
                      </small>
                    </div>
                  )}
                </div>
                <div className="board-caption">
                  <span>
                    <span className="caption-dot" />
                    {game?.moves.length
                      ? `Nước vừa đi: ${String.fromCharCode(65 + game.moves.at(-1)!.col)}${game.moves.at(-1)!.row + 1}`
                      : game
                        ? "Nhấp vào một ô trống để đặt quân"
                        : "Nhấn Bắt đầu để mở bàn cờ"}
                  </span>
                  <span>
                    {ruleText} <span className="caption-divider">·</span>{" "}
                    {actual.win_length} quân thắng
                  </span>
                </div>
                <div className="game-toolbar">
                  <button
                    disabled={
                      !game?.moves.length ||
                      busy ||
                      finished ||
                      actual.mode === "spectate"
                    }
                    onClick={() => void act("undo")}
                    title="Hoàn tác lượt gần nhất"
                  >
                    <Undo2 size={17} />
                    <span>Đi lại</span>
                  </button>
                  <button
                    disabled={!canPlay || actual.mode === "spectate"}
                    onClick={() => void getHint()}
                    className={hint ? "tool-active" : ""}
                  >
                    <Lightbulb size={17} />
                    <span>Gợi ý</span>
                  </button>
                  <span className="toolbar-divider" />
                  <button
                    onClick={() => setShowNumbers(!showNumbers)}
                    aria-pressed={showNumbers}
                    className={showNumbers ? "tool-active" : ""}
                  >
                    <Hash size={17} />
                    <span>Số nước</span>
                  </button>
                  {actual.mode === "spectate" ? (
                    <button
                      onClick={() => setPaused(!paused)}
                      disabled={!game || finished}
                      aria-label={paused ? "Tiếp tục AI" : "Tạm dừng AI"}
                    >
                      {paused ? <Play size={17} /> : <Pause size={17} />}
                      <span>{paused ? "Tiếp tục" : "Tạm dừng"}</span>
                    </button>
                  ) : (
                    <button
                      disabled={!game || finished || busy}
                      onClick={() => setModal("resign")}
                    >
                      <Flag size={16} />
                      <span>Đầu hàng</span>
                    </button>
                  )}
                  <button
                    className="restart-tool"
                    disabled={!game || busy}
                    onClick={() => prepareGame(actual)}
                    aria-label="Thiết lập lại ván cờ"
                    title="Thiết lập lại ván cờ"
                  >
                    <RotateCcw size={17} />
                  </button>
                </div>
                {hint && (
                  <div className="hint-message">
                    <Lightbulb size={16} />
                    <span>
                      Thử{" "}
                      <strong>
                        {String.fromCharCode(65 + hint.col)}
                        {hint.row + 1}
                      </strong>{" "}
                      — {hint.reason.toLowerCase()}.
                    </span>
                    <button
                      className="icon-button"
                      onClick={() => setHint(null)}
                      aria-label="Ẩn gợi ý"
                    >
                      <X size={14} />
                    </button>
                  </div>
                )}
                {finished && game?.ended_reason !== "line" && (
                  <p className="end-reason">
                    {game?.ended_reason === "timeout"
                      ? "Ván cờ kết thúc do hết thời gian lượt đi."
                      : game?.ended_reason === "resign"
                        ? "Ván cờ kết thúc do một bên đầu hàng."
                        : "Bàn cờ đã đầy và không có bên nào thắng."}
                  </p>
                )}
              </section>
              <aside className="insights-column">
                <section className="ai-card">
                  <div className="ai-card-top">
                    <span className="ai-brain">
                      <BrainCircuit size={24} strokeWidth={1.5} />
                    </span>
                    <span className="ai-status">
                      <i />
                      {model?.loaded ? "MODEL READY" : "TACTICAL MODE"}
                    </span>
                  </div>
                  <h2>
                    Đối thủ biết <em>học hỏi.</em>
                  </h2>
                  <p>
                    CaroNet đọc thế cờ bằng deep learning, tìm chiến thuật cho
                    từng nước đi.
                  </p>
                  <div className="ai-mini-stats">
                    <div>
                      <strong>
                        {model
                          ? `${(model.parameters / 1000).toFixed(1)}K`
                          : "—"}
                      </strong>
                      <small>tham số</small>
                    </div>
                    <div>
                      <strong>
                        {model?.training.samples
                          ? `${model.training.samples / 1000}K`
                          : "—"}
                      </strong>
                      <small>mẫu huấn luyện</small>
                    </div>
                  </div>
                  <button onClick={() => setTab("lab")}>
                    Khám phá AI Lab <ArrowUpRight size={16} />
                  </button>
                  <span className="ai-card-decoration" aria-hidden="true">
                    ✳
                  </span>
                </section>
                <section className="panel session-panel">
                  <div className="panel-heading">
                    <h2>Ván cờ hiện tại</h2>
                    <Activity size={16} />
                  </div>
                  <div className="session-metrics">
                    <span>
                      <small>Nước đã đi</small>
                      <strong>
                        {String(moveCount).padStart(2, "0")}
                        <span>/{actual.size ** 2}</span>
                      </strong>
                    </span>
                    <span>
                      <small>Thời gian</small>
                      <strong>
                        {String(Math.floor(elapsed / 60)).padStart(2, "0")}
                        <span>:</span>
                        {String(elapsed % 60).padStart(2, "0")}
                      </strong>
                    </span>
                  </div>
                  <div className="move-history-heading">
                    <span>
                      <History size={14} /> Các nước gần đây
                    </span>
                    {game && <small>{moveCount} nước</small>}
                  </div>
                  <div className="recent-moves">
                    {game?.moves.length ? (
                      [...game.moves]
                        .reverse()
                        .slice(0, 5)
                        .map((m) => (
                          <div className="recent-move" key={m.number}>
                            <span className="move-order">
                              {String(m.number).padStart(2, "0")}
                            </span>
                            <Mark player={m.player} small />
                            <strong>
                              {String.fromCharCode(65 + m.col)}
                              {m.row + 1}
                            </strong>
                            <span>{playerName(m.player)}</span>
                            {m.number === moveCount && <i />}
                          </div>
                        ))
                    ) : (
                      <div className="no-moves">
                        <span className="empty-move-icons">
                          <Mark player={1} small />
                          <Mark player={-1} small />
                        </span>
                        <p>
                          Nước đi đầu tiên
                          <br />
                          luôn mở ra điều mới.
                        </p>
                      </div>
                    )}
                  </div>
                </section>
                {game?.analysis && (
                  <section className="panel analysis-panel">
                    <div className="panel-heading">
                      <h2>
                        <Sparkles size={15} /> AI vừa nghĩ gì?
                      </h2>
                      <small>{game.analysis.elapsed_ms}ms</small>
                    </div>
                    <p>{game.analysis.reason}</p>
                    <button
                      className="text-button"
                      aria-pressed={heatmap}
                      onClick={() => setHeatmap(!heatmap)}
                    >
                      {heatmap ? "Ẩn" : "Xem"} vùng AI đánh giá{" "}
                      <ArrowUpRight size={13} />
                    </button>
                  </section>
                )}
                <button className="quick-guide" onClick={() => setTab("guide")}>
                  <span className="quick-guide-icon">
                    <BookOpen size={19} />
                  </span>
                  <span>
                    <strong>Lần đầu chơi caro?</strong>
                    <small>Nắm luật trong 1 phút</small>
                  </span>
                  <ChevronRight size={17} />
                </button>
              </aside>
            </div>
            <div className="below-game">
              <span>
                <span className="below-icon">✧</span> Chơi một ván. Nghỉ một
                chút. Trở lại với một ý tưởng mới.
              </span>
              <button onClick={() => setTab("guide")}>
                Mẹo chơi hay <ArrowUpRight size={14} />
              </button>
            </div>
          </>
        ) : tab === "guide" ? (
          <Guide />
        ) : tab === "lab" ? (
          <AILab model={model} />
        ) : (
          <Statistics history={history} onReplay={setReplay} />
        )}
      </main>
      <footer>
        <span className="footer-brand">
          carolab<span>.</span>
        </span>
        <span>Trò chơi nhỏ. Tư duy lớn.</span>
        <span className="footer-tech">
          Built with React, FastAPI & a little intelligence <span>✳</span>
        </span>
        <button onClick={() => setTab("guide")}>
          Cách chơi <ArrowUpRight size={12} />
        </button>
      </footer>
      {toast && (
        <div className="toast" role="status">
          <Check size={16} />
          {toast}
        </div>
      )}
      {modal === "settings" && (
        <Dialog title="Tùy chỉnh ván cờ" onClose={() => setModal(null)}>
          <p className="dialog-description">
            Chọn cách chơi phù hợp với bạn. Các thiết lập áp dụng khi bắt đầu
            ván mới.
          </p>
          <AdvancedSettings config={config} setConfig={setConfig} />
          <button
            className="button primary dialog-primary"
            onClick={() => setModal(null)}
          >
            Lưu thiết lập <Check size={17} />
          </button>
        </Dialog>
      )}
      {modal === "resign" && (
        <Dialog title="Dừng ván cờ tại đây?" onClose={() => setModal(null)}>
          <p className="dialog-description">
            Đối thủ sẽ được tính là người thắng. Ván cờ này vẫn được lưu để bạn
            xem lại.
          </p>
          <div className="dialog-buttons">
            <button className="button secondary" onClick={() => setModal(null)}>
              Chơi tiếp
            </button>
            <button
              className="button primary"
              onClick={() => {
                setModal(null);
                void act("resign");
              }}
            >
              Đầu hàng <Flag size={15} />
            </button>
          </div>
        </Dialog>
      )}
      {replay && (
        <Dialog title="Xem lại ván cờ" onClose={() => setReplay(null)} wide>
          <Replay game={replay} onClose={() => setReplay(null)} />
        </Dialog>
      )}
    </div>
  );
}
