export type Mode = "ai" | "local" | "online" | "spectate";
export type Player = 1 | -1;
export type EngineVariant = "hybrid" | "cnn" | "heuristic";
export interface GameConfig {
  size: number;
  win_length: number | null;
  mode: Mode;
  difficulty: "easy" | "medium" | "hard";
  ai_variant: EngineVariant;
  rule: "freestyle" | "exact" | "blocked";
  human_player: Player;
  time_limit: number;
  player_name: string;
}
export interface Move {
  row: number;
  col: number;
  player: Player;
  number: number;
  at: number;
}
export interface Analysis {
  row: number;
  col: number;
  reason: string;
  engine: string;
  elapsed_ms: number;
  value: number;
  candidates: {
    row: number;
    col: number;
    probability: number;
    neural_score?: number | null;
    tactical_score?: number;
    combined_score?: number;
    search_score?: number | null;
    selected?: boolean;
  }[];
  value_used?: boolean;
  weights?: { neural: number; tactical: number };
  search?: { nodes: number; completed_depth: number; budget_ms: number };
  position?: { player: Player; move_number: number };
  revision?: number;
}
export interface Game {
  id: string;
  config: GameConfig;
  board: number[][];
  moves: Move[];
  current_player: Player;
  status: "waiting" | "playing" | "won" | "draw";
  winner: Player | null;
  winning_line: number[][];
  revision: number;
  analysis: Analysis | null;
  your_player: Player;
  room_code: string | null;
  guest_joined: boolean;
  guest_name: string;
  started_at?: number | null;
  created_at: number;
  turn_started_at: number;
  paused_at?: number | null;
  paused_seconds?: number;
  ended_reason: "line" | "timeout" | "resign" | "full" | null;
  ended_at?: number | null;
}
export interface Session {
  id: string;
  token: string;
}
export interface ModelInfo {
  checkpoint_sha256?: string;
  name: string;
  architecture: string;
  loaded: boolean;
  device: string;
  parameters: number;
  description: string;
  load_error?: string | null;
  training: {
    samples?: number;
    validation_samples?: number;
    test_samples?: number;
    best_epoch?: number;
    training_board_sizes?: number[];
    training_rules?: string[];
    dataset?: {
      audit?: {
        duplicate_or_symmetric_positions_between_splits: number;
        shared_games_between_splits: number;
      };
    };
    test_metrics?: EvaluationMetrics;
    epochs?: number;
    seconds?: number;
    best_val_loss?: number;
    method?: string;
    trained_at?: string;
    limitations?: string;
    history?: {
      epoch: number;
      loss: number;
      val_loss: number;
      top1_accuracy: number;
      val_value_mse?: number;
    }[];
  };
}

export interface EvaluationMetrics {
  samples: number;
  loss: number;
  policy_loss: number;
  value_mse: number;
  value_zero_baseline_mse?: number;
  top1_accuracy: number;
  by_size: Record<string, Omit<EvaluationMetrics, "by_size" | "by_rule">>;
  by_rule: Record<string, Omit<EvaluationMetrics, "by_size" | "by_rule">>;
}
export interface Standing {
  engine: EngineVariant;
  games: number;
  wins: number;
  draws: number;
  losses: number;
  score_rate: number;
  mean_ms?: number;
  p95_ms?: number;
}
export interface BenchmarkReport {
  created_at: string;
  seed: number;
  pairs: number;
  budget_ms: number;
  total_games: number;
  checkpoint_sha256: string;
  summary: Standing[];
  by_configuration: { size: number; rule: string; summary: Standing[] }[];
}
export interface ExperimentState {
  status: "idle" | "running" | "completed" | "failed";
  completed: number;
  total: number;
  report: BenchmarkReport | null;
  error: string | null;
  stale?: boolean;
}
export interface GameReview {
  summary: Record<
    string,
    { missed_win: number; missed_block: number; double_threat: number }
  >;
  scope: string;
  events: {
    number: number;
    player: Player;
    kind: "missed_win" | "missed_block" | "double_threat";
    played: number[];
    alternatives: number[][];
    explanation: string;
  }[];
}

export interface HumanLearningReport {
  run_id: string;
  decision: string;
  message: string;
  finished_at: string;
  seconds: number;
  replay_positions?: number;
  data?: {
    selected_train_positions: number;
    selected_validation_positions: number;
  };
  history?: {
    epoch: number;
    loss: number;
    val_loss: number;
    top1_accuracy: number;
  }[];
  arena?: {
    wins: number;
    draws: number;
    losses: number;
    games: number;
    score_rate: number;
  };
  gate?: { accepted: boolean; reasons: string[]; limitation: string };
}
export interface HumanLearningState {
  active_checkpoint_sha256?: string;
  status: "idle" | "running" | "completed" | "failed";
  phase: string;
  message: string;
  completed: number;
  total: number;
  error: string | null;
  min_positions: number;
  settings: { enabled: boolean; min_new_games: number; epochs: number };
  stats: {
    total_games: number;
    pending_games: number;
    total_positions: number;
    validation_games: number;
  };
  report: HumanLearningReport | null;
}
