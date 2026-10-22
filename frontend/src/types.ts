export type Mode = "ai" | "local" | "online" | "spectate";
export type Player = 1 | -1;
export interface GameConfig {
  size: number;
  win_length: number | null;
  mode: Mode;
  difficulty: "easy" | "medium" | "hard";
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
  candidates: { row: number; col: number; probability: number }[];
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
  ended_reason: "line" | "timeout" | "resign" | "full" | null;
  ended_at?: number | null;
}
export interface Session {
  id: string;
  token: string;
}
export interface ModelInfo {
  name: string;
  architecture: string;
  loaded: boolean;
  device: string;
  parameters: number;
  description: string;
  training: {
    samples?: number;
    validation_samples?: number;
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
    }[];
  };
}
