import { useRef, type KeyboardEvent } from "react";
import type { Analysis, Game, Player } from "./types";

export function Mark({
  player,
  small = false,
}: {
  player: number;
  small?: boolean;
}) {
  return (
    <span
      className={`mark ${player === 1 ? "mark-x" : "mark-o"} ${small ? "mark-small" : ""}`}
      aria-hidden="true"
    />
  );
}

export default function Board({
  game,
  canPlay,
  hint,
  onMove,
  showNumbers,
  heatmap = false,
  compact = false,
}: {
  game: Game;
  canPlay: boolean;
  hint: Analysis | null;
  onMove: (r: number, c: number) => void;
  showNumbers: boolean;
  heatmap?: boolean;
  compact?: boolean;
}) {
  const boardRef = useRef<HTMLDivElement>(null);
  const n = game.config.size;
  const last = game.moves.at(-1);
  const winning = new Set(game.winning_line.map(([r, c]) => `${r},${c}`));
  const numbered = new Map(
    game.moves.map((m) => [`${m.row},${m.col}`, m.number]),
  );
  const candidates = new Map(
    game.analysis?.candidates.map((c) => [
      `${c.row},${c.col}`,
      c.probability,
    ]) || [],
  );
  function onKey(
    event: KeyboardEvent<HTMLButtonElement>,
    r: number,
    c: number,
  ) {
    const directions: Record<string, [number, number]> = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1],
    };
    if (!directions[event.key]) return;
    event.preventDefault();
    const [dr, dc] = directions[event.key];
    const row = (r + dr + n) % n,
      col = (c + dc + n) % n;
    boardRef.current
      ?.querySelector<HTMLButtonElement>(`[data-cell="${row}-${col}"]`)
      ?.focus();
  }
  return (
    <div className={`board-scroll ${compact ? "compact" : ""}`}>
      <div
        className="board-coordinates"
        style={{ "--size": n } as React.CSSProperties}
      >
        <div className="column-labels">
          {Array.from({ length: n }, (_, c) => (
            <span key={c}>{String.fromCharCode(65 + c)}</span>
          ))}
        </div>
        <div className="row-labels">
          {Array.from({ length: n }, (_, r) => (
            <span key={r}>{r + 1}</span>
          ))}
        </div>
        <div
          ref={boardRef}
          className={`board ${n <= 6 ? "board-large" : ""}`}
          role="grid"
          aria-label={`Bàn cờ ${n} nhân ${n}`}
          aria-rowcount={n}
          aria-colcount={n}
        >
          {game.board.map((row, r) => (
            <div className="board-row" role="row" key={r}>
              {row.map((cell, c) => {
                const key = `${r},${c}`;
                const isLast = last?.row === r && last?.col === c;
                const isHint = hint?.row === r && hint?.col === c && !cell;
                return (
                  <button
                    key={c}
                    role="gridcell"
                    data-cell={`${r}-${c}`}
                    aria-rowindex={r + 1}
                    aria-colindex={c + 1}
                    aria-label={`${String.fromCharCode(65 + c)}${r + 1}${cell ? `: ${cell === 1 ? "X" : "O"}` : ": ô trống"}`}
                    aria-disabled={!canPlay || !!cell}
                    className={`cell ${cell ? "occupied" : ""} ${isLast ? "last-move" : ""} ${winning.has(key) ? "winning" : ""} ${isHint ? "hint-cell" : ""}`}
                    style={
                      heatmap && !cell && candidates.has(key)
                        ? {
                            backgroundColor: `rgba(70, 136, 96, ${0.1 + candidates.get(key)! * 0.5})`,
                          }
                        : undefined
                    }
                    onKeyDown={(e) => onKey(e, r, c)}
                    onClick={() => canPlay && !cell && onMove(r, c)}
                  >
                    {cell ? (
                      <>
                        <Mark player={cell as Player} />
                        {showNumbers && (
                          <span className="move-number">
                            {numbered.get(key)}
                          </span>
                        )}
                      </>
                    ) : isHint ? (
                      <span className="hint-dot" />
                    ) : (
                      <span
                        className={`cell-preview preview-${game.current_player === 1 ? "x" : "o"}`}
                      />
                    )}
                    {isLast && <span className="last-dot" />}
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
