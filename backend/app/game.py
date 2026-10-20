from dataclasses import dataclass, field
from time import time
import secrets
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .rules import winning_line


class GameConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    size: int = Field(default=15, ge=3, le=25)
    win_length: int | None = Field(default=None, ge=3, le=6)
    mode: Literal["ai", "local", "spectate"] = "ai"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    rule: Literal["freestyle", "exact", "blocked"] = "freestyle"
    human_player: Literal[1, -1] = 1
    time_limit: Literal[0, 15, 30, 60] = 0
    player_name: str = Field(default="Bạn", min_length=1, max_length=24)

    @model_validator(mode="after")
    def validate_length(self):
        if self.win_length is None:
            self.win_length = min(self.size, 5)
        if self.win_length > self.size:
            raise ValueError("Số quân liên tiếp không được lớn hơn kích thước bàn cờ.")
        self.player_name = self.player_name.strip() or "Bạn"
        return self


@dataclass
class Game:
    config: GameConfig
    id: str = field(default_factory=lambda: secrets.token_urlsafe(16))
    host_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    board: list[list[int]] = field(init=False)
    moves: list[dict] = field(default_factory=list)
    current_player: int = 1
    status: str = "playing"
    winner: int | None = None
    win_line: list = field(default_factory=list)
    revision: int = 0
    analysis: dict | None = None
    created_at: float = field(default_factory=time)
    turn_started_at: float = field(default_factory=time)
    ended_reason: str | None = None
    ended_at: float | None = None
    saved: bool = False

    def __post_init__(self):
        self.board = [[0] * self.config.size for _ in range(self.config.size)]

    def player_for(self, token):
        if token == self.host_token:
            return self.config.human_player if self.config.mode == "ai" else 1
        raise ValueError("Bạn không có quyền truy cập ván cờ này.")

    def timeout(self):
        if self.status == "playing" and self.config.time_limit and time() - self.turn_started_at >= self.config.time_limit:
            self.status, self.winner, self.ended_reason = "won", -self.current_player, "timeout"
            self.ended_at = time()
            self.revision += 1
            return True
        return False

    def place(self, row, col):
        self.timeout()
        if self.status != "playing":
            raise ValueError("Ván cờ chưa bắt đầu hoặc đã kết thúc.")
        if not (0 <= row < self.config.size and 0 <= col < self.config.size):
            raise ValueError("Nước đi nằm ngoài bàn cờ.")
        if self.board[row][col]:
            raise ValueError("Ô này đã có quân cờ.")
        player = self.current_player
        self.board[row][col] = player
        self.moves.append({"row": row, "col": col, "player": player,
                           "number": len(self.moves) + 1, "at": time()})
        self.win_line = winning_line(self.board, row, col, self.config.win_length, self.config.rule)
        if self.win_line:
            self.status, self.winner, self.ended_reason = "won", player, "line"
        elif len(self.moves) == self.config.size ** 2:
            self.status, self.ended_reason = "draw", "full"
        else:
            self.current_player *= -1
        self.turn_started_at = time()
        if self.status in ("won", "draw"):
            self.ended_at = self.turn_started_at
        self.revision += 1

    def undo(self):
        if self.config.mode == "spectate":
            raise ValueError("Chế độ này không hỗ trợ đi lại.")
        if not self.moves or self.status != "playing":
            raise ValueError("Không thể đi lại lúc này.")
        count = 1
        if self.config.mode == "ai" and self.current_player == self.config.human_player:
            count = min(2, len(self.moves))
        for _ in range(count):
            move = self.moves.pop()
            self.board[move["row"]][move["col"]] = 0
            self.current_player = move["player"]
        self.analysis = None
        self.turn_started_at = time()
        self.revision += 1

    def view(self, token):
        player = self.player_for(token)
        return {"id": self.id, "config": self.config.model_dump(), "board": self.board,
                "moves": self.moves, "current_player": self.current_player, "status": self.status,
                "winner": self.winner, "winning_line": self.win_line, "revision": self.revision,
                "analysis": self.analysis, "your_player": player,
                "created_at": self.created_at, "turn_started_at": self.turn_started_at,
                "ended_reason": self.ended_reason, "ended_at": self.ended_at}
