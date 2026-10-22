"""Validate and anonymize completed games involving human players."""
from .game import Game, GameConfig
import numpy as np
from ..dataset import position_key, trajectory_key


def human_experience(payload):
    """Return a training record, or None for incomplete/invalid/AI-only games."""
    try:
        config = GameConfig(**payload["config"])
        if config.mode not in ("ai", "local", "online"):
            return None
        if payload.get("ended_reason") not in ("line", "full"):
            return None
        moves = payload["moves"]
        if not 5 <= len(moves) <= config.size**2 or not isinstance(payload.get("id"), str):
            return None
        replay = Game(GameConfig(size=config.size, win_length=config.win_length,
                                 rule=config.rule, mode="local"))
        coordinates, keys = [], []
        for ply, move in enumerate(moves):
            r, c = move["row"], move["col"]
            if type(r) is not int or type(c) is not int or move.get("player", replay.current_player) != replay.current_player:
                return None
            keys.append(position_key(np.asarray(replay.board, dtype=np.int8), replay.current_player,
                                     config.win_length, config.rule).hex() if ply >= 2 else None)
            replay.place(r, c)
            coordinates.append([r, c])
        if (replay.status not in ("won", "draw") or replay.status != payload.get("status")
                or replay.winner != payload.get("winner") or replay.ended_reason != payload.get("ended_reason")
                or payload.get("board", replay.board) != replay.board):
            return None
        key = trajectory_key(config.size, config.win_length, config.rule, coordinates).hex()
        return {"game_hash": key, "source_id": payload["id"], "size": config.size,
                "win_length": config.win_length, "rule": config.rule, "mode": config.mode,
                "winner": replay.winner or 0, "moves": coordinates,
                "position_keys": keys,
                "split": "validation" if int(key[:8], 16) % 5 == 0 else "train",
                "position_count": len(moves)-2}
    except (ValueError, TypeError, KeyError, IndexError):
        return None
