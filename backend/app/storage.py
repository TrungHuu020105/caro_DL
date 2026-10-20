"""Finished games survive API restarts; active games stay in memory."""
import json
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from .game import Game


class Storage:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS games (id TEXT PRIMARY KEY, created REAL, payload TEXT)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path)
        try:
            with db:
                yield db
        finally:
            db.close()

    def save(self, game: Game):
        if game.saved or game.status not in ("won", "draw"):
            return
        payload = game.view(game.host_token)
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO games VALUES (?, ?, ?)",
                       (game.id, game.created_at, json.dumps(payload, ensure_ascii=False)))
        game.saved = True

    def history(self, ids: list[str]):
        # Only return games whose unpredictable ids are held by this browser.
        if not ids:
            return []
        ids = ids[:100]
        marks = ",".join("?" * len(ids))
        with self.connect() as db:
            rows = db.execute(f"SELECT payload FROM games WHERE id IN ({marks}) ORDER BY created DESC LIMIT 50", ids).fetchall()
        return [json.loads(row[0]) for row in rows]
