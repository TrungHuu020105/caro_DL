"""SQLite snapshots for active games and public, token-free finished history."""
import json
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from time import time
from .game import Game


class Storage:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS games (id TEXT PRIMARY KEY, created REAL, payload TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS live_games (id TEXT PRIMARY KEY, updated REAL, payload TEXT)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path)
        try:
            with db:
                yield db
        finally:
            db.close()

    def save(self, game: Game):
        with self.connect() as db:
            if not game.saved and game.status in ("won", "draw"):
                payload = game.view(game.host_token)
                db.execute("INSERT OR REPLACE INTO games VALUES (?, ?, ?)",
                           (game.id, game.created_at, json.dumps(payload, ensure_ascii=False)))
                game.saved = True
            db.execute("INSERT OR REPLACE INTO live_games VALUES (?, ?, ?)",
                       (game.id, time(), json.dumps(game.record(), ensure_ascii=False)))

    def load_active(self):
        with self.connect() as db:
            db.execute("DELETE FROM live_games WHERE updated < ?", (time()-12*3600,))
            rows = db.execute("SELECT payload FROM live_games ORDER BY updated DESC LIMIT 2000").fetchall()
        return [Game.restore(json.loads(row[0])) for row in rows]

    def forget_active(self, game_id):
        with self.connect() as db:
            db.execute("DELETE FROM live_games WHERE id = ?", (game_id,))

    def history(self, ids: list[str]):
        # Only return games whose unpredictable ids are held by this browser.
        if not ids:
            return []
        ids = ids[:100]
        marks = ",".join("?" * len(ids))
        with self.connect() as db:
            rows = db.execute(f"SELECT payload FROM games WHERE id IN ({marks}) ORDER BY created DESC LIMIT 50", ids).fetchall()
        return [json.loads(row[0]) for row in rows]
