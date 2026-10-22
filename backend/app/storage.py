"""SQLite snapshots for active games and public, token-free finished history."""
import json
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from time import time
from .game import Game
from .experience import human_experience


class Storage:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS games (id TEXT PRIMARY KEY, created REAL, payload TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS live_games (id TEXT PRIMARY KEY, updated REAL, payload TEXT)")
            db.execute("""CREATE TABLE IF NOT EXISTS human_games (
                game_hash TEXT PRIMARY KEY, source_id TEXT UNIQUE, collected REAL,
                split TEXT, positions INTEGER, payload TEXT, last_run TEXT)""")
            db.execute("CREATE INDEX IF NOT EXISTS human_games_pending ON human_games(last_run)")
            db.execute("CREATE TABLE IF NOT EXISTS human_position_split (position_key TEXT PRIMARY KEY, split TEXT)")

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
                self._collect(db, payload)
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

    @staticmethod
    def _collect(db, payload):
        record = human_experience(payload)
        if record is not None:
            inserted = db.execute("INSERT OR IGNORE INTO human_games VALUES (?, ?, ?, ?, ?, ?, NULL)",
                       (record["game_hash"], record["source_id"], time(), record["split"],
                        record["position_count"], json.dumps(record, ensure_ascii=False))).rowcount
            if not inserted:
                stored = json.loads(db.execute("SELECT payload FROM human_games WHERE game_hash=?", (record["game_hash"],)).fetchone()[0])
                if "position_keys" in stored:
                    return
                db.execute("UPDATE human_games SET payload=? WHERE game_hash=?", (json.dumps(record, ensure_ascii=False), record["game_hash"]))
            db.executemany("INSERT OR IGNORE INTO human_position_split VALUES (?, ?)",
                           [(key, record["split"]) for key in record["position_keys"] if key is not None])

    def backfill_experience(self):
        with self.connect() as db:
            for row in db.execute("SELECT payload FROM games ORDER BY created").fetchall():
                try:
                    self._collect(db, json.loads(row[0]))
                except (ValueError, TypeError):
                    continue

    def experience_stats(self):
        with self.connect() as db:
            row = db.execute("""SELECT COUNT(*), COALESCE(SUM(positions),0),
                COALESCE(SUM(last_run IS NULL),0), COALESCE(SUM(CASE WHEN last_run IS NULL THEN positions ELSE 0 END),0),
                COALESCE(SUM(split='train'),0), COALESCE(SUM(split='validation'),0) FROM human_games""").fetchone()
        return dict(zip(("total_games", "total_positions", "pending_games", "pending_positions", "train_games", "validation_games"), row))

    def experience_snapshot(self, limit=500):
        with self.connect() as db:
            rows = db.execute("""SELECT payload FROM human_games
                ORDER BY (last_run IS NULL) DESC, collected DESC LIMIT ?""", (limit,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def experience_position_splits(self):
        with self.connect() as db:
            return dict(db.execute("SELECT position_key, split FROM human_position_split").fetchall())

    def mark_experience_used(self, hashes, run_id):
        with self.connect() as db:
            for start in range(0, len(hashes), 500):
                batch = hashes[start:start+500]
                marks = ",".join("?" for _ in batch)
                if batch:
                    db.execute(f"UPDATE human_games SET last_run=? WHERE game_hash IN ({marks})", [run_id, *batch])
