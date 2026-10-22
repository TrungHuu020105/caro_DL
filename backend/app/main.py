import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import secrets
import os
from time import time
from typing import Annotated
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool
from .ai import AIEngine
from .game import Game, GameConfig
from .storage import Storage
from .review import replay_prefix, review_game
from ..benchmark import ExperimentRunner

games: dict[str, Game] = {}
locks: dict[str, asyncio.Lock] = {}
peers: dict[str, set[WebSocket]] = {}
engine: AIEngine | None = None
storage: Storage | None = None
experiments: ExperimentRunner | None = None
experiment_tasks: set[asyncio.Task] = set()


async def broadcast(game):
    for ws in list(peers.get(game.id, set())):
        try:
            await asyncio.wait_for(ws.send_json({"type": "state", "game": game.view(ws.state.token)}), timeout=2)
        except (RuntimeError, WebSocketDisconnect, OSError, asyncio.TimeoutError):
            peers.get(game.id, set()).discard(ws)


async def clock_loop():
    while True:
        await asyncio.sleep(.5)
        for game_id, game in list(games.items()):
            async with locks[game_id]:
                if game.timeout():
                    storage.save(game)
                    await broadcast(game)
            if time() - max(game.created_at, game.turn_started_at, game.paused_at or 0) > 12 * 3600 and not peers.get(game_id):
                games.pop(game_id, None)
                locks.pop(game_id, None)
                peers.pop(game_id, None)
                storage.forget_active(game_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, storage, experiments
    engine = AIEngine()
    storage = Storage(Path(os.environ.get("CARO_DATABASE_PATH", Path(__file__).resolve().parents[1] / "data" / "caro.sqlite3")))
    games.clear(); locks.clear(); peers.clear()
    for game in storage.load_active():
        games[game.id] = game
        locks[game.id] = asyncio.Lock()
    experiments = ExperimentRunner(Path(os.environ.get("CARO_BENCHMARK_PATH", Path(__file__).resolve().parents[1]/"models"/"benchmark.json")))
    task = asyncio.create_task(clock_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    experiments.cancel.set()
    if experiment_tasks:
        await asyncio.gather(*experiment_tasks, return_exceptions=True)


app = FastAPI(title="Caro Lab API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware,
                   allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173", "http://127.0.0.1:4173"],
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-Player-Token"])
Token = Annotated[str, Header(alias="X-Player-Token")]


def get_game(game_id, token):
    game = games.get(game_id)
    if not game:
        raise HTTPException(404, "Không tìm thấy ván cờ. Ván cờ có thể đã hết hạn hoặc máy chủ đã khởi động lại.")
    try:
        game.player_for(token)
    except ValueError as exc:
        raise HTTPException(403, str(exc)) from exc
    return game


async def finish(game, token):
    storage.save(game)
    await broadcast(game)
    return game.view(token)


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": engine.loaded}


@app.get("/api/model")
def model_info():
    return engine.info()


@app.post("/api/games", status_code=201)
async def create_game(config: GameConfig):
    if len(games) >= 2000:
        raise HTTPException(503, "Máy chủ đang đầy, vui lòng thử lại sau.")
    if config.mode in ("ai", "spectate") and config.ai_variant == "cnn" and not engine.loaded:
        raise HTTPException(400, "Chưa có checkpoint CNN. Hãy huấn luyện hoặc chọn engine khác.")
    game = Game(config)
    if config.mode == "online":
        used = {g.room_code for g in games.values()}
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        while not game.room_code or game.room_code in used:
            game.room_code = "".join(secrets.choice(alphabet) for _ in range(6))
    games[game.id] = game
    locks[game.id] = asyncio.Lock()
    storage.save(game)
    return {"game": game.view(game.host_token), "token": game.host_token}


class JoinRequest(BaseModel):
    code: str = Field(min_length=6, max_length=6, pattern=r"^[A-Za-z0-9]{6}$")
    name: str = Field(default="Người chơi 2", min_length=1, max_length=24)


@app.post("/api/rooms/join")
async def join_room(body: JoinRequest):
    game = next((g for g in games.values() if g.room_code == body.code.upper()), None)
    if not game:
        raise HTTPException(404, "Mã phòng không tồn tại hoặc đã hết hạn.")
    async with locks[game.id]:
        if game.guest_token or game.status != "waiting":
            raise HTTPException(409, "Phòng đã đủ người hoặc ván cờ đã kết thúc.")
        game.guest_token = secrets.token_urlsafe(32)
        game.guest_name = body.name.strip() or "Người chơi 2"
        game.status = "playing"
        game.started_at = game.turn_started_at = time()
        game.revision += 1
        storage.save(game)
        await broadcast(game)
        return {"game": game.view(game.guest_token), "token": game.guest_token}


@app.get("/api/games/{game_id}")
async def game_state(game_id: str, x_player_token: Token):
    game = get_game(game_id, x_player_token)
    async with locks[game_id]:
        if game.timeout():
            return await finish(game, x_player_token)
        return game.view(x_player_token)


class MoveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    row: int = Field(ge=0, le=24)
    col: int = Field(ge=0, le=24)
    revision: int = Field(ge=0)


@app.post("/api/games/{game_id}/move")
async def move(game_id: str, body: MoveRequest, x_player_token: Token):
    game = get_game(game_id, x_player_token)
    async with locks[game_id]:
        if body.revision != game.revision:
            raise HTTPException(409, "Bàn cờ đã thay đổi. Hãy thử lại.")
        mode = game.config.mode
        if mode == "spectate" or (mode != "local" and game.player_for(x_player_token) != game.current_player):
            raise HTTPException(403, "Chưa đến lượt của bạn.")
        try:
            game.place(body.row, body.col)
        except ValueError as exc:
            if game.status in ("won", "draw") and not game.saved:
                await finish(game, x_player_token)
            raise HTTPException(400, str(exc)) from exc
        return await finish(game, x_player_token)


@app.post("/api/games/{game_id}/ai")
async def ai_move(game_id: str, x_player_token: Token):
    game = get_game(game_id, x_player_token)
    async with locks[game_id]:
        if game.config.mode not in ("ai", "spectate"):
            raise HTTPException(400, "Chế độ này không dùng AI.")
        if game.config.mode == "ai" and game.current_player == game.config.human_player:
            raise HTTPException(400, "Đang là lượt của bạn.")
        game.timeout()
        if game.status != "playing" or game.paused_at is not None:
            return await finish(game, x_player_token)
        analysis = await run_in_threadpool(engine.choose, game.board, game.current_player,
                                            game.config.win_length, game.config.rule, game.config.difficulty, game.config.ai_variant)
        game.timeout()
        if game.status == "playing" and game.paused_at is None:
            analysis["revision"] = game.revision
            game.place(analysis["row"], analysis["col"])
            game.analysis = analysis
        return await finish(game, x_player_token)


@app.post("/api/games/{game_id}/hint")
async def hint(game_id: str, x_player_token: Token):
    game = get_game(game_id, x_player_token)
    async with locks[game_id]:
        if game.timeout():
            await finish(game, x_player_token)
        if game.config.mode in ("online", "spectate") or game.status != "playing":
            raise HTTPException(400, "Gợi ý chỉ dùng khi đang đấu AI hoặc chơi hai người tại máy.")
        analysis = await run_in_threadpool(engine.choose, game.board, game.current_player,
                                            game.config.win_length, game.config.rule, "hard")
        return {**analysis, "revision": game.revision}


@app.post("/api/games/{game_id}/undo")
async def undo(game_id: str, x_player_token: Token):
    game = get_game(game_id, x_player_token)
    async with locks[game_id]:
        try:
            game.undo()
        except ValueError as exc:
            if game.status in ("won", "draw") and not game.saved:
                await finish(game, x_player_token)
            raise HTTPException(400, str(exc)) from exc
        return await finish(game, x_player_token)


class PauseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paused: bool
    revision: int = Field(ge=0)


@app.post("/api/games/{game_id}/pause")
async def pause_game(game_id: str, body: PauseRequest, x_player_token: Token):
    game = get_game(game_id, x_player_token)
    async with locks[game_id]:
        if body.revision != game.revision:
            raise HTTPException(409, "Bàn cờ đã thay đổi. Hãy thử lại.")
        try:
            game.set_paused(body.paused)
        except ValueError as exc:
            if game.status in ("won", "draw") and not game.saved:
                await finish(game, x_player_token)
            raise HTTPException(400, str(exc)) from exc
        return await finish(game, x_player_token)


@app.post("/api/games/{game_id}/resign")
async def resign(game_id: str, x_player_token: Token):
    game = get_game(game_id, x_player_token)
    async with locks[game_id]:
        if game.timeout():
            return await finish(game, x_player_token)
        if game.status != "playing":
            raise HTTPException(400, "Ván cờ đã kết thúc.")
        loser = game.current_player if game.config.mode == "local" else game.player_for(x_player_token)
        game.status, game.winner, game.ended_reason = "won", -loser, "resign"
        game.ended_at = time()
        game.revision += 1
        return await finish(game, x_player_token)


class HistoryRequest(BaseModel):
    ids: list[str] = Field(max_length=100)


@app.post("/api/history")
def history(body: HistoryRequest):
    return storage.history(body.ids)


def completed_game(game_id):
    rows = storage.history([game_id])
    if not rows:
        raise HTTPException(404, "Không tìm thấy ván đã hoàn thành.")
    # Same capability as /api/history: the caller holds an unpredictable game ID.
    return rows[0]


@app.post("/api/history/{game_id}/review")
async def review_history(game_id: str):
    return await run_in_threadpool(review_game, completed_game(game_id))


class TrialMove(BaseModel):
    model_config = ConfigDict(extra="forbid")
    row: int = Field(ge=0, le=24)
    col: int = Field(ge=0, le=24)


class VariationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: int = Field(ge=0, le=625)
    moves: list[TrialMove] = Field(max_length=625)
    reply: bool = True


@app.post("/api/history/{game_id}/variation")
async def try_variation(game_id: str, body: VariationRequest):
    try:
        game = replay_prefix(completed_game(game_id), body.step)
        for move in body.moves:
            game.place(move.row, move.col)
        if body.reply and game.status == "playing" and body.moves:
            analysis = await run_in_threadpool(engine.choose, game.board, game.current_player,
                                              game.config.win_length, game.config.rule, "hard", "hybrid", 150)
            game.place(analysis["row"], analysis["col"])
            game.analysis = analysis
        return game.view(game.host_token)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pairs: int = Field(default=1, ge=1, le=3)
    sizes: list[int] = Field(default=[5,9,15], min_length=1, max_length=3)
    rules: list[str] = Field(default=["freestyle"], min_length=1, max_length=3)
    variants: list[str] = Field(default=["cnn","heuristic","hybrid"], min_length=2, max_length=3)
    budget_ms: int = Field(default=80, ge=20, le=150)
    seed: int = Field(default=2026, ge=0, le=2**32-1)

    @model_validator(mode="after")
    def validate_choices(self):
        if (any(n < 3 or n > 25 for n in self.sizes) or
                not set(self.rules) <= {"freestyle","exact","blocked"} or
                not set(self.variants) <= {"cnn","heuristic","hybrid"} or
                any(len(v) != len(set(v)) for v in (self.sizes,self.rules,self.variants))):
            raise ValueError("Kích thước/luật/engine không hợp lệ hoặc bị trùng.")
        return self


@app.get("/api/experiments")
def experiment_state():
    state = experiments.snapshot()
    state["stale"] = bool(state["report"] and state["report"].get("checkpoint_sha256") != engine.checkpoint_sha256)
    return state


@app.post("/api/experiments", status_code=202)
async def start_experiment(body: ExperimentRequest):
    if "cnn" in body.variants and not engine.loaded:
        raise HTTPException(400, "Cần checkpoint CNN để chạy so sánh này.")
    try:
        experiments.prepare(body.model_dump())
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    task = asyncio.create_task(run_in_threadpool(experiments.run, engine, body.model_dump()))
    experiment_tasks.add(task)
    task.add_done_callback(experiment_tasks.discard)
    return experiments.snapshot()


@app.websocket("/ws/games/{game_id}")
async def room_socket(ws: WebSocket, game_id: str):
    await ws.accept()
    try:
        auth = await asyncio.wait_for(ws.receive_json(), timeout=10)
        if not isinstance(auth, dict) or not isinstance(auth.get("token"), str):
            raise ValueError("Invalid authentication frame")
        game = get_game(game_id, auth["token"])
        if game.config.mode != "online":
            raise ValueError("Not an online room")
    except (HTTPException, asyncio.TimeoutError, ValueError, WebSocketDisconnect):
        await ws.close(code=1008)
        return
    ws.state.token = auth["token"]
    peers.setdefault(game_id, set()).add(ws)
    try:
        await ws.send_json({"type": "state", "game": game.view(ws.state.token)})
        while True:
            if await ws.receive_text() == "ping":
                await ws.send_json({"type": "pong"})
    except (WebSocketDisconnect, RuntimeError, OSError):
        pass
    finally:
        peers.get(game_id, set()).discard(ws)
