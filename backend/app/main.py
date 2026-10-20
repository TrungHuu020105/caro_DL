import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from time import time
from typing import Annotated
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from .ai import AIEngine
from .game import Game, GameConfig
from .storage import Storage

games: dict[str, Game] = {}
locks: dict[str, asyncio.Lock] = {}
engine: AIEngine | None = None
storage: Storage | None = None


async def clock_loop():
    while True:
        await asyncio.sleep(.5)
        for game_id, game in list(games.items()):
            async with locks[game_id]:
                if game.timeout():
                    storage.save(game)
            if time() - game.created_at > 12 * 3600:
                games.pop(game_id, None)
                locks.pop(game_id, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, storage
    engine = AIEngine()
    storage = Storage(Path(__file__).resolve().parents[1] / "data" / "caro.sqlite3")
    task = asyncio.create_task(clock_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


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
    game = Game(config)
    games[game.id] = game
    locks[game.id] = asyncio.Lock()
    return {"game": game.view(game.host_token), "token": game.host_token}


@app.get("/api/games/{game_id}")
async def game_state(game_id: str, x_player_token: Token):
    game = get_game(game_id, x_player_token)
    async with locks[game_id]:
        game.timeout()
        return await finish(game, x_player_token)


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
        if game.status != "playing":
            return await finish(game, x_player_token)
        analysis = await run_in_threadpool(engine.choose, game.board, game.current_player,
                                            game.config.win_length, game.config.rule, game.config.difficulty)
        game.timeout()
        if game.status == "playing":
            game.place(analysis["row"], analysis["col"])
            game.analysis = analysis
        return await finish(game, x_player_token)


@app.post("/api/games/{game_id}/hint")
async def hint(game_id: str, x_player_token: Token):
    game = get_game(game_id, x_player_token)
    async with locks[game_id]:
        if game.config.mode == "spectate" or game.status != "playing":
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
            raise HTTPException(400, str(exc)) from exc
        return await finish(game, x_player_token)


@app.post("/api/games/{game_id}/resign")
async def resign(game_id: str, x_player_token: Token):
    game = get_game(game_id, x_player_token)
    async with locks[game_id]:
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
