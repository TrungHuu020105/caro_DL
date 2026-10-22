import itertools
from pathlib import Path
import tempfile
import time
import unittest
import os
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from backend.app.ai import AIEngine
from backend.app.game import Game, GameConfig
from backend.app.rules import winning_line
from backend.app.storage import Storage
from backend.app.main import app


class RulesTests(unittest.TestCase):
    def test_waiting_room_does_not_run_the_clock(self):
        game=Game(GameConfig(mode="online",time_limit=15))
        game.turn_started_at=time.time()-60
        self.assertFalse(game.timeout())
        self.assertEqual(game.status,"waiting")
        self.assertIsNone(game.started_at)

    def test_all_board_sizes_and_directions(self):
        for size, (dr, dc) in itertools.product(range(3, 26), [(0,1),(1,0),(1,1),(1,-1)]):
            k = min(size, 5)
            board = [[0]*size for _ in range(size)]
            row, col = 0, k-1 if dc < 0 else 0
            for i in range(k):
                board[row+i*dr][col+i*dc] = 1
            with self.subTest(size=size, direction=(dr,dc)):
                self.assertEqual(len(winning_line(board, row, col, k)), k)

    def test_exact_and_blocked_rules(self):
        board = [[0]*8 for _ in range(8)]
        board[3] = [-1,1,1,1,1,1,-1,0]
        self.assertTrue(winning_line(board,3,3,5,"exact"))
        self.assertFalse(winning_line(board,3,3,5,"blocked"))
        board[3][0] = 0
        self.assertTrue(winning_line(board,3,3,5,"blocked"))
        board[3][6] = 1
        self.assertTrue(winning_line(board,3,3,5,"freestyle"))
        self.assertFalse(winning_line(board,3,3,5,"exact"))

    def test_board_edge_is_not_opponent(self):
        board = [[1,1,1,-1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        self.assertEqual(len(winning_line(board,0,1,3,"blocked")),3)

    def test_draw_undo_and_timeout(self):
        game = Game(GameConfig(size=3,mode="local"))
        for rc in [(0,0),(0,1),(0,2),(1,1),(1,0),(1,2),(2,1),(2,0),(2,2)]:
            game.place(*rc)
        self.assertEqual(game.status,"draw")
        game = Game(GameConfig(size=3,mode="ai"))
        game.place(0,0); game.place(1,1); game.undo()
        self.assertEqual(len(game.moves),0)
        self.assertEqual(game.current_player,1)
        game = Game(GameConfig(size=3,time_limit=15))
        game.turn_started_at = time.time()-16
        self.assertTrue(game.timeout())
        self.assertEqual(game.winner,-1)


class AITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = AIEngine()

    def test_checkpoint_loaded_and_variable_sizes(self):
        self.assertTrue(self.engine.loaded)
        for size in [3,4,5,6,10,15,19,25]:
            board = np.zeros((size,size),dtype=np.int8)
            probs, value = self.engine.predict(board,1,min(size,5))
            self.assertEqual(probs.shape,(size*size,))
            self.assertAlmostEqual(float(probs.sum()),1,places=4)
            self.assertTrue(-1 <= value <= 1)

    def test_wins_and_blocks(self):
        for player in [1,-1]:
            board = [[0]*8 for _ in range(8)]
            board[3][0:4] = [player]*4
            result = self.engine.choose(board,1,5,difficulty="hard")
            self.assertEqual((result["row"],result["col"]),(3,4))
            self.assertTrue(all(0<=c["probability"]<=1 for c in result["candidates"]))

    def test_legal_choices_on_larger_boards(self):
        for size in [4,5,6,10,15,25]:
            for difficulty in ["easy","medium","hard"]:
                board=np.zeros((size,size),dtype=np.int8)
                board[size//2,size//2]=1
                board[size//2,size//2-1]=-1
                result=self.engine.choose(board.tolist(),1,min(size,5),difficulty=difficulty)
                with self.subTest(size=size,difficulty=difficulty):
                    self.assertEqual(board[result["row"],result["col"]],0)

    def test_hard_3x3_never_loses_exhaustive(self):
        # Explore every human reply, following the actual engine's hard moves.
        for human in (1,-1):
            def visit(board, turn):
                empty = [(r,c) for r in range(3) for c in range(3) if not board[r][c]]
                if not empty:
                    return
                choices = empty if turn == human else [self.engine.choose(board,turn,3,difficulty="hard")]
                for item in choices:
                    r,c = item if isinstance(item,tuple) else (item["row"],item["col"])
                    next_board = [row[:] for row in board]; next_board[r][c] = turn
                    if winning_line(next_board,r,c,3):
                        self.assertNotEqual(turn,human, "AI allowed a human win")
                    else:
                        visit(next_board,-turn)
            visit([[0]*3 for _ in range(3)],1)


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.environment = patch.dict(os.environ, {"CARO_DATABASE_PATH":str(Path(cls.temp.name)/"tests.sqlite3"),
                                                 "CARO_BENCHMARK_PATH":str(Path(cls.temp.name)/"benchmark.json")})
        cls.environment.start()
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None,None,None)
        cls.environment.stop()
        cls.temp.cleanup()

    def create(self, **config):
        response = self.client.post("/api/games",json={"size":3,**config})
        self.assertEqual(response.status_code,201,response.text)
        data=response.json()
        return data["game"],{"X-Player-Token":data["token"]}

    def test_health_auth_validation_and_stale_revision(self):
        self.assertEqual(self.client.get("/api/health").status_code,200)
        self.assertEqual(self.client.post("/api/games",json={"size":2}).status_code,422)
        self.assertEqual(self.client.post("/api/games",json={"size":3,"win_length":5}).status_code,422)
        game,headers=self.create()
        url=f'/api/games/{game["id"]}'
        self.assertEqual(self.client.get(url,headers={"X-Player-Token":"wrong"}).status_code,403)
        self.assertEqual(self.client.post(url+"/move",headers=headers,json={"row":0,"col":0,"revision":0}).status_code,200)
        self.assertEqual(self.client.post(url+"/move",headers=headers,json={"row":0,"col":1,"revision":0}).status_code,409)
        self.assertEqual(self.client.post(url+"/move",headers=headers,json={"row":0,"col":1,"revision":1}).status_code,403)
        self.assertEqual(self.client.post(url+"/ai",headers=headers).status_code,200)

    def test_complete_game_and_history_privacy(self):
        game,headers=self.create(mode="local")
        url=f'/api/games/{game["id"]}'
        for revision,(r,c) in enumerate([(0,0),(1,0),(0,1),(1,1),(0,2)]):
            response=self.client.post(url+"/move",headers=headers,json={"row":r,"col":c,"revision":revision})
            self.assertEqual(response.status_code,200,response.text)
        state=response.json()
        self.assertEqual(state["winner"],1)
        self.assertEqual(len(state["winning_line"]),3)
        self.assertIsNotNone(state["ended_at"])
        self.assertEqual(self.client.post("/api/history",json={"ids":[]}).json(),[])
        stored=self.client.post("/api/history",json={"ids":[game["id"]]}).json()
        self.assertEqual(stored[0]["id"],game["id"])
        self.assertNotIn("token",str(stored))

    def test_online_room_websocket_sync_and_turn_ownership(self):
        game,headers=self.create(size=6,win_length=4,mode="online",time_limit=15)
        url=f'/api/games/{game["id"]}'
        self.assertEqual(game["status"],"waiting")
        self.assertIsNone(game["started_at"])
        self.assertEqual(len(game["room_code"]),6)
        self.assertEqual(self.client.post(url+"/move",headers=headers,json={"row":0,"col":0,"revision":0}).status_code,400)
        with self.client.websocket_connect(f'/ws/games/{game["id"]}') as host_ws:
            host_ws.send_json({"token":headers["X-Player-Token"]})
            self.assertEqual(host_ws.receive_json()["game"]["status"],"waiting")
            joined=self.client.post("/api/rooms/join",json={"code":game["room_code"].lower(),"name":"Khách"})
            self.assertEqual(joined.status_code,200,joined.text)
            guest=joined.json()
            guest_headers={"X-Player-Token":guest["token"]}
            self.assertEqual(guest["game"]["config"]["win_length"],4)
            self.assertEqual(guest["game"]["your_player"],-1)
            self.assertIsNotNone(host_ws.receive_json()["game"]["started_at"])
            with self.client.websocket_connect(f'/ws/games/{game["id"]}') as guest_ws:
                guest_ws.send_json({"token":guest["token"]})
                self.assertEqual(guest_ws.receive_json()["game"]["status"],"playing")
                self.assertEqual(self.client.post(url+"/move",headers=guest_headers,json={"row":1,"col":0,"revision":1}).status_code,403)
                self.assertEqual(self.client.post(url+"/move",headers=headers,json={"row":0,"col":0,"revision":1}).status_code,200)
                self.assertEqual(host_ws.receive_json()["game"]["board"][0][0],1)
                self.assertEqual(guest_ws.receive_json()["game"]["board"][0][0],1)
                self.assertEqual(self.client.post(url+"/move",headers=guest_headers,json={"row":1,"col":0,"revision":2}).status_code,200)
                self.assertEqual(host_ws.receive_json()["game"]["board"][1][0],-1)
                self.assertEqual(guest_ws.receive_json()["game"]["board"][1][0],-1)
                self.assertEqual(self.client.post(url+"/move",headers=headers,json={"row":0,"col":1,"revision":2}).status_code,409)
                self.assertEqual(self.client.post(url+"/undo",headers=headers).status_code,400)
                self.assertEqual(self.client.post(url+"/hint",headers=guest_headers).status_code,400)
                self.assertEqual(self.client.post("/api/rooms/join",json={"code":game["room_code"]}).status_code,409)
                self.assertEqual(self.client.post(url+"/resign",headers=guest_headers).json()["winner"],1)
                self.assertEqual(host_ws.receive_json()["game"]["status"],"won")
                self.assertEqual(guest_ws.receive_json()["game"]["status"],"won")

    def test_invalid_room_and_websocket_authentication(self):
        self.assertEqual(self.client.post("/api/rooms/join",json={"code":"ABC234"}).status_code,404)
        game,headers=self.create(mode="online")
        for auth in (["invalid"], {"token":"wrong"}):
            with self.client.websocket_connect(f'/ws/games/{game["id"]}') as ws:
                ws.send_json(auth)
                with self.assertRaises(WebSocketDisconnect) as closed:
                    ws.receive_json()
                self.assertEqual(closed.exception.code,1008)

    def test_custom_winning_length_on_larger_board(self):
        game,headers=self.create(size=6,win_length=4,mode="local")
        url=f'/api/games/{game["id"]}'
        for revision,(r,c) in enumerate([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(0,3)]):
            response=self.client.post(url+"/move",headers=headers,json={"row":r,"col":c,"revision":revision})
            self.assertEqual(response.status_code,200,response.text)
            if revision < 6:
                self.assertEqual(response.json()["status"],"playing")
        self.assertEqual(response.json()["winner"],1)
        self.assertEqual(len(response.json()["winning_line"]),4)

    def test_hint_spectate_and_resignation(self):
        game,headers=self.create(mode="local")
        url=f'/api/games/{game["id"]}'
        hint=self.client.post(url+"/hint",headers=headers).json()
        self.assertIn("candidates",hint)
        self.assertEqual(self.client.get(url,headers=headers).json()["moves"],[])
        result=self.client.post(url+"/resign",headers=headers).json()
        self.assertEqual(result["ended_reason"],"resign")
        self.assertEqual(result["winner"],-1)
        game,headers=self.create(mode="spectate")
        url=f'/api/games/{game["id"]}'
        self.assertEqual(self.client.post(url+"/move",headers=headers,json={"row":0,"col":0,"revision":0}).status_code,403)
        for _ in range(3):
            self.assertEqual(self.client.post(url+"/ai",headers=headers).status_code,200)
        self.assertEqual(len(self.client.get(url,headers=headers).json()["moves"]),3)


if __name__ == "__main__":
    unittest.main()
