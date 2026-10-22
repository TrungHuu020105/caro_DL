import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from backend.app.ai import AIEngine
from backend.app.game import Game, GameConfig
from backend.app.main import app
from backend.app.network import RULES, CaroNet, encode
from backend.app.review import review_game
from backend.app.rules import safe_defenses, winning_line, winning_moves
from backend.app.storage import Storage
from backend.benchmark import ExperimentRunner, run_benchmark


class ImprovedAITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.engine = AIEngine()

    def test_checkpoint_metadata_mismatch_hides_stale_training_metrics(self):
        from backend.app.ai import MODEL_DIR
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"caro_policy.pt").write_bytes((MODEL_DIR/"caro_policy.pt").read_bytes())
            (root/"training.json").write_text(json.dumps({"checkpoint_sha256":"incorrect","samples":999}),encoding="utf-8")
            engine = AIEngine(root)
            self.assertTrue(engine.loaded)
            self.assertEqual(engine.info()["training"],{})
            self.assertTrue(engine.info()["load_error"])
            self.assertEqual(engine.checkpoint_sha256,self.engine.checkpoint_sha256)

    def test_rule_channels_and_padding_mask(self):
        import torch
        board = np.zeros((3,3),dtype=np.int8); board[0,0] = 1
        encoded = [encode(board,-1,3,rule,9) for rule in RULES]
        self.assertFalse(np.array_equal(encoded[0],encoded[1]))
        self.assertEqual(encoded[0].shape,(8,9,9))
        self.assertEqual(int(encoded[0][4].sum()),9)
        with torch.inference_mode():
            logits,value = CaroNet().eval()(torch.from_numpy(encoded[0]).unsqueeze(0))
        self.assertLess(float(logits[0,0]),-1e8)
        self.assertLess(float(logits[0,80]),-1e8)
        self.assertTrue(-1 <= float(value[0]) <= 1)

    def test_vectorized_threats_match_exact_bruteforce(self):
        rng = np.random.default_rng(21)
        for n,k in ((3,3),(5,3),(9,5)):
            for rule in RULES:
                board = rng.choice(np.array([0,1,-1],dtype=np.int8),size=(n,n),p=[.5,.25,.25])
                for player in (1,-1):
                    expected = set()
                    for r,c in np.argwhere(board==0):
                        cells = board.tolist(); cells[r][c] = player
                        if winning_line(cells,int(r),int(c),k,rule): expected.add((int(r),int(c)))
                    self.assertEqual(set(winning_moves(board,player,k,rule)),expected)

    def test_cnn_and_heuristic_are_independent_ablations(self):
        board = [[1,1,0],[-1,-1,0],[0,0,0]]
        probs = np.zeros(9,dtype=np.float32); probs[8] = 1
        with patch.object(self.engine,"predict",return_value=(probs,.3)):
            result = self.engine.choose(board,1,3,difficulty="hard",variant="cnn")
        self.assertEqual((result["row"],result["col"]),(2,2))
        with patch.object(self.engine,"predict",side_effect=AssertionError("heuristic must not call CNN")):
            result = self.engine.choose(board,1,3,difficulty="hard",variant="heuristic")
        self.assertEqual((result["row"],result["col"]),(0,2))

    def test_search_creates_a_double_threat_without_mutating_input(self):
        board = np.zeros((5,5),dtype=np.int8)
        for r,c in [(1,2),(2,1),(2,3),(3,2)]: board[r,c] = 1
        for r,c in [(0,0),(0,4),(4,0),(4,4)]: board[r,c] = -1
        original = board.copy()
        result = self.engine.choose(board,1,4,difficulty="hard",budget_ms=150)
        self.assertEqual((result["row"],result["col"]),(2,2))
        np.testing.assert_array_equal(board,original)
        self.assertTrue(result["value_used"])
        self.assertGreater(result["search"]["nodes"],0)
        self.assertTrue(result["candidates"][0]["selected"])

    def test_learned_value_is_consumed_at_search_leaves(self):
        board = np.zeros((9,9),dtype=np.int8)
        board[0,0]=board[0,8]=1; board[8,0]=board[8,8]=-1
        desired,other = (4,4),(4,5)
        def prediction(cells,side,*args):
            probs = (cells.ravel()==0).astype(np.float32); probs /= probs.sum()
            outcome = .9 if cells[desired] == 1 else -.9
            return probs,outcome*side
        with patch.object(self.engine,"predict",side_effect=prediction) as predict:
            chosen,scores,stats = self.engine.search(board,1,5,"freestyle",[other,desired],100)
        self.assertEqual(chosen,desired)
        self.assertGreater(predict.call_count,0)
        self.assertGreater(stats["nodes"],0)

    def test_blocked_rule_shared_endpoint_can_stop_two_threats(self):
        game = Game(GameConfig(size=9,win_length=3,rule="blocked",mode="local"))
        for rc in [(0,0),(5,2),(5,1),(5,3),(1,5),(2,5),(4,4),(3,5)]: game.place(*rc)
        board = np.asarray(game.board,dtype=np.int8)
        threats = winning_moves(board,-1,3,"blocked")
        self.assertEqual(set(threats),{(5,4),(4,5)})
        self.assertEqual(safe_defenses(board,1,3,"blocked",threats),[(5,5)])
        result = self.engine.choose(board,1,3,"blocked","hard")
        self.assertEqual((result["row"],result["col"]),(5,5))
        game.place(5,5)
        report = review_game(game.view(game.host_token))
        self.assertFalse(any(event["number"]==9 for event in report["events"]))


class PersistenceAndReviewTests(unittest.TestCase):
    def test_pause_preserves_remaining_time_including_restart(self):
        game = Game(GameConfig(mode="spectate",time_limit=15))
        initial = game.turn_started_at
        with patch("backend.app.game.time",return_value=initial+4): game.set_paused(True)
        with tempfile.TemporaryDirectory() as folder:
            store = Storage(Path(folder)/"game.sqlite3"); store.save(game)
            restored = store.load_active()[0]
        with patch("backend.app.game.time",return_value=initial+100):
            self.assertFalse(restored.timeout())
            with self.assertRaises(ValueError): restored.place(0,0)
            restored.set_paused(False)
        self.assertAlmostEqual(restored.paused_seconds,96)
        with patch("backend.app.game.time",return_value=initial+110): self.assertFalse(restored.timeout())
        with patch("backend.app.game.time",return_value=initial+111): self.assertTrue(restored.timeout())

    def test_api_restart_restores_board_room_code_and_authorization(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{"CARO_DATABASE_PATH":str(Path(folder)/"games.sqlite3"),"CARO_BENCHMARK_PATH":str(Path(folder)/"benchmark.json")}):
            with TestClient(app) as client:
                data = client.post("/api/games",json={"mode":"local","size":3}).json()
                url = f'/api/games/{data["game"]["id"]}'
                headers = {"X-Player-Token":data["token"]}
                self.assertEqual(client.post(url+"/move",headers=headers,json={"row":0,"col":0,"revision":0}).status_code,200)
                room = client.post("/api/games",json={"mode":"online","size":5}).json()
            with TestClient(app) as client:
                response = client.get(url,headers=headers)
                self.assertEqual(response.status_code,200)
                self.assertEqual(response.json()["board"][0][0],1)
                self.assertEqual(client.get(url,headers={"X-Player-Token":"bad"}).status_code,403)
                joined = client.post("/api/rooms/join",json={"code":room["game"]["room_code"]})
                self.assertEqual(joined.status_code,200)
                self.assertEqual(joined.json()["game"]["id"],room["game"]["id"])

    def test_api_pause_stale_revision_and_expired_undo(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{"CARO_DATABASE_PATH":str(Path(folder)/"games.sqlite3"),"CARO_BENCHMARK_PATH":str(Path(folder)/"benchmark.json")}):
            with TestClient(app) as client:
                data = client.post("/api/games",json={"mode":"spectate","size":3,"time_limit":15}).json()
                url,headers = f'/api/games/{data["game"]["id"]}',{"X-Player-Token":data["token"]}
                paused = client.post(url+"/pause",headers=headers,json={"paused":True,"revision":0})
                self.assertEqual(paused.status_code,200)
                self.assertIsNotNone(paused.json()["paused_at"])
                self.assertEqual(client.post(url+"/ai",headers=headers).json()["moves"],[])
                self.assertEqual(client.post(url+"/pause",headers=headers,json={"paused":False,"revision":0}).status_code,409)
                self.assertIsNone(client.post(url+"/pause",headers=headers,json={"paused":False,"revision":1}).json()["paused_at"])
                data = client.post("/api/games",json={"mode":"local","size":3,"time_limit":15}).json()
                url,headers = f'/api/games/{data["game"]["id"]}',{"X-Player-Token":data["token"]}
                client.post(url+"/move",headers=headers,json={"row":0,"col":0,"revision":0})
                import backend.app.main as main
                main.games[data["game"]["id"]].turn_started_at -= 30
                self.assertEqual(client.post(url+"/undo",headers=headers).status_code,400)
                self.assertEqual(client.get(url,headers=headers).json()["ended_reason"],"timeout")
                self.assertEqual(len(client.post("/api/history",json={"ids":[data["game"]["id"]]}).json()),1)

    def test_review_finds_missed_win_and_block_and_sandbox_preserves_history(self):
        game = Game(GameConfig(size=3,mode="local"))
        for rc in [(0,0),(1,0),(0,1),(1,1),(2,2),(1,2)]: game.place(*rc)
        payload = game.view(game.host_token)
        review = review_game(payload)
        self.assertEqual(review["summary"]["1"]["missed_win"],1)
        self.assertEqual(review["summary"]["-1"]["missed_block"],1)
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{"CARO_DATABASE_PATH":str(Path(folder)/"games.sqlite3"),"CARO_BENCHMARK_PATH":str(Path(folder)/"benchmark.json")}):
            with TestClient(app) as client:
                import backend.app.main as main
                main.storage.save(game)
                self.assertEqual(client.post(f"/api/history/{game.id}/review").json()["summary"],review["summary"])
                trial = client.post(f"/api/history/{game.id}/variation",json={"step":4,"moves":[{"row":0,"col":2}],"reply":True})
                self.assertEqual(trial.status_code,200,trial.text)
                self.assertEqual(trial.json()["winner"],1)
                self.assertEqual(main.storage.history([game.id])[0],payload)
                bad = client.post(f"/api/history/{game.id}/variation",json={"step":4,"moves":[{"row":0,"col":0}]})
                self.assertEqual(bad.status_code,400)


class BenchmarkTests(unittest.TestCase):
    def test_tournament_has_paired_colors_valid_games_and_real_counts(self):
        report = run_benchmark(AIEngine(),pairs=1,sizes=(3,),rules=("freestyle",),variants=("cnn","heuristic"),budget_ms=20,seed=8)
        self.assertEqual(report["total_games"],2)
        a,b = report["matches"]
        self.assertEqual(a["opening"],b["opening"])
        self.assertEqual(a["x_engine"],b["o_engine"])
        for match in report["matches"]:
            game = Game(GameConfig(size=3,mode="local"))
            for rc in match["moves"]: game.place(*rc)
            self.assertIn(game.status,("won","draw"))
            self.assertEqual(game.winner or 0,match["winner"])
        for row in report["summary"]:
            self.assertEqual(row["wins"]+row["draws"]+row["losses"],2)
            self.assertGreaterEqual(row["p95_ms"],0)

    def test_only_one_experiment_runs_at_a_time(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = ExperimentRunner(Path(folder)/"benchmark.json")
            config = {"pairs":1,"sizes":[3],"rules":["freestyle"],"variants":["cnn","heuristic"],"budget_ms":20,"seed":1}
            runner.prepare(config)
            with self.assertRaises(ValueError): runner.prepare(config)
            runner.run(AIEngine(),config)
            state = runner.snapshot()
            self.assertEqual(state["status"],"completed",state["error"])
            self.assertEqual(state["completed"],state["total"])
            self.assertEqual(json.loads(runner.path.read_text(encoding="utf-8"))["total_games"],2)


if __name__ == "__main__": unittest.main()
