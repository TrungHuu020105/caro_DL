import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from fastapi.testclient import TestClient
from backend.app.ai import AIEngine
from backend.app.experience import human_experience
from backend.app.game import Game,GameConfig
from backend.app.main import app
from backend.app.network import CaroNet
from backend.app.storage import Storage
from backend.dataset import export_dataset,file_sha256,generate_dataset,position_key,rollout,write_json
from backend.learning import HumanLearning,compare_models,promotion_decision,publish_candidate
from backend.learning_data import build_human_data
from backend.model_store import ModelBusyError,model_writer


def played(moves,size=3,k=3,mode="local",rule="freestyle"):
    game = Game(GameConfig(size=size,win_length=k,mode=mode,rule=rule))
    if mode == "online": game.status="playing"
    for rc in moves: game.place(*rc)
    return game


def seed_games(storage,count=12):
    rng = np.random.default_rng(71)
    while storage.experience_stats()["total_games"] < count:
        _,_,moves = rollout(rng,5,4,"freestyle")
        storage.save(played(moves,5,4))


def arena(score=0.,games=16):
    wins = int(score*games)
    summary = {"games":games,"wins":wins,"draws":0,"losses":games-wins,"score_rate":score}
    return {**summary,"by_variant":{"cnn":dict(summary),"hybrid":dict(summary)},"matches":[]}


class ExperienceTests(unittest.TestCase):
    def test_collects_human_modes_and_excludes_ai_only_timeout_and_resign(self):
        moves=[(0,0),(1,0),(0,1),(1,1),(0,2)]
        for mode in ("ai","local","online"):
            game=played(moves,mode=mode)
            record=human_experience(game.view(game.host_token))
            self.assertIsNotNone(record)
            self.assertNotIn("player_name",str(record))
            self.assertNotIn("token",str(record))
        game=played(moves,mode="spectate")
        self.assertIsNone(human_experience(game.view(game.host_token)))
        game=played(moves); payload=game.view(game.host_token)
        for reason in ("timeout","resign"):
            payload["ended_reason"]=reason
            self.assertIsNone(human_experience(payload))
        payload=game.view(game.host_token); payload["winner"]=-1
        self.assertIsNone(human_experience(payload))
        payload=game.view(game.host_token); payload["moves"]=payload["moves"]+[payload["moves"][0]]
        self.assertIsNone(human_experience(payload))

    def test_backfill_deduplicates_symmetric_games_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            storage=Storage(Path(folder)/"games.sqlite3")
            moves=[(0,0),(1,0),(0,1),(1,1),(0,2)]
            game=played(moves)
            storage.save(game); storage.save(game)
            rotated=[(2-c,r) for r,c in moves]
            storage.save(played(rotated))
            storage.backfill_experience()
            self.assertEqual(storage.experience_stats()["total_games"],1)
            reopened=Storage(storage.path)
            self.assertEqual(reopened.experience_stats()["total_games"],1)
            self.assertGreater(len(reopened.experience_position_splits()),0)

    def test_policy_is_reanalyzed_instead_of_copying_a_losing_human_move(self):
        with tempfile.TemporaryDirectory() as folder:
            storage=Storage(Path(folder)/"games.sqlite3")
            game=played([(0,0),(1,0),(0,1),(1,1),(2,2),(1,2)])
            storage.save(game)
            train,val,info=build_human_data(storage.experience_snapshot(),storage.experience_position_splits(),(set(),set()))
            split=train if train is not None else val
            found=False
            for i,board in enumerate(split.boards):
                if int((board!=0).sum())==4:
                    found=True
                    self.assertEqual(split.policy[i].reshape(25,25)[0,2],1)
                    self.assertEqual(split.policy[i].reshape(25,25)[2,2],0)
                    self.assertEqual(split.value[i],-1)
            self.assertTrue(found)

    def test_new_validation_game_cannot_reuse_previously_assigned_training_positions(self):
        with tempfile.TemporaryDirectory() as folder:
            storage=Storage(Path(folder)/"games.sqlite3")
            seed_games(storage,30)
            records=storage.experience_snapshot()
            ledger=storage.experience_position_splits()
            train,val,info=build_human_data(records,ledger,(set(),set()),max_train=2000,max_validation=2000)
            self.assertIsNotNone(train); self.assertIsNotNone(val)
            train_keys=set()
            for i in range(len(train.inputs)):
                n,k,p=map(int,(train.sizes[i],train.win_lengths[i],train.players[i]))
                key=position_key(train.boards[i,:n,:n],p,k,"freestyle").hex()
                self.assertEqual(ledger[key],"train"); train_keys.add(key)
            for i in range(len(val.inputs)):
                n,k,p=map(int,(val.sizes[i],val.win_lengths[i],val.players[i]))
                key=position_key(val.boards[i,:n,:n],p,k,"freestyle").hex()
                self.assertEqual(ledger[key],"validation"); self.assertNotIn(key,train_keys)
            self.assertGreater(info["ignored_split_conflicts"],0)

    def test_original_validation_and_test_positions_are_excluded_from_human_training(self):
        with tempfile.TemporaryDirectory() as folder:
            storage=Storage(Path(folder)/"games.sqlite3"); seed_games(storage,6)
            records,ledger=storage.experience_snapshot(),storage.experience_position_splits()
            blocked=set(ledger)
            train,val,info=build_human_data(records,ledger,(blocked,set()))
            self.assertIsNone(train); self.assertIsNone(val)
            self.assertGreater(info["ignored_base_positions"],0)


class GateTests(unittest.TestCase):
    def test_gate_rejects_regressions_ties_and_variant_losses(self):
        base={"loss":3.,"policy_loss":2.5,"value_mse":1.}
        self.assertTrue(promotion_decision(base,base,None,None,arena(.75))["accepted"])
        self.assertFalse(promotion_decision(base,base,None,None,arena(.5))["accepted"])
        worse={**base,"policy_loss":2.7}
        self.assertFalse(promotion_decision(base,worse,None,None,arena(.75))["accepted"])
        score=arena(.75); score["by_variant"]["hybrid"]["score_rate"]=.25
        self.assertFalse(promotion_decision(base,base,None,None,score)["accepted"])
        self.assertFalse(promotion_decision(base,base,{"loss":1.},{"loss":1.2},arena(.75))["accepted"])

    def test_real_arena_uses_paired_colors_and_equal_checkpoints_do_not_improve(self):
        engine=AIEngine()
        result=compare_models(engine,engine,[(3,3,"freestyle")],pairs=1,budget_ms=10)
        self.assertEqual(result["games"],4)
        self.assertEqual(result["score_rate"],.5)
        for i in (0,2):
            a,b=result["matches"][i:i+2]
            self.assertEqual(a["opening"],b["opening"])
            self.assertEqual(a["candidate_side"],-b["candidate_side"])

    def test_model_writer_excludes_a_second_writer(self):
        with tempfile.TemporaryDirectory() as folder:
            with model_writer(folder):
                with self.assertRaises(ModelBusyError):
                    with model_writer(folder): pass


class LearningPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle=generate_dataset(100,19,sizes=(3,5,9))

    def workspace(self,root):
        dataset=export_dataset(self.bundle,root/"data")
        models=root/"models"; models.mkdir()
        torch.manual_seed(8); model=CaroNet()
        torch.save(model.state_dict(),models/"caro_policy.pt")
        write_json(models/"training.json",{"model_version":2,"samples":100,"validation_samples":100,"test_samples":100,
            "checkpoint_sha256":file_sha256(models/"caro_policy.pt"),"dataset":{"file_sha256":file_sha256(dataset),"fingerprint_sha256":self.bundle.metadata.get("fingerprint_sha256")},"history":[]})
        storage=Storage(root/"games.sqlite3"); seed_games(storage)
        return storage,models,dataset

    def test_rejected_candidate_really_trains_but_preserves_active_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); storage,models,dataset=self.workspace(root)
            old=(models/"caro_policy.pt").read_bytes(); meta=(models/"training.json").read_bytes()
            learner=HumanLearning(storage,models,dataset)
            learner.prepare(manual=True,epochs=1)
            with patch("backend.learning.compare_models",return_value=arena(0.)):
                learner.run()
            state=learner.snapshot(); report=state["report"]
            self.assertEqual(state["status"],"completed",state["error"])
            self.assertEqual(report["decision"],"rejected")
            self.assertNotEqual(report["candidate_sha256"],report["parent_sha256"])
            self.assertEqual((models/"caro_policy.pt").read_bytes(),old)
            self.assertEqual((models/"training.json").read_bytes(),meta)
            self.assertTrue((learner.directory/"runs"/report["run_id"]/"experience.npz").exists())
            self.assertEqual(storage.experience_stats()["pending_games"],0)
            self.assertGreaterEqual(report["data"]["selected_train_positions"],8)

    def test_passing_gate_publishes_backup_and_hot_reload_callback(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); storage,models,dataset=self.workspace(root)
            old_hash=file_sha256(models/"caro_policy.pt")
            learner=HumanLearning(storage,models,dataset); learner.prepare(manual=True,epochs=1)
            activated=[]
            with patch("backend.learning.compare_models",return_value=arena(.75)),patch("backend.learning.promotion_decision",return_value={"accepted":True,"reasons":[]}):
                learner.run(on_promote=activated.append)
            state=learner.snapshot(); report=state["report"]
            self.assertEqual(report["decision"],"promoted",state["error"])
            self.assertEqual(len(activated),1)
            current=AIEngine(models)
            self.assertEqual(current.checkpoint_sha256,report["candidate_sha256"])
            self.assertEqual(current.metadata["parent_checkpoint_sha256"],old_hash)
            self.assertEqual(file_sha256(models/"learning"/"versions"/old_hash/"caro_policy.pt"),old_hash)
            self.assertIn("test_metrics",current.metadata)
            self.assertEqual(HumanLearning(storage,models,dataset).snapshot()["stats"]["pending_games"],0)

    def test_publish_io_failure_restores_weights_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); storage,models,dataset=self.workspace(root)
            parent=file_sha256(models/"caro_policy.pt"); before=(models/"training.json").read_bytes()
            candidate=root/"candidate"; candidate.mkdir(); torch.save(CaroNet().state_dict(),candidate/"caro_policy.pt")
            with model_writer(models),patch("backend.learning.write_json",side_effect=OSError("disk unavailable")):
                with self.assertRaises(OSError): publish_candidate(models,candidate,{"checkpoint_sha256":file_sha256(candidate/"caro_policy.pt")},parent)
            self.assertEqual(file_sha256(models/"caro_policy.pt"),parent)
            self.assertEqual((models/"training.json").read_bytes(),before)

    def test_changed_base_dataset_cannot_replace_the_active_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); storage,models,dataset=self.workspace(root)
            parent=file_sha256(models/"caro_policy.pt")
            dataset.write_bytes(b"changed dataset")
            learner=HumanLearning(storage,models,dataset); learner.prepare(manual=True,epochs=1); learner.run()
            self.assertEqual(learner.snapshot()["status"],"failed")
            self.assertEqual(file_sha256(models/"caro_policy.pt"),parent)
            self.assertEqual(storage.experience_stats()["pending_games"],12)

    def test_automatic_threshold_settings_and_interrupted_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); storage,models,dataset=self.workspace(root)
            learner=HumanLearning(storage,models,dataset)
            self.assertTrue(learner.ready())
            learner.configure({"enabled":False}); self.assertFalse(learner.ready())
            learner.configure({"enabled":True}); learner.prepare()
            with self.assertRaises(ValueError): learner.prepare(manual=True)
            resumed=HumanLearning(storage,models,dataset)
            self.assertEqual(resumed.snapshot()["phase"],"interrupted")
            self.assertTrue(resumed.ready())

    def test_learning_api_runs_and_serializes_with_benchmark(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); storage,models,dataset=self.workspace(root)
            # Avoid a startup automatic job; this test explicitly starts one.
            learner=HumanLearning(storage,models,dataset); learner.configure({"enabled":False})
            env={"CARO_DATABASE_PATH":str(storage.path),"CARO_MODEL_DIR":str(models),"CARO_DATASET_PATH":str(dataset),
                 "CARO_LEARNING_DIR":str(learner.directory),"CARO_BENCHMARK_PATH":str(root/"benchmark.json")}
            with patch.dict(os.environ,env),patch("backend.learning.compare_models",return_value=arena(0.)),TestClient(app) as client:
                self.assertGreater(client.get("/api/learning").json()["stats"]["total_games"],0)
                self.assertEqual(client.post("/api/learning/settings",json={"epochs":100}).status_code,422)
                self.assertEqual(client.post("/api/learning/run",json={"epochs":1}).status_code,202)
                self.assertEqual(client.post("/api/experiments",json={"sizes":[3]}).status_code,409)
                deadline=time.monotonic()+20
                while time.monotonic() < deadline:
                    state=client.get("/api/learning").json()
                    if state["status"] != "running": break
                    time.sleep(.05)
                self.assertEqual(state["report"]["decision"],"rejected",state.get("error"))
                self.assertEqual(client.post("/api/learning/run",json={}).status_code,409)

    def test_finishing_new_games_schedules_automatic_learning(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); _,models,dataset=self.workspace(root)
            storage=Storage(root/"automatic.sqlite3")
            started=threading.Event()
            env={"CARO_DATABASE_PATH":str(storage.path),"CARO_MODEL_DIR":str(models),"CARO_DATASET_PATH":str(dataset),
                 "CARO_LEARNING_DIR":str(root/"automatic-learning"),"CARO_BENCHMARK_PATH":str(root/"benchmark.json")}
            with patch.dict(os.environ,env),patch.object(HumanLearning,"run",side_effect=lambda *args:started.set()) as run,TestClient(app) as client:
                self.assertEqual(client.post("/api/learning/settings",json={"enabled":True,"min_new_games":3}).status_code,200)
                rng=np.random.default_rng(71)
                for _ in range(3):
                    _,_,moves=rollout(rng,5,4,"freestyle")
                    created=client.post("/api/games",json={"mode":"local","size":5,"win_length":4}).json()
                    headers={"X-Player-Token":created["token"]}
                    for revision,(r,c) in enumerate(moves):
                        response=client.post(f'/api/games/{created["game"]["id"]}/move',headers=headers,json={"row":r,"col":c,"revision":revision})
                        self.assertEqual(response.status_code,200,response.text)
                self.assertTrue(started.wait(2))
                self.assertEqual(run.call_count,1)
                self.assertEqual(client.get("/api/learning").json()["stats"]["total_games"],3)

    def test_api_activates_accepted_model_without_restart_or_losing_current_game(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); storage,models,dataset=self.workspace(root)
            learner=HumanLearning(storage,models,dataset); learner.configure({"enabled":False})
            env={"CARO_DATABASE_PATH":str(storage.path),"CARO_MODEL_DIR":str(models),"CARO_DATASET_PATH":str(dataset),
                 "CARO_LEARNING_DIR":str(learner.directory),"CARO_BENCHMARK_PATH":str(root/"benchmark.json")}
            with patch.dict(os.environ,env),patch("backend.learning.compare_models",return_value=arena(.75)),patch("backend.learning.promotion_decision",return_value={"accepted":True,"reasons":[]}),TestClient(app) as client:
                parent=client.get("/api/model").json()["checkpoint_sha256"]
                game=client.post("/api/games",json={"mode":"ai","size":5}).json()
                url=f'/api/games/{game["game"]["id"]}'
                headers={"X-Player-Token":game["token"]}
                client.post(url+"/move",headers=headers,json={"row":2,"col":2,"revision":0})
                self.assertEqual(client.post("/api/learning/run",json={"epochs":1}).status_code,202)
                deadline=time.monotonic()+20
                while time.monotonic() < deadline:
                    state=client.get("/api/learning").json()
                    if state["status"] != "running": break
                    time.sleep(.05)
                self.assertEqual(state["report"]["decision"],"promoted",state.get("error"))
                model=client.get("/api/model").json()
                self.assertNotEqual(model["checkpoint_sha256"],parent)
                self.assertEqual(model["checkpoint_sha256"],state["report"]["candidate_sha256"])
                self.assertEqual(client.get("/api/health").json()["checkpoint_sha256"],model["checkpoint_sha256"])
                self.assertEqual(client.get(url,headers=headers).json()["board"][2][2],1)
                response=client.post(url+"/ai",headers=headers)
                self.assertEqual(response.status_code,200)
                self.assertEqual(len(response.json()["moves"]),2)


if __name__ == "__main__": unittest.main()
