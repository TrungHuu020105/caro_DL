import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import fields
from unittest.mock import patch
import numpy as np
import torch
from backend import train
from backend.app.network import CaroNet, RULES, encode
from backend.app.rules import winning_line
from backend.dataset import (DatasetSplit, export_dataset, file_sha256, fingerprint,
                             generate_dataset, load_dataset, position_key)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = generate_dataset(100,seed=7,sizes=(3,5,9))

    def test_reproducibility_legal_turns_and_symmetry_disjoint_splits(self):
        repeated = generate_dataset(100,seed=7,sizes=(3,5,9))
        self.assertEqual(fingerprint(self.bundle),fingerprint(repeated))
        keys,game_ids = set(),set()
        for split in (self.bundle.train,self.bundle.validation,self.bundle.test):
            ids = set(split.game_ids.tolist())
            self.assertFalse(ids & game_ids); game_ids.update(ids)
            for i in range(len(split.inputs)):
                n,k,p = map(int,(split.sizes[i],split.win_lengths[i],split.players[i]))
                rule = RULES[int(split.rules[i])]
                board = split.boards[i,:n,:n]
                x,o = int((board==1).sum()),int((board==-1).sum())
                self.assertTrue(x==o and p==1 or x==o+1 and p==-1)
                np.testing.assert_array_equal(split.inputs[i],encode(board,p,k,rule,9))
                key = position_key(board,p,k,rule)
                self.assertNotIn(key,keys); keys.add(key)
                self.assertEqual(key,position_key(np.fliplr(np.rot90(board)),p,k,rule))
                for r,c in np.argwhere(board):
                    self.assertFalse(winning_line(board.tolist(),int(r),int(c),k,rule))
                target = split.policy[i].reshape(9,9)
                self.assertEqual(float(target[split.inputs[i,2] < .5].sum()),0.)

    def test_outcomes_and_samples_come_from_completed_legal_games(self):
        for name,split in (("train",self.bundle.train),("val",self.bundle.validation),("test",self.bundle.test)):
            for game in self.bundle.metadata["games"][name]:
                n,k,rule = game["size"],game["win_length"],game["rule"]
                board,player,winner = np.zeros((n,n),dtype=np.int8),1,0
                positions = set()
                for r,c in game["moves"]:
                    self.assertEqual(winner,0,"Rollout continued after a win")
                    positions.add(position_key(board,player,k,rule))
                    self.assertEqual(board[r,c],0)
                    board[r,c] = player
                    if winning_line(board.tolist(),r,c,k,rule): winner = player
                    player = -player
                self.assertEqual(winner,game["winner"])
                self.assertTrue(winner or not (board==0).any())
                for i in np.flatnonzero(split.game_ids==game["id"]):
                    p = int(split.players[i])
                    self.assertIn(position_key(split.boards[i,:n,:n],p,k,rule),positions)
                    self.assertEqual(split.value[i],winner*p)
        self.assertGreater(float(np.abs(self.bundle.train.value).max()),.9)

    def test_npz_csv_round_trip_and_python310_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = export_dataset(self.bundle,root)
            restored = load_dataset(path)
            self.assertEqual(fingerprint(restored),fingerprint(self.bundle))
            for name in ("train","validation","test"):
                original,loaded = getattr(self.bundle,name),getattr(restored,name)
                for field in fields(DatasetSplit):
                    np.testing.assert_array_equal(getattr(original,field.name),getattr(loaded,field.name))
                with (root/f"{name}.csv").open(encoding="utf-8-sig",newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows),len(original.inputs))
                n = int(original.sizes[0])
                self.assertEqual(json.loads(rows[0]["board_json"]),original.boards[0,:n,:n].tolist())
                np.testing.assert_array_equal(np.asarray(json.loads(rows[0]["policy_json"]),dtype=np.float32),original.policy[0].reshape(9,9)[:n,:n].ravel())
            manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
            for name,details in manifest["files"].items():
                self.assertEqual(file_sha256(root/name),details["sha256"])
                self.assertEqual(file_sha256(root/name),hashlib.sha256((root/name).read_bytes()).hexdigest())

    def test_rejects_invalid_targets_and_changed_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            path = export_dataset(self.bundle,Path(folder))
            with np.load(path,allow_pickle=False) as archive:
                arrays = {key:archive[key].copy() for key in archive.files}
            arrays["train_policy"][0] = 0
            broken = Path(folder)/"invalid.npz"
            np.savez_compressed(broken,**arrays)
            with self.assertRaisesRegex(ValueError,"policy distribution"): load_dataset(broken)
            arrays["train_policy"] = self.bundle.train.policy.copy()
            arrays["train_value"][0] = 0 if arrays["train_value"][0] else 1
            np.savez_compressed(broken,**arrays)
            with self.assertRaisesRegex(ValueError,"fingerprint mismatch"): load_dataset(broken)
            arrays["train_value"] = self.bundle.train.value.copy()
            metadata = json.loads(str(arrays["metadata"].item()))
            metadata["games"]["train"][0]["moves"][0] = [25,25]
            arrays["metadata"] = np.array(json.dumps(metadata))
            np.savez_compressed(broken,**arrays)
            with self.assertRaisesRegex(ValueError,"Illegal completed rollout"): load_dataset(broken)

    def test_export_only_preserves_model_and_training_uses_held_out_test(self):
        with tempfile.TemporaryDirectory() as folder:
            root,model_dir = Path(folder),Path(folder)/"models"
            model_dir.mkdir(); checkpoint = model_dir/"caro_policy.pt"
            checkpoint.write_bytes(b"existing checkpoint")
            args = ["backend.train","--samples","100","--export-only","--data-dir",str(root/"data"),"--model-dir",str(model_dir)]
            with patch.object(sys,"argv",args),patch.object(train,"generate_dataset",return_value=self.bundle),redirect_stdout(io.StringIO()): train.main()
            self.assertEqual(checkpoint.read_bytes(),b"existing checkpoint")
            data = root/"data"/"caro_dataset.npz"
            args = ["backend.train","--dataset",str(data),"--epochs","3","--batch-size","32","--model-dir",str(model_dir)]
            with patch.object(sys,"argv",args),patch.object(train,"generate_dataset",side_effect=AssertionError("must reuse data")),redirect_stdout(io.StringIO()): train.main()
            metadata = json.loads((model_dir/"training.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["best_val_loss"],min(row["val_loss"] for row in metadata["history"]))
            self.assertEqual(metadata["checkpoint_sha256"],file_sha256(checkpoint))
            self.assertEqual(metadata["test_samples"],100)
            model = CaroNet().eval()
            model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True))
            self.assertEqual(train.evaluate(model,self.bundle.validation,32)["loss"],metadata["best_val_loss"])
            self.assertEqual(train.evaluate(model,self.bundle.test,32),metadata["test_metrics"])


if __name__ == "__main__": unittest.main()
