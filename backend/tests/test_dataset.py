import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from backend import train
from backend.app.network import CaroNet, encode
from backend.dataset import (export_dataset, file_sha256, fingerprint,
                             generate_dataset, load_dataset)


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = generate_dataset(100, seed=7)

    def test_reproducible_positions_and_separate_validation(self):
        repeated = generate_dataset(100, seed=7)
        self.assertEqual(fingerprint(self.bundle), fingerprint(repeated))
        self.assertFalse(np.array_equal(self.bundle.train.boards, self.bundle.validation.boards))
        for split in (self.bundle.train, self.bundle.validation):
            for i in range(len(split.inputs)):
                np.testing.assert_array_equal(
                    split.inputs[i],
                    encode(split.boards[i], int(split.players[i]), int(split.win_lengths[i])),
                )
                best = int(split.policy[i].argmax())
                self.assertEqual(split.boards[i].ravel()[best], 0)

    def test_npz_round_trip_csv_and_file_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = export_dataset(self.bundle, Path(folder))
            restored = load_dataset(path)
            self.assertEqual(fingerprint(restored), fingerprint(self.bundle))
            for name in ("train", "validation"):
                original, loaded = getattr(self.bundle, name), getattr(restored, name)
                for field in ("boards", "players", "win_lengths", "inputs", "policy", "value"):
                    np.testing.assert_array_equal(getattr(original, field), getattr(loaded, field))
                with (Path(folder) / f"{name}.csv").open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), len(original.inputs))
                row = rows[0]
                self.assertEqual(json.loads(row["board_json"]), original.boards[0].tolist())
                np.testing.assert_array_equal(np.asarray(json.loads(row["policy_json"]), dtype=np.float32), original.policy[0])
                self.assertEqual(float(row["value_target"]), float(original.value[0]))
                self.assertEqual(row["player_to_move"], "X" if original.players[0] == 1 else "O")
            manifest = json.loads((Path(folder) / "manifest.json").read_text(encoding="utf-8"))
            for filename, details in manifest["files"].items():
                self.assertEqual(file_sha256(Path(folder) / filename), details["sha256"])

    def test_rejects_invalid_targets_and_modified_data(self):
        with tempfile.TemporaryDirectory() as folder:
            path = export_dataset(self.bundle, Path(folder))
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files}
            arrays["train_policy"][0] = 0
            broken = Path(folder) / "invalid.npz"
            np.savez_compressed(broken, **arrays)
            with self.assertRaisesRegex(ValueError, "policy distribution"):
                load_dataset(broken)
            arrays["train_policy"] = self.bundle.train.policy.copy()
            arrays["train_value"][0] = .5  # Valid shape/range; fingerprint must still reject edits.
            np.savez_compressed(broken, **arrays)
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_dataset(broken)

    def test_export_only_leaves_existing_model_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model_dir = root / "models"
            model_dir.mkdir()
            checkpoint = model_dir / "caro_policy.pt"
            checkpoint.write_bytes(b"existing checkpoint")
            arguments = ["backend.train", "--samples", "100", "--export-only",
                         "--data-dir", str(root / "data"), "--model-dir", str(model_dir)]
            with patch.object(sys, "argv", arguments), redirect_stdout(io.StringIO()):
                train.main()
            self.assertEqual(checkpoint.read_bytes(), b"existing checkpoint")
            self.assertFalse((model_dir / "training.json").exists())
            self.assertEqual(len(load_dataset(root / "data" / "caro_dataset.npz").train.inputs), 100)

    def test_training_reuses_export_and_saves_lowest_loss_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = export_dataset(self.bundle, root / "data")
            model_dir = root / "models"
            arguments = ["backend.train", "--dataset", str(data), "--epochs", "4",
                         "--batch-size", "32", "--model-dir", str(model_dir)]
            with (patch.object(sys, "argv", arguments),
                  patch.object(train, "generate_dataset", side_effect=AssertionError("must reuse exported data")),
                  redirect_stdout(io.StringIO())):
                train.main()
            metadata = json.loads((model_dir / "training.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["samples"], 100)
            self.assertEqual(metadata["dataset"]["file_sha256"], file_sha256(data))
            self.assertEqual(metadata["dataset"]["fingerprint_sha256"], fingerprint(self.bundle))
            self.assertEqual(metadata["best_val_loss"], min(item["val_loss"] for item in metadata["history"]))
            best = metadata["history"][metadata["best_epoch"] - 1]
            self.assertEqual(best["val_loss"], metadata["best_val_loss"])
            checkpoint = model_dir / "caro_policy.pt"
            self.assertEqual(metadata["checkpoint_sha256"], file_sha256(checkpoint))
            model = CaroNet().eval()
            model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
            vx, vp, _ = self.bundle.validation.tensors()
            with torch.inference_mode():
                losses = []
                for ids in torch.arange(len(vx)).split(32):
                    logits, _ = model(vx[ids])
                    losses.append(float(-(vp[ids]*F.log_softmax(logits, dim=1)).sum()))
            self.assertAlmostEqual(sum(losses) / len(vx), metadata["best_val_loss"], places=4)


if __name__ == "__main__":
    unittest.main()
