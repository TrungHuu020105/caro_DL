"""Generate, export, and reload CaroNet's synthetic tactical dataset.

NPZ stores all tensors and original board positions. CSV files expose the same
positions and targets for inspection without PyTorch. No pickle is used.
"""
from dataclasses import dataclass, fields
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .app.network import encode
from .app.rules import DIRECTIONS, tactical_scores

FORMAT_VERSION = 1
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "training"
DATASET_FILENAME = "caro_dataset.npz"


@dataclass
class DatasetSplit:
    boards: np.ndarray
    players: np.ndarray
    win_lengths: np.ndarray
    inputs: np.ndarray
    policy: np.ndarray
    value: np.ndarray

    def tensors(self):
        return (torch.from_numpy(self.inputs), torch.from_numpy(self.policy),
                torch.from_numpy(self.value))


@dataclass
class DatasetBundle:
    train: DatasetSplit
    validation: DatasetSplit
    metadata: dict


def generate_split(count: int, seed: int) -> DatasetSplit:
    """Same RNG sequence and teacher as the original in-memory generator."""
    if count < 1:
        raise ValueError("Dataset must contain at least one position.")
    rng = np.random.default_rng(seed)
    inputs, policies, values, boards, players, lengths = [], [], [], [], [], []
    n = 9
    for _ in range(count):
        length = int(rng.choice([3, 4, 5, 6], p=[.20, .20, .50, .10]))
        board = np.zeros((n, n), dtype=np.int8)
        # Motifs are synthetic teaching examples, not records of legal self-play.
        moves = int(rng.integers(0, 33))
        indexes = rng.choice(n*n, moves, replace=False)
        board.ravel()[indexes] = np.where(np.arange(moves) % 2 == 0, 1, -1)
        player = 1 if moves % 2 == 0 else -1
        if rng.random() < .65:
            dr, dc = DIRECTIONS[int(rng.integers(4))]
            r, c = int(rng.integers(n)), int(rng.integers(n))
            if 0 <= r+(length-1)*dr < n and 0 <= c+(length-1)*dc < n:
                side = int(rng.choice([-1, 1]))
                gap = int(rng.integers(length))
                for i in range(length):
                    board[r+i*dr, c+i*dc] = 0 if i == gap else side
        board = np.rot90(board, int(rng.integers(4))).copy()
        if rng.random() < .5:
            board = np.fliplr(board).copy()
        scores = tactical_scores(board, player, length).ravel()
        logits = np.where(scores >= 0, np.log1p(np.maximum(scores, 0)) * 1.1, -1e9)
        target = np.exp(logits - logits.max())
        target /= target.sum()
        own = float(tactical_scores(board, player, length).max())
        opp = float(tactical_scores(board, -player, length).max())
        values.append(np.tanh((np.log1p(own)-np.log1p(opp)) / 4))
        inputs.append(encode(board, player, length))
        policies.append(target.astype(np.float32))
        boards.append(board)
        players.append(player)
        lengths.append(length)
    return DatasetSplit(
        boards=np.asarray(boards, dtype=np.int8),
        players=np.asarray(players, dtype=np.int8),
        win_lengths=np.asarray(lengths, dtype=np.int8),
        inputs=np.asarray(inputs, dtype=np.float32),
        policy=np.asarray(policies, dtype=np.float32),
        value=np.asarray(values, dtype=np.float32),
    )


def generate_dataset(samples: int = 6000, seed: int = 42) -> DatasetBundle:
    train = generate_split(samples, seed)
    validation = generate_split(max(100, samples // 5), seed + 1)
    return DatasetBundle(train, validation, {
        "format_version": FORMAT_VERSION,
        "generator": "synthetic-tactical-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "validation_seed": seed + 1,
        "samples": samples,
        "validation_samples": len(validation.inputs),
        "board_size": 9,
        "win_lengths": [3, 4, 5, 6],
        "board_encoding": {"X": 1, "O": -1, "empty": 0},
        "input_channels": ["own_stones", "opponent_stones", "empty_cells", "win_length/6", "board_mask"],
        "policy_index": "row * board_size + col (zero-based)",
        "value_target": "tanh((log1p(own_teacher_max)-log1p(opponent_teacher_max))/4)",
        "limitations": "Synthetic tactical positions, not legal self-play records. Value targets are heuristic scores, not calibrated win probabilities.",
    })


def fingerprint(bundle: DatasetBundle) -> str:
    """Stable hash of actual arrays; independent of export time/ZIP metadata."""
    digest = hashlib.sha256()
    for name, split in (("train", bundle.train), ("val", bundle.validation)):
        for field in fields(DatasetSplit):
            array = np.ascontiguousarray(getattr(split, field.name))
            digest.update(f"{name}_{field.name}:{array.dtype}:{array.shape}".encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: dict):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, split: DatasetSplit, prefix: str):
    temporary = path.with_name(path.name + ".tmp")
    # BOM helps Excel on Windows recognize UTF-8. JSON cells preserve full arrays.
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "player_to_move", "win_length", "teacher_move",
                         "board_json", "policy_json", "value_target"])
        for i in range(len(split.inputs)):
            best = int(split.policy[i].argmax())
            n = split.boards.shape[1]
            writer.writerow([
                f"{prefix}_{i+1:06d}", "X" if split.players[i] == 1 else "O",
                int(split.win_lengths[i]), f"{chr(65+best % n)}{best // n+1}",
                json.dumps(split.boards[i].tolist(), separators=(",", ":")),
                json.dumps(split.policy[i].tolist(), separators=(",", ":")),
                float(split.value[i]),
            ])
    temporary.replace(path)


def export_dataset(bundle: DatasetBundle, directory: Path = DEFAULT_DATA_DIR) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / DATASET_FILENAME
    metadata = {**bundle.metadata, "fingerprint_sha256": fingerprint(bundle)}
    arrays = {}
    for prefix, split in (("train", bundle.train), ("val", bundle.validation)):
        arrays.update({f"{prefix}_{field.name}": getattr(split, field.name) for field in fields(DatasetSplit)})
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps(metadata)), **arrays)
    temporary.replace(path)
    write_csv(directory / "train.csv", bundle.train, "train")
    write_csv(directory / "validation.csv", bundle.validation, "val")
    files = {}
    for filename in (DATASET_FILENAME, "train.csv", "validation.csv"):
        file = directory / filename
        files[filename] = {"bytes": file.stat().st_size, "sha256": file_sha256(file)}
    write_json(directory / "manifest.json", {
        **metadata,
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                   for key, value in arrays.items()},
        "files": files,
    })
    return path


def validate_split(split: DatasetSplit, name: str, n: int):
    if split.inputs.ndim != 4:
        raise ValueError(f"Invalid {name}_inputs: expected a four-dimensional tensor.")
    count = len(split.inputs)
    expected = {
        "boards": ((count, n, n), np.dtype("int8")),
        "players": ((count,), np.dtype("int8")),
        "win_lengths": ((count,), np.dtype("int8")),
        "inputs": ((count, 5, n, n), np.dtype("float32")),
        "policy": ((count, n*n), np.dtype("float32")),
        "value": ((count,), np.dtype("float32")),
    }
    if not count:
        raise ValueError(f"{name} is empty.")
    for field, (shape, dtype) in expected.items():
        array = getattr(split, field)
        if array.shape != shape or array.dtype != dtype or not np.isfinite(array).all():
            raise ValueError(f"Invalid {name}_{field}: expected finite {dtype} array of shape {shape}.")
    if not np.isin(split.boards, [-1, 0, 1]).all() or not np.isin(split.players, [-1, 1]).all():
        raise ValueError(f"Invalid board/player encoding in {name}.")
    if not np.isin(split.win_lengths, [3, 4, 5, 6]).all():
        raise ValueError(f"Invalid winning lengths in {name}.")
    if (split.policy < 0).any() or not np.allclose(split.policy.sum(1), 1, atol=1e-5):
        raise ValueError(f"Invalid policy distribution in {name}.")
    occupied = split.boards.reshape(count, -1) != 0
    if (split.policy[occupied] != 0).any() or (np.abs(split.value) > 1).any():
        raise ValueError(f"Illegal policy/value targets in {name}.")
    # Ensure saved raw positions really encode the tensors used for training.
    for i in range(count):
        if not np.array_equal(split.inputs[i], encode(split.boards[i], int(split.players[i]), int(split.win_lengths[i]))):
            raise ValueError(f"Input does not match board in {name} sample {i}.")


def load_dataset(path: Path) -> DatasetBundle:
    with np.load(Path(path), allow_pickle=False) as archive:
        required = {f"{prefix}_{field.name}" for prefix in ("train", "val") for field in fields(DatasetSplit)} | {"metadata"}
        if not required.issubset(archive.files):
            raise ValueError(f"Missing dataset arrays: {', '.join(sorted(required-set(archive.files)))}")
        metadata = json.loads(str(archive["metadata"].item()))
        if not isinstance(metadata, dict) or metadata.get("format_version") != FORMAT_VERSION or metadata.get("board_size") != 9:
            raise ValueError("Unsupported dataset format or board size.")
        splits = [DatasetSplit(**{field.name: archive[f"{prefix}_{field.name}"] for field in fields(DatasetSplit)})
                  for prefix in ("train", "val")]
    bundle = DatasetBundle(*splits, metadata)
    validate_split(bundle.train, "train", 9)
    validate_split(bundle.validation, "val", 9)
    if metadata.get("samples") != len(bundle.train.inputs) or metadata.get("validation_samples") != len(bundle.validation.inputs):
        raise ValueError("Dataset sizes do not match metadata.")
    if metadata.get("fingerprint_sha256") != fingerprint(bundle):
        raise ValueError("Dataset fingerprint mismatch: the arrays have changed.")
    return bundle
