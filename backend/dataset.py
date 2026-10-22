"""Legal rollouts, outcome labels, and game/symmetry-disjoint splits (Python 3.10)."""
from dataclasses import dataclass, fields
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from .app.network import INPUT_CHANNELS, RULES, encode
from .app.rules import tactical_scores, winning_line, winning_moves

FORMAT_VERSION = 2
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "training"
DATASET_FILENAME = "caro_dataset.npz"
BOARD_SIZES = tuple(range(3, 26))


@dataclass
class DatasetSplit:
    boards: np.ndarray
    players: np.ndarray
    win_lengths: np.ndarray
    inputs: np.ndarray
    policy: np.ndarray
    value: np.ndarray
    sizes: np.ndarray
    rules: np.ndarray
    game_ids: np.ndarray

    def tensors(self):
        return (torch.from_numpy(self.inputs), torch.from_numpy(self.policy), torch.from_numpy(self.value))


@dataclass
class DatasetBundle:
    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit
    metadata: dict


def position_key(board, player, length, rule):
    relative = np.asarray(board, dtype=np.int8) * player
    variants = [np.ascontiguousarray(v).tobytes() for k in range(4)
                for v in (np.rot90(relative, k), np.fliplr(np.rot90(relative, k)))]
    return hashlib.sha256(bytes([len(board), length, RULES.index(rule)]) + min(variants)).digest()


def trajectory_key(size, length, rule, moves):
    coordinates = np.asarray(moves,dtype=np.int8)
    variants = []
    for _ in range(4):
        variants.append(coordinates.tobytes())
        reflected = coordinates.copy(); reflected[:,1] = size-1-reflected[:,1]
        variants.append(reflected.tobytes())
        coordinates = np.column_stack((size-1-coordinates[:,1],coordinates[:,0])).astype(np.int8)
    return hashlib.sha256(bytes([size,length,RULES.index(rule)])+min(variants)).digest()


def teacher_policy(board, player, length, rule):
    forced = winning_moves(board, player, length, rule) or winning_moves(board, -player, length, rule)
    if forced:
        target = np.zeros(board.size, dtype=np.float32)
        for r, c in forced:
            target[r*len(board)+c] = 1 / len(forced)
        return target
    scores = tactical_scores(board, player, length, rule).ravel()
    logits = np.where(scores >= 0, np.log1p(np.maximum(scores, 0)) * 1.1, -1e9)
    target = np.exp(logits-logits.max())
    return (target / target.sum()).astype(np.float32)


def rollout(rng, size, length, rule):
    """Alternate legal X/O moves and stop at the first terminal state."""
    board = np.zeros((size, size), dtype=np.int8)
    positions, moves, player, winner = [], [], 1, 0
    for ply in range(size*size):
        positions.append((board.copy(), player))
        available = np.flatnonzero(board.ravel() == 0)
        forced = winning_moves(board, player, length, rule) or winning_moves(board, -player, length, rule)
        if forced:
            r, c = forced[int(rng.integers(len(forced)))]
        elif ply == 0:
            r, c = divmod(int(rng.choice(available)), size)
        else:
            scores = tactical_scores(board, player, length, rule).ravel()
            ranked = available[np.argsort(scores[available])[::-1]]
            pool = ranked[:min(12, len(ranked))]
            if rng.random() < .18:
                chosen = int(rng.choice(available))
            else:
                weights = 1 / (np.arange(len(pool)) + 1.)
                chosen = int(rng.choice(pool, p=weights/weights.sum()))
            r, c = divmod(chosen, size)
        board[r, c] = player
        moves.append([r, c])
        if winning_line(board.tolist(), r, c, length, rule):
            winner = player
            break
        player = -player
    return positions, winner, moves


def _make_split(count, seed, sizes, seen, offset, seen_games=None):
    if count < 1:
        raise ValueError("Dataset must contain at least one position.")
    rng, canvas = np.random.default_rng(seed), max(sizes)
    if seen_games is None:
        seen_games = set()
    split = DatasetSplit(
        boards=np.zeros((count, canvas, canvas), dtype=np.int8),
        players=np.empty(count, dtype=np.int8), win_lengths=np.empty(count, dtype=np.int8),
        inputs=np.zeros((count, INPUT_CHANNELS, canvas, canvas), dtype=np.float32),
        policy=np.zeros((count, canvas*canvas), dtype=np.float32), value=np.empty(count, dtype=np.float32),
        sizes=np.empty(count, dtype=np.int8), rules=np.empty(count, dtype=np.int8),
        game_ids=np.empty(count, dtype=np.int64))
    records, index, game_number = [], 0, 0
    while index < count:
        n = int(rng.choice(sizes))
        length = int(rng.integers(3, min(n, 6)+1))
        rule_id = int(rng.integers(len(RULES)))
        rule = RULES[rule_id]
        positions, winner, moves = rollout(rng, n, length, rule)
        trajectory = trajectory_key(n,length,rule,moves)
        if trajectory in seen_games:
            continue
        game_number += 1
        game_id, used = offset + game_number, False
        # Assign full games to a split before augmenting any selected position.
        for pos in rng.permutation(len(positions))[:12]:
            board, player = positions[int(pos)]
            key = position_key(board, player, length, rule)
            if key in seen:
                continue
            seen.add(key)
            board = np.rot90(board, int(rng.integers(4))).copy()
            if rng.random() < .5:
                board = np.fliplr(board).copy()
            split.boards[index, :n, :n] = board
            split.inputs[index] = encode(board, player, length, rule, canvas)
            split.policy[index].reshape(canvas, canvas)[:n, :n] = teacher_policy(board, player, length, rule).reshape(n, n)
            split.players[index], split.win_lengths[index] = player, length
            split.value[index] = winner * player
            split.sizes[index], split.rules[index], split.game_ids[index] = n, rule_id, game_id
            index, used = index+1, True
            if index == count:
                break
        if used:
            seen_games.add(trajectory)
            records.append({"id": game_id, "size": n, "win_length": length,
                            "rule": rule, "winner": winner, "moves": moves})
    return split, records


def generate_split(count: int, seed: int) -> DatasetSplit:
    return _make_split(count, seed, BOARD_SIZES, set(), 0)[0]


def generate_dataset(samples: int = 6000, seed: int = 42,
                     sizes: tuple[int, ...] = BOARD_SIZES) -> DatasetBundle:
    sizes = tuple(sorted(set(sizes)))
    if not sizes or min(sizes) < 3 or max(sizes) > 25 or samples < 1 or seed < 0:
        raise ValueError("Use board sizes 3..25, positive samples and a non-negative seed.")
    seen, seen_games, records, splits = set(), set(), {}, []
    for prefix, count, extra in (("train", samples, 0), ("val", max(100, samples//5), 1),
                                 ("test", max(100, samples//5), 2)):
        split, games = _make_split(count, seed+extra, sizes, seen, extra*100000000, seen_games)
        splits.append(split)
        records[prefix] = games
    return DatasetBundle(*splits, {
        "format_version": FORMAT_VERSION, "generator": "legal-rollout-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "samples": samples, "validation_samples": len(splits[1].inputs), "test_samples": len(splits[2].inputs),
        "board_size": max(sizes), "board_sizes": list(sizes), "win_lengths": [3, 4, 5, 6],
        "rules": list(RULES), "board_encoding": {"X": 1, "O": -1, "empty": 0},
        "input_channels": ["own_stones", "opponent_stones", "empty_cells", "win_length/6", "board_mask",
                           "rule_freestyle", "rule_exact", "rule_blocked"],
        "policy_index": "row * canvas_size + col; padded cells have probability zero",
        "value_target": "Completed rollout outcome * player_to_move (-1/0/+1)",
        "split_strategy": "Whole games; global perspective/D4 deduplication before stored augmentation",
        "audit": {"duplicate_or_symmetric_positions_between_splits": 0, "shared_games_between_splits": 0},
        "games": records,
        "limitations": "Supervised imitation of a tactical rollout policy; outcome value is policy-dependent, not a calibrated win probability. Not reinforcement learning.",
    })


def fingerprint(bundle: DatasetBundle) -> str:
    digest = hashlib.sha256()
    for name, split in (("train", bundle.train), ("val", bundle.validation), ("test", bundle.test)):
        for field in fields(DatasetSplit):
            array = np.ascontiguousarray(getattr(split, field.name))
            digest.update(f"{name}_{field.name}:{array.dtype}:{array.shape}".encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    """Streaming SHA-256, including on Python 3.10 (no hashlib.file_digest)."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, split: DatasetSplit, prefix: str):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "game_id", "size", "rule", "player_to_move", "win_length",
                         "teacher_move", "board_json", "policy_json", "value_target"])
        canvas = split.boards.shape[1]
        for i in range(len(split.inputs)):
            n = int(split.sizes[i])
            target = split.policy[i].reshape(canvas, canvas)[:n, :n]
            r, c = divmod(int(target.argmax()), n)
            writer.writerow([f"{prefix}_{i+1:06d}", int(split.game_ids[i]), n, RULES[int(split.rules[i])],
                             "X" if split.players[i] == 1 else "O", int(split.win_lengths[i]), f"{chr(65+c)}{r+1}",
                             json.dumps(split.boards[i,:n,:n].tolist(), separators=(",", ":")),
                             json.dumps(target.ravel().tolist(), separators=(",", ":")), float(split.value[i])])
    temporary.replace(path)


def export_dataset(bundle: DatasetBundle, directory: Path = DEFAULT_DATA_DIR) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / DATASET_FILENAME
    metadata = {**bundle.metadata, "fingerprint_sha256": fingerprint(bundle)}
    arrays = {f"{prefix}_{field.name}": getattr(split, field.name)
              for prefix, split in (("train",bundle.train), ("val",bundle.validation), ("test",bundle.test))
              for field in fields(DatasetSplit)}
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps(metadata)), **arrays)
    temporary.replace(path)
    for name, split in (("train",bundle.train), ("validation",bundle.validation), ("test",bundle.test)):
        write_csv(directory/f"{name}.csv", split, name)
    files = {name: {"bytes": (directory/name).stat().st_size, "sha256": file_sha256(directory/name)}
             for name in (DATASET_FILENAME,"train.csv","validation.csv","test.csv")}
    manifest = {key: value for key, value in metadata.items() if key != "games"}
    write_json(directory/"manifest.json", {**manifest,
        "games_per_split": {key: len(value) for key, value in metadata["games"].items()},
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
        "files": files})
    return path


def validate_split(split: DatasetSplit, name: str, n: int):
    count = len(split.inputs)
    expected = {"boards": ((count,n,n),"int8"), "players": ((count,),"int8"),
        "win_lengths": ((count,),"int8"), "inputs": ((count,INPUT_CHANNELS,n,n),"float32"),
        "policy": ((count,n*n),"float32"), "value": ((count,),"float32"),
        "sizes": ((count,),"int8"), "rules": ((count,),"int8"), "game_ids": ((count,),"int64")}
    if not count:
        raise ValueError(f"{name} is empty.")
    for field, (shape, dtype) in expected.items():
        array = getattr(split, field)
        if array.shape != shape or array.dtype != np.dtype(dtype) or not np.isfinite(array).all():
            raise ValueError(f"Invalid {name}_{field}: expected finite {dtype} array of shape {shape}.")
    if not np.isin(split.boards, [-1,0,1]).all() or not np.isin(split.players, [-1,1]).all():
        raise ValueError(f"Invalid board/player encoding in {name}.")
    if (split.policy < 0).any() or not np.allclose(split.policy.sum(1), 1, atol=1e-5):
        raise ValueError(f"Invalid policy distribution in {name}.")
    if not np.isin(split.value, [-1,0,1]).all():
        raise ValueError(f"Invalid outcome target in {name}.")
    for i in range(count):
        size, k, rid, player = map(int, (split.sizes[i],split.win_lengths[i],split.rules[i],split.players[i]))
        if not (3 <= size <= n and 3 <= k <= min(6,size) and 0 <= rid < len(RULES)):
            raise ValueError(f"Invalid size/rule/length in {name}.")
        board = split.boards[i,:size,:size]
        x, o = int((board == 1).sum()), int((board == -1).sum())
        if not (x == o and player == 1 or x == o+1 and player == -1):
            raise ValueError(f"Illegal stone counts/turn in {name} sample {i}.")
        if not np.array_equal(split.inputs[i], encode(board, player, k, RULES[rid], n)):
            raise ValueError(f"Input does not match board in {name} sample {i}.")
        padding = split.boards[i].copy(); padding[:size,:size] = 0
        legal = split.inputs[i,2].ravel() > .5
        if padding.any() or (split.policy[i,~legal] != 0).any():
            raise ValueError(f"Illegal padded board/policy in {name} sample {i}.")
        cells = board.tolist()
        for r,c in np.argwhere(board != 0):
            if winning_line(cells, int(r), int(c), k, RULES[rid]):
                raise ValueError(f"Terminal training position in {name} sample {i}.")


def load_dataset(path: Path) -> DatasetBundle:
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if not isinstance(metadata, dict) or metadata.get("format_version") != FORMAT_VERSION:
            raise ValueError("Dataset v2 required; regenerate with python -m backend.train --export-only.")
        required = {f"{prefix}_{field.name}" for prefix in ("train","val","test") for field in fields(DatasetSplit)}
        if not required.issubset(archive.files):
            raise ValueError("Missing dataset arrays.")
        splits = [DatasetSplit(**{field.name: archive[f"{prefix}_{field.name}"] for field in fields(DatasetSplit)})
                  for prefix in ("train","val","test")]
    bundle = DatasetBundle(*splits, metadata)
    # Validate distributions before integrity hashes so malformed targets are explicit.
    for name, split in (("train",bundle.train), ("val",bundle.validation), ("test",bundle.test)):
        validate_split(split, name, metadata["board_size"])
    if metadata.get("fingerprint_sha256") != fingerprint(bundle):
        raise ValueError("Dataset fingerprint mismatch: the arrays have changed.")
    seen, game_ids, trajectories = set(), set(), set()
    for name, split, count_key in (("train",bundle.train,"samples"), ("val",bundle.validation,"validation_samples"),
                                    ("test",bundle.test,"test_samples")):
        if metadata.get(count_key) != len(split.inputs):
            raise ValueError("Dataset sizes do not match metadata.")
        ids = set(map(int, split.game_ids))
        if ids & game_ids:
            raise ValueError("A game occurs in multiple data splits.")
        game_ids.update(ids)
        games = {game["id"]: game for game in metadata["games"][name]}
        possible = {}
        for game_id,game in games.items():
            size,k,rule = game["size"],game["win_length"],game["rule"]
            trajectory = trajectory_key(size,k,rule,game["moves"])
            if trajectory in trajectories:
                raise ValueError("Repeated or symmetric game trajectory in data splits.")
            trajectories.add(trajectory)
            board = np.zeros((size,size),dtype=np.int8)
            player,winner = 1,0
            needed = set(int((split.boards[i]!=0).sum()) for i in np.flatnonzero(split.game_ids==game_id))
            keys = set()
            for ply,(r,c) in enumerate(game["moves"]):
                if winner or not (0 <= r < size and 0 <= c < size) or board[r,c]:
                    raise ValueError("Illegal completed rollout in dataset metadata.")
                if ply in needed:
                    keys.add(position_key(board,player,k,rule))
                board[r,c] = player
                if winning_line(board.tolist(),r,c,k,rule): winner = player
                player = -player
            if winner != game["winner"] or not winner and (board==0).any():
                raise ValueError("Rollout is unfinished or its outcome is incorrect.")
            possible[game_id] = keys
        for i in range(len(split.inputs)):
            n, k, rid, player = map(int, (split.sizes[i],split.win_lengths[i],split.rules[i],split.players[i]))
            key = position_key(split.boards[i,:n,:n], player, k, RULES[rid])
            if key in seen:
                raise ValueError("Repeated or symmetric position in data splits.")
            seen.add(key)
            game = games.get(int(split.game_ids[i]))
            if game is None or split.value[i] != game["winner"] * player:
                raise ValueError("Value target does not match the completed game outcome.")
            if key not in possible[int(split.game_ids[i])]:
                raise ValueError("Training position is not a prefix of its completed legal game.")
    return bundle
