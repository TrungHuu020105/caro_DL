"""Replay human game states, relabel policy, and keep stable validation boundaries."""
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import numpy as np
from .app.game import Game, GameConfig
from .app.network import INPUT_CHANNELS, RULES, encode
from .app.rules import safe_defenses, winning_moves
from .dataset import DatasetSplit, position_key, teacher_policy, trajectory_key


def select(split, ids):
    return DatasetSplit(**{field.name: getattr(split, field.name)[ids].copy() for field in fields(DatasetSplit)})


def concatenate(*splits):
    splits = [split for split in splits if split is not None and len(split.inputs)]
    if not splits:
        return None
    return DatasetSplit(**{field.name: np.concatenate([getattr(s, field.name) for s in splits]) for field in fields(DatasetSplit)})


def pad(split, canvas=25):
    old = split.boards.shape[1]
    if old == canvas:
        return split
    result = select(split, np.arange(len(split.inputs)))
    result.boards = np.zeros((len(split.inputs),canvas,canvas),dtype=np.int8)
    result.inputs = np.zeros((len(split.inputs),INPUT_CHANNELS,canvas,canvas),dtype=np.float32)
    result.policy = np.zeros((len(split.inputs),canvas*canvas),dtype=np.float32)
    result.boards[:,:old,:old] = split.boards
    result.inputs[:,:,:old,:old] = split.inputs
    result.policy.reshape(-1,canvas,canvas)[:,:old,:old] = split.policy.reshape(-1,old,old)
    return result


def exclusion_keys(bundle):
    positions = set()
    for split in (bundle.train,bundle.validation,bundle.test):
        for i in range(len(split.inputs)):
            n,k,p = map(int,(split.sizes[i],split.win_lengths[i],split.players[i]))
            positions.add(position_key(split.boards[i,:n,:n],p,k,RULES[int(split.rules[i])]).hex())
    games = {trajectory_key(g["size"],g["win_length"],g["rule"],g["moves"]).hex()
             for group in bundle.metadata["games"].values() for g in group}
    return positions,games


def reanalyzed_policy(board,player,length,rule):
    wins = winning_moves(board,player,length,rule)
    threats = winning_moves(board,-player,length,rule)
    defenses = safe_defenses(board,player,length,rule,threats) if threats and not wins else []
    forced = wins or defenses
    if not forced:
        return teacher_policy(board,player,length,rule)
    target = np.zeros(board.size,dtype=np.float32)
    for r,c in forced:
        target[r*len(board)+c] = 1/len(forced)
    return target


def _pack(rows,canvas=25):
    if not rows:
        return None
    count = len(rows)
    split = DatasetSplit(np.zeros((count,canvas,canvas),dtype=np.int8),np.empty(count,dtype=np.int8),
        np.empty(count,dtype=np.int8),np.zeros((count,INPUT_CHANNELS,canvas,canvas),dtype=np.float32),
        np.zeros((count,canvas*canvas),dtype=np.float32),np.empty(count,dtype=np.float32),
        np.empty(count,dtype=np.int8),np.empty(count,dtype=np.int8),np.empty(count,dtype=np.int64))
    for i,(board,player,record,game_id) in enumerate(rows):
        n,k,rule = record["size"],record["win_length"],record["rule"]
        split.boards[i,:n,:n] = board
        split.players[i],split.win_lengths[i],split.sizes[i] = player,k,n
        split.rules[i],split.game_ids[i] = RULES.index(rule),game_id
        split.value[i] = record["winner"]*player
        split.inputs[i] = encode(board,player,k,rule,canvas)
        split.policy[i].reshape(canvas,canvas)[:n,:n] = reanalyzed_policy(board,player,k,rule).reshape(n,n)
    return split


def build_human_data(records,position_splits,base_keys,seed=2026,max_train=256,max_validation=128):
    blocked,base_games = base_keys
    seen,rows = set(),{"train":[],"validation":[]}
    info = {"ignored_base_positions":0,"ignored_duplicate_positions":0,"ignored_split_conflicts":0,"ignored_base_games":0}
    sources = {}
    for index,record in enumerate(records):
        if record["game_hash"] in base_games:
            info["ignored_base_games"] += 1
            continue
        game = Game(GameConfig(size=record["size"],win_length=record["win_length"],rule=record["rule"],mode="local"))
        group = record["split"]
        for ply,(r,c) in enumerate(record["moves"]):
            if ply >= 2:
                board = np.asarray(game.board,dtype=np.int8)
                key = position_key(board,game.current_player,game.config.win_length,game.config.rule).hex()
                if key in blocked:
                    info["ignored_base_positions"] += 1
                elif position_splits.get(key) != group:
                    info["ignored_split_conflicts"] += 1
                elif key in seen:
                    info["ignored_duplicate_positions"] += 1
                else:
                    seen.add(key)
                    game_id = 10**12+index
                    sources[game_id] = record["game_hash"]
                    rows[group].append((board.copy(),game.current_player,record,game_id))
            game.place(r,c)
        if game.status not in ("won","draw") or (game.winner or 0) != record["winner"]:
            raise ValueError("Dữ liệu ván người chơi có kết quả không hợp lệ.")
    rng = np.random.default_rng(seed)
    info.update(available_train_positions=len(rows["train"]),available_validation_positions=len(rows["validation"]))
    selected = {}
    for name,cap in (("train",max_train),("validation",max_validation)):
        indexes = rng.permutation(len(rows[name]))[:cap]
        selected[name] = _pack([rows[name][int(i)] for i in indexes])
    info["selected_train_positions"] = len(selected["train"].inputs) if selected["train"] is not None else 0
    info["selected_validation_positions"] = len(selected["validation"].inputs) if selected["validation"] is not None else 0
    info["selected_source_hashes"] = {name:sorted({sources[int(i)] for i in split.game_ids}) if split is not None else []
                                     for name,split in selected.items()}
    return selected["train"],selected["validation"],info


def export_human_data(path: Path,train,validation,replay_ids,metadata):
    arrays = {f"{name}_{field.name}":getattr(split,field.name)
              for name,split in (("human_train",train),("human_validation",validation)) if split is not None
              for field in fields(DatasetSplit)}
    arrays["replay_indices"] = np.asarray(replay_ids,dtype=np.int64)
    digest = hashlib.sha256()
    for name,array in sorted(arrays.items()):
        digest.update(name.encode()); digest.update(np.ascontiguousarray(array).tobytes())
    fingerprint = digest.hexdigest()
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,metadata=np.array(json.dumps({**metadata,"fingerprint_sha256":fingerprint})),**arrays)
    return fingerprint
