"""Reproducible knowledge distillation on synthetic tactical positions.

Run from the project root: python -m backend.train --samples 6000 --epochs 12
This is a small educational model, not an AlphaZero-strength engine.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch.nn import functional as F
from .app.network import CaroNet, encode
from .app.rules import DIRECTIONS, tactical_scores


def dataset(count, seed):
    rng = np.random.default_rng(seed)
    inputs, policies, values = [], [], []
    n = 9
    for _ in range(count):
        length = int(rng.choice([3, 4, 5, 6], p=[.20, .20, .50, .10]))
        board = np.zeros((n, n), dtype=np.int8)
        # Synthetic positions start with alternating stone counts. Injected motifs
        # below are teaching examples, not a record of legal self-play games.
        moves = int(rng.integers(0, 33))
        indexes = rng.choice(n*n, moves, replace=False)
        board.ravel()[indexes] = np.where(np.arange(moves) % 2 == 0, 1, -1)
        player = 1 if moves % 2 == 0 else -1
        # Deliberately include line-building and blocking motifs for learning.
        if rng.random() < .65:
            dr, dc = DIRECTIONS[int(rng.integers(4))]
            r, c = int(rng.integers(n)), int(rng.integers(n))
            if 0 <= r+(length-1)*dr < n and 0 <= c+(length-1)*dc < n:
                side = int(rng.choice([-1, 1]))
                gap = int(rng.integers(length))
                for i in range(length):
                    board[r+i*dr, c+i*dc] = 0 if i == gap else side
        # Random rotations/reflections preserve directional symmetry.
        board = np.rot90(board, int(rng.integers(4))).copy()
        if rng.random() < .5:
            board = np.fliplr(board).copy()
        scores = tactical_scores(board, player, length).ravel()
        logits = np.where(scores >= 0, np.log1p(np.maximum(scores, 0)) * 1.1, -1e9)
        target = np.exp(logits - logits.max())
        target /= target.sum()
        own = float(tactical_scores(board, player, length).max())
        opp = float(tactical_scores(board, -player, length).max())
        value = np.tanh((np.log1p(own)-np.log1p(opp)) / 4)
        inputs.append(encode(board, player, length))
        policies.append(target.astype(np.float32))
        values.append(value)
    return (torch.from_numpy(np.asarray(inputs)), torch.from_numpy(np.asarray(policies)),
            torch.tensor(values, dtype=torch.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples < 100 or args.epochs < 1 or args.batch_size < 1:
        parser.error("Use at least 100 samples, 1 epoch, and a positive batch size")
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    started = perf_counter()
    print(f"Generating {args.samples} training and {max(100, args.samples//5)} validation positions...", flush=True)
    x, p, v = dataset(args.samples, args.seed)
    vx, vp, vv = dataset(max(100, args.samples//5), args.seed+1)
    model = CaroNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=1e-4)
    history, best = [], float("inf")
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        indexes = torch.randperm(len(x))
        total = 0.0
        for batch in indexes.split(args.batch_size):
            pred, value = model(x[batch])
            loss = -(p[batch] * F.log_softmax(pred, dim=1)).sum(1).mean() + .25 * F.mse_loss(value, v[batch])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        model.eval()
        with torch.inference_mode():
            vl, acc, count = 0.0, 0, 0
            for batch in torch.arange(len(vx)).split(args.batch_size):
                pred, value = model(vx[batch])
                vl += float(-(vp[batch]*F.log_softmax(pred, dim=1)).sum(1).sum())
                acc += int((pred.argmax(1) == vp[batch].argmax(1)).sum())
                count += len(batch)
            vl /= count
        history.append({"epoch": epoch+1, "loss": round(total/len(x), 5),
                        "val_loss": round(vl, 5), "top1_accuracy": round(acc/count, 4)})
        if vl < best:
            best = vl
            torch.save(model.state_dict(), model_dir / "caro_policy.pt")
        print(f"Epoch {epoch+1:02d}/{args.epochs}: loss={total/len(x):.4f} val={vl:.4f} teacher top1={acc/count:.1%}", flush=True)
    metadata = {"method": "Supervised distillation from open-segment tactical teacher",
                "samples": args.samples, "validation_samples": len(vx), "epochs": args.epochs,
                "seed": args.seed, "training_board_size": 9, "training_win_lengths": [3, 4, 5, 6],
                "seconds": round(perf_counter()-started, 1), "trained_at": datetime.now(timezone.utc).isoformat(),
                "best_val_loss": round(best, 5), "history": history,
                "limitations": "Synthetic training positions; value is a heuristic target, not a calibrated win probability. Larger boards rely on convolutional generalization and tactical search."}
    (model_dir / "training.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved checkpoint and actual training metrics to {model_dir}", flush=True)


if __name__ == "__main__":
    main()
