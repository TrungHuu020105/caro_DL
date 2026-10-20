"""Reproducible knowledge distillation on synthetic tactical positions.

Run from the project root: python -m backend.train --samples 6000 --epochs 12
Export without training: python -m backend.train --export-only
Reuse exported data: python -m backend.train --dataset backend/data/training/caro_dataset.npz
This is a small educational model, not an AlphaZero-strength engine.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import torch
from torch.nn import functional as F
from .app.network import CaroNet
from .dataset import (DEFAULT_DATA_DIR, export_dataset, file_sha256, fingerprint,
                      generate_dataset, generate_split, load_dataset, write_json)


def dataset(count, seed):
    """Backward-compatible helper for callers using the original generator."""
    return generate_split(count, seed).tensors()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=Path, help="Load an existing NPZ instead of generating new positions")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory for generated NPZ, CSV files, and manifest")
    parser.add_argument("--export-only", action="store_true", help="Export positions and targets without changing the model")
    parser.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "models")
    args = parser.parse_args()
    if args.samples < 100 or args.epochs < 1 or args.batch_size < 1:
        parser.error("Use at least 100 samples, 1 epoch, and a positive batch size")
    if args.seed < 0:
        parser.error("Seed must be non-negative")
    if args.export_only and args.dataset:
        parser.error("--export-only generates new files; omit --dataset")
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    started = perf_counter()
    try:
        if args.dataset:
            dataset_path = args.dataset.resolve()
            bundle = load_dataset(dataset_path)
            print(f"Loaded dataset: {dataset_path}", flush=True)
        else:
            print(f"Generating {args.samples} training and {max(100, args.samples//5)} validation positions...", flush=True)
            bundle = generate_dataset(args.samples, args.seed)
            dataset_path = export_dataset(bundle, args.data_dir.resolve())
            print(f"Exported NPZ, train.csv, validation.csv, and manifest.json to {dataset_path.parent}", flush=True)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(f"Cannot prepare dataset: {exc}")
    if args.export_only:
        print("Export complete. Model weights and training metrics were not changed.", flush=True)
        return
    x, p, v = bundle.train.tensors()
    vx, vp, vv = bundle.validation.tensors()
    model = CaroNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=1e-4)
    history, best, best_epoch = [], float("inf"), 0
    model_dir = args.model_dir.resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "caro_policy.pt"
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
            best_epoch = epoch + 1
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(model.state_dict(), temporary)
            temporary.replace(checkpoint_path)
        print(f"Epoch {epoch+1:02d}/{args.epochs}: loss={total/len(x):.4f} val={vl:.4f} teacher top1={acc/count:.1%}", flush=True)
    metadata = {"method": "Supervised distillation from open-segment tactical teacher",
                "samples": len(x), "validation_samples": len(vx), "epochs": args.epochs,
                "seed": args.seed, "training_board_size": 9, "training_win_lengths": [3, 4, 5, 6],
                "seconds": round(perf_counter()-started, 1), "trained_at": datetime.now(timezone.utc).isoformat(),
                "best_val_loss": round(best, 5), "best_epoch": best_epoch,
                "checkpoint_selection": "Lowest validation policy loss within this training run",
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "dataset": {"path": dataset_path.as_posix(), "file_sha256": file_sha256(dataset_path),
                            "fingerprint_sha256": fingerprint(bundle), "generator_seed": bundle.metadata["seed"]},
                "history": history,
                "limitations": "Synthetic training positions; value is a heuristic target, not a calibrated win probability. Larger boards rely on convolutional generalization and tactical search."}
    write_json(model_dir / "training.json", metadata)
    print(f"Best model: epoch {best_epoch}, validation loss {best:.5f} -> {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
