"""Train a rule-conditioned policy/value CNN on legal completed games.

python -m backend.train --samples 6000 --epochs 18
python -m backend.train --export-only
python -m backend.train --dataset backend/data/training/caro_dataset.npz
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch.nn import functional as F
from .app.network import CaroNet, MODEL_VERSION, RULES
from .dataset import (DEFAULT_DATA_DIR, export_dataset, file_sha256, fingerprint,
                      generate_dataset, generate_split, load_dataset, write_json)

VALUE_WEIGHT = .5


def dataset(count, seed):
    return generate_split(count, seed).tensors()


def batches(split, batch_size, rng=None):
    result = []
    for n in np.unique(split.sizes):
        ids = np.flatnonzero(split.sizes == n)
        if rng is not None:
            rng.shuffle(ids)
        result.extend((int(n), ids[start:start+batch_size]) for start in range(0,len(ids),batch_size))
    if rng is not None:
        rng.shuffle(result)
    return result


def batch_tensors(split, n, ids):
    canvas = split.boards.shape[1]
    x = torch.from_numpy(split.inputs[ids,:,:n,:n].copy())
    p = torch.from_numpy(split.policy[ids].reshape(-1,canvas,canvas)[:,:n,:n].copy())
    v = torch.from_numpy(split.value[ids])
    return x, p, v


def evaluate(model, split, batch_size=128):
    model.eval()
    count = len(split.inputs)
    policy, value, correct = (np.zeros(count,dtype=np.float64) for _ in range(3))
    with torch.inference_mode():
        for n, ids in batches(split,batch_size):
            x,p,v = batch_tensors(split,n,ids)
            logits,predicted = model(x)
            target = p.flatten(1)
            policy[ids] = -(target*F.log_softmax(logits,dim=1)).sum(1).numpy()
            value[ids] = ((predicted-v)**2).numpy()
            correct[ids] = (logits.argmax(1)==target.argmax(1)).numpy()
    def summary(mask):
        return {"samples": int(mask.sum()), "loss": round(float((policy[mask]+VALUE_WEIGHT*value[mask]).mean()),5),
                "policy_loss": round(float(policy[mask].mean()),5), "value_mse": round(float(value[mask].mean()),5),
                "value_zero_baseline_mse": round(float(np.square(split.value[mask]).mean()),5),
                "top1_accuracy": round(float(correct[mask].mean()),4)}
    return {**summary(np.ones(count,dtype=bool)),
            "by_size": {str(int(n)): summary(split.sizes==n) for n in np.unique(split.sizes)},
            "by_rule": {rule: summary(split.rules==i) for i,rule in enumerate(RULES) if (split.rules==i).any()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples",type=int,default=6000)
    parser.add_argument("--epochs",type=int,default=18)
    parser.add_argument("--batch-size",type=int,default=128)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--dataset",type=Path)
    parser.add_argument("--data-dir",type=Path,default=DEFAULT_DATA_DIR)
    parser.add_argument("--export-only",action="store_true")
    parser.add_argument("--model-dir",type=Path,default=Path(__file__).resolve().parent/"models")
    args = parser.parse_args()
    if args.samples < 100 or args.epochs < 1 or args.batch_size < 1 or args.seed < 0:
        parser.error("Use at least 100 samples, 1 epoch, a positive batch size and non-negative seed.")
    if args.export_only and args.dataset:
        parser.error("--export-only generates new data; omit --dataset.")
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    rng, started = np.random.default_rng(args.seed), perf_counter()
    try:
        if args.dataset:
            dataset_path = args.dataset.resolve()
            bundle = load_dataset(dataset_path)
        else:
            print("Generating legal train/validation/test games (sizes 3..25, all three rules)...",flush=True)
            bundle = generate_dataset(args.samples,args.seed)
            dataset_path = export_dataset(bundle,args.data_dir.resolve())
        print(f"Dataset: {len(bundle.train.inputs)}/{len(bundle.validation.inputs)}/{len(bundle.test.inputs)}; {dataset_path}",flush=True)
    except (OSError,ValueError,KeyError) as exc:
        parser.error(f"Cannot prepare dataset: {exc}")
    if args.export_only:
        print("Export complete; model weights and metrics were not changed.",flush=True)
        return
    model = CaroNet()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=1e-4)
    history,best,best_epoch = [],float("inf"),0
    model_dir = args.model_dir.resolve()
    model_dir.mkdir(parents=True,exist_ok=True)
    checkpoint_path = model_dir/"caro_policy.pt"
    for epoch in range(args.epochs):
        model.train()
        totals = np.zeros(3)
        for n, ids in batches(bundle.train,args.batch_size,rng):
            x,p,v = batch_tensors(bundle.train,n,ids)
            turns = int(rng.integers(4))
            x,p = torch.rot90(x,turns,(-2,-1)),torch.rot90(p,turns,(-2,-1))
            if rng.random() < .5:
                x,p = x.flip(-1),p.flip(-1)
            logits,predicted = model(x)
            policy_loss = -(p.flatten(1)*F.log_softmax(logits,dim=1)).sum(1).mean()
            value_loss = F.mse_loss(predicted,v)
            loss = policy_loss+VALUE_WEIGHT*value_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2)
            optimizer.step()
            totals += np.array([float(loss.detach()),float(policy_loss.detach()),float(value_loss.detach())])*len(ids)
        totals /= len(bundle.train.inputs)
        metrics = evaluate(model,bundle.validation,args.batch_size)
        history.append({"epoch":epoch+1,"loss":round(totals[0],5),"policy_loss":round(totals[1],5),
                        "value_mse":round(totals[2],5),"val_loss":metrics["loss"],
                        "val_policy_loss":metrics["policy_loss"],"val_value_mse":metrics["value_mse"],
                        "top1_accuracy":metrics["top1_accuracy"]})
        if metrics["loss"] < best:
            best,best_epoch = metrics["loss"],epoch+1
            temporary = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(model.state_dict(),temporary)
            temporary.replace(checkpoint_path)
        print(f"Epoch {epoch+1:02d}/{args.epochs}: train={totals[0]:.4f} val={metrics['loss']:.4f} policy-top1={metrics['top1_accuracy']:.1%} value-MSE={metrics['value_mse']:.4f}",flush=True)
    model.load_state_dict(torch.load(checkpoint_path,weights_only=True,map_location="cpu"))
    # The test split is consulted once, after all checkpoint selection has ended.
    test_metrics = evaluate(model,bundle.test,args.batch_size)
    validation_metrics = evaluate(model,bundle.validation,args.batch_size)
    metadata = {"model_version":MODEL_VERSION,"method":"Supervised policy distillation + completed-rollout outcome regression",
        "samples":len(bundle.train.inputs),"validation_samples":len(bundle.validation.inputs),"test_samples":len(bundle.test.inputs),
        "epochs":args.epochs,"seed":args.seed,"training_board_sizes":bundle.metadata["board_sizes"],
        "training_win_lengths":[3,4,5,6],"training_rules":list(RULES),"input_channels":8,"value_loss_weight":VALUE_WEIGHT,
        "seconds":round(perf_counter()-started,1),"trained_at":datetime.now(timezone.utc).isoformat(),
        "best_val_loss":best,"best_epoch":best_epoch,
        "checkpoint_selection":"Lowest validation policy CE + 0.5 * outcome value MSE; test is held out",
        "checkpoint_sha256":file_sha256(checkpoint_path),
        "dataset":{"path":dataset_path.as_posix(),"file_sha256":file_sha256(dataset_path),
                   "fingerprint_sha256":fingerprint(bundle),"generator_seed":bundle.metadata["seed"],
                   "split_strategy":bundle.metadata["split_strategy"],"audit":bundle.metadata["audit"]},
        "history":history,"validation_metrics":validation_metrics,"test_metrics":test_metrics,
        "limitations":bundle.metadata["limitations"]}
    write_json(model_dir/"training.json",metadata)
    print(f"Best checkpoint epoch {best_epoch}; held-out test={test_metrics['loss']:.5f}, top1={test_metrics['top1_accuracy']:.1%}",flush=True)


if __name__ == "__main__":
    main()
