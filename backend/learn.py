"""Learn from saved human games: python -m backend.learn --epochs 3."""
import argparse
import json
from pathlib import Path
from .app.ai import MODEL_DIR
from .app.storage import Storage
from .dataset import DEFAULT_DATA_DIR
from .learning import HumanLearning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database",type=Path,default=DEFAULT_DATA_DIR.parent/"caro.sqlite3")
    parser.add_argument("--model-dir",type=Path,default=MODEL_DIR)
    parser.add_argument("--dataset",type=Path,default=DEFAULT_DATA_DIR/"caro_dataset.npz")
    parser.add_argument("--epochs",type=int,default=3,choices=range(1,9))
    args = parser.parse_args()
    storage = Storage(args.database); storage.backfill_experience()
    learner = HumanLearning(storage,args.model_dir,args.dataset)
    try: learner.prepare(manual=True,epochs=args.epochs)
    except ValueError as exc: parser.error(str(exc))
    learner.run()
    result = learner.snapshot()
    print(json.dumps({"status":result["status"],"message":result["message"],"stats":result["stats"],"report":result["report"]},indent=2,ensure_ascii=True))
    if result["status"] == "failed": raise SystemExit(1)


if __name__ == "__main__": main()
