"""Paired, seeded tournaments for CNN-only, tactical-only and hybrid engines.

python -m backend.benchmark --pairs 2 --sizes 5 9 15 --rules freestyle exact blocked
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import threading
from time import perf_counter
import numpy as np
from .app.ai import AIEngine, MODEL_DIR, VARIANTS
from .app.game import Game, GameConfig
from .app.network import RULES
from .dataset import file_sha256, write_json

REPORT_PATH = MODEL_DIR/"benchmark.json"


def run_benchmark(engine, pairs=1, sizes=(5,9,15), rules=RULES, variants=VARIANTS,
                  budget_ms=80, seed=2026, progress=None, cancel=None):
    if not (1 <= pairs <= 5 and 2 <= len(set(variants)) <= 3 and set(variants) <= set(VARIANTS)
            and sizes and all(3 <= n <= 25 for n in sizes) and rules and set(rules) <= set(RULES)
            and 10 <= budget_ms <= 1000 and seed >= 0):
        raise ValueError("Invalid tournament configuration.")
    if "cnn" in variants and not engine.loaded:
        raise ValueError("A loaded CNN checkpoint is required for this experiment.")
    rng = np.random.default_rng(seed)
    matches, timings = [], {name:[] for name in variants}
    total = len(list(combinations(variants,2)))*len(sizes)*len(rules)*pairs*2
    started = perf_counter()
    for n in sizes:
        for rule in rules:
            k = min(n,5)
            for left,right in combinations(variants,2):
                for pair in range(pairs):
                    # Four nonterminal random opening plies (two on 3x3); both
                    # color assignments receive exactly the same opening.
                    indexes = rng.choice(n*n,2 if n==3 else 4,replace=False)
                    opening = [list(divmod(int(index),n)) for index in indexes]
                    # K=3 on 3x3 has only two opening plies, so it cannot have won.
                    for x_engine,o_engine in ((left,right),(right,left)):
                        game = Game(GameConfig(size=n,win_length=k,rule=rule,mode="spectate"))
                        for r,c in opening:
                            game.place(r,c)
                        sides = {1:x_engine,-1:o_engine}
                        game_start = perf_counter()
                        while game.status == "playing":
                            if cancel is not None and cancel.is_set():
                                raise InterruptedError("Experiment stopped during server shutdown.")
                            name = sides[game.current_player]
                            analysis = engine.choose(game.board,game.current_player,k,rule,"hard",name,budget_ms)
                            timings[name].append(analysis["elapsed_ms"])
                            game.place(analysis["row"],analysis["col"])
                        matches.append({"size":n,"win_length":k,"rule":rule,"pair":pair+1,
                                        "x_engine":x_engine,"o_engine":o_engine,"winner":game.winner or 0,
                                        "winner_engine":sides[game.winner] if game.winner else None,
                                        "opening":opening,"moves":[[m["row"],m["col"]] for m in game.moves],
                                        "seconds":round(perf_counter()-game_start,3)})
                        if progress:
                            progress(len(matches),total)
    def standings(items):
        rows = []
        for name in variants:
            played = [m for m in items if name in (m["x_engine"],m["o_engine"])]
            wins = sum(m["winner_engine"]==name for m in played)
            draws = sum(m["winner"]==0 for m in played)
            rows.append({"engine":name,"games":len(played),"wins":wins,"draws":draws,
                         "losses":len(played)-wins-draws,
                         "score_rate":round((wins+.5*draws)/len(played),4) if played else 0})
        return rows
    summary = standings(matches)
    for row in summary:
        values = timings[row["engine"]]
        row.update({"mean_ms":round(float(np.mean(values)),2),"p95_ms":round(float(np.percentile(values,95)),2),
                    "moves":len(values)})
    return {"created_at":datetime.now(timezone.utc).isoformat(),"seed":seed,"pairs":pairs,
            "sizes":list(sizes),"rules":list(rules),"variants":list(variants),"budget_ms":budget_ms,
            "total_games":len(matches),"seconds":round(perf_counter()-started,1),
            "checkpoint_sha256":engine.checkpoint_sha256,
            "dataset_fingerprint":engine.metadata.get("dataset",{}).get("fingerprint_sha256"),
            "method":"Identical seeded openings in color-swapped pairs; hard difficulty; shared maximum search budget, actual latency recorded",
            "notes":"Small local sample, not an Elo estimate. CNN uses policy argmax only; heuristic uses rules/tactics only; hybrid includes value and bounded search. Time budgets are best-effort at node boundaries.",
            "summary":summary,"by_configuration":[{"size":n,"rule":rule,"summary":standings([m for m in matches if m['size']==n and m['rule']==rule])}
                                                       for n in sizes for rule in rules],"matches":matches}


class ExperimentRunner:
    def __init__(self,path=REPORT_PATH):
        self.path,self.lock,self.cancel = Path(path),threading.Lock(),threading.Event()
        self.state = {"status":"idle","completed":0,"total":0,"report":None,"error":None}
        if self.path.exists():
            try:
                self.state.update(status="completed",report=json.loads(self.path.read_text(encoding="utf-8")))
            except (OSError,ValueError):
                pass

    def snapshot(self):
        with self.lock:
            return deepcopy(self.state)

    def prepare(self,config):
        with self.lock:
            if self.state["status"] == "running":
                raise ValueError("Một thí nghiệm đang chạy. Hãy chờ hoàn thành.")
            total = len(list(combinations(config["variants"],2)))*len(config["sizes"])*len(config["rules"])*config["pairs"]*2
            self.state.update(status="running",completed=0,total=total,error=None)
            self.cancel.clear()

    def run(self,engine,config):
        def progress(done,total):
            with self.lock:
                self.state.update(completed=done,total=total)
        try:
            report = run_benchmark(engine,**config,progress=progress,cancel=self.cancel)
            write_json(self.path,report)
            with self.lock:
                self.state.update(status="completed",report=report)
        except Exception as exc:
            with self.lock:
                self.state.update(status="failed",error=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs",type=int,default=1)
    parser.add_argument("--sizes",type=int,nargs="+",default=[5,9,15])
    parser.add_argument("--rules",nargs="+",choices=RULES,default=list(RULES))
    parser.add_argument("--budget-ms",type=int,default=80)
    parser.add_argument("--seed",type=int,default=2026)
    parser.add_argument("--output",type=Path,default=REPORT_PATH)
    args = parser.parse_args()
    engine = AIEngine()
    if not engine.loaded:
        parser.error("Train a v2 checkpoint first.")
    if engine.metadata.get("checkpoint_sha256") != file_sha256(MODEL_DIR/"caro_policy.pt"):
        parser.error("Checkpoint and training metadata differ; finish/re-run training first.")
    try:
        report = run_benchmark(engine,pairs=args.pairs,sizes=args.sizes,rules=args.rules,budget_ms=args.budget_ms,seed=args.seed,
                               progress=lambda done,total: print(f"Games {done}/{total}",flush=True))
    except ValueError as exc:
        parser.error(str(exc))
    write_json(args.output,report)
    print(json.dumps(report["summary"],indent=2),flush=True)


if __name__ == "__main__":
    main()
