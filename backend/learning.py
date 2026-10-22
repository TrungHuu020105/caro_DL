"""Incremental supervised learning from human games, replay and promotion gates."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import hashlib
import math
from pathlib import Path
import threading
from time import perf_counter, time
import uuid
import numpy as np
import torch
from torch.nn import functional as F
from .app.ai import AIEngine, MODEL_DIR
from .app.game import Game, GameConfig
from .app.network import CaroNet
from .dataset import DEFAULT_DATA_DIR, file_sha256, load_dataset, write_json
from .learning_data import (build_human_data, concatenate, exclusion_keys,
                            export_human_data, pad, select)
from .model_store import ModelBusyError, model_writer
from .train import VALUE_WEIGHT, batch_tensors, batches, evaluate

DEFAULT_SETTINGS = {"enabled":True,"min_new_games":5,"epochs":3}
MIN_HUMAN_POSITIONS = 8


def compare_models(candidate,incumbent,configs,pairs=2,budget_ms=40,seed=2026,progress=None,cancel=None):
    """Compare both pure CNN and gameplay hybrid on paired, shared openings."""
    rng,matches = np.random.default_rng(seed),[]
    total = len(configs)*pairs*2*2
    for n,k,rule in configs:
        for variant in ("cnn","hybrid"):
            for pair in range(pairs):
                indexes = rng.choice(n*n,2 if n==3 else 4,replace=False)
                # n=3 uses K=3; otherwise four opening plies cannot have K>=3
                # for one player (only two stones each).
                opening = [list(divmod(int(index),n)) for index in indexes]
                for candidate_side in (1,-1):
                    game = Game(GameConfig(size=n,win_length=k,rule=rule,mode="spectate"))
                    for rc in opening: game.place(*rc)
                    while game.status == "playing":
                        if cancel is not None and cancel.is_set(): raise InterruptedError("Tác vụ học đã dừng cùng máy chủ.")
                        engine = candidate if game.current_player == candidate_side else incumbent
                        move = engine.choose(game.board,game.current_player,k,rule,"hard",variant,budget_ms)
                        game.place(move["row"],move["col"])
                    result = "draw" if game.winner is None else "win" if game.winner==candidate_side else "loss"
                    matches.append({"size":n,"win_length":k,"rule":rule,"variant":variant,"pair":pair+1,
                                    "candidate_side":candidate_side,"result":result,"opening":opening,
                                    "moves":[[m["row"],m["col"]] for m in game.moves]})
                    if progress: progress(len(matches),total)
    def summary(rows):
        wins = sum(m["result"]=="win" for m in rows)
        draws = sum(m["result"]=="draw" for m in rows)
        return {"games":len(rows),"wins":wins,"draws":draws,"losses":len(rows)-wins-draws,
                "score_rate":round((wins+.5*draws)/len(rows),4)}
    return {**summary(matches),"by_variant":{name:summary([m for m in matches if m["variant"]==name]) for name in ("cnn","hybrid")},
            "seed":seed,"pairs":pairs,"budget_ms":budget_ms,"matches":matches,
            "candidate_sha256":candidate.checkpoint_sha256,"incumbent_sha256":incumbent.checkpoint_sha256}


def promotion_decision(baseline_base,candidate_base,baseline_human,candidate_human,arena):
    reasons = []
    for key,label in (("loss","loss tổng"),("policy_loss","policy CE"),("value_mse","value MSE")):
        before,after = baseline_base[key],candidate_base[key]
        if not math.isfinite(after) or after > before*1.02+.01:
            reasons.append(f"{label} trên validation nền tăng quá ngưỡng 2% + 0,01.")
    if baseline_human is not None:
        if not math.isfinite(candidate_human["loss"]) or candidate_human["loss"] > baseline_human["loss"]*1.10+.02:
            reasons.append("Loss trên các ván người chơi giữ riêng tăng quá ngưỡng 10% + 0,02.")
    if arena["score_rate"] < .55:
        reasons.append("Điểm đấu với model cũ chưa đạt 55% (hòa tính nửa điểm).")
    if any(row["score_rate"] < .5 for row in arena["by_variant"].values()):
        reasons.append("Một trong hai chế độ CNN/lai có điểm đấu dưới 50%.")
    return {"accepted":not reasons,"reasons":reasons,
            "thresholds":{"arena_score_min":.55,"each_variant_score_min":.5,"base_relative_tolerance":.02,
                          "base_absolute_tolerance":.01,"human_relative_tolerance":.10,"human_absolute_tolerance":.02},
            "limitation":"Cổng kiểm tra thực nghiệm với mẫu nhỏ, không bảo đảm mạnh hơn trong mọi ván."}


def publish_candidate(model_dir,candidate_dir,metadata,expected_parent):
    """Caller holds model_writer. Keep an immutable parent backup and restore on I/O failure."""
    model_dir,candidate_dir = Path(model_dir),Path(candidate_dir)
    checkpoint,meta = model_dir/"caro_policy.pt",model_dir/"training.json"
    if file_sha256(checkpoint) != expected_parent:
        raise RuntimeError("Checkpoint gốc đã thay đổi; không ghi đè bằng ứng viên đang học.")
    old_weights,old_metadata = checkpoint.read_bytes(),meta.read_bytes()
    archive = model_dir/"learning"/"versions"/expected_parent
    archive.mkdir(parents=True,exist_ok=True)
    if not (archive/"caro_policy.pt").exists(): (archive/"caro_policy.pt").write_bytes(old_weights)
    if not (archive/"training.json").exists(): (archive/"training.json").write_bytes(old_metadata)
    temporary = checkpoint.with_name("caro_policy.pt.learning.tmp")
    try:
        temporary.write_bytes((candidate_dir/"caro_policy.pt").read_bytes())
        temporary.replace(checkpoint)
        write_json(meta,metadata)
        activated = AIEngine(model_dir)
        if not activated.loaded or activated.checkpoint_sha256 != metadata["checkpoint_sha256"] or not activated.metadata:
            raise RuntimeError("Không kiểm chứng được checkpoint mới sau khi lưu.")
        return activated
    except Exception:
        temporary.write_bytes(old_weights); temporary.replace(checkpoint)
        restore = meta.with_name("training.json.restore.tmp")
        restore.write_bytes(old_metadata); restore.replace(meta)
        raise


class HumanLearning:
    def __init__(self,storage,model_dir=MODEL_DIR,data_path=None,directory=None):
        self.storage,self.model_dir = storage,Path(model_dir)
        self.data_path = Path(data_path or DEFAULT_DATA_DIR/"caro_dataset.npz")
        self.directory = Path(directory or self.model_dir/"learning")
        self.path,self.lock,self.cancel = self.directory/"state.json",threading.RLock(),threading.Event()
        self._bundle,self._base_keys,self._base_sha = None,None,None
        self.state = {"status":"idle","phase":"idle","message":"Đang tích lũy các ván người chơi.",
                      "completed":0,"total":0,"settings":dict(DEFAULT_SETTINGS),"report":None,"error":None,
                      "last_attempt_total":-1,"last_attempt_pending":-1,"retry_after":0}
        if self.path.exists():
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                self.state.update({key:value for key,value in saved.items() if key in self.state})
                self.state["settings"] = {**DEFAULT_SETTINGS,**self.state["settings"]}
                if self.state["status"] == "running":
                    self.state.update(status="idle",phase="interrupted",last_attempt_total=-1,
                                      message="Lần học trước bị ngắt. Dữ liệu vẫn được giữ để thử lại.")
            except (ValueError,OSError,TypeError):
                pass
        meta = self.model_dir/"training.json"
        if meta.exists():
            try:
                learned = json.loads(meta.read_text(encoding="utf-8")).get("human_learning",{})
                if learned.get("run_id") and learned.get("source_game_hashes"):
                    self.storage.mark_experience_used(learned["source_game_hashes"],learned["run_id"])
            except (ValueError,OSError,TypeError):
                pass

    def snapshot(self):
        with self.lock: result = deepcopy(self.state)
        result["stats"] = self.storage.experience_stats()
        result["min_positions"] = MIN_HUMAN_POSITIONS
        return result

    def _update(self,**changes):
        with self.lock:
            self.state.update(changes)
            write_json(self.path,self.state)

    def configure(self,changes):
        with self.lock:
            if self.state["status"] == "running": raise ValueError("Chờ lần học đang chạy hoàn thành rồi đổi thiết lập.")
            settings = {**self.state["settings"],**changes}
            if not (type(settings["enabled"]) is bool and 3 <= settings["min_new_games"] <= 50 and 1 <= settings["epochs"] <= 8):
                raise ValueError("Thiết lập học không hợp lệ.")
            self._update(settings=settings)
        return self.snapshot()

    def ready(self):
        stats = self.storage.experience_stats()
        with self.lock:
            return (self.state["settings"]["enabled"] and self.state["status"] != "running"
                    and stats["pending_games"] >= self.state["settings"]["min_new_games"]
                    and (stats["total_games"] != self.state["last_attempt_total"] or stats["pending_games"] != self.state["last_attempt_pending"])
                    and time() >= self.state["retry_after"])

    def prepare(self,manual=False,epochs=None):
        stats = self.storage.experience_stats()
        with self.lock:
            if self.state["status"] == "running": raise ValueError("Một lần học từ người chơi đang chạy.")
            if not stats["pending_games"]: raise ValueError("Chưa có ván hợp lệ mới để học. Hãy hoàn thành một ván trước.")
            if not manual and not self.ready(): return False
            self.job_epochs = epochs or self.state["settings"]["epochs"]
            if not 1 <= self.job_epochs <= 8: raise ValueError("Số epoch phải từ 1 đến 8.")
            self.cancel.clear()
            self._update(status="running",phase="preparing",completed=0,total=0,error=None,
                         last_attempt_total=stats["total_games"],last_attempt_pending=stats["pending_games"],message="Đang lọc dữ liệu và tạo model ứng viên.")
        return True

    def _check(self):
        if self.cancel.is_set(): raise InterruptedError("Lần học dừng khi máy chủ tắt; các ván chưa xử lý vẫn được giữ.")

    def _load_base(self,incumbent):
        expected = incumbent.metadata.get("dataset",{}).get("file_sha256")
        actual = file_sha256(self.data_path)
        if actual != expected: raise ValueError("Dataset nền không khớp checkpoint. Hãy dùng đúng dataset của lần huấn luyện nền.")
        if self._base_sha != actual:
            self._bundle = load_dataset(self.data_path)
            self._base_keys = exclusion_keys(self._bundle)
            self._base_sha = actual
        if file_sha256(self.data_path) != actual: raise ValueError("Dataset nền thay đổi trong lúc đọc; hãy chạy lại.")
        return self._bundle

    def run(self,incumbent=None,on_promote=None):
        started,run_id = perf_counter(),datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid.uuid4().hex[:8]
        report = {"run_id":run_id,"started_at":datetime.now(timezone.utc).isoformat(),"decision":"running"}
        run_dir = self.directory/"runs"/run_id
        try:
            with model_writer(self.model_dir):
                incumbent = incumbent or AIEngine(self.model_dir)
                if not incumbent.loaded or not incumbent.metadata: raise ValueError("Cần checkpoint và metadata nền hợp lệ để học bổ sung.")
                if file_sha256(self.model_dir/"caro_policy.pt") != incumbent.checkpoint_sha256:
                    raise ValueError("Model trên đĩa đã đổi. Khởi động lại backend để nạp đúng model trước khi học.")
                report["parent_sha256"] = incumbent.checkpoint_sha256
                base = self._load_base(incumbent)
                self._check()
                records = self.storage.experience_snapshot()
                captured = [record["game_hash"] for record in records]
                human_train,human_val,data_info = build_human_data(records,self.storage.experience_position_splits(),self._base_keys)
                report.update(data=data_info,source_game_hashes=captured,epochs=self.job_epochs,seed=2026)
                if human_train is None or len(human_train.inputs) < MIN_HUMAN_POSITIONS:
                    report.update(decision="insufficient_data",message=f"Cần ít nhất {MIN_HUMAN_POSITIONS} vị trí train mới sau lọc trùng; hãy chơi thêm ván.")
                    self._finish(report,run_dir,started)
                    return
                rng = np.random.default_rng(2026)
                replay_ids = rng.permutation(len(base.train.inputs))[:min(1024,4*len(human_train.inputs))]
                mixed = concatenate(pad(select(base.train,replay_ids)),human_train)
                validation = concatenate(pad(base.validation),human_val)
                data_hash = export_human_data(run_dir/"experience.npz",human_train,human_val,replay_ids,
                                             {"parent_sha256":incumbent.checkpoint_sha256,"base_sha256":self._base_sha,"data":data_info})
                report.update(replay_positions=len(replay_ids),training_positions=len(mixed.inputs),human_dataset_fingerprint=data_hash)
                self._update(phase="training",completed=0,total=self.job_epochs,message="Đang học từ các ván mới và ôn lại dữ liệu cũ.")
                candidate_model = CaroNet()
                candidate_model.load_state_dict(incumbent.model.state_dict())
                optimizer = torch.optim.AdamW(candidate_model.parameters(),lr=.0002,weight_decay=1e-4)
                history,best,best_epoch = [],float("inf"),0
                checkpoint = run_dir/"caro_policy.pt"
                for epoch in range(self.job_epochs):
                    self._check(); candidate_model.train(); total_loss=0.
                    for n,ids in batches(mixed,128,rng):
                        self._check()
                        x,p,v = batch_tensors(mixed,n,ids)
                        turns = int(rng.integers(4)); x,p = torch.rot90(x,turns,(-2,-1)),torch.rot90(p,turns,(-2,-1))
                        if rng.random()<.5: x,p = x.flip(-1),p.flip(-1)
                        logits,value = candidate_model(x)
                        loss = -(p.flatten(1)*F.log_softmax(logits,dim=1)).sum(1).mean()+VALUE_WEIGHT*F.mse_loss(value,v)
                        if not math.isfinite(float(loss.detach())): raise RuntimeError("Loss huấn luyện không hữu hạn.")
                        optimizer.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(candidate_model.parameters(),1.5); optimizer.step()
                        total_loss += float(loss.detach())*len(ids)
                    metrics = evaluate(candidate_model,validation)
                    history.append({"epoch":epoch+1,"loss":round(total_loss/len(mixed.inputs),5),"val_loss":metrics["loss"],
                                    "top1_accuracy":metrics["top1_accuracy"],"val_value_mse":metrics["value_mse"]})
                    if metrics["loss"] < best:
                        best,best_epoch = metrics["loss"],epoch+1
                        torch.save(candidate_model.state_dict(),checkpoint)
                    self._update(completed=epoch+1,message=f"Đã học epoch {epoch+1}/{self.job_epochs}.")
                candidate = AIEngine(run_dir)
                if not candidate.loaded: raise RuntimeError("Không nạp được model ứng viên.")
                self._update(phase="evaluating",completed=0,total=1,message="Đang so sánh loss trên dữ liệu kiểm định giữ riêng.")
                baseline_base,new_base = evaluate(incumbent.model,base.validation),evaluate(candidate.model,base.validation)
                baseline_human = evaluate(incumbent.model,human_val) if human_val is not None else None
                new_human = evaluate(candidate.model,human_val) if human_val is not None else None
                self._check()
                common = sorted({(g["size"],g["win_length"],g["rule"]) for g in records},
                                key=lambda config:(-sum((g["size"],g["win_length"],g["rule"])==config for g in records),*config))
                n,k,rule = common[0]
                configs = [(min(n,15),k,rule)]
                configs.append((9,5,"freestyle") if configs[0] != (9,5,"freestyle") else (5,5,"exact"))
                self._update(phase="arena",completed=0,total=16,message="Đang cho ứng viên đấu model cũ, đổi bên X/O.")
                arena_seed = int(hashlib.sha256((incumbent.checkpoint_sha256+data_hash).encode()).hexdigest()[:8],16)
                arena = compare_models(candidate,incumbent,configs,seed=arena_seed,
                    progress=lambda done,total:self._update(completed=done,total=total),cancel=self.cancel)
                gate = promotion_decision(baseline_base,new_base,baseline_human,new_human,arena)
                report.update(candidate_sha256=candidate.checkpoint_sha256,history=history,best_epoch=best_epoch,
                              baseline_validation=baseline_base,candidate_validation=new_base,
                              baseline_human_validation=baseline_human,candidate_human_validation=new_human,
                              arena=arena,gate=gate)
                self._check()
                if file_sha256(self.data_path) != self._base_sha: raise ValueError("Dataset nền đã thay đổi; giữ nguyên model cũ.")
                metadata = deepcopy(incumbent.metadata)
                metadata.update(method="Incremental supervised replay from human games + original training data",epochs=self.job_epochs,
                    samples=len(mixed.inputs),validation_samples=len(validation.inputs),best_epoch=best_epoch,best_val_loss=best,
                    history=history,checkpoint_sha256=candidate.checkpoint_sha256,trained_at=datetime.now(timezone.utc).isoformat(),
                    validation_metrics=evaluate(candidate.model,validation),base_validation_metrics=new_base,human_validation_metrics=new_human,
                    checkpoint_selection="Lowest combined validation loss; promotion also requires validation and paired arena gates",
                    parent_checkpoint_sha256=incumbent.checkpoint_sha256,
                    human_learning={"run_id":run_id,"source_game_hashes":captured,"data":data_info,
                                    "human_dataset_fingerprint":data_hash,"replay_positions":len(replay_ids),"arena_score":arena["score_rate"]})
                # Never use the original test split to select epochs or decide promotion.
                if gate["accepted"]:
                    self._update(phase="publishing",message="Ứng viên đã qua kiểm tra; đang lưu phiên bản và nạp model mới.")
                    metadata["test_metrics"] = evaluate(candidate.model,base.test)
                    metadata["seconds"] = round(perf_counter()-started,1)
                    write_json(run_dir/"training.json",metadata)
                    self._check()
                    activated = publish_candidate(self.model_dir,run_dir,metadata,incumbent.checkpoint_sha256)
                    if on_promote: on_promote(activated)
                    report.update(decision="promoted",message="Đã học từ lịch sử và áp dụng model mới cho các lượt AI tiếp theo.")
                else:
                    metadata.pop("test_metrics",None)
                    metadata["candidate_not_activated"] = True
                    write_json(run_dir/"training.json",metadata)
                    report.update(decision="rejected",message="Đã huấn luyện và đánh giá ứng viên; giữ model cũ vì chưa qua cổng kiểm tra.")
                self.storage.mark_experience_used(captured,run_id)
                self._finish(report,run_dir,started)
        except ModelBusyError as exc:
            self._update(status="idle",phase="waiting",last_attempt_total=-1,retry_after=time()+30,message=str(exc),error=None)
        except Exception as exc:
            report.update(decision="interrupted" if isinstance(exc,InterruptedError) else "failed",message=str(exc),error=str(exc))
            self._finish(report,run_dir,started,failed=True)

    def _finish(self,report,run_dir,started,failed=False):
        report.update(seconds=round(perf_counter()-started,1),finished_at=datetime.now(timezone.utc).isoformat())
        write_json(run_dir/"report.json",report)
        self._update(status="failed" if failed else "completed",phase="finished",report=report,
                     message=report["message"],error=report.get("error"),retry_after=0)
