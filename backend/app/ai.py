"""Ablatable CNN/tactical engines and time-bounded alpha-beta with learned value."""
from functools import lru_cache
from pathlib import Path
from time import perf_counter
import json
import math
import random
import threading
import hashlib
import io
import pickle
import numpy as np
import torch
from .network import CaroNet, encode
from .rules import safe_defenses, segments, tactical_scores, winning_line, winning_moves

MODEL_DIR = Path(__file__).resolve().parents[1]/"models"
VARIANTS = ("cnn", "heuristic", "hybrid")
torch.set_num_threads(min(4,torch.get_num_threads()))


class SearchTimeout(Exception):
    pass


class AIEngine:
    def __init__(self, model_dir=MODEL_DIR):
        self.model, self.lock = CaroNet().eval(), threading.Lock()
        self.loaded, self.metadata, self.load_error = False, {}, None
        self.checkpoint_sha256 = None
        model_dir = Path(model_dir)
        checkpoint, meta = model_dir/"caro_policy.pt", model_dir/"training.json"
        if checkpoint.exists():
            try:
                data = checkpoint.read_bytes()
                self.model.load_state_dict(torch.load(io.BytesIO(data),map_location="cpu",weights_only=True))
                self.checkpoint_sha256 = hashlib.sha256(data).hexdigest()
                self.loaded = True
            except (RuntimeError,ValueError,OSError,EOFError,TypeError,pickle.UnpicklingError):
                self.load_error = "Checkpoint không tương thích. Hãy huấn luyện lại CaroNet v2."
        if meta.exists():
            try:
                metadata = json.loads(meta.read_text(encoding="utf-8"))
                if self.loaded and metadata.get("checkpoint_sha256") == self.checkpoint_sha256:
                    self.metadata = metadata
                elif self.loaded:
                    self.load_error = "Metadata không khớp checkpoint; đang ẩn số liệu huấn luyện cũ."
            except (ValueError,OSError,AttributeError):
                self.load_error = "Không đọc được metadata huấn luyện."

    def info(self):
        return {"name":"CaroNet","architecture":"Rule-conditioned residual CNN · policy + outcome value",
                "loaded":self.loaded,"device":"CPU","parameters":sum(p.numel() for p in self.model.parameters()),
                "training":self.metadata,"load_error":self.load_error,
                "checkpoint_sha256":self.checkpoint_sha256,
                "description":"CNN 8 kênh, value học từ kết quả ván hợp lệ; tìm kiếm alpha-beta giới hạn thời gian."
                if self.loaded else self.load_error or "Chưa có checkpoint; dùng chiến thuật dự phòng."}

    def predict(self, board, player, length, rule="freestyle"):
        if not self.loaded:
            scores = tactical_scores(board,player,length,rule).ravel()
            logits = np.where(scores >= 0,np.log1p(np.maximum(scores,0)),-1e9)
            probs = np.exp(logits-logits.max())
            return probs/probs.sum(),0.
        with self.lock,torch.inference_mode():
            x = torch.from_numpy(encode(board,player,length,rule)).unsqueeze(0)
            policy,value = self.model(x)
            return torch.softmax(policy[0],dim=0).numpy(),float(value[0])

    @staticmethod
    def immediate(board, player, length, rule, candidates):
        allowed = set(candidates)
        return [rc for rc in winning_moves(board,player,length,rule) if rc in allowed]

    @staticmethod
    @lru_cache(maxsize=40000)
    def minimax(flat, player, length, rule):
        n,best = math.isqrt(len(flat)),-2.
        for i,cell in enumerate(flat):
            if cell:
                continue
            next_board = list(flat); next_board[i] = player
            rows = [next_board[j:j+n] for j in range(0,n*n,n)]
            if winning_line(rows,i//n,i%n,length,rule):
                value = 1.
            elif 0 not in next_board:
                value = 0.
            else:
                value = -.97*AIEngine.minimax(tuple(next_board),-player,length,rule)
            best = max(best,value)
        return 0. if best == -2. else best

    @staticmethod
    def positional_value(board,player,length):
        windows = board.ravel()[segments(len(board),length)]
        own,other = (windows==player).sum(1),(windows==-player).sum(1)
        attack = float(np.where(other==0,4.**own-1,0).sum())
        defend = float(np.where(own==0,4.**other-1,0).sum())
        return float(np.tanh((np.log1p(attack)-np.log1p(defend))/3))

    def search(self,board,player,length,rule,ranked,budget_ms):
        deadline = perf_counter()+budget_ms/1000
        stats = {"nodes":0,"completed_depth":0,"budget_ms":budget_ms,"value_evaluations":0}
        cache,root_scores = {},{}

        def check():
            if perf_counter() >= deadline:
                raise SearchTimeout()

        def features(side):
            check()
            key = (board.tobytes(),side)
            if key not in cache:
                probabilities,value = self.predict(board,side,length,rule)
                if self.loaded:
                    stats["value_evaluations"] += 1
                scores = np.log1p(np.maximum(tactical_scores(board,side,length,rule),0))
                scores /= max(float(scores.max()),1.)
                neural = probabilities.reshape(board.shape)
                neural = neural/max(float(neural.max()),1e-9)
                cache[key] = (.55*neural+.45*scores,
                              .65*value+.35*self.positional_value(board,side,length))
            return cache[key]

        def negamax(side,depth,alpha,beta,ply):
            check(); stats["nodes"] += 1
            available = [tuple(map(int,rc)) for rc in np.argwhere(board==0)]
            if not available:
                return 0.
            if winning_moves(board,side,length,rule):
                return 1.-.003*ply
            blocks = winning_moves(board,-side,length,rule)
            defenses = safe_defenses(board,side,length,rule,blocks) if blocks else []
            if blocks and not defenses:
                return -1.+.003*(ply+1)
            combined,value = features(side)
            if depth <= 0:
                return .9*value
            ordered = defenses or sorted(available,key=lambda rc: combined[rc],reverse=True)[:6]
            best = -2.
            for r,c in ordered:
                board[r,c] = side
                try:
                    score = -negamax(-side,depth-1,-beta,-alpha,ply+1)
                finally:
                    board[r,c] = 0
                best,alpha = max(best,score),max(alpha,score)
                if alpha >= beta:
                    break
            return best

        chosen = ranked[0]
        # A completed shallower iteration remains usable if a deeper one times out.
        for depth in range(1,5):
            iteration,alpha = {},-2.
            try:
                order = sorted(ranked[:10],key=lambda rc: root_scores.get(rc,-2.),reverse=True)
                for r,c in order:
                    check(); board[r,c] = player
                    try:
                        score = -negamax(-player,depth-1,-2.,-alpha,1)
                    finally:
                        board[r,c] = 0
                    iteration[(r,c)] = score
                    alpha = max(alpha,score)
            except SearchTimeout:
                if not root_scores and iteration:
                    root_scores = iteration
                    chosen = max(iteration,key=iteration.get)
                break
            root_scores = iteration
            chosen = max(root_scores,key=root_scores.get)
            stats["completed_depth"] = depth
            if root_scores[chosen] > .98:
                break
        return chosen,root_scores,stats

    def choose(self,cells,player,length,rule="freestyle",difficulty="medium",variant="hybrid",budget_ms=250):
        started = perf_counter()
        if variant not in VARIANTS:
            raise ValueError("Engine không hợp lệ.")
        if variant == "cnn" and not self.loaded:
            raise ValueError("Chưa tải được checkpoint CNN.")
        board = np.array(cells,dtype=np.int8,copy=True)
        n = len(board)
        available = [tuple(map(int,rc)) for rc in np.argwhere(board==0)]
        if not available:
            raise ValueError("Bàn cờ đã đầy.")
        use_neural = self.loaded and variant != "heuristic"
        probabilities,value = self.predict(board,player,length,rule) if use_neural else (None,0.)
        raw = tactical_scores(board,player,length,rule)
        tactical = np.log1p(np.maximum(raw,0))
        tactical /= max(float(tactical.max()),1.)
        neural = probabilities.reshape(n,n) if use_neural else np.zeros((n,n))
        neural = neural/max(float(neural.max()),1e-9)
        weight = 1. if variant == "cnn" else {"easy":.9,"medium":.55,"hard":.45}[difficulty] if use_neural else 0.
        combined = weight*neural+(1-weight)*tactical
        ranked = sorted(available,key=lambda rc: combined[rc],reverse=True)
        chosen,reason = ranked[0],"cnn" if variant == "cnn" else "hybrid" if use_neural else "heuristic"
        root_scores,search = {},{"nodes":0,"completed_depth":0,"budget_ms":budget_ms,"value_evaluations":0}
        if variant != "cnn":
            wins,blocks = winning_moves(board,player,length,rule),winning_moves(board,-player,length,rule)
            if wins:
                chosen,reason = max(wins,key=lambda rc: combined[rc]),"win"
            elif blocks and (difficulty != "easy" or variant == "heuristic"):
                defenses = safe_defenses(board,player,length,rule,blocks)
                chosen,reason = max(defenses or blocks,key=lambda rc: combined[rc]),"block" if defenses else "forced_loss"
            elif variant == "hybrid" and n == 3 and difficulty == "hard":
                def exact(rc):
                    flat = board.ravel().tolist(); flat[rc[0]*n+rc[1]] = player
                    return -self.minimax(tuple(flat),-player,length,rule)
                root_scores = {rc:exact(rc) for rc in available}
                chosen,reason = max(available,key=lambda rc:(root_scores[rc],combined[rc])),"minimax"
            elif variant == "hybrid" and difficulty == "hard" and len(available) < n*n-2:
                chosen,root_scores,search = self.search(board,player,length,rule,ranked,budget_ms)
                reason = "search"
            elif difficulty == "easy" and variant == "hybrid":
                chosen = random.choices(ranked[:6],weights=[1/(i+1) for i in range(min(6,len(ranked)))])[0]
        labels = {"win":"Hoàn thành chuỗi thắng theo luật đang chọn", "block":"Chặn nước thắng của đối thủ",
                  "forced_loss":"Không có nước loại bỏ mọi đe dọa thắng ngay; chặn một hướng",
                  "minimax":"Minimax chính xác trên bàn 3×3", "search":"Alpha-beta dùng policy và value từ kết quả ván đấu" if use_neural else "Alpha-beta dùng đánh giá chiến thuật dự phòng",
                  "cnn":"Chọn trực tiếp từ policy CNN", "hybrid":"Kết hợp policy CNN và điểm chiến thuật",
                  "heuristic":"Chọn theo chiến thuật; không dùng mạng neural"}
        shown = [chosen]+[rc for rc in ranked if rc != chosen][:4]
        heuristic_probs = np.where(raw>=0,np.maximum(raw,0)+1,0)
        heuristic_probs /= heuristic_probs.sum()
        candidates = [{"row":r,"col":c,"probability":round(float(probabilities[r*n+c] if use_neural else heuristic_probs[r,c]),6),
                       "neural_score":round(float(neural[r,c]),4) if use_neural else None,
                       "tactical_score":round(float(tactical[r,c]),4),"combined_score":round(float(combined[r,c]),4),
                       "search_score":round(float(root_scores[(r,c)]),4) if (r,c) in root_scores else None,
                       "selected":(r,c)==chosen} for r,c in shown]
        return {"row":chosen[0],"col":chosen[1],"reason":labels[reason],"decision_source":reason,
                "engine":variant if use_neural or variant == "heuristic" else "heuristic-fallback",
                "elapsed_ms":round((perf_counter()-started)*1000,1),"value":round(value,4),
                "value_used":search["value_evaluations"]>0,"weights":{"neural":weight,"tactical":1-weight},
                "search":search,"position":{"player":player,"move_number":int((board!=0).sum())+1},"candidates":candidates}
