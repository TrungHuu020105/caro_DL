"""Neural policy + tactical safety + bounded adversarial search."""
from functools import lru_cache
from pathlib import Path
from time import perf_counter
import json
import math
import random
import threading
import numpy as np
import torch
from .network import CaroNet, encode
from .rules import tactical_scores, winning_line

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
torch.set_num_threads(min(4, torch.get_num_threads()))


class AIEngine:
    def __init__(self):
        self.model = CaroNet().eval()
        self.lock = threading.Lock()
        self.loaded = False
        self.metadata = {}
        checkpoint = MODEL_DIR / "caro_policy.pt"
        if checkpoint.exists():
            self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            self.loaded = True
        meta = MODEL_DIR / "training.json"
        if meta.exists():
            self.metadata = json.loads(meta.read_text(encoding="utf-8"))

    def info(self):
        return {"name": "CaroNet", "architecture": "Residual CNN · policy + value",
                "loaded": self.loaded, "device": "CPU",
                "parameters": sum(p.numel() for p in self.model.parameters()),
                "training": self.metadata,
                "description": "CNN học từ dữ liệu chiến thuật tổng hợp, kết hợp kiểm tra thắng/chặn và tìm kiếm."
                if self.loaded else "Chưa có checkpoint. Đang dùng thuật toán chiến thuật dự phòng."}

    def predict(self, board, player, length):
        if not self.loaded:
            scores = tactical_scores(board, player, length).ravel()
            scores = np.where(scores >= 0, np.log1p(np.maximum(scores, 0)), -1e9)
            probs = np.exp(scores - scores.max())
            return probs / probs.sum(), 0.0
        with self.lock, torch.inference_mode():
            x = torch.from_numpy(encode(board, player, length)).unsqueeze(0)
            policy, value = self.model(x)
            return torch.softmax(policy[0], dim=0).numpy(), float(value[0])

    @staticmethod
    def immediate(board, player, length, rule, candidates):
        result = []
        cells = board.tolist()
        for r, c in candidates:
            cells[r][c] = player
            if winning_line(cells, r, c, length, rule):
                result.append((r, c))
            cells[r][c] = 0
        return result

    @staticmethod
    @lru_cache(maxsize=40000)
    def minimax(flat, player, length, rule):
        """Exact 3x3 play; memoized and independent of CNN calibration."""
        n = math.isqrt(len(flat))
        best = -2.0
        for i, cell in enumerate(flat):
            if cell:
                continue
            next_board = list(flat)
            next_board[i] = player
            rows = [next_board[j:j+n] for j in range(0, n*n, n)]
            if winning_line(rows, i // n, i % n, length, rule):
                value = 1.0
            elif 0 not in next_board:
                value = 0.0
            else:
                value = -.97 * AIEngine.minimax(tuple(next_board), -player, length, rule)
            best = max(best, value)
        return 0.0 if best == -2.0 else best

    def choose(self, cells, player, length, rule="freestyle", difficulty="medium"):
        started = perf_counter()
        board = np.asarray(cells, dtype=np.int8)
        n = len(board)
        available = [tuple(map(int, rc)) for rc in np.argwhere(board == 0)]
        if not available:
            raise ValueError("Bàn cờ đã đầy.")
        probabilities, value = self.predict(board, player, length)
        tactical = tactical_scores(board, player, length)
        # Check every legal square with the exact selected rules, including blocked ends.
        wins = self.immediate(board, player, length, rule, available)
        blocks = self.immediate(board, -player, length, rule, available)
        reason = "neural"
        if wins:
            chosen, reason = max(wins, key=lambda rc: probabilities[rc[0]*n+rc[1]]), "win"
        elif blocks and difficulty != "easy":
            chosen, reason = max(blocks, key=lambda rc: tactical[rc]), "block"
        elif n == 3 and difficulty == "hard":
            def exact(rc):
                flat = board.ravel().tolist()
                flat[rc[0]*n+rc[1]] = player
                return -self.minimax(tuple(flat), -player, length, rule)
            chosen, reason = max(available, key=lambda rc: (exact(rc), probabilities[rc[0]*n+rc[1]])), "search"
        else:
            teacher = np.log1p(np.maximum(tactical, 0))
            teacher /= max(float(teacher.max()), 1)
            neural = probabilities.reshape(n, n).copy()
            neural /= max(float(neural.max()), 1e-9)
            weight = {"easy": .90, "medium": .55, "hard": .35}[difficulty] if self.loaded else 0
            combined = neural * weight + teacher * (1 - weight)
            ranked = sorted(available, key=lambda rc: combined[rc], reverse=True)
            if difficulty == "easy":
                chosen = random.choices(ranked[:min(6, len(ranked))], weights=[1/(i+1) for i in range(min(6, len(ranked)))])[0]
            elif difficulty == "hard" and len(available) < n*n - 2:
                # Bounded two-ply lookahead. Penalize moves that hand the opponent
                # an immediate win and evaluate the opponent's strongest reply.
                best_score, chosen = -float("inf"), ranked[0]
                for r, c in ranked[:12]:
                    board[r, c] = player
                    replies = tactical_scores(board, -player, length)
                    reply_cells = [tuple(map(int, rc)) for rc in np.argwhere(board == 0)]
                    unsafe = bool(self.immediate(board, -player, length, rule, reply_cells))
                    danger = float(np.log1p(max(float(replies.max()), 0))) / 20
                    score = float(combined[r, c]) - .3 * danger - 10 * unsafe
                    board[r, c] = 0
                    if score > best_score:
                        best_score, chosen = score, (r, c)
                reason = "search"
            else:
                chosen = ranked[0]
        suggestions = sorted(available, key=lambda rc: probabilities[rc[0]*n+rc[1]], reverse=True)[:5]
        labels = {"win": "Hoàn thành chuỗi thắng", "block": "Chặn nước thắng của đối thủ",
                  "search": "Tìm kiếm nước đi kết hợp CNN", "neural": "Đánh giá thế cờ bằng CNN"}
        return {"row": chosen[0], "col": chosen[1], "reason": labels[reason],
                "engine": "CaroNet + tactical search" if self.loaded else "Tactical fallback",
                "elapsed_ms": round((perf_counter() - started) * 1000, 1),
                "value": round(value, 4),
                "candidates": [{"row": r, "col": c, "probability": round(float(probabilities[r*n+c]), 5)}
                               for r, c in suggestions]}
