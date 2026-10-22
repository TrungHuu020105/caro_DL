"""Pure game rules shared by the API, search, and tests."""
from functools import lru_cache
import numpy as np

DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def winning_line(board: list[list[int]], row: int, col: int, length: int,
                 rule: str = "freestyle") -> list[list[int]]:
    n, player = len(board), board[row][col]
    if not player:
        return []
    for dr, dc in DIRECTIONS:
        line = [[row, col]]
        ends = []
        for sign in (-1, 1):
            r, c = row + dr * sign, col + dc * sign
            part = []
            while 0 <= r < n and 0 <= c < n and board[r][c] == player:
                part.append([r, c])
                r, c = r + dr * sign, c + dc * sign
            ends.append(0 <= r < n and 0 <= c < n and board[r][c] == -player)
            line = part[::-1] + line if sign == -1 else line + part
        if len(line) >= length and (rule == "freestyle" or len(line) == length):
            if rule != "blocked" or not all(ends):
                return line
    return []


@lru_cache(maxsize=128)
def segments(n: int, length: int) -> np.ndarray:
    lines = []
    for r in range(n):
        for c in range(n):
            for dr, dc in DIRECTIONS:
                if 0 <= r + (length - 1) * dr < n and 0 <= c + (length - 1) * dc < n:
                    lines.append([(r + i * dr) * n + c + i * dc for i in range(length)])
    return np.asarray(lines, dtype=np.int64)


@lru_cache(maxsize=128)
def segment_ends(n: int, length: int) -> np.ndarray:
    result = []
    for line in segments(n, length):
        r, c = divmod(int(line[0]), n)
        er, ec = divmod(int(line[-1]), n)
        dr, dc = (er - r) // (length - 1), (ec - c) // (length - 1)
        result.append([rr * n + cc if 0 <= rr < n and 0 <= cc < n else n*n
                       for rr, cc in ((r-dr, c-dc), (er+dr, ec+dc))])
    return np.asarray(result, dtype=np.int64)


def winning_moves(board: np.ndarray, player: int, length: int,
                  rule: str = "freestyle") -> list[tuple[int, int]]:
    """Find threats vectorially, then verify overlines/blocked ends exactly."""
    n = len(board)
    windows = board.ravel()[segments(n, length)]
    ready = ((windows == player).sum(1) == length-1) & ((windows == 0).sum(1) == 1)
    indexes = segments(n, length)[ready]
    if not len(indexes):
        return []
    candidates = np.unique(indexes[board.ravel()[indexes] == 0])
    cells, result = board.tolist(), []
    for index in candidates:
        r, c = divmod(int(index), n)
        cells[r][c] = player
        if winning_line(cells, r, c, length, rule):
            result.append((r, c))
        cells[r][c] = 0
    return result


def tactical_scores(board: np.ndarray, player: int, length: int,
                    rule: str = "freestyle") -> np.ndarray:
    """Vectorized open-segment teacher; a heuristic, not the neural model."""
    n = board.shape[0]
    flat = board.ravel()
    idx = segments(n, length)
    windows = flat[idx]
    own, other = (windows == player).sum(1), (windows == -player).sum(1)
    weights = np.array([0.0, 2.0, 18.0, 180.0, 20000.0, 2000000.0, 200000000.0])
    # Scale relative to the requested winning length, including 3x3 and 4x4.
    weights[length - 1] = 1e7
    attack = np.where(other == 0, weights[own], 0)
    defend = np.where(own == 0, weights[other] * .96, 0)
    if rule != "freestyle":
        ends = np.append(flat, 0)[segment_ends(n, length)]
        attack = np.where((ends == player).any(1), 0, attack)
        defend = np.where((ends == -player).any(1), 0, defend)
        if rule == "blocked":
            attack = np.where((ends == -player).all(1), 0, attack)
            defend = np.where((ends == player).all(1), 0, defend)
    scores = np.zeros(n * n, dtype=np.float64)
    np.add.at(scores, idx.ravel(), np.repeat(attack + defend, length) * (windows == 0).ravel())
    rr, cc = np.indices((n, n))
    scores += (1 / (1 + np.abs(rr - (n - 1) / 2) + np.abs(cc - (n - 1) / 2))).ravel()
    scores[flat != 0] = -1
    return scores.reshape(n, n)


def safe_defenses(board: np.ndarray, player: int, length: int, rule: str,
                  threats: list[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
    """Moves removing every immediate threat, including closing Caro line ends.

    Under the blocked rule, occupying a shared endpoint can neutralize several
    different winning squares. Counting threats alone cannot prove a forced loss.
    """
    threats = winning_moves(board,-player,length,rule) if threats is None else threats
    if not threats:
        return []
    if rule != "blocked":
        return threats if len(threats)==1 else []
    n = len(board)
    flat,idx = board.ravel(),segments(n,length)
    windows = flat[idx]
    ready = ((windows == -player).sum(1)==length-1) & ((windows==0).sum(1)==1)
    candidates = {r*n+c for r,c in threats}
    for endpoint in segment_ends(n,length)[ready].ravel():
        if endpoint < n*n and flat[endpoint] == 0:
            candidates.add(int(endpoint))
    result = []
    for index in sorted(candidates):
        r,c = divmod(index,n)
        board[r,c] = player
        try:
            if not winning_moves(board,-player,length,rule): result.append((r,c))
        finally:
            board[r,c] = 0
    return result
