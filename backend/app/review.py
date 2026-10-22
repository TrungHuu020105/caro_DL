"""Explain provable one-move tactical opportunities and sandbox alternatives."""
import numpy as np
from .game import Game, GameConfig
from .rules import safe_defenses, winning_moves


def replay_prefix(payload,step):
    if not 0 <= step <= len(payload["moves"]):
        raise ValueError("Mốc xem lại nằm ngoài ván cờ.")
    config = {**payload["config"],"mode":"local","time_limit":0}
    game = Game(GameConfig(**config))
    for move in payload["moves"][:step]:
        game.place(move["row"],move["col"])
    return game


def review_game(payload):
    game = replay_prefix(payload,0)
    events = []
    summary = {str(p):{"missed_win":0,"missed_block":0,"double_threat":0} for p in (1,-1)}
    for index,move in enumerate(payload["moves"]):
        board,player = np.asarray(game.board,dtype=np.int8),game.current_player
        wins = winning_moves(board,player,game.config.win_length,game.config.rule)
        threats = winning_moves(board,-player,game.config.win_length,game.config.rule)
        defenses = safe_defenses(board,player,game.config.win_length,game.config.rule,threats) if threats and not wins else []
        played,kind,alternatives = (move["row"],move["col"]),None,[]
        if wins and played not in wins:
            kind,alternatives = "missed_win",wins
        elif not wins and defenses and played not in defenses:
            kind,alternatives = "missed_block",defenses
        elif not wins and threats and not defenses:
            kind = "double_threat"
        if kind:
            summary[str(player)][kind] += 1
            events.append({"number":index+1,"player":player,"kind":kind,"played":list(played),
                           "alternatives":[list(rc) for rc in alternatives],
                           "explanation":{"missed_win":"Có nước thắng ngay nhưng đã chọn ô khác.",
                                          "missed_block":"Đối thủ có nước thắng ngay; các ô đề xuất loại bỏ mọi đe dọa đó.",
                                          "double_threat":"Đối thủ có nhiều ô thắng ngay. Một nước chặn đơn lẻ không cứu được thế cờ."}[kind]})
        game.place(*played)
    return {"game_id":payload["id"],"summary":summary,"events":events,
            "scope":"Chỉ kết luận các cơ hội thắng/chặn ngay theo đúng luật; không gán nhãn sai lầm chiến lược từ một value chưa hiệu chỉnh."}
