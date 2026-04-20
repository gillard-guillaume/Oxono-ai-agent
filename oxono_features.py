from __future__ import annotations

from typing import List
from oxono import Game, State


def _window_score(count: int) -> float:
    if count >= 4:
        return 1.0
    if count == 3:
        return 0.35
    if count == 2:
        return 0.08
    if count == 1:
        return 0.01
    return 0.0


def _count_windows(state: State, player: int) -> List[float]:
    board = state.board
    opp = 1 - player

    my_color_2 = my_color_3 = my_color_4 = 0
    opp_color_2 = opp_color_3 = opp_color_4 = 0
    my_x_2 = my_x_3 = my_x_4 = 0
    opp_x_2 = opp_x_3 = opp_x_4 = 0
    my_o_2 = my_o_3 = my_o_4 = 0
    opp_o_2 = opp_o_3 = opp_o_4 = 0

    windows = []
    for r in range(6):
        for c in range(3):
            windows.append([board[r][c + i] for i in range(4)])
    for c in range(6):
        for r in range(3):
            windows.append([board[r + i][c] for i in range(4)])

    for window in windows:
        empty = sum(cell is None for cell in window)

        my_color = sum(cell is not None and cell[1] == player for cell in window)
        opp_color = sum(cell is not None and cell[1] == opp for cell in window)
        my_x = sum(cell is not None and cell[1] == player and cell[0] == 'x' for cell in window)
        opp_x = sum(cell is not None and cell[1] == opp and cell[0] == 'x' for cell in window)
        my_o = sum(cell is not None and cell[1] == player and cell[0] == 'o' for cell in window)
        opp_o = sum(cell is not None and cell[1] == opp and cell[0] == 'o' for cell in window)

        if opp_color == 0:
            if my_color == 2 and empty == 2:
                my_color_2 += 1
            elif my_color == 3 and empty == 1:
                my_color_3 += 1
            elif my_color == 4:
                my_color_4 += 1
        if my_color == 0:
            if opp_color == 2 and empty == 2:
                opp_color_2 += 1
            elif opp_color == 3 and empty == 1:
                opp_color_3 += 1
            elif opp_color == 4:
                opp_color_4 += 1

        if opp_x == 0:
            if my_x == 2 and empty == 2:
                my_x_2 += 1
            elif my_x == 3 and empty == 1:
                my_x_3 += 1
            elif my_x == 4:
                my_x_4 += 1
        if my_x == 0:
            if opp_x == 2 and empty == 2:
                opp_x_2 += 1
            elif opp_x == 3 and empty == 1:
                opp_x_3 += 1
            elif opp_x == 4:
                opp_x_4 += 1

        if opp_o == 0:
            if my_o == 2 and empty == 2:
                my_o_2 += 1
            elif my_o == 3 and empty == 1:
                my_o_3 += 1
            elif my_o == 4:
                my_o_4 += 1
        if my_o == 0:
            if opp_o == 2 and empty == 2:
                opp_o_2 += 1
            elif opp_o == 3 and empty == 1:
                opp_o_3 += 1
            elif opp_o == 4:
                opp_o_4 += 1

    return [
        my_color_2, my_color_3, my_color_4,
        opp_color_2, opp_color_3, opp_color_4,
        my_x_2, my_x_3, my_x_4,
        opp_x_2, opp_x_3, opp_x_4,
        my_o_2, my_o_3, my_o_4,
        opp_o_2, opp_o_3, opp_o_4,
    ]


def count_immediate_wins(state: State, player: int) -> int:
    current = state.current_player
    if current != player:
        test_state = state.copy()
        test_state.current_player = player
    else:
        test_state = state

    wins = 0
    for action in Game.actions(test_state):
        next_state = test_state.copy()
        Game.apply(next_state, action)
        if Game.is_terminal(next_state) and Game.utility(next_state, player) == 1:
            wins += 1
    return wins


def extract_features(state: State, player: int) -> List[float]:
    window_counts = _count_windows(state, player)
    opp = 1 - player

    current_player_flag = 1.0 if state.current_player == player else 0.0

    my_actions = len(Game.actions(state if state.current_player == player else _state_with_player(state, player)))
    opp_actions = len(Game.actions(state if state.current_player == opp else _state_with_player(state, opp)))

    my_immediate_wins = count_immediate_wins(state, player)
    opp_immediate_wins = count_immediate_wins(state, opp)

    features = window_counts + [
        float(state.pieces_x[player]),
        float(state.pieces_o[player]),
        float(state.pieces_x[opp]),
        float(state.pieces_o[opp]),
        float(my_actions),
        float(opp_actions),
        float(my_immediate_wins),
        float(opp_immediate_wins),
        float(current_player_flag),
        float(state.totem_X[0]), float(state.totem_X[1]),
        float(state.totem_O[0]), float(state.totem_O[1]),
    ]
    return features


def _state_with_player(state: State, player: int) -> State:
    copied = state.copy()
    copied.current_player = player
    return copied


def heuristic_value(state: State, player: int) -> float:
    feats = extract_features(state, player)
    (
        my_color_2, my_color_3, my_color_4,
        opp_color_2, opp_color_3, opp_color_4,
        my_x_2, my_x_3, my_x_4,
        opp_x_2, opp_x_3, opp_x_4,
        my_o_2, my_o_3, my_o_4,
        opp_o_2, opp_o_3, opp_o_4,
        my_px, my_po, opp_px, opp_po,
        my_actions, opp_actions,
        my_wins, opp_wins,
        current_player_flag,
        *_
    ) = feats

    score = 0.0
    score += 0.10 * my_color_2 + 0.45 * my_color_3 + 1.0 * my_color_4
    score -= 0.12 * opp_color_2 + 0.55 * opp_color_3 + 1.0 * opp_color_4
    score += 0.08 * my_x_2 + 0.35 * my_x_3 + 1.0 * my_x_4
    score -= 0.08 * opp_x_2 + 0.40 * opp_x_3 + 1.0 * opp_x_4
    score += 0.08 * my_o_2 + 0.35 * my_o_3 + 1.0 * my_o_4
    score -= 0.08 * opp_o_2 + 0.40 * opp_o_3 + 1.0 * opp_o_4
    score += 0.02 * (my_px + my_po - opp_px - opp_po)
    score += 0.005 * (my_actions - opp_actions)
    score += 0.35 * my_wins - 0.45 * opp_wins
    score += 0.01 * current_player_flag

    if Game.is_terminal(state):
        return float(Game.utility(state, player))

    return max(-0.99, min(0.99, score))
