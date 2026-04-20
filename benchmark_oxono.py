#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import itertools
import random
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from agent import Agent
from oxono import Game, State

TOTAL_TIME_PER_PLAYER = 300.0  # 5 minutes


@dataclass
class GameResult:
    matchup: str
    game_index: int
    pink_agent: str
    black_agent: str
    winner: str
    winner_agent: str
    loser_agent: str
    reason: str
    plies: int
    pink_time_used: float
    black_time_used: float
    pink_avg_move_time: float
    black_avg_move_time: float
    pink_moves: int
    black_moves: int


@dataclass
class MatchSummary:
    matchup: str
    agent_a: str
    agent_b: str
    games: int
    wins_agent_a: int
    wins_agent_b: int
    draws: int
    forfeits: int
    avg_plies: float
    avg_time_agent_a: float
    avg_time_agent_b: float
    avg_move_time_agent_a: float
    avg_move_time_agent_b: float
    wins_as_pink_agent_a: int
    wins_as_black_agent_a: int
    wins_as_pink_agent_b: int
    wins_as_black_agent_b: int


def load_module_from_path(path: Path):
    module_name = f"benchmark_mod_{path.stem}_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def discover_agent_class(module) -> Type[Agent]:
    candidates = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, Agent) and obj is not Agent:
            candidates.append(obj)
    if not candidates:
        raise ValueError(f"No Agent subclass found in module {module.__name__}")
    for cls in candidates:
        if cls.__name__ == "MyAgent":
            return cls
    return candidates[0]


def agent_label(path: Path) -> str:
    return path.name


def validate_action(state: State, action: Any) -> bool:
    try:
        return action in Game.actions(state)
    except Exception:
        return False


def initial_state() -> State:
    return State()


def forfeit_result(
    matchup: str,
    game_index: int,
    pink_label: str,
    black_label: str,
    forfeiting_player: int,
    reason: str,
    plies: int,
    time_used: Dict[int, float],
    move_count: Dict[int, int],
) -> GameResult:
    if forfeiting_player == 0:
        winner = "black"
        winner_agent = black_label
        loser_agent = pink_label
    else:
        winner = "pink"
        winner_agent = pink_label
        loser_agent = black_label

    return GameResult(
        matchup=matchup,
        game_index=game_index,
        pink_agent=pink_label,
        black_agent=black_label,
        winner=winner,
        winner_agent=winner_agent,
        loser_agent=loser_agent,
        reason=reason,
        plies=plies,
        pink_time_used=time_used[0],
        black_time_used=time_used[1],
        pink_avg_move_time=(time_used[0] / move_count[0]) if move_count[0] else 0.0,
        black_avg_move_time=(time_used[1] / move_count[1]) if move_count[1] else 0.0,
        pink_moves=move_count[0],
        black_moves=move_count[1],
    )


def play_one_game(
    pink_cls: Type[Agent],
    black_cls: Type[Agent],
    pink_label: str,
    black_label: str,
    matchup: str,
    game_index: int,
) -> GameResult:
    state = initial_state()
    pink_agent = pink_cls(0)
    black_agent = black_cls(1)

    remaining = {0: TOTAL_TIME_PER_PLAYER, 1: TOTAL_TIME_PER_PLAYER}
    time_used = {0: 0.0, 1: 0.0}
    move_count = {0: 0, 1: 0}
    plies = 0

    while not Game.is_terminal(state):
        player = Game.to_move(state)
        agent = pink_agent if player == 0 else black_agent

        before = time.perf_counter()
        try:
            action = agent.act(state.copy(), remaining[player])
        except Exception:
            elapsed = time.perf_counter() - before
            remaining[player] -= elapsed
            time_used[player] += elapsed
            return forfeit_result(
                matchup, game_index, pink_label, black_label,
                forfeiting_player=player, reason="exception", plies=plies,
                time_used=time_used, move_count=move_count
            )

        elapsed = time.perf_counter() - before
        remaining[player] -= elapsed
        time_used[player] += elapsed
        move_count[player] += 1

        if remaining[player] < 0:
            return forfeit_result(
                matchup, game_index, pink_label, black_label,
                forfeiting_player=player, reason="timeout", plies=plies,
                time_used=time_used, move_count=move_count
            )

        if action is None:
            return forfeit_result(
                matchup, game_index, pink_label, black_label,
                forfeiting_player=player, reason="no_action", plies=plies,
                time_used=time_used, move_count=move_count
            )

        if not validate_action(state, action):
            return forfeit_result(
                matchup, game_index, pink_label, black_label,
                forfeiting_player=player, reason="invalid_action", plies=plies,
                time_used=time_used, move_count=move_count
            )

        Game.apply(state, action)
        plies += 1

    utility_for_pink = Game.utility(state, 0)
    if utility_for_pink > 0:
        winner = "pink"
        winner_agent = pink_label
        loser_agent = black_label
    elif utility_for_pink < 0:
        winner = "black"
        winner_agent = black_label
        loser_agent = pink_label
    else:
        winner = "draw"
        winner_agent = ""
        loser_agent = ""

    return GameResult(
        matchup=matchup,
        game_index=game_index,
        pink_agent=pink_label,
        black_agent=black_label,
        winner=winner,
        winner_agent=winner_agent,
        loser_agent=loser_agent,
        reason="normal",
        plies=plies,
        pink_time_used=time_used[0],
        black_time_used=time_used[1],
        pink_avg_move_time=(time_used[0] / move_count[0]) if move_count[0] else 0.0,
        black_avg_move_time=(time_used[1] / move_count[1]) if move_count[1] else 0.0,
        pink_moves=move_count[0],
        black_moves=move_count[1],
    )


def summarize_matchup(
    agent_a: str,
    agent_b: str,
    results: List[GameResult],
) -> MatchSummary:
    wins_a = 0
    wins_b = 0
    draws = 0
    forfeits = 0
    wins_as_pink_a = 0
    wins_as_black_a = 0
    wins_as_pink_b = 0
    wins_as_black_b = 0

    total_plies = 0
    total_time_a = 0.0
    total_time_b = 0.0
    total_move_time_a = 0.0
    total_move_time_b = 0.0

    for r in results:
        total_plies += r.plies

        if r.pink_agent == agent_a:
            total_time_a += r.pink_time_used
            total_move_time_a += r.pink_avg_move_time
            total_time_b += r.black_time_used
            total_move_time_b += r.black_avg_move_time
        else:
            total_time_a += r.black_time_used
            total_move_time_a += r.black_avg_move_time
            total_time_b += r.pink_time_used
            total_move_time_b += r.pink_avg_move_time

        if r.reason != "normal":
            forfeits += 1

        if r.winner == "draw":
            draws += 1
        elif r.winner_agent == agent_a:
            wins_a += 1
            if r.pink_agent == agent_a:
                wins_as_pink_a += 1
            else:
                wins_as_black_a += 1
        elif r.winner_agent == agent_b:
            wins_b += 1
            if r.pink_agent == agent_b:
                wins_as_pink_b += 1
            else:
                wins_as_black_b += 1

    n = len(results)
    matchup_name = f"{agent_a} vs {agent_b}"
    return MatchSummary(
        matchup=matchup_name,
        agent_a=agent_a,
        agent_b=agent_b,
        games=n,
        wins_agent_a=wins_a,
        wins_agent_b=wins_b,
        draws=draws,
        forfeits=forfeits,
        avg_plies=(total_plies / n) if n else 0.0,
        avg_time_agent_a=(total_time_a / n) if n else 0.0,
        avg_time_agent_b=(total_time_b / n) if n else 0.0,
        avg_move_time_agent_a=(total_move_time_a / n) if n else 0.0,
        avg_move_time_agent_b=(total_move_time_b / n) if n else 0.0,
        wins_as_pink_agent_a=wins_as_pink_a,
        wins_as_black_agent_a=wins_as_black_a,
        wins_as_pink_agent_b=wins_as_pink_b,
        wins_as_black_agent_b=wins_as_black_b,
    )


def write_csv(path: Path, rows: List[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_match_summary(summary: MatchSummary):
    print(f"\n=== {summary.matchup} ===")
    print(f"Games: {summary.games}")
    print(f"{summary.agent_a}: {summary.wins_agent_a} wins")
    print(f"{summary.agent_b}: {summary.wins_agent_b} wins")
    print(f"Draws: {summary.draws}")
    print(f"Forfeits: {summary.forfeits}")
    print(f"Average plies: {summary.avg_plies:.2f}")
    print(
        f"Average total time used: "
        f"{summary.agent_a}={summary.avg_time_agent_a:.3f}s, "
        f"{summary.agent_b}={summary.avg_time_agent_b:.3f}s"
    )
    print(
        f"Average move time: "
        f"{summary.agent_a}={summary.avg_move_time_agent_a:.5f}s, "
        f"{summary.agent_b}={summary.avg_move_time_agent_b:.5f}s"
    )
    print(
        f"{summary.agent_a} wins as pink/black: "
        f"{summary.wins_as_pink_agent_a}/{summary.wins_as_black_agent_a}"
    )
    print(
        f"{summary.agent_b} wins as pink/black: "
        f"{summary.wins_as_pink_agent_b}/{summary.wins_as_black_agent_b}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Oxono agents.")
    parser.add_argument("--agents", nargs="+", required=True, help="Paths to agent Python files.")
    parser.add_argument("--games-per-pair", type=int, default=20, help="Number of games per agent pair.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducible pairing order.")
    parser.add_argument("--outdir", type=str, default="benchmark_results", help="Directory to save CSV outputs.")
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help="Optional explicit pairs in the form A.py:B.py. If omitted, all unordered pairs are benchmarked.",
    )
    return parser.parse_args()


def build_pairs(agent_paths: List[Path], explicit_pairs: Optional[List[str]]):
    by_name = {p.name: p for p in agent_paths}
    if explicit_pairs:
        pairs = []
        for item in explicit_pairs:
            a_name, b_name = item.split(":")
            pairs.append((by_name[a_name], by_name[b_name]))
        return pairs
    return list(itertools.combinations(agent_paths, 2))


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    agent_paths = [Path(p).resolve() for p in args.agents]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    loaded = {}
    for path in agent_paths:
        module = load_module_from_path(path)
        cls = discover_agent_class(module)
        loaded[path] = cls

    all_game_rows = []
    all_summary_rows = []
    pairs = build_pairs(agent_paths, args.pairs)

    for path_a, path_b in pairs:
        label_a = agent_label(path_a)
        label_b = agent_label(path_b)
        matchup = f"{label_a} vs {label_b}"
        results = []

        seatings = []
        for i in range(args.games_per_pair):
            if i % 2 == 0:
                seatings.append((path_a, path_b))
            else:
                seatings.append((path_b, path_a))
        rng.shuffle(seatings)

        for game_index, (pink_path, black_path) in enumerate(seatings, start=1):
            pink_cls = loaded[pink_path]
            black_cls = loaded[black_path]
            pink_label = agent_label(pink_path)
            black_label = agent_label(black_path)

            try:
                result = play_one_game(
                    pink_cls=pink_cls,
                    black_cls=black_cls,
                    pink_label=pink_label,
                    black_label=black_label,
                    matchup=matchup,
                    game_index=game_index,
                )
            except Exception:
                print(traceback.format_exc())
                raise

            results.append(result)
            all_game_rows.append(asdict(result))

        summary = summarize_matchup(label_a, label_b, results)
        print_match_summary(summary)
        all_summary_rows.append(asdict(summary))

    write_csv(outdir / "games.csv", all_game_rows)
    write_csv(outdir / "summary.csv", all_summary_rows)

    print(f"\nSaved detailed results to: {outdir / 'games.csv'}")
    print(f"Saved matchup summaries to: {outdir / 'summary.csv'}")


if __name__ == "__main__":
    main()
