#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Oxono benchmark results.")
    parser.add_argument(
        "--summary",
        type=str,
        default="benchmark_results/summary.csv",
        help="Path to summary.csv produced by benchmark_oxono.py",
    )
    parser.add_argument(
        "--games",
        type=str,
        default="benchmark_results/games.csv",
        help="Path to games.csv produced by benchmark_oxono.py",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="benchmark_plots",
        help="Directory where plots will be saved",
    )
    return parser.parse_args()


def save_bar_plot(series, title, ylabel, output_path):
    plt.figure(figsize=(10, 6))
    series.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    summary_path = Path(args.summary)
    games_path = Path(args.games)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(summary_path)
    games_df = pd.read_csv(games_path)

    # 1. Win / draw rates by matchup
    win_rate_rows = []
    for _, row in summary_df.iterrows():
        games = row["games"]
        if games == 0:
            continue
        win_rate_rows.append(
            {
                "matchup": row["matchup"],
                row["agent_a"]: 100.0 * row["wins_agent_a"] / games,
                row["agent_b"]: 100.0 * row["wins_agent_b"] / games,
                "Draws": 100.0 * row["draws"] / games,
            }
        )

    if win_rate_rows:
        win_rate_df = pd.DataFrame(win_rate_rows).set_index("matchup").fillna(0.0)
        ax = win_rate_df.plot(kind="bar", figsize=(12, 7))
        ax.set_title("Win / Draw Rates by Matchup")
        ax.set_ylabel("Percentage")
        ax.set_xlabel("")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / "win_rates_by_matchup.png", dpi=200)
        plt.close()

    # 2. Average game length by matchup
    avg_plies = summary_df.set_index("matchup")["avg_plies"]
    save_bar_plot(
        avg_plies,
        "Average Game Length by Matchup",
        "Average plies",
        outdir / "avg_plies_by_matchup.png",
    )

    # 3. Average move time by matchup
    move_time_rows = []
    for _, row in summary_df.iterrows():
        move_time_rows.append(
            {
                "matchup": row["matchup"],
                row["agent_a"]: row["avg_move_time_agent_a"],
                row["agent_b"]: row["avg_move_time_agent_b"],
            }
        )

    if move_time_rows:
        move_time_df = pd.DataFrame(move_time_rows).set_index("matchup").fillna(0.0)
        ax = move_time_df.plot(kind="bar", figsize=(12, 7))
        ax.set_title("Average Move Time by Matchup")
        ax.set_ylabel("Seconds")
        ax.set_xlabel("")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / "avg_move_time_by_matchup.png", dpi=200)
        plt.close()

    # 4. Overall win rate by agent
    agents = sorted(set(games_df["pink_agent"]).union(set(games_df["black_agent"])))
    overall = []
    for agent in agents:
        total_games = ((games_df["pink_agent"] == agent) | (games_df["black_agent"] == agent)).sum()
        wins = (games_df["winner_agent"] == agent).sum()
        overall.append(
            {
                "agent": agent,
                "win_rate": 100.0 * wins / total_games if total_games else 0.0,
            }
        )

    overall_df = pd.DataFrame(overall).set_index("agent").sort_values("win_rate", ascending=False)
    save_bar_plot(
        overall_df["win_rate"],
        "Overall Win Rate by Agent",
        "Win rate (%)",
        outdir / "overall_win_rate_by_agent.png",
    )

    # 5. Win rate by color
    split_rows = []
    for agent in agents:
        pink_games = games_df[games_df["pink_agent"] == agent]
        black_games = games_df[games_df["black_agent"] == agent]
        pink_wins = (pink_games["winner"] == "pink").sum()
        black_wins = (black_games["winner"] == "black").sum()

        split_rows.append(
            {
                "agent": agent,
                "Pink win rate": 100.0 * pink_wins / len(pink_games) if len(pink_games) else 0.0,
                "Black win rate": 100.0 * black_wins / len(black_games) if len(black_games) else 0.0,
            }
        )

    split_df = pd.DataFrame(split_rows).set_index("agent").fillna(0.0)
    ax = split_df.plot(kind="bar", figsize=(12, 7))
    ax.set_title("Win Rate by Color")
    ax.set_ylabel("Win rate (%)")
    ax.set_xlabel("")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "win_rate_by_color.png", dpi=200)
    plt.close()

    print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    main()
