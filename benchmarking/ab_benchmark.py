import argparse
import time
from statistics import mean

from oxono.manager import Manager


def compute_stats(times):
    if not times:
        return (0.0, 0.0, 0.0)
    return (min(times), mean(times), max(times))


def format_stats(stats):
    return f"{stats[0]:.4f} / {stats[1]:.4f} / {stats[2]:.4f}"


def play_games(agent1_file, agent2_file, n_games, time_limit):
    results = {"agent1": 0, "agent2": 0, "draw": 0}
    game_stats = []

    for i in range(n_games):
        print(f"Game {i+1}/{n_games}")

        # Alternate players
        if i % 2 == 0:
            files = [agent1_file, agent2_file]
            perspective = "normal"
        else:
            files = [agent2_file, agent1_file]
            perspective = "swapped"

        manager = Manager(files, time_limit)

        start = time.time()
        result = manager.play()
        elapsed = time.time() - start

        # Interpret result
        if perspective == "normal":
            if result == (1, -1):
                winner = "Agent1"
                results["agent1"] += 1
            elif result == (-1, 1):
                winner = "Agent2"
                results["agent2"] += 1
            else:
                winner = "Draw"
                results["draw"] += 1
        else:
            if result == (1, -1):
                winner = "Agent2"
                results["agent2"] += 1
            elif result == (-1, 1):
                winner = "Agent1"
                results["agent1"] += 1
            else:
                winner = "Draw"
                results["draw"] += 1

        game_stats.append({
            "winner": winner,
            "time": elapsed
        })

    return results, game_stats


def print_results(results, game_stats):
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print("\nGlobal results:")
    print(f"Agent1 wins : {results['agent1']}")
    print(f"Agent2 wins : {results['agent2']}")
    print(f"Draws       : {results['draw']}")

    times = [g["time"] for g in game_stats]
    stats = compute_stats(times)

    print("\nGame duration stats (min / avg / max):")
    print(format_stats(stats))

    print("\nPer-game details:")
    print("-" * 60)
    print(f"{'Game':<6} | {'Winner':<10} | {'Duration (s)'}")
    print("-" * 60)

    for i, g in enumerate(game_stats):
        print(f"{i+1:<6} | {g['winner']:<10} | {g['time']:.4f}")

    print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark using Manager (safe timeout)")

    parser.add_argument("--agent1", type=str, required=True,
                        help="Path to agent1 file (e.g. agents/ab1_agent.py)")
    parser.add_argument("--agent2", type=str, required=True,
                        help="Path to agent2 file")

    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--time", type=int, default=10)

    args = parser.parse_args()

    results, game_stats = play_games(
        args.agent1,
        args.agent2,
        args.games,
        args.time
    )

    print_results(results, game_stats)