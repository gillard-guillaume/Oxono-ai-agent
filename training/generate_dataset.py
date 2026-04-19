import random
import multiprocessing as mp
import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime

from agents.ab5_agent import AB5
from agents.ab1_agent import AB1
from agents.random_agent import RandomAgent
from agents.rl_agent import RLAgent
from oxono.oxono import Game, State


GLOBAL_ARGS = None

AGENTS = {
    "ab5": AB5,
    "ab1": AB1,
    "random": RandomAgent
}

def play_game(args):

    strong, easy, time_per_move, opening_moves = args

    strong = AGENTS[strong]
    easy = AGENTS[easy]


    rd_agent0 = RandomAgent(0)
    rd_agent1 = RandomAgent(1)

    if random.random() < 0.5:
        agent0 = easy(0)
        agent1 = strong(1)
    else :
        agent0 = strong(0)
        agent1 = easy(1)

    turn = 0; state = State()



    history = []

    # random opening (always random)
    for _ in range(opening_moves):
        if turn % 2 == 0: 
            Game.apply(state, rd_agent0.act(state, time_per_move))
        else :
            Game.apply(state, rd_agent1.act(state, time_per_move))
        encoded = RLAgent.encode_state(state, perspective_player=0)
        history.append(encoded)
        turn += 1


    # main game
    while True:
        actions = Game.actions(state)
        if not actions:
            break
        if Game.is_terminal(state):
            break
        if turn % 2 == 0:
            Game.apply(state, agent0.act(state, time_per_move))
        else :
            Game.apply(state, agent1.act(state, time_per_move))
        encoded = RLAgent.encode_state(state, perspective_player=0)
        history.append(encoded)
        turn += 1


    winner = Game.utility(state, 0)

    gamma = 0.95
    n = len(history)
    powers = np.arange(n - 1, -1, -1)
    targets = winner * (gamma ** powers)

    data = list(zip(history, targets))
    return data

def worker(_):
    return play_game(GLOBAL_ARGS)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--time", type=float, default=3)
    parser.add_argument("--strong", type=str, default="ab5")
    parser.add_argument("--easy", type=str, default="ab1")
    parser.add_argument("--opening", type=int, default=4)
    parser.add_argument("--workers", type=int, default=mp.cpu_count())

    args = parser.parse_args()

    # dataset name auto
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = (
        f"{args.strong}_vs_{args.easy}"
        f"_t{args.time}"
        f"_g{args.games}"
        f"_{timestamp}.npz"
    )

    os.makedirs("datasets", exist_ok=True)
    output_path = os.path.join("datasets", filename)

    print("Starting dataset generation")
    print(f"Games: {args.games}, Workers: {args.workers}")
    print(f"Strong: {args.strong}, Easy: {args.easy}")
    print(f"Output: {output_path}")

    global GLOBAL_ARGS
    GLOBAL_ARGS = (args.strong, args.easy, args.time, args.opening)

    pbar = tqdm(
        total=args.games,
        desc="Generating",
        unit="game",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[elapsed: {elapsed} | ETA: {remaining} | {rate_fmt}]"
    )
    dataset = []

    with mp.Pool(processes=args.workers) as pool:
        for i, game_data in enumerate(pool.imap_unordered(worker, range(args.games))):
            dataset.extend(game_data)
            if (i + 1) % 1000 == 0:
                pbar.write(f"{i+1} games done")
                backup_path = output_path.replace(".npz", ".pkl")
                with open(backup_path, "wb") as f:
                    pickle.dump(dataset, f)
            pbar.update(1)
    pbar.close()


    random.shuffle(dataset)
    X = np.array([s for s, v in dataset], dtype=np.float32)
    y = np.array([v for s, v in dataset], dtype=np.float32)
    np.savez(output_path, X=X, y=y)

    print(f" Done | games: {args.games} | samples: {len(dataset)}")

if __name__ == "__main__":
    main()