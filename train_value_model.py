from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from oxono import Game, State
from oxono_features import extract_features, heuristic_value


@dataclass
class SearchPolicy:
    depth: int
    epsilon: float = 0.10

    def choose(self, state: State) -> tuple[str, tuple[int, int], tuple[int, int]]:
        actions = list(Game.actions(state))
        if not actions:
            raise RuntimeError("No legal actions available.")

        if random.random() < self.epsilon:
            return random.choice(actions)

        player = state.current_player
        best_action = actions[0]
        best_value = -math.inf
        for action in actions:
            next_state = state.copy()
            Game.apply(next_state, action)
            value = self._min_value(next_state, self.depth - 1, player, -math.inf, math.inf)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    def _max_value(self, state: State, depth: int, player: int, alpha: float, beta: float) -> float:
        if Game.is_terminal(state):
            return float(Game.utility(state, player))
        if depth == 0:
            return heuristic_value(state, player)

        value = -math.inf
        for action in Game.actions(state):
            next_state = state.copy()
            Game.apply(next_state, action)
            value = max(value, self._min_value(next_state, depth - 1, player, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    def _min_value(self, state: State, depth: int, player: int, alpha: float, beta: float) -> float:
        if Game.is_terminal(state):
            return float(Game.utility(state, player))
        if depth == 0:
            return heuristic_value(state, player)

        value = math.inf
        for action in Game.actions(state):
            next_state = state.copy()
            Game.apply(next_state, action)
            value = min(value, self._max_value(next_state, depth - 1, player, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


def generate_dataset(num_games: int, depth_a: int, depth_b: int, gamma: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)

    policy_a = SearchPolicy(depth=depth_a, epsilon=0.15)
    policy_b = SearchPolicy(depth=depth_b, epsilon=0.15)

    X: List[List[float]] = []
    y: List[float] = []

    for game_idx in range(num_games):
        state = State()
        history: List[Tuple[State, int, int]] = []  # state, player, ply_index
        ply = 0

        # Alternate which policy starts to diversify the dataset.
        first_policy_is_a = (game_idx % 2 == 0)

        while not Game.is_terminal(state):
            history.append((state.copy(), state.current_player, ply))

            if first_policy_is_a:
                actor = policy_a if state.current_player == 0 else policy_b
            else:
                actor = policy_b if state.current_player == 0 else policy_a

            action = actor.choose(state)
            Game.apply(state, action)
            ply += 1

        total_plies = max(1, ply)
        for saved_state, player, ply_index in history:
            X.append(extract_features(saved_state, player))
            outcome = float(Game.utility(state, player))
            discount = gamma ** (total_plies - ply_index - 1)
            y.append(outcome * discount)

        if (game_idx + 1) % max(1, num_games // 10) == 0:
            print(f"Generated {game_idx + 1}/{num_games} games -> {len(X)} samples")

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def train_model(X: np.ndarray, y: np.ndarray, seed: int) -> Pipeline:
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "mlp",
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=60,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=8,
                random_state=seed,
                verbose=True,
            ),
        ),
    ])
    model.fit(X, y)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Oxono self-play samples and train an MLP value model.")
    parser.add_argument("--games", type=int, default=500, help="Number of self-play games.")
    parser.add_argument("--depth-a", type=int, default=1, help="Search depth for policy A.")
    parser.add_argument("--depth-b", type=int, default=3, help="Search depth for policy B.")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount applied to earlier states.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--out", type=str, default="value_model.joblib", help="Output model path.")
    parser.add_argument("--dataset-out", type=str, default="", help="Optional .npz dataset output path.")
    args = parser.parse_args()

    X, y = generate_dataset(args.games, args.depth_a, args.depth_b, args.gamma, args.seed)
    print(f"Training on {len(X)} samples with {X.shape[1]} features")

    model = train_model(X, y, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"Saved model to {out_path}")

    if args.dataset_out:
        dataset_path = Path(args.dataset_out)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dataset_path, X=X, y=y)
        print(f"Saved dataset to {dataset_path}")


if __name__ == "__main__":
    main()
