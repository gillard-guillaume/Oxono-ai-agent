import numpy as np
import random as rd
import tensorflow as tf
from keras import layers
import time
import os
import glob

from oxono.oxono import Game, State
from agents.alphabeta_agent import AlphaBeta
from agents.rl_agent import RLAgent


# =========================
# MODEL
# =========================
def create_model(filters1=32, filters2=64, dense_size=256, lr=1e-3):
    inputs = tf.keras.Input(shape=(6, 6, 7))

    x = layers.Conv2D(filters1, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters2, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(dense_size, activation='relu')(x)

    outputs = layers.Dense(1, activation='tanh')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mse'
    )

    return model


# =========================
# TRAIN
# =========================
def train(model, n_games=100, iteration=0, n_iterations=50,
          epochs=5, batch_size=32, time_per_move=0.3):

    dataset_X = []
    dataset_y = []

    progress = iteration / n_iterations
    if progress < 1/3:
        epsilon = 0.5
    elif progress < 2/3:
        epsilon = 0.2
    else:
        epsilon = 0.05

    print(f"\n[Train] Iter {iteration+1}/{n_iterations} - epsilon={epsilon:.2f}")

    game_times = []

    for g in range(n_games):
        game_start = time.time()

        state = State()
        states = []
        players = []

        if rd.random() < 0.5:
            agent_0 = RLAgent(0, model)
            agent_1 = AlphaBeta(1)
        else:
            agent_0 = RLAgent(0, model)
            agent_1 = RLAgent(1, model)

        if rd.random() < 0.5:
            agent_0, agent_1 = agent_1, agent_0

        step = 0

        while not Game.is_terminal(state):
            step += 1
            current_player = state.current_player

            states.append(RLAgent.encode_state(state, current_player))
            players.append(current_player)

            if rd.random() < epsilon:
                action = rd.choice(Game.actions(state))
            else:
                agent = agent_0 if current_player == 0 else agent_1
                action = agent.act(state, remaining_time=time_per_move)

            Game.apply(state, action)

        for s, p in zip(states, players):
            dataset_X.append(s)
            dataset_y.append(Game.utility(state, p))

        game_time = time.time() - game_start
        game_times.append(game_time)

        if (g+1) % 20 == 0:
            avg_time = np.mean(game_times[-20:])
            print(f"Game {g+1}/{n_games} | avg last 20: {avg_time:.2f}s")

    print(f"Dataset size: {len(dataset_X)} samples")

    X = np.array(dataset_X, dtype=np.float32)
    y = np.array(dataset_y, dtype=np.float32)

    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    print("Training model...")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    return model


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--f1", type=int, default=32)
    parser.add_argument("--f2", type=int, default=64)
    parser.add_argument("--dense", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--time", type=float, default=0.3)

    args = parser.parse_args()

    print("=" * 40)
    print("TRAINING START")
    print("=" * 40)

    global_start = time.time()

    if args.model:
        model = tf.keras.models.load_model(args.model)
        print(f"Loaded model: {args.model}")
    else:
        model = create_model(
            filters1=args.f1,
            filters2=args.f2,
            dense_size=args.dense,
            lr=args.lr
        )
        print("Created new model")

    os.makedirs("agents/models", exist_ok=True)

    for i in range(args.iterations):
        iter_start = time.time()

        model = train(
            model,
            n_games=args.games,
            iteration=i,
            n_iterations=args.iterations,
            epochs=args.epochs,
            time_per_move=args.time
        )

        iter_time = time.time() - iter_start
        total_elapsed = time.time() - global_start

        remaining_iters = args.iterations - (i + 1)
        eta = remaining_iters * iter_time

        print(f"\nIteration {i+1} finished")
        print(f"Time this iter: {iter_time/60:.2f} min")
        print(f"Total elapsed: {total_elapsed/3600:.2f} h")
        print(f"ETA: {eta/3600:.2f} h")

        # 🔥 checkpoint
        checkpoint_name = f"iter{i+1}_{int(time.time())}.keras"
        checkpoint_path = os.path.join("agents/models", checkpoint_name)
        model.save(checkpoint_path)

        print(f"Saved checkpoint: {checkpoint_path}")

    # =========================
    # FINAL CLEAN MODEL
    # =========================
    final_name = (
        f"cnn_f{args.f1}_{args.f2}"
        f"_d{args.dense}"
        f"_lr{args.lr}"
        f"_g{args.games}"
        f"_it{args.iterations}.keras"
    )

    final_path = os.path.join("agents/models", final_name)
    model.save(final_path)

    print(f"\nFinal model saved as: {final_path}")

    total_time = time.time() - global_start

    print("\n" + "=" * 40)
    print(f"TOTAL TIME: {total_time/3600:.2f} hours")
    print("=" * 40)

    """
    systemd-inhibit python -m training.rl_training2 \
    --iterations 15 \
    --games 400 \
    --epochs 5 \
    --time 0.3
    """