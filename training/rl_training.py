import numpy as np
import random as rd
import tensorflow as tf
from tensorflow.keras import layers
import time

from oxono.oxono import Game, State
from agents.alphabeta_agent import AlphaBeta
from agents.rl_agent import RLAgent


def create_model():
    """
    Create a convolutional neural network (CNN) used to evaluate a game position

    Architecture:
        - Input layer: 6x6x7 tensor representing the board state
        - Convolutional layer 1: 32 filters of size 3x3 (captures local patterns)
        - Convolutional layer 2: 64 filters of size 3x3 (combines local features)
        - Fully connected layer: 256 neurons (global reasoning)
        - Output layer: 1 neuron (state value)

    The model predicts a value between -1 and 1:
        -1 → losing position
         0 → neutral
        +1 → winning position

    Activation functions:
        - ReLU for hidden layers (introduces non-linearity), LeakyReLU could be considered to mitigate dead neurons and preserve negative signals
        - Tanh for output (bounded value suitable for reinforcement learning)

    Design choices:
        - Convolutional layers preserve spatial structure of the board and capture
          local interactions between pieces (e.g., alignments, threats)
        - Kernel size 3x3 is used to model local neighborhoods efficiently
        - Padding='same' ensures spatial dimensions are preserved
        - Dense layer aggregates extracted features into a global evaluation

    Returns:
        tf.keras.Model: compiled model ready for training
    """
    inputs = tf.keras.Input(shape=(6, 6, 7))
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(1, activation='tanh')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')
    return model




def train(model, n_games=100, iteration=0, n_iterations=50):
    start_total = time.time()

    dataset_X = []
    dataset_y = []

    # Epsilon
    progress = iteration / n_iterations
    if progress < 1/3:
        epsilon = 0.5
    elif progress < 2/3:
        epsilon = 0.2
    else:
        epsilon = 0.05

    print(f"[Train] Start - epsilon={epsilon:.2f}")

    agent_0 = RLAgent(0, model)
    agent_1 = RLAgent(1, model)

    for g in range(n_games):
        game_start = time.time()

        state = State()

        states = []
        players = []

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
                action = agent.act(state, remaining_time=300.0)

            Game.apply(state, action)

        # rewards
        for i, (s, p) in enumerate(zip(states, players)):
            reward = Game.utility(state, p)
            dataset_X.append(s)
            dataset_y.append(reward)

        game_time = time.time() - game_start
        print(f"  Game {g+1}/{n_games} - steps={step} - time={game_time:.2f}s")

    # Dataset creation
    t0 = time.time()
    X = np.array(dataset_X, dtype=np.float32)
    y = np.array(dataset_y, dtype=np.float32)
    print(f"[Data] Conversion time: {time.time() - t0:.2f}s")

    # Shuffle
    t0 = time.time()
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    print(f"[Data] Shuffle time: {time.time() - t0:.2f}s")

    # Training
    t0 = time.time()
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)
    print(f"[Model] Fit time: {time.time() - t0:.2f}s")

    total_time = time.time() - start_total
    print(f"[Train] Total iteration time: {total_time:.2f}s")

    return model




if __name__ == "__main__":

    global_start = time.time()

    iterations = 10
    print("=" * iterations)
    print("Training started")
    print("=" * iterations)

    model = create_model()

    for i in range(iterations):
        # Train
        t0 = time.time()
        model = train(model, n_games=100, iteration=i, n_iterations=iterations)
        train_time = time.time() - t0
        print(f"[Time] Train       : {train_time:.2f}s")


    print("\n" + "=" * iterations)
    print("Training finished")
    print("=" * iterations)

    total_time = time.time() - global_start

    print(f"Total training time : {total_time:.2f}s")

    model.save("agents/models/model1.keras")
    print("Model saved at agents/models/model1.keras")