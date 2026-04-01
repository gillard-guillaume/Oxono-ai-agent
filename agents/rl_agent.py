import random
import numpy as np
import tensorflow as tf

from oxono import State, Game


# =========================
# 1. ENCODAGE DU STATE
# =========================
def encode_state(state: State):
    board_tensor = np.zeros((6, 6, 3))

    for i in range(6):
        for j in range(6):
            cell = state.board[i][j]
            if cell:
                symbol, player = cell
                if symbol == 'x':
                    board_tensor[i][j][0] = 1
                else:
                    board_tensor[i][j][1] = 1
                board_tensor[i][j][2] = player

    return board_tensor.flatten()


# =========================
# 2. MODELE TENSORFLOW
# =========================
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(108,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='tanh')  # sortie entre -1 et 1
    ])

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    return model


# =========================
# 3. GENERER UNE PARTIE (SELF-PLAY)
# =========================
def generate_game():
    state = State()
    history = []

    while not Game.is_terminal(state):
        history.append(encode_state(state))

        actions = Game.actions(state)
        action = random.choice(actions)  # random au début

        Game.apply(state, action)

    # récompense finale
    reward = Game.utility(state, 0)

    return history, reward


# =========================
# 4. CREER DATASET
# =========================
def generate_dataset(n_games=1000):
    X = []
    y = []

    for _ in range(n_games):
        states, reward = generate_game()

        for s in states:
            X.append(s)
            y.append(reward)

    return np.array(X), np.array(y)


# =========================
# 5. ENTRAINEMENT
# =========================
def train_model(model, n_games=1000, epochs=5):
    X, y = generate_dataset(n_games)

    model.fit(X, y, epochs=epochs, verbose=1)


# =========================
# 6. EVALUATION (à utiliser dans alpha-beta)
# =========================
def evaluate(model, state):
    s = encode_state(state)
    value = model.predict(s.reshape(1, -1), verbose=0)[0][0]
    return value


# =========================
# 7. CHOISIR UNE ACTION AVEC LE MODELE
# =========================
def select_action(model, state):
    actions = Game.actions(state)

    best_value = -float('inf')
    best_action = None

    for action in actions:
        new_state = state.copy()
        Game.apply(new_state, action)

        value = evaluate(model, new_state)

        if value > best_value:
            best_value = value
            best_action = action

    return best_action


# =========================
# 8. SELF-PLAY AVEC LE MODELE
# =========================
def generate_game_with_model(model, epsilon=0.1):
    state = State()
    history = []

    while not Game.is_terminal(state):
        history.append(encode_state(state))

        actions = Game.actions(state)

        # exploration vs exploitation
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = select_action(model, state)

        Game.apply(state, action)

    reward = Game.utility(state, 0)

    return history, reward


# =========================
# 9. ENTRAINEMENT ITERATIF (STYLE ALPHAZERO LIGHT)
# =========================
def train_iterative(model, iterations=10, games_per_iter=500):
    for i in range(iterations):
        print(f"\n=== ITERATION {i+1} ===")

        X = []
        y = []

        for _ in range(games_per_iter):
            states, reward = generate_game_with_model(model)

            for s in states:
                X.append(s)
                y.append(reward)

        X = np.array(X)
        y = np.array(y)

        model.fit(X, y, epochs=3, verbose=1)


# =========================
# 10. MAIN
# =========================
if __name__ == "__main__":
    model = create_model()

    print("🔹 Training initial (random play)...")
    train_model(model, n_games=1000, epochs=5)

    print("🔹 Training amélioré (self-play)...")
    train_iterative(model, iterations=5, games_per_iter=500)

    print("✅ Training terminé !")