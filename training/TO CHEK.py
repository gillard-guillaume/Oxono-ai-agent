import tensorflow as tf
from keras import layers, models, optimizers
import numpy as np
import random
import os

# =============================================
# 1. Buffer circulaire (sans collections.deque)
# =============================================
class ReplayBuffer:
    def __init__(self, max_size=5000):
        self.buffer = []
        self.max_size = max_size
        self.index = 0

    def add(self, transition):
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.index] = transition
            self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = random.sample(range(min(len(self.buffer), self.max_size)), batch_size)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

# =============================================
# 2. CNN pour évaluer les positions (Q-values)
# =============================================
class CNNEvaluator:
    def __init__(self, input_shape=(8, 8, 1), num_actions=64, learning_rate=1e-3):
        self.model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_actions)  # Q-values pour chaque action
        ])
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='mse'
        )
        self.gamma = 0.99  # Facteur d'actualisation

    def evaluate(self, board_state):
        """Prend un état du jeu (board_state) et retourne les Q-values."""
        input_state = preprocess_board(board_state)  # À adapter à ton jeu
        return self.model.predict(np.array([input_state]), verbose=0)[0]

    def train(self, states, targets):
        """Entraîne le CNN sur un mini-batch de transitions."""
        self.model.fit(states, targets, verbose=0)

    def save_model(self, path="cnn_model.keras"):
        """Sauvegarde le modèle entraîné."""
        self.model.save(path)

    @staticmethod
    def load_model(path="cnn_model.keras", input_shape=(8, 8, 1), num_actions=64):
        """Charge un modèle sauvegardé."""
        model = models.load_model(path)
        evaluator = CNNEvaluator.__new__(CNNEvaluator)
        evaluator.model = model
        evaluator.gamma = 0.99
        return evaluator

# =============================================
# 3. Fonctions utilitaires pour le training
# =============================================
def preprocess_board(board_state):
    """Convertit l'état du jeu en entrée pour le CNN.
    À adapter selon ton jeu (ex : Othello, Go, échecs)."""
    # Exemple pour un jeu de plateau 8x8 :
    board_array = np.array(board_state.get_grid(), dtype=np.float32)  # Remplace par ta méthode
    board_array = board_array.reshape(8, 8, 1)  # Normalisation si besoin
    return board_array

def get_initial_board_state():
    """Retourne un état initial du jeu.
    À adapter selon ton jeu."""
    return YourGameBoardClass()  # Remplace par ta classe

# =============================================
# 4. Boucle d'entraînement (TD Coherent Learning)
# =============================================
def setup_training(buffer_size=5000, batch_size=32, learning_rate=1e-3):
    """Initialise le CNN et le buffer."""
    cnn_evaluator = CNNEvaluator(input_shape=(8, 8, 1), num_actions=64, learning_rate=learning_rate)
    replay_buffer = ReplayBuffer(max_size=buffer_size)
    return cnn_evaluator, replay_buffer, batch_size

def train_td_coherent(cnn_evaluator, replay_buffer, batch_size=32, gamma=0.99):
    """Entraîne le CNN avec TD Coherent Learning."""
    if len(replay_buffer) < batch_size:
        return False  # Pas assez de données

    batch = replay_buffer.sample(batch_size)
    states = np.array([x[0] for x in batch])
    actions = np.array([x[1] for x in batch])
    rewards = np.array([x[2] for x in batch])
    next_states = np.array([x[3] for x in batch])

    # Calcule les Q-values cibles (TD target)
    next_q_values = cnn_evaluator.model.predict(next_states, verbose=0)
    max_next_q = np.max(next_q_values, axis=1)
    target_q_values = rewards + gamma * max_next_q

    # Calcule les Q-values prédites
    q_values = cnn_evaluator.model.predict(states, verbose=0)
    for i in range(batch_size):
        q_values[i][actions[i]] = target_q_values[i]

    # Entraîne le CNN
    cnn_evaluator.train(states, q_values)
    return True

def run_training_episodes(num_episodes=1000, max_depth=3, exploration_rate=0.1):
    """Lance l'entraînement sur plusieurs épisodes."""
    cnn_evaluator, replay_buffer, batch_size = setup_training()
    gamma = 0.99

    for episode in range(num_episodes):
        board_state = get_initial_board_state()
        total_reward = 0

        while not board_state.is_terminal():
            # Alpha-Beta choisit l'action (en utilisant le CNN pour évaluer les feuilles)
            # TODO: Remplace par un appel à ta classe AlphaBetaAgent existante
            # action = alpha_beta_agent.get_best_action(board_state, evaluator=cnn_evaluator.evaluate)

            # Pour l'exemple, on simule une action aléatoire
            action = random.choice(board_state.get_legal_moves())

            # Joue l'action et observe le reward
            next_board_state = board_state.play(action)
            reward = next_board_state.get_reward()  # À adapter

            # Stocke la transition
            replay_buffer.add((
                preprocess_board(board_state),
                action,
                reward,
                preprocess_board(next_board_state)
            ))

            # Entraîne le CNN
            train_td_coherent(cnn_evaluator, replay_buffer, batch_size, gamma)

            board_state = next_board_state
            total_reward += reward

        print(f"Episode {episode}, Reward: {total_reward}")

        # Sauvegarde le modèle toutes les 100 épisodes
        if episode % 100 == 0:
            cnn_evaluator.save_model(f"cnn_model_episode_{episode}.keras")

    # Sauvegarde le modèle final
    cnn_evaluator.save_model("cnn_model_final.keras")
    return cnn_evaluator

# =============================================
# 5. Point d'entrée pour exécuter le training
# =============================================
if __name__ == "__main__":
    # Exécute l'entraînement si ce fichier est lancé directement
    run_training_episodes(num_episodes=100, max_depth=3)