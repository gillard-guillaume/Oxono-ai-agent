import numpy as np
import random as rd
import tensorflow as tf
from keras import layers
import time

from oxono.oxono import Game, State
from agents.alphabeta_agent import AlphaBeta
from agents.rl_agent import RLAgent

# cmd systemd-inhibit python -m training.rl_training --iterations 2 --games 100
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


def train(model, n_games=100, iteration=0, n_iterations=50, epochs=5, batch_size=32):
    return None