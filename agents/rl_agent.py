from agents.alphabeta_agent import AlphaBeta
from oxono.oxono import Game, State
from tensorflow.keras.models import load_model
import numpy as np

class RLAgent(AlphaBeta):
    """
    Reinforcement Learning-based agent extending AlphaBeta search
    This agent reuses the AlphaBeta search framework but replaces the
    handcrafted evaluation function with a neural network trained via self-play

    Key features:
    - Alpha-beta search with iterative deepening and time management
    - Neural network evaluation of game states (CNN-based)
    - State encoding preserving spatial structure of the board
    - Inherits optimizations such as move ordering, transposition table,
        killer moves, and pruning strategies from AlphaBeta

    The model predicts a value for each state, guiding the search toward
    more promising positions
    """

    def __init__(self, player):
        super().__init__(player)
        self.model = load_model("agents/models/model1.keras")

    @staticmethod
    def encode_state(state: State, perspective_player: int = 0) -> np.ndarray:
        """
        Encode the game state into a 6x6x7 tensor suitable for a convolutional neural network.

        The state is represented as a 3D tensor where each channel encodes a specific
        type of information about the board from the perspective of a given player.

        Features (channels):
            0: current player's X pieces
            1: current player's O pieces
            2: opponent's X pieces
            3: opponent's O pieces
            4: X totem position
            5: O totem position
            6: current player indicator (broadcast over the grid)

        This representation preserves the spatial structure of the board, allowing
        convolutional layers to capture local interactions between pieces.

        Args:
            state (State): Current game state.
            perspective_player (int): Player perspective (0 or 1).

        Returns:
            np.ndarray: Tensor of shape (6, 6, 7).
        """
        board = state.board
        tensor = np.zeros((6, 6, 7), dtype=np.float32)
        for r in range(6):
            for c in range(6):
                cell = board[r][c]
                if cell is None:
                    continue
                sym, player = cell
                is_mine = (player == perspective_player)
                if sym == 'x':
                    tensor[r][c][0 if is_mine else 2] = 1
                elif sym == 'o':
                    tensor[r][c][1 if is_mine else 3] = 1
                elif sym == 'totem_x':
                    tensor[r][c][4] = 1
                elif sym == 'totem_o':
                    tensor[r][c][5] = 1
        tensor[:, :, 6] = state.current_player
        return tensor
    
    def evaluate(self, state: State) -> float:
        """
        Evaluate a state using the neural network model.

        Args:
            state (State): Current game state.

        Returns:
            float: Predicted value of the state.
        """
        encoded = RLAgent.encode_state(state, perspective_player=self.player)
        value = self.model(encoded[np.newaxis], training=False)[0][0]
        return float(value)