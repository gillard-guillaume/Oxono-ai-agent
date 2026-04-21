from agents.ab8_agent import AB8
from oxono.oxono import Game, State
import time
from collections import OrderedDict
from keras.models import load_model
import tensorflow as tf
import numpy as np
Action = tuple[str, tuple[int, int], tuple[int, int]]
EXACT = 0; LOWERBOUND = 1; UPPERBOUND = 2


class RLAgent(AB8):
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

    def __init__(self, player, model=None, debug=False, log_file="RL_log.log"):
        super().__init__(player)
        self.nn_cache = OrderedDict()
        self.nn_cache_size = 100_000
        self.agent_name = "RL agent"
        self.debug =debug
        self.log_file = log_file
        try:
            self.model = model if model is not None else load_model("models/idefix_cnn_ab5_vs_ab1_t6.0_g50000_20260417_2236_c2_f128_d256_lr2.8e-04_hd0.31_b64_n694587_20260419_2339_best.keras")
        except Exception:
            self.model = None

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
            6: indicator of the player to play (broadcast over the grid)

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
        tensor[:, :, 6] = 1 if state.current_player == perspective_player else -1 # symmetry
        return tensor

    
    def move_ordering(self, state: State, actions: list[Action], depth: int, h: int, reverse: bool) -> list[Action]:
        """
        Order actions to improve alpha-beta pruning

        Priority :
            1) The hash move from the transposition table (if available)
            2) Actions evaluated using transposition table values when possible
            3) Otherwise, heuristic evaluation of resulting states

        Args:
            state (State): Current game state
            actions (list[Action]): List of possible actions
            depth (int): Current remaining depth
            h (int): Zobrist hash of the current state
            reverse (bool): Sorting direction for heuristic scores
                True for max nodes (descending)
                False for min nodes (ascending)

        Returns:
            list[Action]: Ordered list of actions
        """
        scored = []
        tt_move = self.tt[h][2] if h in self.tt else None
        for a in actions:
            if a == tt_move:
                continue
            h_child = super().update_hash(h, state, a)
            score = None
            if h_child in self.tt:
                val, d, _, flag = self.tt[h_child]
                if d >= depth - 1 and flag == EXACT:
                    score = val
            if score is None:
                new_state = state.copy()
                Game.apply(new_state, a)
                score = super().evaluate(new_state)
            scored.append((score, a))
        scored.sort(key=lambda x: x[0], reverse=reverse)
        if tt_move in actions:
            return [tt_move] + [a for _, a in scored]
        return [a for _, a in scored][:5]


    @tf.function
    def model_predict(self, x):
        return self.model(x, training=False)
    
    def evaluate(self, state: State) -> float:
        """
        Evaluate a state using the neural network model.

        Args:
            state (State): Current game state.

        Returns:
            float: Predicted value of the state.
        """
        key = self.hash_state(state)
        if key in self.nn_cache:
            return self.nn_cache[key]
        t_encode = time.time()
        encoded = RLAgent.encode_state(state, perspective_player=self.player)
        encoded = tf.convert_to_tensor(encoded, dtype=tf.float32)
        encoded = encoded[None, ...]
        #self.log(f"ENCODE time : {time.time()-t_encode}")
        t_cnn = time.time()
        value = self.model_predict(encoded)[0][0]
        #self.log(f"CNN time : {time.time()-t_cnn}")
        self.nn_cache[key] = value
        if len(self.nn_cache) > self.nn_cache_size:
            self.nn_cache.popitem(last=False)
        return float(value)