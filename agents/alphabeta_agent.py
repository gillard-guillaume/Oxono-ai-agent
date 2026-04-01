from agents.agent import Agent
from oxono.oxono import Game, State
import math


class AlphaBeta(Agent):

    def __init__(self, player):
        super().__init__(player)
        self.move_count = 0
    
    def act(self, state, remaining_time):
        if self.move_count <= 4:
            self.move_count += 1
            return self.alpha_beta(state, depth=3)
        self.move_count += 1
        return self.alpha_beta(state, depth=6)
        
    def alpha_beta(self, state, depth):
        value, move = self.max_value(state, -math.inf, math.inf, depth)
        return move
      
    def max_value(self, state, alpha, beta, depth):
        if Game.is_terminal(state):
            return Game.utility(state, self.player), None
        
        if depth == 0:
            return self.evaluate(state), None
        
        v = -math.inf
        move = None

        for a in Game.actions(state):
            new_state = state.copy()
            Game.apply(new_state, a)
            v2, a2 = self.min_value(new_state, alpha, beta, depth-1)
            if v2 > v :
                v, move = v2, a
                alpha = max(alpha, v)
            if v >= beta :
                return v, move
        return v, move

    def min_value(self, state, alpha, beta, depth):
        if Game.is_terminal(state):
            return Game.utility(state, self.player), None
        
        if depth == 0:
            return self.evaluate(state), None
        
        v = math.inf
        move = None 

        for a in Game.actions(state): 
            new_state = state.copy()
            Game.apply(new_state, a)
            v2, a2 = self.max_value(new_state, alpha, beta, depth-1)
            if v2 < v :
                v, move = v2, a
                beta = min(beta, v)
            if v <= alpha :
                return v, move
        return v, move
    
    def killer_move(self, state):
        """Look for a killer move (win in 1)"""
        return None


    def evaluate(self, state):
        board = state.board
        score = 0

        directions = [(1, 0), (0, 1)]  # vertical, horizontal

        for r in range(6):
            for c in range(6):

                if board[r][c] is None:
                    continue

                symbol, player = board[r][c]

                for dr, dc in directions:
                    count_symbol = 0
                    count_color = 0

                    for i in range(4):
                        nr, nc = r + i*dr, c + i*dc

                        if not (0 <= nr < 6 and 0 <= nc < 6):
                            break

                        cell = board[nr][nc]
                        if cell is None:
                            break

                        sym, pl = cell

                        if sym == symbol:
                            count_symbol += 1
                        else:
                            break

                    for i in range(4):
                        nr, nc = r + i*dr, c + i*dc

                        if not (0 <= nr < 6 and 0 <= nc < 6):
                            break

                        cell = board[nr][nc]
                        if cell is None:
                            break

                        sym, pl = cell

                        if pl == player:
                            count_color += 1
                        else:
                            break

                    # scoring
                    if player == self.player:
                        score += count_color * 5
                        score += count_symbol * 3
                    else:
                        score -= count_color * 5
                        score -= count_symbol * 3

        return score