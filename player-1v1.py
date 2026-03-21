import argparse
from pathlib import Path
import pygame
import math
import sys
import time

from oxono import Game, State

SELECT_TOTEM = 0
SELECT_TOTEM_ACTION = 1
SELECT_PIECE_ACTION = 2

def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n

class VisualManager:
    def __init__(self, agent_files, time_limit=300, path_to_file=None):
        self.agent_files = agent_files
        self.time_limit = time_limit

        self.path = None
        if path_to_file:
            self.path = Path(path_to_file)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as f:
                f.write(f"{self.time_limit}\n")

        self.state = State()
        self.remaining_times = [time_limit, time_limit]
        self.turn = 0
        self.reason = None

        self.dim = (6, 6)

        pygame.init()

        self.screen = pygame.display.set_mode((70*self.dim[1] + 100, 70*self.dim[0] + 150))
        pygame.display.set_caption("Oxono GUI")

        self.number_font = pygame.font.Font(None, 36)
        self.win_font = pygame.font.Font(None, 52)

        self.action_parts = []

        self.pieces = {
            ('x', 0): self._create_piece_surface('x', 0),
            ('x', 1): self._create_piece_surface('x', 1),
            ('o', 0): self._create_piece_surface('o', 0),
            ('o', 1): self._create_piece_surface('o', 1)
        }

        self.totems = {
            'X': self._create_totem_surface('x'),
            'O': self._create_totem_surface('o')
        }

        self.clock = pygame.time.Clock()
        self.running = True

        self.winner = None
        self._turn_start_time = None
        self._turn_start_remaining = None

        self.play()

    def _create_piece_surface(self, symbol, player):
        surface = pygame.Surface((70, 70), pygame.SRCALPHA)
        center = (35, 35)
        pygame.draw.circle(surface, (51, 63, 73) if player == 1 else (254, 44, 135), center, 28)

        if symbol == 'o':
            pygame.draw.circle(surface, (255, 255, 255), center, 15, width=6)
        else:
            w = 3  # half-width of the arms
            s = 12 # spread from center

            points = [
                (35-s, 35-s+w), (35-s+w, 35-s), (35, 35-w), (35+s-w, 35-s),
                (35+s, 35-s+w), (35+w, 35), (35+s, 35+s-w), (35+s-w, 35+s),
                (35, 35+w), (35-s+w, 35+s), (35-s, 35+s-w), (35-w, 35)
            ]
            pygame.draw.polygon(surface, (255, 255, 255), points)

        return surface

    def _create_totem_surface(self, symbol):
        surface = pygame.Surface((70, 70), pygame.SRCALPHA)

        rect_area = pygame.Rect(7, 7, 56, 56)

        pygame.draw.rect(surface, (1, 195, 255), rect_area, border_radius=12)

        if symbol == 'o':
            pygame.draw.circle(surface, (255, 255, 255), (35, 35), 15, width=6)
        else:
            w = 3  # half-width of the arms
            s = 12 # spread from center

            points = [
                (35-s, 35-s+w), (35-s+w, 35-s), (35, 35-w), (35+s-w, 35-s),
                (35+s, 35-s+w), (35+w, 35), (35+s, 35+s-w), (35+s-w, 35+s),
                (35, 35+w), (35-s+w, 35+s), (35-s, 35+s-w), (35-w, 35)
            ]
            pygame.draw.polygon(surface, (255, 255, 255), points)

        return surface

    def is_possible_action(self, action):
        current_phase = len(self.action_parts)
        if current_phase == SELECT_TOTEM:
            return True
        elif current_phase == SELECT_TOTEM_ACTION:
            return action[0] == self.action_parts[0]
        else:
            return action[0] == self.action_parts[0] and action[1] == self.action_parts[1]

    def _handle_mouse_click(self, pos):
        row = (pos[1] - 50) // 70
        col = (pos[0] - 50) // 70
        if row > 5 or col > 5:
            return

        current_phase = len(self.action_parts)
        if current_phase == SELECT_TOTEM:
            if (row, col) == self.state.totem_O and self.state.pieces_o[self.state.current_player] > 0:
                self.action_parts.append('O')
            elif (row, col) == self.state.totem_X and self.state.pieces_x[self.state.current_player] > 0:
                self.action_parts.append('X')
        elif current_phase == SELECT_TOTEM_ACTION:
            if (row, col) in Game._totems_actions(self.state, self.action_parts[SELECT_TOTEM]):
                self.action_parts.append((row, col))
            else:
                self.action_parts = []
        else:
            if (row, col) in map(lambda a: a[2], filter(lambda a: self.is_possible_action(a), Game.actions(self.state))):
                self.action_parts.append((row, col))
            else:
                self.action_parts = []

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not (Game.is_terminal(self.state) or any(t <= 0 for t in self.remaining_times) or (self.reason is not None)):
                self._handle_mouse_click(event.pos)

    def _draw_board(self):
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                pygame.draw.rect(self.screen, 'Black', (70*j + 50, 70*i + 50, 70, 70), 1)

    def _draw_piece(self, position, value):
        self.screen.blit(self.pieces[value], (70*position[1] + 50, 70*position[0] + 50))

    def _draw_totem(self, position, value):
        self.screen.blit(self.totems[value], (70*position[1] + 50, 70*position[0] + 50))

    def _draw_moves(self):
        current_phase = len(self.action_parts)
        if current_phase == SELECT_TOTEM_ACTION:
            for row, col in Game._totems_actions(self.state, self.action_parts[SELECT_TOTEM]):
                pygame.draw.circle(self.screen, '#81a2c5', (70 * col + 85, 70 * row + 85), 8)
        elif current_phase == SELECT_PIECE_ACTION:
            for act in filter(lambda a: self.is_possible_action(a), Game.actions(self.state)):
                row, col = act[2]
                pygame.draw.circle(self.screen, '#81a2c5', (70 * col + 85, 70 * row + 85), 8)

    def _draw_pieces(self):
        selected_totem = self.action_parts[SELECT_TOTEM] if len(self.action_parts) == SELECT_PIECE_ACTION else None
        totem_move = self.action_parts[SELECT_TOTEM_ACTION] if len(self.action_parts) == SELECT_PIECE_ACTION else None
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                if self.state.board[i][j] is not None:
                    self._draw_piece((i, j), self.state.board[i][j])
                else:
                    if (i, j) == self.state.totem_O and not selected_totem == 'O':
                        self._draw_totem((i, j), 'O')
                    elif (i, j) == self.state.totem_X and not selected_totem == 'X':
                        self._draw_totem((i, j), 'X')
                    elif (i, j) == totem_move:
                        self._draw_totem((i, j), selected_totem)

    def draw(self):
        self.screen.fill('White')

        self._draw_board()
        self._draw_pieces()
        self._draw_moves()

        if self.reason is not None:
            text = self.win_font.render(f"{'Black' if self.winner == 1 else 'Pink'} wins!", True, 'Black')
            text_rect = text.get_rect(center=(self.screen.get_width()//2, 70*self.dim[0] + 80))
            pygame.draw.rect(self.screen, 'White', text_rect)
            self.screen.blit(text, text_rect)

            text = self.number_font.render(self.reason, True, 'Black')
            text_rect = text.get_rect(center=(self.screen.get_width()//2, 70*self.dim[0] + 110))
            pygame.draw.rect(self.screen, 'White', text_rect)
            self.screen.blit(text, text_rect)
        elif Game.is_terminal(self.state):
            u0 = Game.utility(self.state, 0)
            u1 = Game.utility(self.state, 1)
            text = None
            if u0 == u1:
                text = self.win_font.render("Draw!", True, 'Black')
            else:
                text = self.win_font.render(f"{'Pink' if u0 > u1 else 'Black'} wins!", True, 'Black')
            text_rect = text.get_rect(center=(self.screen.get_width()//2, 70*self.dim[0] + 100))
            pygame.draw.rect(self.screen, 'White', text_rect)
            self.screen.blit(text, text_rect)
        elif self.remaining_times[0] <= 0:
            text = self.win_font.render("Black wins!", True, 'Black')
            text_rect = text.get_rect(center=(self.screen.get_width()//2, 70*self.dim[0] + 80))
            pygame.draw.rect(self.screen, 'White', text_rect)
            self.screen.blit(text, text_rect)

            text = self.number_font.render("Timeout", True, 'Black')
            text_rect = text.get_rect(center=(self.screen.get_width()//2, 70*self.dim[0] + 110))
            pygame.draw.rect(self.screen, 'White', text_rect)
            self.screen.blit(text, text_rect)
        elif self.remaining_times[1] <= 0:
            text = self.win_font.render("Pink wins!", True, 'Black')
            text_rect = text.get_rect(center=(self.screen.get_width()//2, 70*self.dim[0] + 80))
            pygame.draw.rect(self.screen, 'White', text_rect)
            self.screen.blit(text, text_rect)

            text = self.number_font.render("Timeout", True, 'Black')
            text_rect = text.get_rect(center=(self.screen.get_width()//2, 70*self.dim[0] + 110))
            pygame.draw.rect(self.screen, 'White', text_rect)
            self.screen.blit(text, text_rect)
        else:
            text = self.number_font.render(f"{'Black' if Game.to_move(self.state) == 1 else 'Pink'} to play", True, 'Black')
            text_rect = text.get_rect(center=(self.screen.get_width()//2, 70*self.dim[0] + 75))
            self.screen.blit(text, text_rect)

            text = self.number_font.render(f"Pink: {truncate(self.remaining_times[0], 2)} s", True, 'Black')
            text_rect = text.get_rect(topleft=(50, 70*self.dim[0] + 110))
            self.screen.blit(text, text_rect)
            text = self.number_font.render(f"Black: {truncate(self.remaining_times[1], 2)} s", True, 'Black')
            text_rect = text.get_rect(topleft=(320, 70*self.dim[0] + 110))
            self.screen.blit(text, text_rect)

        text = self.number_font.render(f"Turn {self.turn+1}", True, 'Black')
        text_rect = text.get_rect(topleft=(50, 12))
        self.screen.blit(text, text_rect)

        pygame.display.flip()

    def update(self):
        if Game.is_terminal(self.state) or any(t <= 0 for t in self.remaining_times) or self.reason is not None:
            return

        current = Game.to_move(self.state)

        if self._turn_start_time is None:
            self._turn_start_time = time.perf_counter()
            self._turn_start_remaining = self.remaining_times[current]

        self.remaining_times[current] = self._turn_start_remaining - (time.perf_counter() - self._turn_start_time)

        action = None
        if len(self.action_parts) > SELECT_PIECE_ACTION:  # We have finished selecting an action
            action = tuple(self.action_parts)
            self.action_parts = []

        if action is None:
            return

        if action not in Game.actions(self.state):
            self.reason = "Invalid action"
            if self.path:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write("invalid\n")
            return

        Game.apply(self.state, action)
        self.turn += 1
        self._turn_start_time = None
        self._turn_start_remaining = None

        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(f"{action}, {self.remaining_times[current]}\n")

    def play(self):
        try:
            while self.running:
                self.handle_events()
                self.update()
                self.draw()
                self.clock.tick(60)
        finally:
            pygame.quit()
            sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Oxono games between two human players")
    parser.add_argument("-p0", type=str, default="human", help="First player (default: human)")
    parser.add_argument("-p1", type=str, default="human", help="Second player (default: human)")
    parser.add_argument("-l", type=str, default=None, metavar="LOG_DIR", help="Log file (default: no logging)")
    parser.add_argument("-t", type=int, default=300, help="Time limit for each player (default: 300 seconds)")
    args = parser.parse_args()

    VisualManager(agent_files=[args.p0, args.p1], time_limit=args.t, path_to_file=args.l)
