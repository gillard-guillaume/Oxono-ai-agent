import pygame
import sys
import os
import time
from oxono.visual_manager import VisualManager
from oxono.replayer import Replayer

import tkinter as tk
from tkinter import filedialog

# --- Load agents from folder ---
AGENTS = ["human"] + [
    f for f in os.listdir("agents")
    if f.endswith(".py") and f != "agent.py"
]


# =========================
# Dropdown Component
# =========================
class Dropdown:
    def __init__(self, x, y, w, h, options, font):
        self.rect = pygame.Rect(x, y, w, h)
        self.options = options
        self.selected = 0
        self.open = False
        self.font = font

    def draw(self, screen):
        pygame.draw.rect(screen, (1, 195, 255), self.rect, border_radius=8)

        text = self.font.render(self.options[self.selected], True, (0, 0, 0))
        screen.blit(text, text.get_rect(center=self.rect.center))

        if self.open:
            for i, option in enumerate(self.options):
                rect = pygame.Rect(
                    self.rect.x,
                    self.rect.y + (i + 1) * self.rect.height,
                    self.rect.width,
                    self.rect.height
                )
                pygame.draw.rect(screen, (220, 220, 220), rect, border_radius=8)
                txt = self.font.render(option, True, (0, 0, 0))
                screen.blit(txt, txt.get_rect(center=rect.center))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.open = not self.open
            elif self.open:
                for i in range(len(self.options)):
                    rect = pygame.Rect(
                        self.rect.x,
                        self.rect.y + (i + 1) * self.rect.height,
                        self.rect.width,
                        self.rect.height
                    )
                    if rect.collidepoint(event.pos):
                        self.selected = i
                        self.open = False

    def get_value(self):
        return self.options[self.selected]


# =========================
# MAIN MENU
# =========================
class MainMenu:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((700, 550))
        pygame.display.set_caption("Oxono")

        self.font = pygame.font.Font(None, 50)
        self.big_font = pygame.font.Font(None, 80)

        self.clock = pygame.time.Clock()
        self.running = True

        self.play()

    def draw(self):
        self.screen.fill((255, 255, 255))

        title = self.big_font.render("OXONO", True, (0, 0, 0))
        self.screen.blit(title, title.get_rect(center=(350, 120)))

        # Play button
        pygame.draw.rect(self.screen, (1, 195, 255), (250, 250, 200, 70), border_radius=15)
        play_text = self.font.render("PLAY", True, (0, 0, 0))
        self.screen.blit(play_text, play_text.get_rect(center=(350, 285)))

        # Replay button
        pygame.draw.rect(self.screen, (200, 200, 200), (250, 350, 200, 70), border_radius=15)
        replay_text = self.font.render("REPLAY", True, (0, 0, 0))
        self.screen.blit(replay_text, replay_text.get_rect(center=(350, 385)))

        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos

                if 250 < x < 450 and 250 < y < 320:
                    self.running = False
                    pygame.quit()
                    PlayMenu()

                if 250 < x < 450 and 350 < y < 420:
                    self.running = False
                    pygame.quit()
                    ReplayMenu()

    def play(self):
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(60)


# =========================
# PLAY MENU
# =========================
class PlayMenu:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((750, 900))
        pygame.display.set_caption("Play Menu")

        self.font = pygame.font.Font(None, 42)
        self.big_font = pygame.font.Font(None, 70)

        self.clock = pygame.time.Clock()
        self.running = True

        self.dropdown_p0 = Dropdown(150, 160, 400, 60, AGENTS, self.font)
        self.dropdown_p1 = Dropdown(150, 300, 400, 60, AGENTS, self.font)

        # just for testing the models at the moment
        self.dropdown_p0.selected = AGENTS.index("ab4_agent.py")
        self.dropdown_p1.selected = AGENTS.index("human")

        self.play()

    def draw(self):
        self.screen.fill((255, 255, 255))

        title = self.big_font.render("PLAY", True, (0, 0, 0))
        self.screen.blit(title, title.get_rect(center=(350, 80)))

        p0_text = self.font.render("Player 0 (Pink)", True, (254, 44, 135))
        self.screen.blit(p0_text, (150, 120))

        p1_text = self.font.render("Player 1 (Black)", True, (51, 63, 73))
        self.screen.blit(p1_text, (150, 260))

        self.dropdown_p0.draw(self.screen)
        self.dropdown_p1.draw(self.screen)

        # Play button
        pygame.draw.rect(self.screen, (1, 195, 255), (250, 420, 200, 70), border_radius=15)
        play_text = self.font.render("PLAY", True, (0, 0, 0))
        self.screen.blit(play_text, play_text.get_rect(center=(350, 455)))

        # Back button
        pygame.draw.rect(self.screen, (200, 200, 200), (20, 20, 120, 50), border_radius=10)
        back_text = self.font.render("BACK", True, (0, 0, 0))
        self.screen.blit(back_text, back_text.get_rect(center=(80, 45)))

        # Ensure dropdown on top
        if self.dropdown_p0.open:
            self.dropdown_p0.draw(self.screen)
        if self.dropdown_p1.open:
            self.dropdown_p1.draw(self.screen)

        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            self.dropdown_p0.handle_event(event)
            self.dropdown_p1.handle_event(event)

            if self.dropdown_p0.open:
                self.dropdown_p1.open = False
            if self.dropdown_p1.open:
                self.dropdown_p0.open = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos

                # Back button
                if 20 < x < 140 and 20 < y < 70:
                    self.running = False
                    pygame.quit()
                    MainMenu()

                # Play button
                if 250 < x < 450 and 420 < y < 490:
                    self.running = False
                    pygame.quit()

                    def format_agent(agent):
                        return agent if agent == "human" else os.path.join("agents", agent)

                    def get_agent_name(agent):
                        if agent == "human":
                            return "human"
                        name = os.path.splitext(os.path.basename(agent))[0]
                        if name.endswith("_agent"):
                            name = name[:-6]
                        return name

                    def build_log_path(p0, p1):
                        a, b = sorted([get_agent_name(p0), get_agent_name(p1)])
                        folder = os.path.join("data", f"{a}_vs_{b}")
                        os.makedirs(folder, exist_ok=True)
                        return os.path.join(folder, f"game_{int(time.time())}.log")

                    agent0 = format_agent(self.dropdown_p0.get_value())
                    agent1 = format_agent(self.dropdown_p1.get_value())

                    log_path = build_log_path(agent0, agent1)

                    VisualManager(
                        agent_files=[agent0, agent1],
                        path_to_file=log_path
                    )
                    PlayMenu()

    def play(self):
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(60)



# =========================
# FILE DIALOG FUNCTION
# =========================
def open_file_dialog():
    root = tk.Tk()
    root.withdraw()  # Hide Tkinter window

    file_path = filedialog.askopenfilename(
        initialdir="data",
        title="Select a game log",
        filetypes=[("Log files", "*.log"), ("All files", "*.*")]
    )

    root.destroy()
    return file_path


# =========================
# REPLAY MENU
# =========================
class ReplayMenu:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((700, 550))
        pygame.display.set_caption("Replay")

        self.font = pygame.font.Font(None, 50)
        self.big_font = pygame.font.Font(None, 70)

        self.clock = pygame.time.Clock()
        self.running = True

        self.play()

    def draw(self):
        self.screen.fill((255, 255, 255))

        # Title
        title = self.big_font.render("REPLAY", True, (0, 0, 0))
        self.screen.blit(title, title.get_rect(center=(350, 120)))

        # Load button
        pygame.draw.rect(self.screen, (1, 195, 255), (200, 250, 300, 70), border_radius=12)
        text = self.font.render("LOAD GAME", True, (0, 0, 0))
        self.screen.blit(text, text.get_rect(center=(350, 285)))

        # Back button
        pygame.draw.rect(self.screen, (200, 200, 200), (20, 20, 120, 50), border_radius=10)
        back_text = self.font.render("BACK", True, (0, 0, 0))
        self.screen.blit(back_text, back_text.get_rect(center=(80, 45)))

        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos

                # Back button
                if 20 < x < 140 and 20 < y < 70:
                    self.running = False
                    pygame.quit()
                    from main import MainMenu  # avoid circular import
                    MainMenu()

                # Load game button
                if 200 < x < 500 and 250 < y < 320:
                    self.running = False
                    pygame.quit()

                    file_path = open_file_dialog()

                    if file_path:
                        r = Replayer(file_path)
                        r.play()
                        ReplayMenu()
                    else:
                        from main import MainMenu
                        MainMenu()

    def play(self):
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(60)



# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    MainMenu()