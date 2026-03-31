import pygame
import sys
import os
import time
from oxono.visual_manager import VisualManager

# --- Load agents from folder ---
AGENTS = ["human"] + [
    f for f in os.listdir("agents")
    if f.endswith(".py") and f != "agent.py"
]


class Dropdown:
    def __init__(self, x, y, w, h, options, font):
        self.rect = pygame.Rect(x, y, w, h)
        self.options = options
        self.selected = 0
        self.open = False
        self.font = font

    def draw(self, screen):
        # Main dropdown box (same color as game)
        color = (1, 195, 255)
        pygame.draw.rect(screen, color, self.rect, border_radius=8)

        text = self.font.render(self.options[self.selected], True, (0, 0, 0))
        screen.blit(text, text.get_rect(center=self.rect.center))

        # Draw options if open
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
            # Toggle dropdown
            if self.rect.collidepoint(event.pos):
                self.open = not self.open
            # Select option
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


class MenuManager:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((500, 600))
        pygame.display.set_caption("Oxono Menu")

        self.font = pygame.font.Font(None, 36)
        self.big_font = pygame.font.Font(None, 48)

        self.clock = pygame.time.Clock()
        self.running = True

        # Dropdowns
        self.dropdown_p0 = Dropdown(100, 120, 300, 40, AGENTS, self.font)
        self.dropdown_p1 = Dropdown(100, 220, 300, 40, AGENTS, self.font)

        self.play()

    def draw(self):
        self.screen.fill((255, 255, 255))

        # Title
        title = self.big_font.render("OXONO", True, (0, 0, 0))
        self.screen.blit(title, title.get_rect(center=(250, 50)))

        # Labels
        p0_text = self.font.render("Player 0 (Pink)", True, (254, 44, 135))
        self.screen.blit(p0_text, (100, 90))

        p1_text = self.font.render("Player 1 (Black)", True, (51, 63, 73))
        self.screen.blit(p1_text, (100, 190))

        # Draw dropdowns
        self.dropdown_p0.draw(self.screen)
        self.dropdown_p1.draw(self.screen)

        # Play button
        pygame.draw.rect(self.screen, (1, 195, 255), (150, 300, 200, 50), border_radius=12)
        play_text = self.font.render("PLAY", True, (0, 0, 0))
        self.screen.blit(play_text, play_text.get_rect(center=(250, 325)))

        # Ensure opened dropdown is drawn on top
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

            # Handle dropdown interactions
            self.dropdown_p0.handle_event(event)
            self.dropdown_p1.handle_event(event)

            # Ensure only one dropdown is open at a time
            if self.dropdown_p0.open:
                self.dropdown_p1.open = False
            if self.dropdown_p1.open:
                self.dropdown_p0.open = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos

                # Play button clicked
                if 150 < x < 350 and 300 < y < 350:
                    self.running = False
                    pygame.quit()

                    # Format agent path
                    def format_agent(agent):
                        return agent if agent == "human" else os.path.join("agents", agent)

                    # Extract clean agent name
                    def get_agent_name(agent):
                        if agent == "human":
                            return "human"

                        name = os.path.splitext(os.path.basename(agent))[0]

                        if name.endswith("_agent"):
                            name = name[:-6]

                        return name

                    # Build log path
                    def build_log_path(p0, p1):
                        name0 = get_agent_name(p0)
                        name1 = get_agent_name(p1)

                        a, b = sorted([name0, name1])

                        folder = os.path.join("data", f"{a}_vs_{b}")
                        os.makedirs(folder, exist_ok=True)

                        filename = f"game_{int(time.time())}.log"
                        return os.path.join(folder, filename)

                    # Get selected agents
                    raw_p0 = self.dropdown_p0.get_value()
                    raw_p1 = self.dropdown_p1.get_value()

                    agent0 = format_agent(raw_p0)
                    agent1 = format_agent(raw_p1)

                    # Generate log file path
                    log_path = build_log_path(agent0, agent1)

                    # Launch game
                    VisualManager(
                        agent_files=[agent0, agent1],
                        path_to_file=log_path
                    )

    def play(self):
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(60)


if __name__ == "__main__":
    MenuManager()