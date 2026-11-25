import pygame
from board import Action


class GameGUI:
    """Pygame renderer for the 10x10 Learn2Slither board."""

    MARGIN = 2
    HEADER_HEIGHT = 50

    # Colors
    BG_COLOR = (24, 26, 32)
    GRID_COLOR = (40, 44, 52)
    HEAD_COLOR = (56, 152, 236)
    BODY_COLOR = (30, 100, 180)
    GREEN_APPLE = (46, 204, 113)
    RED_APPLE = (231, 76, 60)
    TEXT_COLOR = (236, 240, 241)

    def __init__(self, board_size, fps=8, sessions=None, max_win_size=1400):
        pygame.init()
        self.fps = fps
        self.sessions = sessions
        self.board_size = board_size
        self.max_win_size = max_win_size
        self.cell_size = min(100, max_win_size // self.board_size)
        width = self.board_size * self.cell_size
        height = self.board_size * self.cell_size + self.HEADER_HEIGHT
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Learn2Slither - 42")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 18, bold=True)

    def get_human_action(self, current_direction):
        """Poll keyboard events and map arrow/WASD keys to Actions."""
        action = current_direction
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_w):
                    action = Action.UP
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    action = Action.DOWN
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    action = Action.LEFT
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    action = Action.RIGHT
                elif event.key == pygame.K_ESCAPE:
                    return None
        return action

    def draw_cell(self, x, y, color, border_radius=4):
        """Draw a single cell at (x, y) with the specified color."""
        rect = pygame.Rect(
            x * self.cell_size + self.MARGIN,
            y * self.cell_size + self.HEADER_HEIGHT + self.MARGIN,
            max(self.cell_size - 2 * self.MARGIN, 4),
            max(self.cell_size - 2 * self.MARGIN, 4),
        )
        pygame.draw.rect(self.screen, color, rect, border_radius=border_radius)

    def render(self, board):
        """Draw the current state of the Board."""
        self.screen.fill(self.BG_COLOR)

        # Draw HUD header
        status = f"Len: {len(board.snake)}/10   Steps: {board.steps}"
        # add the session number to the status
        status += f"   Session: {board.session_number}"
        if self.sessions is not None:
            status += f"/{self.sessions}"
        text_surf = self.font.render(status, True, self.TEXT_COLOR)
        self.screen.blit(text_surf, (15, 15))

        # Draw 10x10 grid background
        for y in range(self.board_size):
            for x in range(self.board_size):
                self.draw_cell(x, y, self.GRID_COLOR, border_radius=4)

        # Draw Green Apples
        for gx, gy in board.green_apples:
            self._draw_circle(gx, gy, self.GREEN_APPLE)

        # Draw Red Apple
        if board.red_apple is not None:
            self._draw_circle(
                board.red_apple[0], board.red_apple[1], self.RED_APPLE
            )

        # Draw Snake Body and Head
        for idx, (sx, sy) in enumerate(board.snake):
            color = self.HEAD_COLOR if idx == 0 else self.BODY_COLOR
            self.draw_cell(sx, sy, color, border_radius=8)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def _draw_circle(self, gx, gy, color):
        cx = gx * self.cell_size + self.cell_size // 2
        cy = gy * self.cell_size + self.HEADER_HEIGHT + self.cell_size // 2
        pygame.draw.circle(
            self.screen, color, (cx, cy), max(self.cell_size // 2 - 8, 4)
        )

    def close(self):
        pygame.quit()
