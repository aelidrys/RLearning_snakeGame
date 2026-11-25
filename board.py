import random
from enum import Enum


class Action(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class Board:
    """10x10 Snake environment matching Learn2Slither specifications."""

    def __init__(self, size=10):
        self.snake = []
        self.green_apples = []
        self.red_apple = None
        self.direction = Action.RIGHT
        self.game_over = False
        self.steps = 0
        self.size = size
        self.session_number = 0
        self.reset()

    def reset(self):
        """Initialize a contiguous 3-cell snake, 2 green apples, 1 red."""
        self.game_over = False
        self.steps = 0
        self.green_apples = []
        self.red_apple = None
        self.session_number += 1

        # Place snake head safely away from borders so 3 cells fit
        hx = random.randint(2, max(2, self.size - 3))
        hy = random.randint(2, max(2, self.size - 3))
        self.direction = random.choice(list(Action))
        dx, dy = self.direction.value

        # Build snake from head to tail opposite to initial movement
        self.snake = [
            (hx, hy),
            (hx - dx, hy - dy),
            (hx - 2 * dx, hy - 2 * dy),
        ]

        # Spawn 2 green apples and 1 red apple
        for _ in range(2):
            self._spawn_apple(is_green=True)
        self._spawn_apple(is_green=False)

        return self.get_vision()

    def _get_empty_cells(self):
        occupied = set(self.snake) | set(self.green_apples)
        if self.red_apple is not None:
            occupied.add(self.red_apple)
        return [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) not in occupied
        ]

    def _spawn_apple(self, is_green):
        empty = self._get_empty_cells()
        if not empty:
            return
        pos = random.choice(empty)
        if is_green:
            self.green_apples.append(pos)
        else:
            self.red_apple = pos

    def step(self, action):
        """
        Execute one move on the board.
        Returns: (vision_dict, reward, done)
        """
        if self.game_over:
            return self.get_vision(), 0.0, True

        self.direction = action
        self.steps += 1
        dx, dy = action.value
        hx, hy = self.snake[0]
        new_head = (hx + dx, hy + dy)

        # 1. Check wall collision
        if not (0 <= new_head[0] < self.size and 0 <= new_head[1] < self.size):
            self.game_over = True
            return self.get_vision(), -100.0, True

        # 2. Check self collision (exclude tail tip if it won't grow)
        will_grow = new_head in self.green_apples
        body_to_check = self.snake if will_grow else self.snake[:-1]
        if new_head in body_to_check:
            self.game_over = True
            return self.get_vision(), -100.0, True

        # Move head forward
        self.snake.insert(0, new_head)

        # 3. Handle Green Apple (+1 length)
        if new_head in self.green_apples:
            self.green_apples.remove(new_head)
            self._spawn_apple(is_green=True)
            return self.get_vision(), 15.0, False

        # Remove tail for normal move
        self.snake.pop()

        # 4. Handle Red Apple (-1 length)
        if new_head == self.red_apple:
            self.red_apple = None
            if self.snake:
                self.snake.pop()
            if len(self.snake) == 0:
                self.game_over = True
                return self.get_vision(), -100.0, True
            self._spawn_apple(is_green=False)
            return self.get_vision(), -10.0, False

        # 5. Empty space move
        return self.get_vision(), -0.1, False

    def _cell_char(self, x, y):
        if not (0 <= x < self.size and 0 <= y < self.size):
            return "W"
        pos = (x, y)
        if pos == self.snake[0]:
            return "H"
        if pos in self.snake:
            return "S"
        if pos in self.green_apples:
            return "G"
        if pos == self.red_apple:
            return "R"
        return "0"

    def get_vision(self):
        """
        Return ONLY the 4 straight sightlines from the snake's head.
        This enforces the strict partial-observability rule.
        """
        if not self.snake:
            return {"UP": ["W"], "DOWN": ["W"], "LEFT": ["W"], "RIGHT": ["W"]}

        hx, hy = self.snake[0]
        vision = {}
        for act in Action:
            dx, dy = act.value
            line = []
            cx, cy = hx + dx, hy + dy
            while 0 <= cx < self.size and 0 <= cy < self.size:
                line.append(self._cell_char(cx, cy))
                cx += dx
                cy += dy
            line.append("W")
            vision[act.name] = line
        return vision

    def print_terminal_vision(self):
        """Render the exact cross-shaped vision required by Part 2."""
        if not self.snake:
            return
        hx, hy = self.snake[0]
        for y in range(-1, self.size + 1):
            row = []
            for x in range(-1, self.size + 1):
                if x == hx or y == hy:
                    row.append(self._cell_char(x, y))
                else:
                    row.append(" ")
            print("".join(row))

    def snake_length(self):
        return len(self.snake)
