#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")
import argparse  # noqa: E402
import sys  # noqa: E402
import pygame  # noqa: E402
from agent import QLearningAgent  # noqa: E402
from board import Board  # noqa: E402
from gui import GameGUI  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Learn2Slither RL Agent")
    parser.add_argument(
        "-sessions", type=int, default=10, help="Number of training sessions"
    )
    parser.add_argument(
        "-save", type=str, default=None, help="Path to save trained model"
    )
    parser.add_argument(
        "-load", type=str, default=None, help="Path to load existing model"
    )
    parser.add_argument(
        "-visual",
        type=str,
        choices=["on", "off"],
        default="on",
        help="Enable or disable graphical and terminal display",
    )
    parser.add_argument(
        "-dontlearn",
        action="store_true",
        help="Disable Q-table updates and exploration",
    )
    parser.add_argument(
        "-step-by-step",
        action="store_true",
        help="Wait for key press before each move",
    )
    parser.add_argument(
        "-speed", type=int, default=15, help="Frames per second when visual=on"
    )
    parser.add_argument(
        "-size", type=int, default=10, help="Size of the board (default 10x10)"
    )
    parser.add_argument(
        '-epsilon', type=float, default=0.2,
        help='Initial exploration rate (default 0.3)'
    )
    parser.add_argument(
        '-max_steps', type=int, default=500,
        help='Maximum steps per game to prevent infinite loops (default 500)'
    )
    parser.add_argument(
        '-max_win_size', type=int, default=1400,
        help='Maximum window size for GUI (default 1400)'
    )
    return parser.parse_args()


def wait_for_step():
    """Pause execution until user presses SPACE/ENTER or closes window."""
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                return True


def handle_gui_events():
    """Keep Pygame window responsive and allow clean exit."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return False
    return True


def main():
    args = parse_args()
    visual_on = args.visual == "on"

    if args.size < 3:
        print("Error: Board size must be at least 3x3.")
        sys.exit(1)
    board = Board(args.size)
    if args.load and not args.dontlearn:
        agent = QLearningAgent(dont_learn=args.dontlearn, epsilon=args.epsilon)
    else:
        agent = QLearningAgent(dont_learn=args.dontlearn)
    if args.load:
        agent.load_model(args.load)

    gui = (
        GameGUI(args.size, fps=args.speed, sessions=args.sessions,
                max_win_size=args.max_win_size)
        if visual_on else None
    )

    max_length = 0
    min_length = float("inf")
    max_duration = 0
    goal_reached_count = 0

    for session in range(1, args.sessions + 1):
        vision = board.get_vision()
        done = False
        session_max_len = len(board.snake)

        # Prevent infinite circling if a model has no apple in sight
        max_steps_per_game = args.max_steps

        while not done and board.steps < max_steps_per_game:
            if visual_on:
                board.print_terminal_vision()
                gui.render(board)
                if args.step_by_step:
                    if not wait_for_step():
                        gui.close()
                        sys.exit(0)
                elif not handle_gui_events():
                    gui.close()
                    sys.exit(0)

            action = agent.choose_action(vision)
            if visual_on:
                print(f"Chosen action: {action.name}\n")

            next_vision, reward, done = board.step(action)
            agent.learn(vision, action, reward, next_vision, done)

            vision = next_vision
            if len(board.snake) > session_max_len:
                session_max_len = len(board.snake)

        agent.decay_epsilon()
        snake_len = len(board.snake)
        if snake_len < min_length:
            min_length = snake_len

        if snake_len > max_length:
            max_length = snake_len
        if board.steps > max_duration:
            max_duration = board.steps
        if snake_len >= 10:
            goal_reached_count += 1
        board.reset()

    if gui:
        gui.close()

    print(
        f"Game over, max length = {max_length}, "
        f"max duration = {max_duration}"
    )
    print(f"Min length reached: {min_length}")
    if args.sessions > 1:
        print(
            f"Sessions reaching length >= 10: "
            f"{goal_reached_count}/{args.sessions}"
        )
    print(f'\n\nQ_table summary: {agent.q_table_info()}')

    if args.save:
        agent.save_model(args.save)


if __name__ == "__main__":
    main()
