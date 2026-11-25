import json
import os
import random
from board import Action


class QLearningAgent:
    """Tabular Q-Learning agent operating strictly on 4-way snake vision."""

    ACTIONS = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    def __init__(
        self,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        dont_learn=False,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 0.01 if dont_learn else epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.dont_learn = dont_learn
        self.q_table = {}

    def encode_state(self, vision):
        """
        Convert raw 4-direction sightlines into a compact 12-bit state tuple.
        Uses ONLY the vision dictionary (no board coordinates).
        """
        danger = []
        green_seen = []
        red_adjacent = []

        for act in self.ACTIONS:
            line = vision[act.name]
            first_cell = line[0] if line else "W"

            # 1. Immediate fatal collision (Wall or Body)
            danger.append(int(first_cell in ("W", "S")))

            # 2. Is a Green Apple in clear line of sight before an obstacle?
            has_green = 0
            for cell in line:
                if cell == "G":
                    has_green = 1
                    break
                if cell in ("W", "S"):
                    break
            green_seen.append(has_green)

            # 3. Is a Red Apple 1 step away?
            red_adjacent.append(int(first_cell == "R"))

        return tuple(danger + green_seen + red_adjacent)

    def _get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0, 0.0]
        return self.q_table[state]

    def choose_action(self, vision):
        """Select an action using epsilon-greedy policy."""
        state = self.encode_state(vision)

        # Exploration (disabled when dont_learn is True)
        if not self.dont_learn and random.random() < self.epsilon:
            return random.choice(self.ACTIONS)

        # Avoiding infinite loops: if no green apple is visible,
        # prefer safe moves with a 15% chance
        green_visible = any(state[4:8])
        if not green_visible and random.random() < 0.15:
            safe_actions = [
                self.ACTIONS[i]
                for i in range(4)
                if state[i] == 0 and state[8 + i] == 0
            ]
            if safe_actions:
                return random.choice(safe_actions)

        # Exploitation with random tie-breaking to prevent infinite loops
        q_vals = self._get_q_values(state)
        max_q = max(q_vals)
        best_indices = [i for i, q in enumerate(q_vals) if q == max_q]
        return self.ACTIONS[random.choice(best_indices)]

    def learn(self, vision, action, reward, next_vision, done):
        """Update Q-value using the Bellman equation."""
        if self.dont_learn:
            return

        state = self.encode_state(vision)
        next_state = self.encode_state(next_vision)
        act_idx = self.ACTIONS.index(action)

        q_vals = self._get_q_values(state)
        current_q = q_vals[act_idx]

        if done:
            target = reward
        else:
            next_max_q = max(self._get_q_values(next_state))
            target = reward + self.gamma * next_max_q

        q_vals[act_idx] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        """Reduce exploration rate at the end of each training session."""
        if not self.dont_learn and self.epsilon > self.epsilon_min:
            self.epsilon = max(
                self.epsilon_min, self.epsilon * self.epsilon_decay
            )

    def save_model(self, filepath):
        """Export Q-table and epsilon state to a JSON text file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        serialized_table = {
            ",".join(map(str, k)): v for k, v in self.q_table.items()
        }
        payload = {
            "epsilon": self.epsilon,
            "q_table": serialized_table,
        }
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"Error saving model to {filepath}: {e}")
            return
        print(f"Save learning state in {filepath}")

    def load_model(self, filepath):
        """Restore Q-table and epsilon state from a saved model file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except FileNotFoundError:
            print(f"Model file {filepath} not found. Starting fresh.")
            return
        if not self.dont_learn:
            saved_epsilon = payload.get("epsilon", self.epsilon)
            self.epsilon = max(self.epsilon, saved_epsilon)
        self.q_table = {
            tuple(map(int, k.split(","))): v
            for k, v in payload["q_table"].items()
        }
        print(f"Load trained model from {filepath}")
        print(f'epsilon: {self.epsilon}')

    def q_table_info(self):
        """Return a summary of the Q-table for debugging purposes."""
        num_states = len(self.q_table)
        num_actions = len(self.ACTIONS)
        return {
            "num_states": num_states,
            "num_actions": num_actions,
            "epsilon": self.epsilon,
        }
