import json
import os
from typing import Any, Dict, Union
import numpy as np
import time


class TreeofThoughts:
    """
    A class representing a tree of thoughts.

    Attributes:
        agent: The agent used for reasoning and evaluation.
        tree: The tree structure containing the nodes and their evaluations.
        best_state: The best state found so far.
        best_value: The best value found so far.
        history: The history of evaluated states.

    Methods:
        save_tree_to_json: Saves the tree structure to a JSON file.
        log_new_state: Logs a new state and its evaluation.
    """

    def __init__(self, agent):
        self.agent = agent
        self.tree: Dict[str, Dict[str, Union[float, Dict[str, Any]]]] = {
            "nodes": {},
        }
        self.best_state = None
        self.best_value = float("-inf")
        self.history = []  # added line initialize history

    def save_tree_to_json(self, file_name):
        """
        Saves the tree structure to a JSON file.

        Args:
            file_name: The name of the JSON file to save the tree structure to.
        """
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w") as json_file:
            json.dump(self.tree, json_file, indent=4)

    def log_new_state(self, state, evaluation):
        """
        Logs a new state and its evaluation.

        Args:
            state: The state to log.
            evaluation: The evaluation of the state.
        """
        if not (type(state) == str):
            state = " | ".join(state)
        if state in self.tree["nodes"]:
            self.tree["nodes"][state]["thoughts"].append(evaluation)
        else:
            self.tree["nodes"][state] = {"thoughts": [evaluation]}


class BFS(TreeofThoughts):
    """Class representing the Breadth-First Search algorithm for Tree of Thoughts."""

    def solve(
        self,
        initial_prompt,
        num_thoughts,
        max_steps,
        max_states,
        consistency=1,
    ):
        """
        Solve the Tree of Thoughts problem using the Breadth-First Search algorithm.

        Args:
            initial_prompt (str): The initial prompt for generating thoughts.
            num_thoughts (int): The number of thoughts to generate at each state.
            max_steps (int): The maximum number of steps to take in the search.
            max_states (int): The maximum number of states to keep track of.

        Returns:
            str or None: The generated solution or the highest rated state.
        """
        current_states = [initial_prompt]
        state_values = {}

        all_time = 0
        all_tht_time = 0
        all_eval_time = 0
        all_acc = 0
        all_tht_acc = 0
        all_eval_acc = 0

        for step in range(1, max_steps + 1):
            selected_states = []
            for state in current_states:
                thoughts, thought_time, thought_acc = self.agent.generate_thoughts(
                    state, num_thoughts, initial_prompt
                )
                all_time += thought_time
                all_tht_time += thought_time

                evaluated_thoughts = {}
                results, evaluated_time, evaluated_acc = self.agent.evaluate_states(
                    thoughts, initial_prompt
                )
                all_time += evaluated_time
                all_eval_time += evaluated_time
                all_tht_acc += thought_acc
                all_eval_acc += evaluated_acc
                all_acc = float((thought_acc + evaluated_acc) / 2)
                evaluated_thoughts = results

                for thought, value in evaluated_thoughts.items():
                    flattened_state = (
                        (state, thought)
                        if isinstance(state, str)
                        else (*state, thought)
                    )
                    selected_states.append((flattened_state, value))

                selected_states.sort(key=lambda x: x[1], reverse=True)
                selected_states = selected_states[:max_states]  # Select only the top states

                for state, value in selected_states:
                    # No pruning condition, simply log the state and value
                    state_values[state] = value
                    self.log_new_state(state, value)

        all_acc = float(all_acc / max_steps)
        all_tht_acc = float(all_tht_acc / max_steps)
        all_eval_acc = float(all_eval_acc / max_steps)

        if state_values:
            highest_rated_solution = max(state_values.items(), key=lambda x: x[1])
            highest_rated_state = highest_rated_solution[0]
            solution = self.agent.generate_solution(
                initial_prompt, highest_rated_state
            )
            return all_time, all_tht_time, all_eval_time, all_acc, all_tht_acc, all_eval_acc, solution if solution else highest_rated_state
        else:
            return all_time, all_tht_time, all_eval_time, all_acc, all_tht_acc, all_eval_acc, None