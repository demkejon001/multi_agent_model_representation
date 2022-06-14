import numpy as np

from src.agents.destination_selector import DestinationSelector
from src.helper_code.navigation import astar, maze_traversal


class Agent:
    def __init__(self, agent_id, num_actions, state_function, reward_function, discount):
        self.id = agent_id
        self.num_actions = num_actions
        self.state_function = state_function
        # reward_function has dim (num_agents, num_goals). It is interpreted as (consuming_agent, consumed_goal)
        self.reward_function = reward_function
        self.discount = discount

    def get_rewards(self, collisions):
        reward = 0
        for collision in collisions:
            if collision[0] == -2:  # Player collision
                reward += 0
            elif collision[0] == -1:
                reward += 0
            else:
                reward += self.reward_function[collision[1]][collision[0]]
        return reward

    def action(self, state):
        pass


class PathfindingAgent(Agent):
    def __init__(self, agent_id, reward_func, destination_selector: DestinationSelector):
        super().__init__(agent_id, 5, None, reward_func, discount=.99)
        self.num_goals = reward_func.num_goals
        self.path = None
        self.agent_position = None
        self.reachable_positions = None
        self.goal_positions = None
        self.destination_selector = destination_selector
        self.destination = None
        self.env_agent_idx = -1
        self.agent_resetted = False
        self.horizon = 100
        self.steps_taken = 0
        self.old_next_position = None

    def get_correct_action(self, current_position, next_position):
        for action, direction in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
            if (current_position[0] + direction[0], current_position[1] + direction[1]) == next_position:
                return action

    def reset(self, state, agent_idx, horizon=100):
        self.agent_resetted = True
        self.env_agent_idx = agent_idx
        self.path = None
        self.reachable_positions = self.get_reachable_positions(state)
        self.goal_positions = None
        self.destination_selector.reset(self.reachable_positions, state)
        self.destination = None
        self.horizon = horizon
        self.steps_taken = 0
        self.old_next_position = None

    def get_next_action(self, state):
        next_action = 4  # Stay put action
        current_position = self.get_current_position(state)
        if self.path is not None:
            current_position_idx = self.path.index(current_position)
            if current_position_idx == len(self.path) - 1:
                return 4
            next_position = self.path[current_position_idx + 1]
            if next_position == self.old_next_position:  # A hack for collision avoidance among agents
                if np.random.random() < .5:
                    self.old_next_position = None
                    return 4
            for i in range(current_position_idx):
                self.path.pop(0)  # Remove previously traversed positions
            next_action = self.get_correct_action(current_position, next_position)
            self.old_next_position = next_position
        return next_action

    def action(self, state):
        self.steps_taken += 1
        if self.steps_taken > self.horizon:
            self.agent_resetted = False
        if self.agent_resetted is False:
            raise RuntimeError('You need to reset the PathfindingAgent everytime you reset the env.')

        destination = self.destination_selector.get_destination(state)
        if self.path is None or destination != self.destination:
            self.destination = destination
            self.get_path(state)
        return self.get_next_action(state)

    def get_current_position(self, state):
        agent_channel = state[self.num_goals + 1 + self.env_agent_idx]
        return list(zip(*np.where(agent_channel == 1)))[0]

    def get_reachable_positions(self, state):
        current_position = self.get_current_position(state)
        return maze_traversal(state[0], current_position)

    def get_goal_positions(self, state):
        goal_channels = state[1: 1 + self.num_goals]
        goal_channel_and_positions = np.where(goal_channels == 1)
        return list(zip(goal_channel_and_positions[1], goal_channel_and_positions[2]))

    def get_path(self, state):
        if self.goal_positions is None:
            self.goal_positions = self.get_goal_positions(state)
        self.path = None
        if self.destination is not None:
            current_position = self.get_current_position(state)
            path = astar(state[0], current_position, self.destination, self.goal_positions)
            if len(path) > 1:
                self.path = path
