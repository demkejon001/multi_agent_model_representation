import random
import numpy as np

from tommas.agents.agents import Agent
from tommas.agents.destination_selector import DestinationSelector
from tommas.env.separated_gridworld import SeparatedGridworld
from tommas.helper_code.navigation import astar, maze_traversal


class HandCodedAgent(Agent):
    def __init__(self, agent_id, num_actions, state_function, rewards, discount, num_goals):
        super().__init__(agent_id, num_actions, state_function, rewards, discount)
        self.num_goals = num_goals
        self.goal_priority = list(range(self.num_goals))
        random.shuffle(self.goal_priority)
        self.path = None
        self.agent_position = None
        self.current_goal = None
        self.current_goal_position = None
        self.reachable_goals = []
        self.old_world = []

        self.old_position = None
        self.old_action = None
        self.old_state = None
        self.sep = SeparatedGridworld(10, num_goals, (100, 000))

    def check_new_world(self, state_walls):
        if np.array_equal(self.old_world, state_walls):
            return False
        self.old_world = state_walls
        return True

    def get_correct_action(self, current_position, next_position):
        for action, direction in enumerate([(-1, 0), (0, 1), (1, 0), (0, -1)]):
            if (current_position[0] + direction[0], current_position[1] + direction[1]) == next_position:
                return action

    def remove_goal(self):
        if self.current_goal is not None:
            self.goal_priority[self.current_goal] = -1
        self.reachable_goals = []
        self.path = None
        self.current_goal = None
        self.current_goal_position = None

    def check_goal_attained(self, state, current_position):
        if self.path is not None and self.current_goal is not None:
            if self.path[-1] == current_position:
                self.remove_goal()
                return True
        return False

    def get_next_action(self, state):
        next_action = 4  # Stay put action
        current_position = self.get_current_position(state)
        if self.check_goal_attained(state, current_position):
            self.get_new_goal(state)
            self.get_path(state)
        if self.path is not None:

            next_position = self.path[self.path.index(current_position) + 1]
            next_action = self.get_correct_action(current_position, next_position)

        self.old_action = next_action  # TODO: REMOVE
        self.old_position = self.get_current_position(state)
        self.old_state = state

        return next_action

    def action(self, state):
        if self.check_new_world(state[0]):
            self.remove_goal()
        if self.path is None:
            self.get_new_goal(state)
            self.get_path(state)
        return self.get_next_action(state)

    def get_current_position(self, state):
        agent_channel = state[self.num_goals + 1 + self.id]
        return list(zip(*np.where(agent_channel == 1)))[0]

    def get_reachable_positions(self, state):
        current_position = self.get_current_position(state)
        return maze_traversal(state[0], current_position)

    def get_reachable_goals(self, state):
        reachable_positions = self.get_reachable_positions(state)
        # print('reachable pos', reachable_positions)
        for position in reachable_positions:
            for goal in range(self.num_goals):
                if state[goal + 1][position[0]][position[1]] == 1:
                    self.reachable_goals.append((goal, position[0], position[1]))

    def get_new_goal(self, state):
        if len(self.reachable_goals) <= 0:
            self.get_reachable_goals(state)
            # print('reachable goals', self.reachable_goals)
        if len(self.reachable_goals) > 0:
            max_val = max([self.goal_priority[goal_position[0]] for goal_position in self.reachable_goals])
            self.current_goal = self.goal_priority.index(max_val)
            # print([self.goal_priority[goal_position[0]] for goal_position in self.reachable_goals])
            # print('goalpriority', self.goal_priority)
            # print('current goal', self.current_goal)
            for goal in self.reachable_goals:
                if goal[0] == self.current_goal:
                    self.current_goal_position = (goal[1], goal[2])
                    # print('current goal pos', self.current_goal_position)

    def get_path(self, state):
        if self.current_goal_position is not None:
            current_position = self.get_current_position(state)
            path = astar(state[0], current_position, self.current_goal_position)
            # print('path', path)
            if len(path) > 0:
                self.path = path


class PathfindingAgent(Agent):
    def __init__(self, agent_id, reward_func, destination_selector: DestinationSelector):
        super().__init__(agent_id, 5, None, reward_func, discount=.99)
        self.num_goals = reward_func.num_goals
        # self.goal_rewards = goal_rewards
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


class OneDirectionAgent(Agent):
    def __init__(self, agent_id, num_actions, state_function, rewards, discount, direction=None):
        super().__init__(agent_id, num_actions, state_function, rewards, discount)
        if direction is None:
            self.one_direction = random.randint(0, self.num_actions - 1)
        else:
            self.one_direction = direction

    def action(self, state):
        return self.one_direction

