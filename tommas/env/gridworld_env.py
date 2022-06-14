import numpy as np
from typing import List, Tuple, Union
import gym
from gym.spaces import Discrete, Box

from tommas.env.gridworld import GridWorld, RepositionGridworld
from tommas.env.separated_gridworld import SeparatedGridworld
from tommas.agents.reward_function import RewardFunction
from tommas.viz.visual import Drawer, ToMnetRLDrawer

# from tommas.agents.hand_coded_agents import PathfindingAgent
from tommas.agents.destination_selector import RepositionPicnicDestinationSelector


class GridworldEnv(gym.Env):
    def __init__(self,
                 num_agents: int,
                 agent_reward_functions: List[RewardFunction],
                 num_goals: int = 4,
                 horizon: int = 30,
                 dim: Tuple[int, int] = (11, 11),
                 min_num_walls: int = 0,
                 max_num_walls: int = 2,
                 seed: Union[None, int] = None):
        self.world = GridWorld(num_agents=num_agents, num_goals=num_goals, dim=dim, min_num_walls=min_num_walls,
                               max_num_walls=max_num_walls, seed=seed)
        self.agent_reward_functions = agent_reward_functions
        self.num_agents = num_agents
        self.num_goals = num_goals
        self.dim = dim
        self.horizon = horizon
        self.num_actions = len(self.world.actions)
        self.state_size = self.world.state.shape
        self.current_step = 0
        self._drawer = None
        self._debug_print = False
        self.action_space = Discrete(self.num_actions)
        self.world.reset()
        state = self.get_world_state()
        self.observation_space = Box(low=0, high=1, shape=state.shape, dtype=np.uint8)

    def get_world_state(self):
        return np.copy(self.world.state).astype(dtype=np.uint8)

    def draw(self):
        if self._drawer is not None:
            self._drawer.draw(self.get_world_state())
        if self._debug_print:
            print(self.world)

    def calculate_rewards_from_collisions(self, collision_record):
        player_rewards = np.zeros(self.num_agents)
        for agent_idx in range(self.num_agents):
            player_rewards[agent_idx] += self.agent_reward_functions[agent_idx].get_reward(agent_idx, collision_record)
        return player_rewards

    def get_goals_consumed_from_collisions(self, collision_record):
        goals_consumed = dict()
        goal_collisions = collision_record.get_goal_collisions()
        for agent_id, agent_goal_collision in enumerate(goal_collisions):
            if int(agent_goal_collision) >= 0:
                goals_consumed[agent_id] = int(agent_goal_collision)
        return goals_consumed

    def reset(self, create_new_world=True, agent_reward_functions=None):
        self.world.reset(create_new_world)
        if agent_reward_functions is not None:
            self.agent_reward_functions = agent_reward_functions
        self.current_step = 0
        self.draw()
        return self.get_world_state()

    def step(self, actions):
        if isinstance(actions, int):
            actions = [actions]

        if self.current_step < self.horizon:
            self.current_step += 1
            done = True if self.current_step == self.horizon else False
            collision_record = self.world.step(actions)
            self.draw()
            player_rewards = self.calculate_rewards_from_collisions(collision_record)
            goals_consumed = self.get_goals_consumed_from_collisions(collision_record)
            return self.get_world_state(), player_rewards, done, goals_consumed
        else:
            return None, np.zeros(self.num_agents), True, None

    def render(self, mode='human'):
        if mode == "human":
            wall_positions, goal_positions, agent_positions = self.world.get_feature_positions()
            self._drawer = Drawer(self.world.dim, wall_positions, goal_positions, agent_positions)
        elif mode == "ansi":
            self._debug_print = True
        else:
            raise ValueError("render mode: %s is not recognized" % (mode, ))

    def seed(self, seed=None):
        self.world.seed = seed


class ToMnetRLGridworldEnv(GridworldEnv):
    def __init__(self, agent_FoV=None):
        agent_reward_functions = [RewardFunction([[1, 1, 0, 0, 0]])]
        super().__init__(1, agent_reward_functions, num_goals=5, horizon=51, dim=(11, 11))
        self.agent_FoV = agent_FoV

    def check_for_early_termination(self, goals_consumed):
        done = False
        if bool(goals_consumed):
            for goal in goals_consumed.values():
                if goal >= 1:  # Goal 0 is subgoal and not a terminal goal
                    self.current_step = np.inf
                    done = True
        return done

    def step(self, action):
        world_state, player_rewards, done, goals_consumed = super().step(action)
        if not done:
            done = self.check_for_early_termination(goals_consumed)
        # return self.mask_state(world_state), player_rewards, done, goals_consumed
        return world_state, player_rewards, done, goals_consumed

    def reset(self, create_new_world=True, agent_reward_functions=None):
        # return self.mask_state(super().reset(create_new_world))
        return super().reset(create_new_world)

    def render(self, debug_print=False):
        self._debug_print = debug_print
        wall_positions, goal_positions, agent_positions = self.world.get_feature_positions()
        self._drawer = ToMnetRLDrawer(self.world.dim, wall_positions, goal_positions, agent_positions)

    def get_mask(self, position):
        x, y = position[0], position[1]
        min_x = max(0, x - self.agent_FoV)
        min_y = max(0, y - self.agent_FoV)
        max_x = min(self.dim[0] - 1, x + self.agent_FoV)
        max_y = min(self.dim[1] - 1, y + self.agent_FoV)
        mask = np.zeros(self.dim)
        mask[min_x:max_x + 1, min_y:max_y + 1] = 1
        return mask

    def get_agent_position(self):
        return self.world.agent_positions[0]

    def mask_state(self, state):
        if self.agent_FoV is None or state is None:
            return state
        position = self.get_agent_position()
        mask = self.get_mask(position)
        return np.multiply(state, mask)


class SeparatedGridworldEnv(GridworldEnv):
    def __init__(self, num_agents, agent_reward_functions, num_goals=4, horizon=30, dim=(11, 11), seed=None):
        super().__init__(num_agents, agent_reward_functions, num_goals, horizon, dim, seed)
        self.world = SeparatedGridworld(num_agents=num_agents, num_goals=num_goals, dim=dim, seed=seed)
        self.state_size = self.world.state.shape


class RepositionGridworldEnv(GridworldEnv):
    def __init__(self,
                 agents,
                 agent_reward_functions: List[RewardFunction],
                 num_goals: int = 4,
                 horizon: int = 30,
                 dim: Tuple[int, int] = (11, 11),
                 min_num_walls: int = 0,
                 max_num_walls: int = 2,
                 seed: Union[None, int] = None):
        super().__init__(len(agents), agent_reward_functions, num_goals, horizon, dim, min_num_walls,
                         max_num_walls, seed)
        self.agents = agents
        for agent in agents:
            if not isinstance(agent.destination_selector, RepositionPicnicDestinationSelector):
                raise ValueError("agents should have the RepositionPicnicDestinationSelector")
        if not self.agents[0].destination_selector.is_primary_selector:
            raise ValueError("agents[0] should be the primary selector")
        if self.num_agents <= 1:
            raise ValueError("RepositionGridworld requires at least 2 agents")
        self.world = RepositionGridworld(num_agents=self.num_agents, num_goals=num_goals, dim=dim,
                                         min_num_walls=min_num_walls, max_num_walls=max_num_walls, seed=seed)

    def step(self, actions):
        if isinstance(actions, int):
            actions = [actions]

        if self.current_step < self.horizon:
            self.current_step += 1
            done = True if self.current_step == self.horizon else False
            collision_record = self.world.step(actions)
            if self.agents[0].destination_selector.finished_picnic:
                self.world.reposition_agents()
                for agent_idx, agent in enumerate(self.agents):
                    agent.reset(self.get_world_state(), agent_idx, self.horizon)
            self.draw()
            player_rewards = self.calculate_rewards_from_collisions(collision_record)
            goals_consumed = self.get_goals_consumed_from_collisions(collision_record)
            return self.get_world_state(), player_rewards, done, goals_consumed
        else:
            return None, np.zeros(self.num_agents), True, None

