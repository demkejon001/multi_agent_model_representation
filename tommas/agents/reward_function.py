from scipy.stats import dirichlet
import numpy as np
from typing import Union, List

from tommas.env.gridworld import Collision


DEFAULT_WALL_COLLISION_PENALTY = -.05
DEFAULT_PLAYER_COLLISION_PENALTY = -.05
DEFAULT_MOVEMENT_PENALTY = -.005


class RewardFunction:
    def __init__(self,
                 goal_rewards: Union[List, np.array],
                 wall_collision_penalty=DEFAULT_WALL_COLLISION_PENALTY,
                 player_collision_penalty=DEFAULT_PLAYER_COLLISION_PENALTY,
                 movement_penalty=DEFAULT_MOVEMENT_PENALTY):
        self.goal_rewards = goal_rewards
        if isinstance(goal_rewards, List):
            self.goal_rewards = np.array(self.goal_rewards)
        self.num_goals = self.goal_rewards.shape[1]
        self.wall_collision_penalty = wall_collision_penalty
        self.player_collision_penalty = player_collision_penalty
        self.movement_penalty = movement_penalty

    def get_reward(self, agent_id, collision_record):
        reward = self.movement_penalty
        for agent_idx, agent_collision in enumerate(collision_record.collisions):
            for collision in agent_collision:
                if agent_idx == agent_id:
                    if collision == Collision.WALL:
                        reward += self.wall_collision_penalty
                    elif collision == Collision.PLAYER:
                        reward += self.player_collision_penalty
                if collision != Collision.WALL and collision != Collision.PLAYER:
                    reward += self.goal_rewards[agent_idx][int(collision)]
        return reward


def get_cooperative_reward_functions(num_agents, single_agent_goal_rewards,
                                     wall_collision_penalty=DEFAULT_WALL_COLLISION_PENALTY,
                                     player_collision_penalty=DEFAULT_PLAYER_COLLISION_PENALTY,
                                     movement_penalty=DEFAULT_MOVEMENT_PENALTY) -> List[RewardFunction]:
    multi_agent_goal_rewards = np.array([single_agent_goal_rewards for _ in range(num_agents)])
    return [RewardFunction(multi_agent_goal_rewards, wall_collision_penalty, player_collision_penalty, movement_penalty)
            for _ in range(num_agents)]


def get_competitive_reward_functions(agent_goal_rewards, wall_collision_penalty=DEFAULT_WALL_COLLISION_PENALTY,
                                     player_collision_penalty=DEFAULT_PLAYER_COLLISION_PENALTY,
                                     movement_penalty=DEFAULT_MOVEMENT_PENALTY) -> List[RewardFunction]:
    agent_goal_rewards = np.array(agent_goal_rewards)
    num_agents, num_goals = agent_goal_rewards.shape
    multi_agent_goal_rewards = [np.zeros((num_agents, num_goals)) for _ in range(num_agents)]
    for agent_a_idx, agent_goal_reward in enumerate(agent_goal_rewards):
        multi_agent_goal_rewards[agent_a_idx][agent_a_idx] = agent_goal_reward
        zero_sum_goal_reward = -agent_goal_reward / (num_agents - 1)
        for agent_b_idx in range(num_agents):
            if agent_a_idx != agent_b_idx:
                multi_agent_goal_rewards[agent_b_idx][agent_a_idx] = zero_sum_goal_reward

    agent_reward_functions = []
    for multi_agent_goal_reward in multi_agent_goal_rewards:
        agent_reward_functions.append(RewardFunction(multi_agent_goal_reward, wall_collision_penalty,
                                                     player_collision_penalty, movement_penalty))
    return agent_reward_functions


def get_random_reward_functions(num_agents, num_goals, wall_collision_penalty=DEFAULT_WALL_COLLISION_PENALTY,
                                player_collision_penalty=DEFAULT_PLAYER_COLLISION_PENALTY,
                                movement_penalty=DEFAULT_MOVEMENT_PENALTY) -> List[RewardFunction]:
    def get_random_reward_function():
        goal_rewards = dirichlet.rvs(alpha=np.ones(num_goals) * alpha, size=num_agents)
        return RewardFunction(goal_rewards, wall_collision_penalty, player_collision_penalty, movement_penalty)
    alpha = 1
    return [get_random_reward_function() for _ in range(num_agents)]


def get_dependent_reward_functions():
    pass


def get_team_reward_functions():
    pass


# Can't think of a better name, but all goal rewards are zero.
def get_zeros_reward_functions(num_agents, num_goals, wall_collision_penalty=DEFAULT_WALL_COLLISION_PENALTY,
                               player_collision_penalty=DEFAULT_PLAYER_COLLISION_PENALTY,
                               movement_penalty=DEFAULT_MOVEMENT_PENALTY) -> List[RewardFunction]:
    multi_agent_goal_rewards= np.zeros((num_agents, num_goals))
    return [RewardFunction(multi_agent_goal_rewards, wall_collision_penalty, player_collision_penalty,
                           movement_penalty) for _ in range(num_agents)]
