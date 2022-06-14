import numpy as np
import gym
from gym.spaces import Box

from src.agents.reward_function import RewardFunction


class FoVWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return self.env.mask_state(obs).astype(dtype=np.uint8)


class BlindWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=1, shape=(7,), dtype=np.float32)
        self.previous_action = 5  # No previous action
        self.previous_reward = 0

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.previous_action = 5  # No previous action
        self.previous_reward = 0
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.previous_action = action
        self.previous_reward = reward
        return self.observation(observation), reward, done, info

    def observation(self, obs):
        new_obs = np.zeros(7, dtype=np.float32)
        new_obs[self.previous_action] = 1
        new_obs[-1] = self.previous_reward
        return new_obs


class GoalSwapWrapper(gym.RewardWrapper):
    def __init__(self, env, goal_preference):
        super().__init__(env)
        new_goals = [0 for _ in range(5)]
        new_goals[0] = 1
        new_goals[goal_preference] = 1
        self.env.env.agent_reward_functions = [RewardFunction([new_goals])]

    def reward(self, reward):
        return reward


class EpisodeRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_reward = 0

    def reset(self, **kwargs):
        self.episode_reward = 0
        return self.env.reset(**kwargs)

    def reward(self, reward):
        self.episode_reward += reward
        return reward

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if done:
            info.update({'episode': {'r': self.episode_reward}})
            return observation, reward, done, info
        return observation, self.reward(reward), done, info


class StationaryAgentsEarlyTermination(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.old_actions = np.ones(self.env.num_agents) * -1
        self.old_agent_positions = [(0, 0) for _ in range(self.env.num_agents)]
        self.stationary_agents_ctr = 0

    def reset(self, **kwargs):
        self.old_actions = np.ones(self.env.num_agents) * -1
        self.old_agent_positions = [(0, 0) for _ in range(self.env.num_agents)]
        self.stationary_agents_ctr = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if np.array_equal(action, self.old_actions) and np.array_equal(self.env.world.agent_positions, self.old_agent_positions):
            self.stationary_agents_ctr += 1
            if self.stationary_agents_ctr > 2:  # i.e. they have been stationary 3 times
                return state, reward, True, info
        else:
            self.stationary_agents_ctr = 0

        self.old_actions = np.copy(action)
        self.old_agent_positions = self.env.world.agent_positions.copy()
        return state, reward, done, info

