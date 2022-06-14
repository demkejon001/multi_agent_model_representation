import pytest
import numpy as np

from src.env.gridworld_env import GridworldEnv
from src.env.gridworld import GridWorld
from src.data.gridworld_trajectory import GridworldTrajectory
from src.data.agent_trajectory_fetcher import AgentGridworldTrajectories
from src.agents.reward_function import get_cooperative_reward_functions

min_num_episodes_per_agent = 2
num_agents = 3
num_goals = 4
horizon = 5
dim = (5, 5)
num_actions = 5
true_actions_taken = [[[1, 0, 4], [0, 3, 0], [0, 3, 1], [4, 3, 0], [4, 4, 4]],
                      [[0, 0, 0], [0, 3, 3], [2, 2, 1], [1, 4, 4], [4, 4, 0]]]


@pytest.fixture(scope="module")
def agent_objectives():
    readable_trajectory = [
        [[[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 51, 0, 0],
          [0, 0, 50, 52, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 51, 50, 52, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 0, 50, 0, 0],
          [0, 51, 0, 52, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 0, 50, 52, 0],
          [0, 51, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 0, 50, 52, 0],
          [0, 51, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]],
         ],

        [[[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 50, 51, 52, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 50, 0, 0, 0],
          [0, 51, 52, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 50, 0, 52, 0],
          [0, 51, 0, 0, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 50, 52, 0],
          [0, 51, 0, 0, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 0],
          [0, 0, 0, 52, 0],
          [0, 0, 50, 0, 0],
          [0, 51, 0, 0, 0],
          [0, 0, 0, 0, 0]],
         ]
    ]

    actions = {
        0:
            [[1, 0, 0, 4, 4], [0, 0, 2, 1, 4]],
        1:
            [[0, 3, 3, 3, 4], [0, 3, 2, 4, 4]],
        2:
            [[4, 0, 1, 0, 4], [0, 3, 1, 4, 0]]
    }

    goals_consumed = {
        0:
            [
                [[0, 1, 0, 0],
                 [0, 1, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[1, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],
            ],
        1:
            [
                [[1, 0, 0, 0],
                 [1, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],
                [[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],
            ],
        2:
            [
                [[0, 0, 1, 1],
                 [0, 0, 1, 1],
                 [0, 0, 0, 1],
                 [0, 0, 0, 1],
                 [0, 0, 0, 0]],
                [[0, 0, 1, 1],
                 [0, 0, 0, 1],
                 [0, 0, 0, 1],
                 [0, 0, 0, 1],
                 [0, 0, 0, 1]],
            ],
    }

    srs = dict()
    for agent_idx in range(num_agents):
        srs[agent_idx] = []
        for trajectory in readable_trajectory:
            agent_traj_srs = []
            agent_occupancy = np.array(trajectory)
            agent_occupancy[agent_occupancy != (50 + agent_idx)] = 0
            agent_occupancy[agent_occupancy == (50 + agent_idx)] = 1
            for starting_step in range(horizon):
                agent_step_srs = []
                for discount in [.5, .9, .99]:
                    traj_discount = np.expand_dims(discount ** np.arange(horizon - starting_step), axis=(1, 2))
                    agent_srs = agent_occupancy[starting_step:] * traj_discount
                    agent_srs = np.sum(agent_srs, axis=0)
                    agent_srs = agent_srs / np.sum(agent_srs)
                    agent_step_srs.append(agent_srs)
                agent_traj_srs.append(agent_step_srs)
            srs[agent_idx].append(agent_traj_srs)
    return actions, goals_consumed, srs


@pytest.fixture(scope="module")
def trajectory_fetcher():
    agent_ids = np.arange(num_agents)
    reward_funcs = get_cooperative_reward_functions(num_agents, np.ones(num_goals))
    history = []
    for traj_idx in range(min_num_episodes_per_agent):
        env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, (11, 11), seed=0)
        env.reset()
        gridworld_import = [[-1, -1, -1, -1, -1],
                            [-1, 0, 2, 4, -1],
                            [-1, 1, 0, 3, -1],
                            [-1, 50, 51, 52, -1],
                            [-1, -1, -1, -1, -1]]

        gridworld = GridWorld(num_agents, num_goals, dim)
        gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
        env.world = gridworld
        env.dim = dim
        state = np.copy(gridworld.state)
        traj = []
        for step in range(horizon):
            actions = true_actions_taken[traj_idx][step]
            new_state, reward, done, info = env.step(actions)
            traj.append((state, reward, actions, info))
            state = new_state
        traj.append((state, 0, -1, {}))
        history.append(GridworldTrajectory(traj, env, list(agent_ids)))

    return AgentGridworldTrajectories(history)
