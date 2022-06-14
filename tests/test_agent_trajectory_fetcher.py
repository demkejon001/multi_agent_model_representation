import numpy as np

from tommas.env.gridworld_env import GridworldEnv
from tommas.env.gridworld import GridWorld
from tommas.agents.reward_function import get_cooperative_reward_functions
from tommas.data.gridworld_trajectory import GridworldTrajectory
from tommas.data.agent_trajectory_fetcher import AgentGridworldTrajectories


def test_agent_gridworld_trajectories_append():
    min_num_episodes_per_agent = 2
    num_agents = 3
    num_goals = 4
    horizon = 5
    dim = (5, 5)
    reward_funcs = get_cooperative_reward_functions(3, np.ones(num_goals))

    agent_ids_a = np.arange(num_agents)
    history_a = []
    for action in range(min_num_episodes_per_agent):
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
        for _ in range(horizon):
            actions = [action for _ in range(num_agents)]
            new_state, reward, done, info = env.step(actions)
            traj.append((state, reward, actions, info))
            state = new_state
        history_a.append(GridworldTrajectory(traj, env, list(agent_ids_a)))

    dataset_a = AgentGridworldTrajectories(history_a)
    combined_dataset = AgentGridworldTrajectories(history_a)

    agent_ids_b = np.arange(num_agents, num_agents * 2)
    history_b = []
    for action in range(min_num_episodes_per_agent):
        env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, (11, 11), seed=0)
        env.reset()
        gridworld_import = [[-1, -1, -1, -1, -1],
                            [-1, 50, 51, 52, -1],
                            [-1, 1, 0, 3, -1],
                            [-1, 0, 2, 4, -1],
                            [-1, -1, -1, -1, -1]]

        gridworld = GridWorld(num_agents, num_goals, dim)
        gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
        env.world = gridworld
        env.dim = dim
        state = np.copy(gridworld.state)
        traj = []
        for _ in range(horizon):
            actions = [action for _ in range(num_agents)]
            new_state, reward, done, info = env.step(actions)
            traj.append((state, reward, actions, info))
            state = new_state
        history_b.append(GridworldTrajectory(traj, env, list(agent_ids_b)))

    dataset_b = AgentGridworldTrajectories(history_b)

    agent_ids_c = np.arange(num_agents * 2, num_agents * 3)
    history_c = []
    for action in range(min_num_episodes_per_agent):
        env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, (11, 11), seed=0)
        env.reset()
        gridworld_import = [[-1, -1, -1, -1, -1],
                            [-1, 50, 0, 0, -1],
                            [-1, 51, 1, 3, -1],
                            [-1, 52, 2, 4, -1],
                            [-1, -1, -1, -1, -1]]

        gridworld = GridWorld(num_agents, num_goals, dim)
        gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
        env.world = gridworld
        env.dim = dim
        state = np.copy(gridworld.state)
        traj = []
        for _ in range(horizon):
            actions = [action for _ in range(num_agents)]
            new_state, reward, done, info = env.step(actions)
            traj.append((state, reward, actions, info))
            state = new_state
        history_c.append(GridworldTrajectory(traj, env, list(agent_ids_c)))

    dataset_c = AgentGridworldTrajectories(history_c)

    datasets_to_combine = [dataset_b, dataset_c]
    for i in range(len(datasets_to_combine)):
        dataset = datasets_to_combine[i]
        combined_dataset.append(dataset)

    assert np.array_equal(combined_dataset.agent_ids, np.arange(num_agents * 3))

    assert np.array_equal(combined_dataset.agent_gridworld_indices[0], [0, 1])
    assert np.array_equal(combined_dataset.agent_gridworld_indices[1], [0, 1])
    assert np.array_equal(combined_dataset.agent_gridworld_indices[2], [0, 1])
    assert np.array_equal(combined_dataset.agent_gridworld_indices[3], [2, 3])
    assert np.array_equal(combined_dataset.agent_gridworld_indices[4], [2, 3])
    assert np.array_equal(combined_dataset.agent_gridworld_indices[5], [2, 3])
    assert np.array_equal(combined_dataset.agent_gridworld_indices[6], [4, 5])
    assert np.array_equal(combined_dataset.agent_gridworld_indices[7], [4, 5])
    assert np.array_equal(combined_dataset.agent_gridworld_indices[8], [4, 5])

    for gridworld_idx, gridworld_traj in enumerate(combined_dataset.gridworld_trajectories):
        if gridworld_idx < 2:
            assert gridworld_traj in dataset_a.gridworld_trajectories
        elif gridworld_idx < 4:
            assert gridworld_traj in dataset_b.gridworld_trajectories
        else:
            assert gridworld_traj in dataset_c.gridworld_trajectories

    dataset_a.append(dataset_b)
    dataset_a.append(dataset_c)

    assert np.array_equal(dataset_a.agent_ids, combined_dataset.agent_ids)
    for grid_traj_a, grid_traj_b in zip(combined_dataset.gridworld_trajectories, dataset_a.gridworld_trajectories):
        for traj_a, traj_b in zip(grid_traj_a.trajectory, grid_traj_b.trajectory):
            assert np.array_equal(traj_a[0], traj_b[0])
