import numpy as np
import pytest

from src.data.gridworld_trajectory import GridworldTrajectory
from src.env.gridworld_env import GridworldEnv
from src.env.gridworld import GridWorld
from src.agents.reward_function import get_cooperative_reward_functions


def test_create_state_tensor():
    def get_true_state_tensor():
        action_tensor = []
        for i in range(num_agents):
            agent_action_tensor = np.zeros((num_actions, mg_dim[0], mg_dim[1]))
            agent_action_tensor[i] = np.ones((mg_dim[0], mg_dim[1]))
            action_tensor.append(agent_action_tensor)
        action_tensor[0], action_tensor[agent_idx] = action_tensor[agent_idx], action_tensor[0]
        action_tensor = np.concatenate(action_tensor, axis=0)

        agent_ids_tensor = []
        for agent_id in agent_ids:
            agent_ids_tensor.append(np.ones((1, mg_dim[0], mg_dim[1])) * agent_id)
        agent_ids_tensor[0], agent_ids_tensor[agent_idx] = agent_ids_tensor[agent_idx], agent_ids_tensor[0]
        agent_ids_tensor = np.concatenate(agent_ids_tensor, axis=0)

        gs = gridworld_state.copy()
        gs[1 + num_goals + 0], gs[1 + num_goals + agent_idx] = gridworld_state[1 + num_goals + agent_idx].copy(), \
                                                               gridworld_state[1 + num_goals + 0].copy()

        return np.concatenate((gs, action_tensor, agent_ids_tensor), axis=0)

    def get_remove_action_indices():
        action_offset = 1 + num_goals + num_agents
        excluding_indices = list(range(action_offset, action_offset + (num_agents * num_actions)))
        return tuple([[i for i in range(len(true_state_tensor)) if i not in excluding_indices]])

    def get_remove_other_agents_actions_indices():
        action_offset = 1 + num_goals + num_agents
        excluding_indices = list(range(action_offset + num_actions, action_offset + (num_agents * num_actions)))
        return tuple([[i for i in range(len(true_state_tensor)) if i not in excluding_indices]])

    def get_remove_other_agents_indices():
        agent_offset = 1 + num_goals
        action_offset = 1 + num_goals + num_agents
        excluding_indices = list(range(agent_offset + 1, agent_offset + num_agents))
        excluding_indices += list(range(action_offset + num_actions, action_offset + (num_agents * num_actions)))
        return tuple([[i for i in range(len(true_state_tensor)) if i not in excluding_indices]])

    mg_dim = (5, 5)
    num_agents = 5
    horizon = 5
    num_goals = 3
    num_actions = 5
    assert num_agents <= num_actions

    gridworld_state = np.zeros((1 + num_goals + num_agents, *mg_dim))
    for agent_idx in range(num_agents):
        gridworld_state[1 + num_goals + agent_idx, agent_idx, agent_idx] = 1

    trajectory = [(gridworld_state, [0 for _ in range(num_agents)], [agent_idx for agent_idx in range(num_agents)],
                   dict()) for _ in range(horizon)]

    class DummyEnv:
        def __init__(self):
            self.horizon = horizon
            self.dim = mg_dim
            self.num_goals = num_goals
            self.num_actions = num_actions
            self.num_agents = num_agents

    env = DummyEnv()

    agent_ids = [11 * i for i in range(num_agents)]
    gridworld_trajectory = GridworldTrajectory(trajectory, env, agent_ids)
    state = trajectory[0][0]
    actions = trajectory[0][2]

    for agent_idx in range(num_agents):
        state_tensor = gridworld_trajectory.create_state_tensor(state, actions, agent_idx, attach_agent_ids=True)
        true_state_tensor = get_true_state_tensor()
        assert np.array_equal(true_state_tensor, state_tensor)

        # Testing without agent ids attached
        state_tensor = gridworld_trajectory.create_state_tensor(state, actions, agent_idx, attach_agent_ids=False)
        true_state_tensor = get_true_state_tensor()
        assert np.array_equal(true_state_tensor[:-num_agents], state_tensor)

        # Making sure no in place modifications occur to the gridworld_trajectory
        true_state_tensor = get_true_state_tensor()
        state_tensor = gridworld_trajectory.create_state_tensor(state, actions, agent_idx, attach_agent_ids=True)
        assert np.array_equal(true_state_tensor, state_tensor)

        # Making sure no in place modifications occur to the gridworld_trajectory without agent ids attached
        state_tensor = gridworld_trajectory.create_state_tensor(state, actions, agent_idx, attach_agent_ids=False)
        true_state_tensor = get_true_state_tensor()
        assert np.array_equal(true_state_tensor[:-num_agents], state_tensor)

    # testing remove_actions
    for agent_idx in range(num_agents):
        state_tensor = gridworld_trajectory.create_state_tensor(state, actions, agent_idx, attach_agent_ids=True, remove_actions=True)
        true_state_tensor = get_true_state_tensor()
        true_state_tensor = true_state_tensor[get_remove_action_indices()]
        assert np.array_equal(true_state_tensor, state_tensor)

        # Testing without agent ids attached
        state_tensor = gridworld_trajectory.create_state_tensor(state, actions, agent_idx, attach_agent_ids=False, remove_actions=True)
        true_state_tensor = get_true_state_tensor()
        true_state_tensor = true_state_tensor[get_remove_action_indices()]
        assert np.array_equal(true_state_tensor[:-num_agents], state_tensor)

        # Making sure no in place modifications occur to the gridworld_trajectory
        true_state_tensor = get_true_state_tensor()
        true_state_tensor = true_state_tensor[get_remove_action_indices()]
        state_tensor = gridworld_trajectory.create_state_tensor(state, actions, agent_idx, attach_agent_ids=True, remove_actions=True)
        assert np.array_equal(true_state_tensor, state_tensor)

        # Making sure no in place modifications occur to the gridworld_trajectory without agent ids attached
        state_tensor = gridworld_trajectory.create_state_tensor(state, actions, agent_idx, attach_agent_ids=False, remove_actions=True)
        true_state_tensor = get_true_state_tensor()
        true_state_tensor = true_state_tensor[get_remove_action_indices()]
        assert np.array_equal(true_state_tensor[:-num_agents], state_tensor)

    # testing remove_other_agents
    for agent_idx in range(num_agents):
        pytest.raises(AssertionError, gridworld_trajectory.create_state_tensor, state, actions, agent_idx, attach_agent_ids=True, remove_other_agents=True)

        # Testing without agent ids attached
        state_tensor = gridworld_trajectory.create_state_tensor(state, actions, agent_idx, attach_agent_ids=False, remove_other_agents=True)
        true_state_tensor = get_true_state_tensor()
        true_state_tensor = true_state_tensor[get_remove_other_agents_indices()]
        assert np.array_equal(true_state_tensor[:-num_agents], state_tensor)

        # Making sure no in place modifications occur to the gridworld_trajectory without agent ids attached
        state_tensor = gridworld_trajectory.create_state_tensor(state, actions, agent_idx, attach_agent_ids=False, remove_other_agents=True)
        true_state_tensor = get_true_state_tensor()
        true_state_tensor = true_state_tensor[get_remove_other_agents_indices()]
        assert np.array_equal(true_state_tensor[:-num_agents], state_tensor)

    # testing remove_other_agents_actions
    for agent_idx in range(num_agents):
        state_tensor = gridworld_trajectory.create_state_tensor(state, actions, agent_idx, attach_agent_ids=True, remove_other_agents_actions=True)
        true_state_tensor = get_true_state_tensor()
        print(get_remove_other_agents_actions_indices())
        true_state_tensor = true_state_tensor[get_remove_other_agents_actions_indices()]
        assert np.array_equal(true_state_tensor, state_tensor)

        # Testing without agent ids attached
        state_tensor = gridworld_trajectory.create_state_tensor(state, actions, agent_idx, attach_agent_ids=False, remove_other_agents_actions=True)
        true_state_tensor = get_true_state_tensor()
        true_state_tensor = true_state_tensor[get_remove_other_agents_actions_indices()]
        assert np.array_equal(true_state_tensor[:-num_agents], state_tensor)

        # Making sure no in place modifications occur to the gridworld_trajectory
        true_state_tensor = get_true_state_tensor()
        true_state_tensor = true_state_tensor[get_remove_other_agents_actions_indices()]
        state_tensor = gridworld_trajectory.create_state_tensor(state, actions, agent_idx, attach_agent_ids=True, remove_other_agents_actions=True)
        assert np.array_equal(true_state_tensor, state_tensor)

        # Making sure no in place modifications occur to the gridworld_trajectory without agent ids attached
        state_tensor = gridworld_trajectory.create_state_tensor(state, actions, agent_idx, attach_agent_ids=False, remove_other_agents_actions=True)
        true_state_tensor = get_true_state_tensor()
        true_state_tensor = true_state_tensor[get_remove_other_agents_actions_indices()]
        assert np.array_equal(true_state_tensor[:-num_agents], state_tensor)


def test_get_successor_representations():
    def add_2d_tuple(x, y):
        return x[0] + y[0], x[1] + y[1]

    def update_agent_position(pos, action):
        new_pos = add_2d_tuple(pos, action)
        return min(max(0, new_pos[0]), mg_dim[0]), min(max(0, new_pos[1]), mg_dim[1])

    mg_dim = (5, 5)
    num_agents = 5
    horizon = 5
    num_goals = 3
    num_actions = 5
    assert num_agents <= num_actions

    up_action = (-1, 0)

    gridworld_states = np.zeros((horizon, 1 + num_goals + num_agents, *mg_dim))
    for agent_idx in range(num_agents):
        agent_pos = (agent_idx, agent_idx)
        for t in range(horizon):
            gridworld_states[t, 1 + num_goals + agent_idx, agent_pos[0], agent_pos[1]] = 1
            agent_pos = update_agent_position(agent_pos, up_action)

    trajectory = [(gridworld_states[t], [0 for _ in range(num_agents)], [0 for agent_idx in range(num_agents)],
                   dict()) for t in range(horizon)]

    class DummyEnv:
        def __init__(self):
            self.horizon = horizon
            self.dim = mg_dim
            self.num_goals = num_goals
            self.num_actions = num_actions
            self.num_agents = num_agents

    env = DummyEnv()

    agent_ids = [11 * i for i in range(num_agents)]
    gridworld_trajectory = GridworldTrajectory(trajectory, env, agent_ids)

    gammas = np.array([.5, .9, .99])
    for agent_idx in range(num_agents):
        for start_index in range(horizon - 1):
            agent_position = (agent_idx, agent_idx)
            true_sr = np.zeros((3, *mg_dim))
            for i in range(horizon):
                if i > start_index:
                    true_sr[:, agent_position[0], agent_position[1]] += gammas ** (i - start_index)
                agent_position = update_agent_position(agent_position, up_action)
            gammas_sum = np.sum([gammas ** i for i in range(1, horizon - start_index)], axis=0).reshape(3, 1, 1)
            true_sr = true_sr / gammas_sum
            sr = gridworld_trajectory.get_successor_representations(start_index, agent_idx=agent_idx)
            assert np.allclose(true_sr, sr)


def test_get_future_goals_consumed():
    num_agents = 3
    num_goals = 4
    horizon = 5
    dim = (5, 5)
    reward_funcs = get_cooperative_reward_functions(3, np.ones(num_goals))
    # .reset() places 2 walls always, so the system will have a hard time finding positions for everything so we assign
    #   a bigger dim, (11, 11), so reset works and then we replace the gridworld and dim.
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
        new_state, reward, done, info = env.step(np.zeros(num_agents))
        traj.append((state, reward, done, info))

    agent_goals_consumed = [[np.zeros(num_goals) for i in range(horizon)] for _ in range(num_agents)]
    # Agent 0 is going to consume goal 0 at time 0 only
    agent_goals_consumed[0][0][0] = 1

    # Agent 1 is going to consume goal 1 at time 0 and 1 only
    agent_goals_consumed[1][0][1] = 1
    agent_goals_consumed[1][1][1] = 1

    # Agent 2 is going to consume goals 2 and 3 at time 0 and goals 3 at time 1
    agent_goals_consumed[2][0][2] = 1
    agent_goals_consumed[2][0][3] = 1
    agent_goals_consumed[2][1][3] = 1

    gridworld_trajectory = GridworldTrajectory(traj, env, np.arange(num_agents))

    for agent_idx in range(num_agents):
        for t in range(horizon):
            goals_consumed = gridworld_trajectory.get_future_goals_consumed(t, agent_idx=agent_idx)
            true_goals_consumed = agent_goals_consumed[agent_idx][t]
            assert np.array_equal(goals_consumed, true_goals_consumed)

