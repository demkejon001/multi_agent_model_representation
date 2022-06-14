import numpy as np
import pytest

from tommas.agents.hand_coded_agents import PathfindingAgent
from tommas.agents.destination_selector import DestinationSelector, MultiAgentDestinationSelector, \
    HighestGoalRanker, ClosestDistanceGoalRanker, DiscountGoalRanker, DistancePenaltyGoalRanker, \
    CollaborativeStatePartitionFilter, PicnicDestinationSelector, StaticCircleFilter, StaticSquareFilter, \
    StaticDiamondFilter, SinglePicnicDestinationSelector
# from tommas.agents.create_handcodedagents import create_agent
from tommas.agents.create_gridworld_agents import get_random_gridworld_agent, RandomAgentParamSampler
from tommas.agents.reward_function import get_cooperative_reward_functions, RewardFunction
from tommas.env.gridworld_env import GridworldEnv, GridWorld


# Tests PathfindingAgent.reset(), pathfinding, not grabbing negative goals, but not avoiding them either
def test_pathfinding():
    num_agents = 3
    num_goals = 5
    horizon = 30
    dim = (11, 11)
    goal_rewards = [4, 3, 2, 1, -1]
    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, (11, 11), seed=0)
    env.reset()
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 50, 0, 0, 51, -1, 0, 52, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 4, 0, 0, 0, -1, 0, 0, 0, 5, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 1, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 2, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 3, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    agents = []
    goal_ranker = HighestGoalRanker(collaborative=False)
    for agent_id in range(num_agents):
        destination_selector = DestinationSelector(reward_funcs[agent_id], agent_id, goal_ranker)
        agent = PathfindingAgent(agent_id, reward_funcs[agent_id], destination_selector=destination_selector)
        agent.reset(state, agent_idx=agent_id)
        agents.append(agent)

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)
        print(env.world)
        if timestep == 5:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[0])
        if timestep == 10:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[1])
        if timestep == 16:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[2])
        if timestep == 22:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[3])

        if timestep > 5:
            assert actions[2] == 4
        if timestep > 22:
            assert actions[0] == 4
            assert actions[1] == 4

        timestep += 1

    player_collision_penalty = -.05
    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env.reset(agent_reward_functions=reward_funcs)
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 50, 0, 0, 0, -1, 0, 52, 0, 0, -1],
                        [-1, -1, -1, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, -1, 0, 0, -1, 0, 1, 0, 0, -1],
                        [-1, 3, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, -1, -1, 0, 0, -1, 0, 51, 0, 0, -1],
                        [-1, 0, -1, -1, 5, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, -1, 0, 0, 0, 0, 0, 2, -1],
                        [-1, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 4, -1, 0, 0, 0, 0, 0, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    for i, agent in enumerate(agents):
        agent.reset(state, i)

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)

        if timestep == 8:
            true_rewards = np.ones(num_agents) * goal_rewards[4]
            true_rewards[1] += player_collision_penalty
            true_rewards[2] += player_collision_penalty
            assert np.array_equal(rewards, true_rewards)
        elif timestep == 14:
            true_rewards = np.ones(num_agents) * goal_rewards[1]
            true_rewards[1] += player_collision_penalty
            true_rewards[2] += player_collision_penalty
            assert np.array_equal(rewards, true_rewards)
        elif timestep == 25:
            true_rewards = np.ones(num_agents) * goal_rewards[2]
            true_rewards[1] += player_collision_penalty
            true_rewards[2] += player_collision_penalty
            assert np.array_equal(rewards, true_rewards)
        else:
            if timestep > 2:
                assert np.array_equal([0, player_collision_penalty, player_collision_penalty], rewards)

        assert actions[1] == 0
        assert actions[2] == 2
        if timestep > 25:
            assert actions[0] == 4

        timestep += 1


# Tests that errors occur when you don't call .reset()
def test_reset_failure():
    num_agents = 1
    num_goals = 5
    horizon = 30
    dim = (11, 11)
    goal_rewards = [4, 3, 2, 1, -1]
    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, dim, seed=0)
    goal_ranker = HighestGoalRanker(collaborative=False)
    destination_selector = DestinationSelector(reward_funcs[0], 0, goal_ranker)
    agent = PathfindingAgent(0, reward_funcs[0], destination_selector=destination_selector)

    state = env.reset()
    # Tests failure to reset initially
    with pytest.raises(RuntimeError, match=r".* reset .*"):
        agent.action(state)

    # Tests failure to reset over multiple games
    with pytest.raises(RuntimeError, match=r".* reset .*"):
        agent.reset(state, 0)
        for i in range(1000):
            print(env.world)
            actions = [agent.action(state)]
            state, rewards, done, info = env.step(actions)
            if done:
                state = env.reset()


def test_colab_highest_goal_preference():
    num_agents = 3
    num_goals = 5
    horizon = 30
    dim = (11, 11)
    goal_rewards = [4, 3, 2, 1, -1]
    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, (11, 11), seed=0)
    env.reset()
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 50, 0, 0, 51, -1, 0, 52, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 4, 0, 0, -1, 0, 0, 0, 5, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 2, 0, 0, 3, -1, 1, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    raps = RandomAgentParamSampler(0, True)
    agents, _ = get_random_gridworld_agent("collaborative", raps, [-1, -1, -1], goal_rewards=goal_rewards,
                                           num_goals=len(goal_rewards), goal_ranker_type="highest")

    for agent_idx, agent in enumerate(agents):
        agent.reset(state, agent_idx)

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)
        if timestep == 4:
            assert np.array_equal(rewards, np.ones(num_agents) * (goal_rewards[1] + goal_rewards[2]))
        if timestep == 5:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[0])
        if timestep == 7:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[3])

        if timestep > 5:
            assert actions[2] == 4
            assert actions[1] == 4
        if timestep > 7:
            assert actions[0] == 4

        timestep += 1


    env.reset(agent_reward_functions=reward_funcs)
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 50, 4, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 51, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 5, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 52, -1],
                        [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 3, 0, 0, 0, 0, 2, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    for i, agent in enumerate(agents):
        agent.reset(state, i)

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)

        if timestep == 2:
            true_rewards = np.ones(num_agents) * goal_rewards[1]
            assert np.array_equal(rewards, true_rewards)
        elif timestep == 5:
            true_rewards = np.ones(num_agents) * goal_rewards[3]
            assert np.array_equal(rewards, true_rewards)
        elif timestep == 6:
            true_rewards = np.ones(num_agents) * goal_rewards[0]
            assert np.array_equal(rewards, true_rewards)
        elif timestep == 7:
            true_rewards = np.ones(num_agents) * goal_rewards[2]
            assert np.array_equal(rewards, true_rewards)

        if timestep > 5:
            assert actions[0] == 4
        if timestep > 6:
            assert actions[1] == 4
        if timestep > 7:
            assert actions[2] == 4

        timestep += 1


def test_closest_goal_preference():
    num_agents = 3
    num_goals = 5
    horizon = 30
    dim = (11, 11)
    goal_rewards = [4, 3, 2, 1, 6]
    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, (11, 11), seed=0)
    env.reset()
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 50, 0, -1, 0, 0, 52, 0, -1],
                        [-1, 0, -1, -1, -1, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 5, -1],
                        [-1, 0, 0, 0, 51, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 1, 0, 0, 0, -1],
                        [-1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 2, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 3, 0, 0, 0, 0, 0, 0, 0, 4, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    agents = []
    goal_ranker = ClosestDistanceGoalRanker()
    for agent_id in range(num_agents):
        destination_selector = DestinationSelector(reward_funcs[agent_id], agent_id, goal_ranker)
        agent = PathfindingAgent(agent_id, reward_funcs[agent_id], destination_selector=destination_selector)
        agent.reset(state, agent_idx=agent_id)
        agents.append(agent)

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)
        if timestep == 3:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[4])
            assert agents[0].destination == (9, 1)  # Agent 50 is going after goal 3
        if timestep == 8:
            assert np.array_equal(rewards, np.ones(num_agents) * (goal_rewards[2] + goal_rewards[0]))
        if timestep == 16:
            assert np.array_equal(rewards, np.ones(num_agents) * (goal_rewards[1] + goal_rewards[3]))

        if timestep > 8:
            assert actions[2] == 4
        if timestep > 16:
            assert actions[0] == 4
            assert actions[1] == 4

        timestep += 1

    player_collision_penalty = -.05
    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env.reset(agent_reward_functions=reward_funcs)
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 52, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 1, 4, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 3, 0, 0, 0, -1, 0, 51, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 5, 0, 0, 50, 0, 0, 2, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    for i, agent in enumerate(agents):
        agent.reset(state, i)

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)

        if timestep == 3:
            true_rewards = np.ones(num_agents) * goal_rewards[4]
            true_rewards[1] += player_collision_penalty
            true_rewards[2] += player_collision_penalty
            assert np.array_equal(rewards, true_rewards)
        elif timestep == 6:
            true_rewards = np.ones(num_agents) * goal_rewards[2]
            true_rewards[1] += player_collision_penalty
            true_rewards[2] += player_collision_penalty
            assert np.array_equal(rewards, true_rewards)
        elif timestep == 15:
            true_rewards = np.ones(num_agents) * goal_rewards[1]
            true_rewards[1] += player_collision_penalty
            true_rewards[2] += player_collision_penalty
            assert np.array_equal(rewards, true_rewards)
        else:
            if timestep > 2:
                assert np.array_equal([0, player_collision_penalty, player_collision_penalty], rewards)

        assert actions[1] == 0
        assert actions[2] == 2
        if timestep > 15:
            assert actions[0] == 4

        timestep += 1

    goal_rewards = [4, 3, 2, 1, 1]  # Changed goal 5 to reward 1 so that agent 50 prefers goal 2
    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env.reset(agent_reward_functions=reward_funcs)
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 52, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 1, 4, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 3, 0, 0, 0, -1, 0, 51, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 5, 0, 0, 50, 0, 0, 2, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    # for i, agent in enumerate(agents):
    #     agent.reset(state, i)
    agents = []
    for agent_id in range(num_agents):
        destination_selector = DestinationSelector(reward_funcs[agent_id], agent_id, goal_ranker)
        agent = PathfindingAgent(agent_id, reward_funcs[agent_id], destination_selector=destination_selector)
        agent.reset(state, agent_idx=agent_id)
        agents.append(agent)

    done = False
    timestep = 1
    while not done:

        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)

        if timestep == 3:
            true_rewards = np.ones(num_agents) * goal_rewards[1]
            true_rewards[1] += player_collision_penalty
            true_rewards[2] += player_collision_penalty
            assert np.array_equal(rewards, true_rewards)
        elif timestep == 9:
            true_rewards = np.ones(num_agents) * goal_rewards[4]
            true_rewards[1] += player_collision_penalty
            true_rewards[2] += player_collision_penalty
            assert np.array_equal(rewards, true_rewards)
        elif timestep == 12:
            true_rewards = np.ones(num_agents) * goal_rewards[2]
            true_rewards[1] += player_collision_penalty
            true_rewards[2] += player_collision_penalty
            assert np.array_equal(rewards, true_rewards)
        else:
            if timestep > 2:
                assert np.array_equal([0, player_collision_penalty, player_collision_penalty], rewards)

        assert actions[1] == 0
        assert actions[2] == 2
        if timestep > 12:
            assert actions[0] == 4

        timestep += 1


def test_other_agents_consuming_an_agents_goal():
    num_agents = 2
    num_goals = 4
    horizon = 30
    dim = (11, 11)
    reward_funcs = [RewardFunction([[-1, -1, 3, 4], [0, 0, 0, 0]]),
                    RewardFunction([[0, 0, 0, 0], [4, 3, -2, -1]])]
    env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, (11, 11), seed=0)
    env.reset()
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 50, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 1, 0, 0, 0, 0, 51, 0, -1],
                        [-1, 0, 0, 4, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 3, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 2, 0, 0, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    true_actions_taken = [[2, 3], [2, 3], [2, 2], [5, 2], [4, 2], [4, 2]]

    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    agents = []
    goal_ranker = ClosestDistanceGoalRanker()
    for agent_id in range(num_agents):
        destination_selector = DestinationSelector(reward_funcs[agent_id], agent_id, goal_ranker)
        agent = PathfindingAgent(agent_id, reward_funcs[agent_id], destination_selector=destination_selector)
        agent.reset(state, agent_idx=agent_id)
        agents.append(agent)

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)
        print(env.world)
        if timestep < 7:
            true_actions = true_actions_taken[timestep-1]
        else:
            true_actions = [4, 4]
        if 5 in true_actions:
            for true_action, action in zip(true_actions, actions):
                if true_action == 5:
                    assert action != 4
                else:
                    assert action == true_action
        else:
            assert actions == true_actions

        timestep += 1


def test_discounted_goal_preference():
    num_agents = 3
    num_goals = 5
    horizon = 30
    dim = (11, 11)
    goal_rewards = [4.1, 2, 1.37, 1, 4]
    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, (11, 11), seed=0)
    env.reset()
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 52, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 50, 0, -1, 0, 0, 0, 5, -1],
                        [-1, 0, 0, 51, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 1, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 2, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 3, 0, 0, 0, 0, 0, 0, 0, 4, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    agents = []
    goal_ranker = DiscountGoalRanker(discount=.95)
    for agent_id in range(num_agents):
        destination_selector = DestinationSelector(reward_funcs[agent_id], agent_id, goal_ranker)
        agent = PathfindingAgent(agent_id, reward_funcs[agent_id], destination_selector)
        agent.reset(state, agent_idx=agent_id)
        agents.append(agent)

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)
        if timestep == 3:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[4])
        if timestep == 8:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[0])
        if timestep == 9:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[1])
        if timestep == 11:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[3])
        if timestep == 18:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[2])

        if timestep > 8:
            assert actions[2] == 4
        if timestep > 18:
            assert actions[0] == 4
            assert actions[1] == 4

        timestep += 1

    player_collision_penalty = -.05
    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env.reset(agent_reward_functions=reward_funcs)
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 52, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 1, 4, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 3, 0, 0, 0, -1, 0, 51, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 5, 0, 0, 50, 0, 0, 2, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    for i, agent in enumerate(agents):
        agent.reset(state, i)

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)

        if timestep == 3:
            true_rewards = np.ones(num_agents) * goal_rewards[4]
            true_rewards[1] += player_collision_penalty
            true_rewards[2] += player_collision_penalty
            assert np.array_equal(rewards, true_rewards)
        elif timestep == 9:
            true_rewards = np.ones(num_agents) * goal_rewards[1]
            true_rewards[1] += player_collision_penalty
            true_rewards[2] += player_collision_penalty
            assert np.array_equal(rewards, true_rewards)
        elif timestep == 18:
            true_rewards = np.ones(num_agents) * goal_rewards[2]
            true_rewards[1] += player_collision_penalty
            true_rewards[2] += player_collision_penalty
            assert np.array_equal(rewards, true_rewards)
        else:
            if timestep > 2:
                assert np.array_equal([0, player_collision_penalty, player_collision_penalty], rewards)

        assert actions[1] == 0
        assert actions[2] == 2
        if timestep > 18:
            assert actions[0] == 4

        timestep += 1


def test_distance_penalty_goal_preference():
    num_agents = 3
    num_goals = 5
    horizon = 30
    dim = (11, 11)
    goal_rewards = [1.6, .79, .71, 1, 1]
    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, (11, 11), seed=0)
    env.reset()
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 52, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 50, 0, 0, -1, 0, 0, 0, 5, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 1, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 2, -1],
                        [-1, 0, 0, 0, 0, 51, 0, 0, 0, 0, -1],
                        [-1, 3, 0, 0, 0, 0, 0, 0, 0, 4, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    agents = []
    goal_ranker = DistancePenaltyGoalRanker(distance_penalty=-.1)
    for agent_id in range(num_agents):
        destination_selector = DestinationSelector(reward_funcs[agent_id], agent_id, goal_ranker)
        agent = PathfindingAgent(agent_id, reward_funcs[agent_id], destination_selector)
        agent.reset(state, agent_idx=agent_id)
        agents.append(agent)

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)
        if timestep == 5:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[3])
        if timestep == 6:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[0])
        if timestep == 7:
            assert np.array_equal(rewards, np.ones(num_agents) * (goal_rewards[2] + goal_rewards[1]))
        if timestep == 11:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[4])

        if timestep > 7:
            assert actions[0] == 4
            assert actions[1] == 4
        if timestep > 11:
            assert actions[2] == 4

        timestep += 1

    player_collision_penalty = -.05
    # goal_rewards = [1.6, .79, .71, 1, 1]
    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env.reset(agent_reward_functions=reward_funcs)
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 52, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 50, 0, 0, -1, 0, 1, 4, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 3, 0, 0, 0, -1, 0, 0, 51, 0, -1],
                        [-1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 5, 0, 0, 0, 0, 0, 2, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    for i, agent in enumerate(agents):
        agent.reset(state, i)

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)

        if timestep == 4:
            true_rewards = np.ones(num_agents) * goal_rewards[4]
            true_rewards[1] += player_collision_penalty
            true_rewards[2] += player_collision_penalty
            assert np.array_equal(rewards, true_rewards)
        elif timestep == 7:
            true_rewards = np.ones(num_agents) * goal_rewards[2]
            true_rewards[1] += player_collision_penalty
            true_rewards[2] += player_collision_penalty
            assert np.array_equal(rewards, true_rewards)
        elif timestep == 16:
            true_rewards = np.ones(num_agents) * goal_rewards[1]
            true_rewards[1] += player_collision_penalty
            true_rewards[2] += player_collision_penalty
            assert np.array_equal(rewards, true_rewards)
        else:
            if timestep > 3:
                assert np.array_equal([0, player_collision_penalty, player_collision_penalty], rewards)

        if timestep > 2:
            assert actions[1] == 0
            assert actions[2] == 2
        if timestep > 16:
            assert actions[0] == 4

        timestep += 1


def test_collaborative_greedy_goal_preference():
    num_agents = 4
    num_goals = 8
    horizon = 30
    dim = (11, 11)
    goal_rewards = [1, 1, 1, 1, -1, 1, 1, 1]
    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, (11, 11), seed=0)
    env.reset()
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 50, 0, 0, 51, -1, 0, 52, 0, 53, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 1, 0, 0, 2, -1, 0, 3, 0, 4, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, -1, -1, -1, 0, -1],
                        [-1, 5, 0, 0, 6, 0, 0, 7, -1, 8, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    agents = []
    goal_ranker = ClosestDistanceGoalRanker()
    destination_selectors = [MultiAgentDestinationSelector(reward_funcs[0], 0, goal_ranker, True)]
    for agent_id in range(1, num_agents):
        destination_selector = MultiAgentDestinationSelector(reward_funcs[agent_id], agent_id, goal_ranker, False)
        destination_selectors[0].add_destination_selector(destination_selector)
        destination_selectors.append(destination_selector)

    for agent_id in range(num_agents):
        agent = PathfindingAgent(agent_id, reward_funcs[agent_id], destination_selectors[agent_id])
        agents.append(agent)

    for agent_idx in range(num_agents):
        agents[agent_idx].reset(state, agent_idx)

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)
        if timestep == 3:
            true_rewards = np.ones(num_agents) * (goal_rewards[0]+goal_rewards[1]+goal_rewards[2]+goal_rewards[3])
            assert np.array_equal(rewards, true_rewards)
        if timestep == 6:
            true_rewards = np.ones(num_agents) * (goal_rewards[5]+goal_rewards[7])
            assert np.array_equal(rewards, true_rewards)
        if timestep == 9:
            true_rewards = np.ones(num_agents) * goal_rewards[6]
            assert np.array_equal(rewards, true_rewards)

        if timestep > 9:
            assert actions == [4, 4, 4, 4]
        elif timestep > 6:
            assert actions[2] == actions[3] == 4
        elif timestep <= 3:
            assert actions == [2, 2, 2, 2]

        timestep += 1

    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env.reset(agent_reward_functions=reward_funcs)
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 50, -1, 0, 51, 0, 0, 0, 0, 0, -1],
                        [-1, 0, -1, 0, 0, 0, 0, 0, 0, 53, -1],
                        [-1, 0, -1, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 2, 0, 0, 0, 0, 4, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 3, 0, 8, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 7, 0, 6, 0, 0, 0, -1],
                        [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 5, 0, 0, 0, 0, 0, 0, 52, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    for i, agent in enumerate(agents):
        agent.reset(state, i)

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)
        if timestep == 2:
            true_rewards = np.ones(num_agents) * goal_rewards[3]
            assert np.array_equal(rewards, true_rewards)
        elif timestep == 3:
            true_rewards = np.ones(num_agents) * (goal_rewards[1] + goal_rewards[7])
            assert np.array_equal(rewards, true_rewards)
        elif timestep == 5:
            true_rewards = np.ones(num_agents) * (goal_rewards[2] + goal_rewards[5])
            assert np.array_equal(rewards, true_rewards)
        elif timestep == 6:
            true_rewards = np.ones(num_agents) * goal_rewards[6]
            assert np.array_equal(rewards, true_rewards)
        elif timestep == 7:
            true_rewards = np.ones(num_agents) * goal_rewards[0]
            assert np.array_equal(rewards, true_rewards)

        timestep += 1

    # Testing differences in optimization
    num_agents = 2
    num_goals = 2
    horizon = 30
    dim = (11, 11)
    goal_rewards = [1, 1]
    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, (11, 11), seed=0)
    env.reset()
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 2, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 50, 0, 1, 0, 51, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    agents = []

    destination_selectors = [MultiAgentDestinationSelector(reward_funcs[0], 0, goal_ranker, True)]
    for agent_id in range(1, num_agents):
        destination_selector = MultiAgentDestinationSelector(reward_funcs[agent_id], agent_id, goal_ranker, False)
        destination_selectors[0].add_destination_selector(destination_selector)
        destination_selectors.append(destination_selector)

    for agent_id in range(num_agents):
        agent = PathfindingAgent(agent_id, reward_funcs[agent_id], destination_selectors[agent_id])
        agents.append(agent)

    for agent_idx in range(num_agents):
        agents[agent_idx].reset(state, agent_idx)

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)
        if timestep == 2:
            true_rewards = np.ones(num_agents) * goal_rewards[0]
            assert np.array_equal(rewards, true_rewards)
        if timestep == 3:
            true_rewards = np.ones(num_agents) * goal_rewards[1]
            assert np.array_equal(rewards, true_rewards)

        if timestep > 3:
            assert actions == [4, 4]

        timestep += 1

    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env.reset(agent_reward_functions=reward_funcs)
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 2, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 50, 0, 1, 0, 0, 51, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    for i, agent in enumerate(agents):
        agent.reset(state, i)

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)
        if timestep == 3:
            true_rewards = np.ones(num_agents) * (goal_rewards[0] + goal_rewards[1])
            assert np.array_equal(rewards, true_rewards)

        if timestep > 3:
            assert actions == [4, 4]

        timestep += 1


def get_state_partition_agents(reward_funcs, goal_ranker, seed=42):
    agents = []
    num_goals = reward_funcs[0].num_goals
    num_agents = len(reward_funcs)
    state_filter = CollaborativeStatePartitionFilter(list(range(num_agents)), 1 + num_goals, seed)
    destination_selectors = [MultiAgentDestinationSelector(reward_funcs[0], 0, goal_ranker, True, state_filter)]
    for agent_id in range(1, num_agents):
        destination_selector = MultiAgentDestinationSelector(reward_funcs[agent_id], agent_id, goal_ranker, False, state_filter)
        destination_selectors[0].add_destination_selector(destination_selector)
        destination_selectors.append(destination_selector)
    for agent_id in range(num_agents):
        agent = PathfindingAgent(agent_id, reward_funcs[agent_id], destination_selectors[agent_id])
        agents.append(agent)
    return agents


def test_state_partition_has_no_collisions():
    # test the we partition the whole space and no partitions have collisions
    num_goals = 1
    horizon = 30
    dim = (11, 11)
    goal_rewards = [1]
    for num_agents in range(2, 10):
        reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
        env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, dim, seed=0)
        env.reset()
        gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],
                                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
        for i in range(num_agents):
            gridworld_import[1][1 + i] = 50 + i
        gridworld = GridWorld(num_agents, num_goals, dim)
        gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
        env.world = gridworld
        env.dim = dim
        state = np.copy(gridworld.state)

        agents = get_state_partition_agents(reward_funcs, HighestGoalRanker(collaborative=False))

        for agent_idx in range(num_agents):
            agents[agent_idx].reset(state, agent_idx)

        partitioned_state = np.zeros_like(state[0])
        used_positions = set()
        for agent in agents:
            agent_filter = agent.destination_selector.state_filters[0]
            agent_pos = agent.destination_selector._get_agent_current_position(state)
            agent_positions = agent_filter.filter(state, agent_pos)
            for position in agent_positions:
                partitioned_state[position] = 1
                assert position not in used_positions
                used_positions.add(position)
        true_partitioned_state = np.ones_like(state[0])
        true_partitioned_state[[0, -1], :] = 0
        true_partitioned_state[:, [0, -1]] = 0
        assert np.array_equal(partitioned_state, true_partitioned_state)


def test_state_partition_highest_goal_preference():
    num_agents = 2
    num_goals = 8
    horizon = 30
    dim = (11, 11)
    goal_rewards = [1, 2, 3, 4, 5, 6, 7, 8]
    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, (11, 11), seed=0)
    env.reset()
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 51, 0, 0, -1],
                        [-1, 1, 2, 3, 4, 0, 50, 5, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 6, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 7, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 8, 0, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    agents = get_state_partition_agents(reward_funcs, HighestGoalRanker(collaborative=False))

    for agent_idx in range(num_agents):
        agents[agent_idx].reset(state, agent_idx)

    agent0_reachable_goals = {0: (2, 1), 1: (2, 2), 2: (2, 3), 3: (2, 4)}
    agent1_reachable_goals = {4: (2, 7), 5: (4, 7), 6: (7, 7), 7: (9, 7)}
    assert agents[0].destination_selector.current_reachable_goals == agent0_reachable_goals
    assert agents[1].destination_selector.current_reachable_goals == agent1_reachable_goals

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)
        if timestep == 1:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[4])
        if timestep == 2:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[3])
        if timestep == 3:
            assert np.array_equal(rewards, np.ones(num_agents) * (goal_rewards[2] + goal_rewards[5]))
        if timestep == 4:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[1])
        if timestep == 5:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[0])
        if timestep == 6:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[6])
        if timestep == 8:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[7])

        if timestep > 8:
            assert actions == [4, 4]

        timestep += 1

    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env.reset(agent_reward_functions=reward_funcs)
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 5, -1, 0, 1, 0, 2, -1],
                        [-1, 0, 0, 6, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 7, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 8, 0, 0, 0, -1, 50, 3, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 51, 0, 0, 4, 0, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    for agent_idx in range(num_agents):
        agents[agent_idx].reset(state, agent_idx)
    agent0_reachable_goals = {4: (4, 4), 5: (5, 3), 6: (6, 2), 7: (7, 1)}
    agent1_reachable_goals = {0: (4, 7), 1: (4, 9), 2: (7, 7), 3: (9, 7)}
    assert agents[0].destination_selector.current_reachable_goals == agent0_reachable_goals
    assert agents[1].destination_selector.current_reachable_goals == agent1_reachable_goals

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)

        if timestep == 3:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[3])
        elif timestep == 5:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[2])
        elif timestep == 7:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[7])
        elif timestep == 7:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[6])
        elif timestep == 10:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[1])
        elif timestep == 11:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[5])
        elif timestep == 12:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[0])
        elif timestep == 13:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[4])

        timestep += 1


def test_state_partition_distance_based_preference():
    np.random.seed(42)
    num_agents = 2
    num_goals = 8
    horizon = 30
    dim = (11, 11)
    goal_rewards = [1, 2, 3, 4, 5, 6, 7, 8]
    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, (11, 11), seed=0)
    env.reset()
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 1, 0, 2, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 7, 6, 5, -1, 0, 0, 0, 0, -1],
                        [-1, 8, 0, 0, 0, -1, 50, 3, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 51, 0, 0, 4, 0, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    agents = get_state_partition_agents(reward_funcs, ClosestDistanceGoalRanker())

    for agent_idx in range(num_agents):
        agents[agent_idx].reset(state, agent_idx)

    agent0_reachable_goals = {4: (6, 4), 5: (6, 3), 6: (6, 2), 7: (7, 1)}
    agent1_reachable_goals = {0: (4, 7), 1: (4, 9), 2: (7, 7), 3: (9, 7)}
    assert agents[0].destination_selector.current_reachable_goals == agent0_reachable_goals
    assert agents[1].destination_selector.current_reachable_goals == agent1_reachable_goals

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)

        if timestep == 3:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[3])
        elif timestep == 5:
            assert np.array_equal(rewards, np.ones(num_agents) * (goal_rewards[2] + goal_rewards[4]))
        elif timestep == 6:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[5])
        elif timestep == 7:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[6])
        elif timestep == 7:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[7])
        elif timestep == 8:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[0])
        elif timestep == 10:
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[1])

        timestep += 1


    num_agents = 3
    num_goals = 8
    horizon = 30
    dim = (11, 11)
    goal_rewards = [1, 2, 3, 4, 5, 6, 7, 8]
    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, (11, 11), seed=0)
    env.reset()
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 50, 0, 0, 0, 0, 0, 0, 0, 52, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 1, 0, 2, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 7, 6, 5, 0, 0, 0, 0, 0, -1],
                        [-1, 8, 0, 0, 0, 0, 0, 3, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 51, 0, 0, 0, 0, 0, 4, 0, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    agents = get_state_partition_agents(reward_funcs, ClosestDistanceGoalRanker(), seed=18)

    for agent_idx in range(num_agents):
        agents[agent_idx].reset(state, agent_idx)

    agent0_reachable_goals = {0: (4, 7), 1: (4, 9)}
    agent1_reachable_goals = {4: (6, 4), 5: (6, 3), 6: (6, 2), 7: (7, 1)}
    agent2_reachable_goals = {2: (7, 7), 3: (9, 7)}
    assert agents[0].destination_selector.current_reachable_goals == agent0_reachable_goals
    assert agents[1].destination_selector.current_reachable_goals == agent1_reachable_goals
    assert agents[2].destination_selector.current_reachable_goals == agent2_reachable_goals

    done = False
    summed_rewards = np.zeros(num_agents)
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)
        summed_rewards += rewards
    assert np.array_equal(summed_rewards, np.ones(num_agents) * sum(goal_rewards))


def test_static_shape_state_filters():
    num_agents = 3
    num_goals = 5
    horizon = 30
    dim = (11, 11)
    goal_rewards = [1, 2, 3, 4, 5]
    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, (11, 11), seed=0)
    env.reset()
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 50, 0, 0, 0, 4, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 52, 0, 0, -1],
                        [-1, 0, 3, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 2, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 51, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 5, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    state_filters = [StaticDiamondFilter(4), StaticCircleFilter(3), StaticSquareFilter(2)]

    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    agents = []
    goal_ranker = HighestGoalRanker(collaborative=False)
    for agent_id in range(num_agents):
        destination_selector = DestinationSelector(reward_funcs[agent_id], agent_id, goal_ranker, state_filters[agent_id])
        agent = PathfindingAgent(agent_id, reward_funcs[agent_id], destination_selector=destination_selector)
        agent.reset(state, agent_idx=agent_id)
        agents.append(agent)

    agent0_reachable_goals = {2: (3, 2), 3: (1, 5)}
    agent1_reachable_goals = {0: (5, 6), 1: (4, 4)}
    agent2_reachable_goals = {3: (1, 5)}

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)

        assert agents[0].destination_selector.current_reachable_goals == agent0_reachable_goals
        assert agents[1].destination_selector.current_reachable_goals == agent1_reachable_goals
        assert agents[2].destination_selector.current_reachable_goals == agent2_reachable_goals

        if timestep == 3:
            del agent0_reachable_goals[3]
            del agent2_reachable_goals[3]
            del agent1_reachable_goals[1]
            assert np.array_equal(rewards, np.ones(num_agents) * (goal_rewards[1] + goal_rewards[3]))
        elif timestep == 6:
            del agent1_reachable_goals[0]
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[0])
        elif timestep == 7:
            del agent0_reachable_goals[2]
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[2])

        timestep += 1

    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env.reset(agent_reward_functions=reward_funcs)
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 50, 0, 0, 0, 4, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 52, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 3, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 2, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 51, 0, 0, 5, 1, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    for i, agent in enumerate(agents):
        agent.reset(state, i)

    agent0_reachable_goals = {3: (1, 5)}
    agent1_reachable_goals = {4: (7, 7), 1: (4, 4)}
    agent2_reachable_goals = {2: (3, 5), 3: (1, 5)}

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)

        assert agents[0].destination_selector.current_reachable_goals == agent0_reachable_goals
        assert agents[1].destination_selector.current_reachable_goals == agent1_reachable_goals
        assert agents[2].destination_selector.current_reachable_goals == agent2_reachable_goals

        if timestep == 3:
            del agent0_reachable_goals[3]
            del agent2_reachable_goals[3]
            del agent1_reachable_goals[4]
            assert np.array_equal(rewards, np.ones(num_agents) * (goal_rewards[4] + goal_rewards[3]))
        elif timestep == 5:
            del agent2_reachable_goals[2]
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[2])
        elif timestep == 9:
            del agent1_reachable_goals[1]
            assert np.array_equal(rewards, np.ones(num_agents) * goal_rewards[1])

        timestep += 1


def test_picnic_destination_selector():
    def compare_actions_at_timestep():
        if timestep <= len(agent0_actions):
            true_actions = [agent0_actions[timestep - 1], agent1_actions[timestep - 1], agent2_actions[timestep - 1],
                            agent3_actions[timestep - 1]]
            for action, true_action in zip(actions, true_actions):
                if isinstance(true_action, list):
                    assert action in true_action
                else:
                    assert action == true_action
        else:
            assert actions == [4, 4, 4, 4]

    num_agents = 4
    num_goals = 12
    horizon = 30
    dim = (12, 12)
    goal_rewards = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, (11, 11), seed=0)
    env.reset()
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 50, 0, 1, 0, 0, 0, 0, 2, 0, 51, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 5, 6, 0, 9, 10, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 11, 0, 0, 0, 0, 12, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 7, 8, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 52, 0, 3, 0, 0, 0, 0, 4, 0, 53, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    agents = []
    goal_ranker = ClosestDistanceGoalRanker()
    destination_selectors = [PicnicDestinationSelector(reward_funcs[0], 0, goal_ranker, True)]
    for agent_id in range(1, num_agents):
        destination_selector = PicnicDestinationSelector(reward_funcs[agent_id], agent_id, goal_ranker, False)
        destination_selectors[0].add_destination_selector(destination_selector)
        destination_selectors.append(destination_selector)

    for agent_id in range(num_agents):
        agent = PathfindingAgent(agent_id, reward_funcs[agent_id], destination_selectors[agent_id])
        agents.append(agent)

    for agent_idx in range(num_agents):
        agents[agent_idx].reset(state, agent_idx)

    agent0_actions = [1, 1, 1, 1, 2, 2, 2, 2, [1, 0], [1, 0], [1, 0], [1, 0], 4, 2, 2, 4, 4]
    agent1_actions = [3, 3, 3, 3, 2, 2, 2, 2, [1, 0], [1, 0], [1, 0], 1, 1, 3, 3, [2, 3], [2, 3]]
    agent2_actions = [1, 1, 1, 1, 0, 0, 0, 0, 3, 3, 4, 4, 4, [0, 1], [0, 1], [0, 1], [0, 1]]
    agent3_actions = [3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 4, 4, 4, [0, 3], 4, 4, 4]

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)
        compare_actions_at_timestep()
        timestep += 1

    env.reset()
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 50, 0, 0, 51, -1, 52, 0, 0, 53, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 1, 0, 0, 2, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 3, 0, 0, 4, -1],
                        [-1, 0, 0, 0, 0, -1, -1, -1, -1, 0, -1],
                        [-1, 0, 5, 6, 0, 0, 0, 7, -1, 8, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1],
                        [-1, 9, 10, 11, 12, 0, 0, 0, -1, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    agent0_actions = [2, 2, 2, 4, 1, 2, 2, 2, 4, 4, 4, 4, 4, 4, 1, 1]
    agent1_actions = [2, 2, 2, 4, 3, 2, 2, 2, 4, 4, 1, 1, 1, 1, 3, 3]
    agent2_actions = [2, 2, 2, 2, 1, 4, 4, 4, 1, 1, 4, 4, 4, 4, 4, 4]
    agent3_actions = [2, 2, 2, 2, 3, 1, 2, 2, 0, 4, 4, 4, 4, 4, 4, 4]

    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    for i, agent in enumerate(agents):
        agent.reset(state, i)
    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)
        compare_actions_at_timestep()
        timestep += 1


    env.reset(agent_reward_functions=reward_funcs)
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 50, 0, 1, 0, 0, 0, 0, 2, 0, 51, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 7, 0, 4, 0, 0, 0, 0, 5, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 9, 0, 0, 0, 0, 0, 0, 0, 8, -1],
                        [-1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 52, 0, 3, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 11, 0, 0, 0, 0, 0, 0, 0, 0, 10, -1],
                        [-1, 12, 0, 0, 0, 0, 0, 0, 0, 0, 53, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    agent0_actions = [1, 1, 1, 1, [2, 3], [2, 3], [2, 3], 4, 4, 4, 4, 4, 4, 2, [0, 3], [0, 3], [0, 3], 4, 4, 4, 4]
    agent1_actions = [3, 3, 3, 3, [1, 2], [1, 2], [1, 2], [1, 2], [1, 2], 4, 4, 4, 4, 4, [1, 2], [1, 2], [1, 2], 3, 3, 3, 3]
    agent2_actions = [1, 1, 4, 4, [0, 1], [0, 1], 4, 4, 4, 4, 4, 4, 4, 0, 3, 3, 4, 1, 1, 1, 4]
    agent3_actions = [0, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 2, 4, 4, 4, 4, 4, 4]

    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    for i, agent in enumerate(agents):
        agent.reset(state, i)

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)
        compare_actions_at_timestep()
        timestep += 1


    env.reset(agent_reward_functions=reward_funcs)
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 50, 0, 0, 1, -1, 2, 0, 0, 0, 51, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1],
                        [-1, 7, 0, 0, 4, -1, 8, 0, 0, 0, 5, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 9, -1, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 6, -1, 0, 0, 0, 0, 0, -1],
                        [-1, 52, 0, 0, 3, -1, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 12, 0, 0, 0, 10, -1],
                        [-1, 0, 0, 0, 0, -1, 11, 0, 0, 0, 53, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    agent0_actions = [2, 2, 4, 4, 1, 1, 1, 4, 4, 0, 0]
    agent1_actions = [2, 2, 4, 4, 3, 3, 3, 3, 4, 0, 0]
    agent2_actions = [1, 1, 1, 4, 0, 4, 4, 4, 4, 0, 4]
    agent3_actions = [0, 4, 4, 4, 3, 3, 3, 3, 4, 2, 4]

    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    for i, agent in enumerate(agents):
        agent.reset(state, i)

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)
        compare_actions_at_timestep()
        timestep += 1


def test_single_picnic_destination_selector():
    def compare_actions_at_timestep():
        if timestep <= len(agent0_actions):
            true_actions = [agent0_actions[timestep - 1], agent1_actions[timestep - 1], agent2_actions[timestep - 1],
                            agent3_actions[timestep - 1]]
            for action, true_action in zip(actions, true_actions):
                if isinstance(true_action, list):
                    assert action in true_action
                else:
                    assert action == true_action
        else:
            assert actions == [4, 4, 4, 4]

    num_agents = 4
    num_goals = 12
    horizon = 30
    dim = (12, 12)
    goal_rewards = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, (11, 11), seed=0)
    env.reset()
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 50, 0, 1, 0, 0, 0, 0, 2, 0, 51, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 5, 6, 0, 9, 10, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 11, 0, 0, 0, 0, 12, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 7, 8, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 52, 0, 3, 0, 0, 0, 0, 4, 0, 53, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    agents = []
    goal_ranker = ClosestDistanceGoalRanker()
    destination_selectors = [SinglePicnicDestinationSelector(reward_funcs[0], 0, goal_ranker, True)]
    for agent_id in range(1, num_agents):
        destination_selector = SinglePicnicDestinationSelector(reward_funcs[agent_id], agent_id, goal_ranker, False)
        destination_selectors[0].add_destination_selector(destination_selector)
        destination_selectors.append(destination_selector)

    for agent_id in range(num_agents):
        agent = PathfindingAgent(agent_id, reward_funcs[agent_id], destination_selectors[agent_id])
        agents.append(agent)

    for agent_idx in range(num_agents):
        agents[agent_idx].reset(state, agent_idx)

    agent0_actions = [1, 1, 1, 1]
    agent1_actions = [3, 3, 3, 3]
    agent2_actions = [1, 1, 1, 1]
    agent3_actions = [3, 3, 3, 3]

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)
        compare_actions_at_timestep()
        timestep += 1

    env.reset()
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 50, 0, 0, 51, -1, 52, 0, 0, 53, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 1, 0, 0, 2, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 3, 0, 0, 4, -1],
                        [-1, 0, 0, 0, 0, -1, -1, -1, -1, 0, -1],
                        [-1, 0, 5, 6, 0, 0, 0, 7, -1, 8, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1],
                        [-1, 9, 10, 11, 12, 0, 0, 0, -1, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    agent0_actions = [2, 2, 2, 4, 1]
    agent1_actions = [2, 2, 2, 4, 3]
    agent2_actions = [2, 2, 2, 2, 1]
    agent3_actions = [2, 2, 2, 2, 3]

    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    for i, agent in enumerate(agents):
        agent.reset(state, i)
    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)
        compare_actions_at_timestep()
        timestep += 1


    env.reset(agent_reward_functions=reward_funcs)
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 50, 0, 1, 0, 0, 0, 0, 2, 0, 51, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 7, 0, 4, 0, 0, 0, 0, 5, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 9, 0, 0, 0, 0, 0, 0, 0, 8, -1],
                        [-1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 52, 0, 3, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 11, 0, 0, 0, 0, 0, 0, 0, 0, 10, -1],
                        [-1, 12, 0, 0, 0, 0, 0, 0, 0, 0, 53, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    agent0_actions = [1, 1, 1, 1]
    agent1_actions = [3, 3, 3, 3]
    agent2_actions = [1, 1, 4, 4]
    agent3_actions = [0, 4, 4, 4]

    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    for i, agent in enumerate(agents):
        agent.reset(state, i)

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)
        compare_actions_at_timestep()
        timestep += 1


    env.reset(agent_reward_functions=reward_funcs)
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 50, 0, 0, 1, -1, 2, 0, 0, 0, 51, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1],
                        [-1, 7, 0, 0, 4, -1, 8, 0, 0, 0, 5, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 9, -1, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 6, -1, 0, 0, 0, 0, 0, -1],
                        [-1, 52, 0, 0, 3, -1, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 12, 0, 0, 0, 10, -1],
                        [-1, 0, 0, 0, 0, -1, 11, 0, 0, 0, 53, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    agent0_actions = [2, 2, 4, 4]
    agent1_actions = [2, 2, 4, 4]
    agent2_actions = [1, 1, 1, 4]
    agent3_actions = [0, 4, 4, 4]

    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.world = gridworld
    env.dim = dim
    state = np.copy(gridworld.state)

    for i, agent in enumerate(agents):
        agent.reset(state, i)

    done = False
    timestep = 1
    while not done:
        actions = [agent.action(state) for agent in agents]
        state, rewards, done, info = env.step(actions)
        compare_actions_at_timestep()
        timestep += 1

