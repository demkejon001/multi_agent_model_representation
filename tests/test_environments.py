import numpy as np
import random

from src.env.gridworld import GridWorld
from src.env.gridworld_env import GridworldEnv
from src.agents.reward_function import get_competitive_reward_functions, get_cooperative_reward_functions


def test_collisions():
    gridworld = GridWorld()
    gridworld_array = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 4, 0, 0, 3, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 50, 0, 0, 0, 2, -1],
                       [-1, 0, 0, 0, 53, 0, 51, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 52, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    gridworld.import_gridworld(gridworld_array, num_agents=4, num_goals=4)

    # Test agent's trying to occupy same space
    gridworld.step([2, 3, 0, 1])
    assert repr(gridworld) == np.array_repr(np.array(gridworld_array))
    gridworld.step([1, 0, 3, 2])
    assert repr(gridworld) == np.array_repr(np.array(gridworld_array))

    gridworld_steps = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 4, 0, 0, 3, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 53, 0, 50, 0, 0, 2, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 52, 0, 51, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    gridworld.step([1, 2, 3, 0])
    assert repr(gridworld) == np.array_repr(np.array(gridworld_steps))


    gridworld_array = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 4, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 51, 50, 52, 3, 0, 0, 2, -1],
                       [-1, 0, 0, 0, 0, 53, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    gridworld.import_gridworld(gridworld_array, num_agents=4, num_goals=4)

    # Test agents bumping into each other
    gridworld_step1 = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 4, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 51, 50, 0, 52, 0, 0, 2, -1],
                       [-1, 0, 0, 0, 0, 53, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    gridworld.step([1, 1, 1, 0])
    assert repr(gridworld) == np.array_repr(np.array(gridworld_step1))

    # Test agents 51 and 50 can't swap places
    gridworld.import_gridworld(gridworld_array, num_agents=4, num_goals=4)
    gridworld_step2 = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 4, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 51, 50, 53, 52, 0, 0, 2, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    gridworld.step([3, 1, 1, 0])
    assert repr(gridworld) == np.array_repr(np.array(gridworld_step2))

    # Test agents can shift in same direction
    # No import so continuing on from step2
    gridworld_step3 = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 4, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 51, 50, 53, 52, 0, 2, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    gridworld.step([1,1,1,1])
    assert repr(gridworld) == np.array_repr(np.array(gridworld_step3))


    gridworld_array = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                       [-1, 0, 4, 0, 52, -1, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 51, 50, 53, 3, 0, 0, 2, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    gridworld.import_gridworld(gridworld_array, num_agents=4, num_goals=4)

    # Testing agents bumping into walls and each other
    gridworld.step([0, 1, 1, 3])
    assert repr(gridworld) == np.array_repr(np.array(gridworld_array))

    # Test agents shifting together, but bumping into wall
    gridworld_array = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                       [-1, 0, 4, 0, 0, -1, 0, 0, 0, 0, -1],
                       [-1, 52, 53, 51, 50, 0, 3, 0, 0, 2, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    gridworld.import_gridworld(gridworld_array, num_agents=4, num_goals=4)
    gridworld.step([3, 3, 3, 3])
    assert repr(gridworld) == np.array_repr(np.array(gridworld_array))


    gridworld_array = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 4, 0, 50, 3, 51, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 52, 2, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 53, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    gridworld.import_gridworld(gridworld_array, num_agents=4, num_goals=4)

    # Test agents try to consume goal, but bump into each other
    gridworld.step([1, 3, 1, 0])
    assert repr(gridworld) == np.array_repr(np.array(gridworld_array))

    # Test agents are able to consume goals
    gridworld_step1 = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 4, 0, 0, 0, 51, 0, 0, 53, -1],
                       [-1, 0, 0, 0, 0, 50, 0, 0, 52, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    gridworld.step([1, 4, 4, 0])
    gridworld.step([2, 4, 4, 0])
    assert repr(gridworld) == np.array_repr(np.array(gridworld_step1))


def test_import_gridworld():
    gridworld_array = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1],
                       [-1, 0, 51, 0, 0, 0, 0, -1, 0, 0, -1],
                       [-1, 0, 4, 0, 0, 3, 0, -1, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, -1, 0, 2, -1],
                       [-1, 0, 0, 0, 0, 0, 0, -1, 0, 50, -1],
                       [-1, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1],
                       [-1, 0, 0, 0, 0, -1, -1, -1, -1, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 1, 0, -1, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    imported_gridworld = GridWorld()
    imported_gridworld.import_gridworld(gridworld_array, num_goals=4, num_agents=2)

    assert repr(imported_gridworld) == np.array_repr(np.array(gridworld_array))

    imported_gridworld.step([0, 2])
    imported_gridworld.step([3, 1])
    imported_gridworld.step([3, 1])
    gridworld_steps = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1],
                       [-1, 0, 0, 0, 51, 3, 0, -1, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, -1, 50, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1],
                       [-1, 0, 0, 0, 0, -1, -1, -1, -1, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 1, 0, -1, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

    assert repr(imported_gridworld) == np.array_repr(np.array(gridworld_steps))


def test_random_gen():
    gridworld1 = GridWorld(seed=42)

    # Test gridworld creates new worlds and keeps old world
    gridworld1.reset(create_new_world=False)
    gridworld1_str = str(gridworld1)
    gridworld1.reset(create_new_world=False)
    assert gridworld1_str == str(gridworld1)
    gridworld1.reset(create_new_world=True)
    # Creating new world should be different
    assert gridworld1_str != str(gridworld1)
    gridworld1_str = str(gridworld1)
    gridworld1.reset(create_new_world=False)
    # The new world should now be the same
    assert gridworld1_str == str(gridworld1)

    # Test gridworlds with same seed stay the same
    gridworld1 = GridWorld(seed=42)
    gridworld2 = GridWorld(seed=42)
    gridworld1.reset(create_new_world=False)
    gridworld2.reset(create_new_world=False)
    assert str(gridworld2) == str(gridworld1)
    gridworld1.reset(create_new_world=True)
    gridworld2.reset(create_new_world=True)
    assert str(gridworld2) == str(gridworld1)

    # Test steps don't affect generation
    gridworld1.step([0])
    gridworld1.reset(create_new_world=False)
    gridworld2.reset(create_new_world=False)
    assert str(gridworld2) == str(gridworld1)
    gridworld1.step([0])
    gridworld1.reset(create_new_world=True)
    gridworld2.reset(create_new_world=True)
    assert str(gridworld2) == str(gridworld1)

    # Random seed makes different worlds
    gridworld1 = GridWorld()
    gridworld2 = GridWorld()
    gridworld1.reset(create_new_world=False)
    gridworld2.reset(create_new_world=False)
    assert str(gridworld2) != str(gridworld1)
    gridworld1.reset(create_new_world=True)
    gridworld2.reset(create_new_world=True)
    assert str(gridworld2) != str(gridworld1)


def test_cooperative_rewards():
    num_agents = 4
    num_goals = 4
    goal_rewards = np.random.random(num_goals)
    movement_penalty = random.random() * -1
    reward_functions = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=movement_penalty)
    env = GridworldEnv(num_agents, reward_functions, num_goals)
    gridworld_array = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 4, 50, 2, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 53, 3, 51, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 52, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    env.world.import_gridworld(gridworld_array, num_agents=num_agents, num_goals=num_goals)
    _, rewards, _, _ = env.step([0,0,0,0])
    assert np.allclose(rewards, [np.sum(goal_rewards) + movement_penalty for _ in range(num_agents)])


def test_competitive_rewards():
    num_agents = 4
    num_goals = 4
    goal_rewards = np.random.random((num_agents, num_goals))
    movement_penalty = random.random() * -1
    reward_functions = get_competitive_reward_functions(goal_rewards, movement_penalty=movement_penalty)
    env = GridworldEnv(num_agents, reward_functions, num_goals)
    gridworld_array = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 4, 50, 2, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 53, 3, 51, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 52, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    env.world.import_gridworld(gridworld_array, num_agents=num_agents, num_goals=num_goals)
    _, rewards, _, _ = env.step([0,0,0,0])
    agent_rewards = np.ones(num_agents) * movement_penalty
    for agent_idx, goal_reward in enumerate(goal_rewards):
        agent_rewards[agent_idx] += goal_reward[agent_idx]
        agent_rewards[[i for i in range(num_agents) if i != agent_idx]] += -goal_reward[agent_idx]/(num_agents - 1)
    assert np.allclose(rewards, agent_rewards)
