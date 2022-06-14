from src.agents.reward_function import get_cooperative_reward_functions
from src.env.gridworld_env import GridworldEnv, GridWorld
from src.env.gridworld_wrappers import StationaryAgentsEarlyTermination


def test_stationary_agents_early_termination():
    num_agents = 3
    num_goals = 5
    horizon = 30
    dim = (11, 11)
    goal_rewards = [4, 3, 2, 1, 6]
    reward_funcs = get_cooperative_reward_functions(num_agents, goal_rewards, movement_penalty=0)
    env = GridworldEnv(num_agents, reward_funcs, num_goals, horizon, (11, 11), seed=0)
    env = StationaryAgentsEarlyTermination(env)

    env.reset()
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 50, 0, -1, 0, 0, 0, 52, -1],
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
    env.env.world = gridworld
    env.dim = dim

    actions_to_take = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [4, 4, 4], [4, 4, 4], [4, 4, 4]]

    for actions in actions_to_take:
        _, _, done, _ = env.step(actions)
        assert done is False
    _, _, done, _ = env.step([4, 4, 4])
    assert done is True

    env.reset()
    gridworld_import = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 52, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 1, 4, 0, -1],
                        [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
                        [-1, 3, 0, 0, 0, -1, 0, 51, 0, 0, -1],
                        [-1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1],
                        [-1, 0, 5, 0, 0, 0, 0, 50, 2, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]
    gridworld = GridWorld(num_agents, num_goals, dim)
    gridworld.import_gridworld(gridworld_import, num_agents, num_goals)
    env.env.world = gridworld

    actions_to_take = [[4, 0, 2], [4, 0, 2], [4, 0, 2]]
    for actions in actions_to_take:
        _, _, done, _ = env.step(actions)
        assert done is False
    _, _, done, _ = env.step([4, 0, 2])
    assert done is True

    env.reset()
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
    env.env.world = gridworld

    actions_to_take = [[4, 0, 2], [4, 0, 2], [4, 0, 2]]
    for actions in actions_to_take:
        _, _, done, _ = env.step(actions)
        assert done is False
    for i in range(horizon-len(actions_to_take)-1):
        actions = [4, i % 5, i % 5]
        _, _, done, _ = env.step(actions)
        assert done is False

    env.reset()
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
    env.env.world = gridworld

    actions_to_take = [[4, 0, 2], [4, 0, 2], [4, 0, 2], [4, 0, 0], [4, 1, 1], [4, 2, 2], [4, 3, 3],
                       [4, 0, 2], [4, 0, 2], [4, 0, 0], [4, 1, 1], [4, 2, 2], [4, 3, 3], [4, 0, 2],
                       [4, 0, 2], [4, 0, 2]]
    for actions in actions_to_take:
        _, _, done, _ = env.step(actions)
        print(env.world, actions)
        assert done is False
    _, _, done, _ = env.step([4, 0, 2])
    assert done is True


