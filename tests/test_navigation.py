from tommas.helper_code.navigation import djikstra_actions, maze_traversal
from tommas.env.gridworld import GridWorld


def test_djikstra_actions():
    def tuple_add(x, y):
        return tuple([sum(a) for a in zip(x, y)])

    maze = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]


    action_dict = GridWorld().actions
    actions = []
    for action in range(4):
        actions.append(action_dict[action])

    start = (1, 1)

    possible_paths = djikstra_actions(maze, start)
    for dest, action_traj in possible_paths.items():
        end_position = start
        for action in action_traj:
            end_position = tuple_add(end_position, actions[action])
        assert end_position == dest

    possible_positions = maze_traversal(maze, start)
    for position in possible_positions:
        if position != start:
            assert position in possible_paths

