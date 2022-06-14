import random

from tommas.env.gridworld import *


class SeparatedGridworld(GridWorld):
    def __init__(self, num_agents=1, num_goals=1, dim=(11, 11), seed=None, num_walls=2):
        super().__init__(num_agents, num_goals, dim, seed, num_walls)
        self.rooms = {}
        self.current_room = -1

    def reset(self, create_new_world=True):
        random.seed(self.seed)
        if create_new_world:
            self.seed = random.random()
        self.state = np.zeros((self.num_goals + self.num_agents + 1, self.dim[0], self.dim[1]))  # +1 for the wall
        self.agent_positions = []
        self.wall_positions = set()
        self.goal_positions = dict()
        self.create_world(self.num_walls)  # TODO: Make this a variable

    def assign_feature_positions(self):
        position = 0
        self.feature_positions['walls'] = position
        position += 1
        for goal in range(self.num_goals):
            self.feature_positions['goal' + str(goal)] = position
            position += 1
        for agent in range(self.num_agents):
            self.feature_positions['agent' + str(agent)] = position
            position += 1

    def get_feature_positions(self):
        wall_positions = []
        goal_positions = []
        agent_positions = []
        for feature, position in self.feature_positions.items():
            if 'wall' in feature:
                wall_positions.append(position)
            if 'goal' in feature:
                goal_positions.append(position)
            if 'agent' in feature:
                agent_positions.append(position)
        return wall_positions, goal_positions, agent_positions

    def random_position(self):
        # print(self.current_room)
        # print(self.rooms[self.current_room])
        min_extent, max_extent = self.rooms[self.current_room]
        x = random.randint(min_extent[0], max_extent[0])
        y = random.randint(min_extent[1], max_extent[1])
        return x, y

    def find_mult_factors(self):
        factor_pairs = []
        for factor in range(1, self.num_agents + 1):
            if self.num_agents % factor == 0:
                factor_pairs.append((factor, self.num_agents // factor))
        return factor_pairs[len(factor_pairs) // 2]

    def create_separate_rooms(self):
        room_dimensions = self.find_mult_factors()
        row_walls_to_add = room_dimensions[0] - 1
        col_walls_to_add = room_dimensions[1] - 1
        row_spacing = (self.dim[0] - 2 - row_walls_to_add) // max(room_dimensions)
        col_spacing = (self.dim[1] - 2 - col_walls_to_add) // min(room_dimensions)

        for row in range(1, room_dimensions[0]):
            for col in range(self.dim[1]):
                self.add_position('walls', ((row_spacing + 1) * row, col))
        for col in range(room_dimensions[1]):
            for row in range(self.dim[0]):
                self.add_position('walls', (row, (col_spacing + 1) * col))

        agent = 0
        for row in range(1, room_dimensions[0] + 1):
            for col in range(1, room_dimensions[1] + 1):
                min_extent = ((row_spacing + 1) * row) - row_spacing, ((col_spacing + 1) * col) - col_spacing
                max_extent = ((row_spacing + 1) * row) - 1, ((col_spacing + 1) * col) - 1
                self.rooms[agent] = (min_extent, max_extent)
                agent += 1

    def create_border(self):
        for i in range(self.dim[0]):
            self.add_position('walls', (i, 0))
            self.add_position('walls', (i, self.dim[1] - 1))
        for i in range(self.dim[1]):
            self.add_position('walls', (0, i))
            self.add_position('walls', (self.dim[0] - 1, i))
        self.create_separate_rooms()

    def create_world(self, num_walls=0):
        self.create_border()

        for agent in range(self.num_agents):
            self.current_room = agent
            for _ in range(num_walls):
                self.create_wall()

            for goal in range(self.num_goals):
                goal_position = self.get_unoccupied_position()
                self.add_position('goal' + str(goal), goal_position)

            agent_position = self.get_unoccupied_position()
            self.add_position('agent' + str(agent), agent_position)
