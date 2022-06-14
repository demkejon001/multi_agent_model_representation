import numpy as np
import collections
from enum import IntEnum


class Collision(IntEnum):
    WALL = -1
    PLAYER = -2


class CollisionRecord:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.collisions = [[] for _ in range(self.num_agents)]

    def add_collision(self, agent, collision_type):
        self.collisions[agent].append(int(collision_type))

    def get_agent_collisions(self, agent):
        return self.collisions[agent]

    def get_goal_collisions(self):
        goal_collisions = [-1 for _ in range(self.num_agents)]
        for agent_id, agent_collisions in enumerate(self.collisions):
            for collision in agent_collisions:
                if collision >= 0:
                    goal_collisions[agent_id] = collision
        return goal_collisions

    def clear(self):
        self.collisions = [[] for _ in range(self.num_agents)]


class GridWorld:
    def __init__(self, num_agents=1, num_goals=1, dim=(11, 11), min_num_walls=0, max_num_walls=2, seed=None):
        self.dim = dim
        self.num_agents = num_agents
        self.num_goals = num_goals
        self.min_num_walls = min_num_walls
        self.max_num_walls = max_num_walls
        self.state = np.zeros((self.num_goals + self.num_agents + 1, self.dim[0], self.dim[1]))  # +1 for the wall

        self.feature_positions = dict()
        self.assign_feature_positions()

        self.agent_positions = []
        self.wall_positions = set()
        self.goal_positions = dict()

        self.collision_record = CollisionRecord(self.num_agents)

        if seed is None:
            self._seed = np.random.SeedSequence().generate_state(1)[0]
        else:
            self._seed = seed
        self.rng = self._get_rng()

        self.actions = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (0, 0)}

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, new_seed):
        if new_seed is None:
            new_seed = np.random.SeedSequence().generate_state(1)[0]
        self._seed = new_seed
        self.rng = self._get_rng()

    def _get_rng(self):
        return np.random.default_rng(self._seed)

    def import_gridworld(self, gridworld, num_agents, num_goals):
        wall_value = -1
        goal_min_value = 1
        goal_max_value = 49
        agent_min_value = 50
        agent_max_value = 99
        self.num_agents = num_agents
        self.num_goals = num_goals
        self.feature_positions = dict()
        self.assign_feature_positions()
        self.goal_positions = dict()
        goal_count = 0
        self.wall_positions = set()
        self.agent_positions = [None for _ in range(self.num_agents)]
        agent_count = 0
        self.dim = (len(gridworld), len(gridworld[0]))
        self.state = np.zeros((self.num_goals + self.num_agents + 1, self.dim[0], self.dim[1]))
        for row in range(self.dim[0]):
            for col in range(self.dim[1]):
                gridworld_value = gridworld[row][col]
                if goal_min_value <= gridworld_value <= goal_max_value:
                    self.add_position('goal' + str(gridworld_value - 1), (row, col))
                    goal_count += 1
                    if goal_count > self.num_goals:
                        raise ValueError('num_goals is less than number of goals in imported gridworld')
                elif gridworld_value == wall_value:
                    self.add_position('walls', (row, col))
                elif agent_min_value <= gridworld_value <= agent_max_value:
                    self.state[self.feature_positions['agent' + str(gridworld_value - agent_min_value)]][(row, col)] = 1
                    self.agent_positions[gridworld_value - agent_min_value] = (row, col)
                    agent_count += 1
                    if agent_count > self.num_agents:
                        raise ValueError('num_agents is greater than number of agents in gridworld')
                elif gridworld_value == 0:
                    continue
                else:
                    raise TypeError(gridworld_value,
                                    'is not an acceptable gridworld symbol, i.e. not in range [-1, 99]')

        if goal_count < self.num_goals:
            raise ValueError('num_goals is less than number of goals in gridworld')
        if agent_count < self.num_agents:
            raise ValueError('num_agents is less than number of agents in gridworld')
        if len(set(goal for goal in self.goal_positions.values())) < self.num_goals:
            raise ValueError('you have the same goal value in the imported gridworld')
        if None in self.agent_positions:
            raise ValueError('you have the same agent value in the imported gridworld')

        self.collision_record = CollisionRecord(self.num_agents)

    def get_random_seed(self):
        return self.rng.integers(1, 9223372036854775807, dtype=np.int64)

    def reset(self, create_new_world=True):
        if create_new_world:
            self._seed = self.get_random_seed()
        self.rng = self._get_rng()
        self.state = np.zeros((self.num_goals + self.num_agents + 1, self.dim[0], self.dim[1]))  # +1 for the wall
        self.agent_positions = []
        self.wall_positions = set()
        self.goal_positions = dict()
        self.create_world(self.rng.integers(self.min_num_walls, self.max_num_walls, endpoint=True))

    def __repr__(self):
        agent_min_value = 50
        state = np.copy(self.state[0])
        state = state * -1
        # state = state.sum(axis=0)
        for goal_position, goal in self.goal_positions.items():
            state[goal_position] = str(int(goal) + 1)
        for agent, agent_position in enumerate(self.agent_positions):
            state[agent_position] = str(agent_min_value + agent)
        return state.astype(np.int).__repr__()

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
        x = self.rng.integers(1, self.dim[1] - 2, dtype=int, endpoint=True)
        y = self.rng.integers(1, self.dim[0] - 2, dtype=int, endpoint=True)
        return x, y

    def add_position(self, feature, position):
        self.state[self.feature_positions[feature]][position] = 1
        if 'walls' in feature:
            self.wall_positions.add(position)
        elif 'goal' in feature:
            self.goal_positions[position] = feature.split('goal')[1]
        elif 'agent' in feature:
            self.agent_positions.append(position)

    def create_border(self):
        for i in range(self.dim[0]):
            self.add_position('walls', (i, 0))
            self.add_position('walls', (i, self.dim[1] - 1))
        for i in range(self.dim[1]):
            self.add_position('walls', (0, i))
            self.add_position('walls', (self.dim[0] - 1, i))

    # Create Walls only make L-shaped walls.
    def create_wall(self):
        start_position = self.random_position()
        end_position = self.random_position()
        x_len = start_position[0] - end_position[0]
        y_len = start_position[1] - end_position[1]

        positions = lambda x: range(0, x + 1) if x > -1 else range(0, x - 1, -1)

        for x in positions(x_len * -1):
            #       self.state[self.feature_positions['walls']][start_position[0] + x, start_position[1]] = 1
            self.add_position('walls', (start_position[0] + x, start_position[1]))
        for y in positions(y_len):
            #       self.state[self.feature_positions['walls']][end_position[0], end_position[1]+y] = 1
            self.add_position('walls', (end_position[0], end_position[1] + y))

    def get_unoccupied_position(self):
        position = self.random_position()
        while np.any(self.state[:, position[0], position[1]]):
            position = self.random_position()
        return position

    def create_world(self, num_walls=0):
        self.create_border()

        for _ in range(num_walls):
            self.create_wall()

        for goal in range(self.num_goals):
            goal_position = self.get_unoccupied_position()
            self.add_position('goal' + str(goal), goal_position)

        for agent in range(self.num_agents):
            agent_position = self.get_unoccupied_position()
            self.add_position('agent' + str(agent), agent_position)

    def check_wall_collisions(self, agent_positions):
        for agent, agent_position in enumerate(agent_positions):
            if agent_position in self.wall_positions:
                agent_positions[agent] = self.agent_positions[agent]  # Placing agent in original position
                self.collision_record.add_collision(agent, Collision.WALL)

    # Agents atm can collide with each other if they occupy positions adjacent to each other and they move to each other's position (i.e. they run into each other)
    # Agents can also collide with each other if they try to occupy the same position
    # Any agent that collides with another agent is placed in it's original position.
    def check_agents_occupy(self, agent_positions):
        collision = False
        duplicates = collections.defaultdict(list)
        for agent, agent_position in enumerate(agent_positions):
            duplicates[agent_position].append(agent)
        for colliding_agents in sorted(duplicates.values()):
            if len(colliding_agents) >= 2:
                for colliding_agent in colliding_agents:
                    collision = True
                    agent_positions[colliding_agent] = self.agent_positions[colliding_agent]
                    self.collision_record.add_collision(colliding_agent, Collision.PLAYER)
        return collision

    def check_agents_swap(self, agent_positions):
        collision = False
        for agentA, agentA_position in enumerate(agent_positions):
            for agentB, agentB_position in enumerate(agent_positions):
                if agentA != agentB:
                    if self.agent_positions[agentB] == agentA_position and self.agent_positions[
                        agentA] == agentB_position:  # (i.e. Did they swap?)
                        collision = True
                        agent_positions[agentB] = self.agent_positions[agentB]
                        agent_positions[agentA] = self.agent_positions[agentA]
                        self.collision_record.add_collision(agentA, Collision.PLAYER)
                        self.collision_record.add_collision(agentB, Collision.PLAYER)
        return collision

    def check_agent_collisions(self, agent_positions):
        collision = True
        while collision:
            collision = self.check_agents_occupy(agent_positions)

        collision = True
        while collision:
            collision = self.check_agents_swap(agent_positions)

    def check_goal_collisions(self, agent_positions):
        for agent, agent_position in enumerate(agent_positions):
            if agent_position in self.goal_positions:
                goal = self.goal_positions.pop(agent_position)
                self.state[self.feature_positions['goal' + str(goal)]][agent_position] = 0
                self.collision_record.add_collision(agent, goal)

    def record_collisions(self, new_positions):
        self.check_wall_collisions(new_positions)
        self.check_agent_collisions(new_positions)
        self.check_goal_collisions(new_positions)

    def transition(self, agent_positions):
        for agent, agent_position in enumerate(agent_positions):
            self.state[self.feature_positions['agent' + str(agent)]][self.agent_positions[agent]] = 0
            self.state[self.feature_positions['agent' + str(agent)]][agent_position] = 1
            self.agent_positions[agent] = agent_position

    def get_new_positions(self, actions):
        return [(self.agent_positions[agent][0] + self.actions[action][0],
                 self.agent_positions[agent][1] + self.actions[action][1]) for agent, action in enumerate(actions)]

    def step(self, actions):
        assert len(actions) == self.num_agents
        self.collision_record.clear()
        new_positions = self.get_new_positions(actions)
        self.record_collisions(new_positions)
        self.transition(new_positions)
        return self.collision_record


class RepositionGridworld(GridWorld):
    def __init__(self, num_agents=1, num_goals=1, dim=(11, 11), min_num_walls=0, max_num_walls=2, seed=None):
        super().__init__(num_agents, num_goals, dim, min_num_walls, max_num_walls, seed)
        self.agent_starting_positions = []

    def reset(self, create_new_world=True):
        self.agent_starting_positions = []
        super().reset(create_new_world)

    def add_position(self, feature, position):
        super().add_position(feature, position)
        if 'agent' in feature:
            self.agent_starting_positions.append(position)

    def reposition_agents(self):
        new_starting_agent_positions = []
        for i in range(len(self.agent_starting_positions)):
            agent_position = self.agent_starting_positions[(i + 1) % len(self.agent_starting_positions)]
            new_starting_agent_positions.append(agent_position)
        self.transition(new_starting_agent_positions)
        self.agent_starting_positions = new_starting_agent_positions
