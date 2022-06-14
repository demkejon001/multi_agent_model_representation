from PIL import Image, ImageDraw, ImageColor
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


class Drawer:
    def __init__(self, world_dim, wall_positions, goal_positions, agent_positions, img_dimensions=(660, 660)):
        self._img_dimensions = img_dimensions
        self.rectangle_dim = int(min(img_dimensions[0] / world_dim[1], img_dimensions[1] / world_dim[0]))
        self.rectangle_offset = ((self._img_dimensions[0] - self.rectangle_dim * world_dim[1]) / 2,
                                 (self._img_dimensions[1] - self.rectangle_dim * world_dim[0]) / 2)
        self.use_outline = False
        self.goal_colors = []
        self.agent_colors = []

        rng = np.random.default_rng(42)
        def random_color_gen(): return tuple(rng.choice(range(256), size=3))
        for _ in range(len(agent_positions)):
            self.agent_colors.append(random_color_gen())
        for _ in range(len(goal_positions)):
            self.goal_colors.append(random_color_gen())

        plt.ion()
        plt.show()
        self.image = None
        self._wall_positions = wall_positions
        self._goal_positions = goal_positions
        self._agent_positions = agent_positions

    def show(self, data):
        cv2.imshow('Gridworld', data)
        cv2.waitKey(1000)

    def draw(self, state):
        im = Image.new('RGB', self._img_dimensions, 'white')
        drawing = ImageDraw.Draw(im)
        self.drawGrid(drawing, state)
        self.show(np.array(im))

    def get_world_position(self, gridworld_position):
        return [self.rectangle_dim * gridworld_position[1] + self.rectangle_offset[0],
                self.rectangle_dim * gridworld_position[0] + self.rectangle_offset[1],
                self.rectangle_dim * (gridworld_position[1] + 1) + self.rectangle_offset[0],
                self.rectangle_dim * (gridworld_position[0] + 1) + self.rectangle_offset[1]]

    def drawRectangle(self, drawing, color, position):
        if self.use_outline:
            drawing.rectangle(position, fill=color, outline=ImageColor.getrgb('black'))
        else:
            drawing.rectangle(position, fill=color)

    def drawCircle(self, drawing, color, position):
        drawing.ellipse(position, fill=color)

    def drawGoals(self, drawing, positions):
        for position in positions:
            self.drawRectangle(drawing, self.goal_colors[position[0]],
                               self.get_world_position((position[1], position[2])))

    def drawAgents(self, drawing, positions):
        for position in positions:
            self.drawCircle(drawing, self.agent_colors[position[0]],
                            self.get_world_position((position[1], position[2])))

    def drawWalls(self, drawing, walls):
        walls = walls[0]
        for row, grid in enumerate(walls):
            for col, wall in enumerate(grid):
                self.drawRectangle(drawing, 'black' if wall == 1 else 'white', self.get_world_position((row, col)))

    def get_feature_positions(self, state):
        return zip(*np.where(state == 1))

    def drawGrid(self, drawing, state):
        walls = state[self._wall_positions]
        goals = state[self._goal_positions]
        agents = state[self._agent_positions]
        self.drawWalls(drawing, walls)
        self.drawGoals(drawing, self.get_feature_positions(goals))
        self.drawAgents(drawing, self.get_feature_positions(agents))


class ToMnetRLDrawer(Drawer):
    def __init__(self, world_dim, wall_positions, goal_positions, agent_positions, img_dimensions=(660, 660)):
        super().__init__(world_dim, wall_positions, goal_positions, agent_positions, img_dimensions)
        self.goal_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 0), (0, 122, 255), (255, 0, 255)]
        self.agent_colors = [(0, 0, 0)]

    def drawGoals(self, drawing, positions):
        for position in positions:
            if position[0] > 0:
                self.drawRectangle(drawing, self.goal_colors[position[0]],
                                   self.get_world_position((position[1], position[2])))
            else:
                self.drawStar(drawing, self.goal_colors[position[0]],
                              self.get_star_positions((position[1], position[2])))

    def drawStar(self, drawing, color, position):
        drawing.polygon(position, fill=color, outline=ImageColor.getrgb('black') if self.use_outline else None)

    def get_star_positions(self, gridworld_position):
        star_points = []
        star_point_radian_offset = (2 * math.pi) / 5
        interior_point_scale = .427
        ext_point_offset = -math.pi / 2
        int_point_offset = ext_point_offset + (star_point_radian_offset / 2)
        half_rectangle_dim = self.rectangle_dim / 2

        for i in range(5):
            ext_xy = (
                (math.cos(ext_point_offset + i * star_point_radian_offset) * half_rectangle_dim) + half_rectangle_dim,
                (math.sin(ext_point_offset + i * star_point_radian_offset) * half_rectangle_dim) + half_rectangle_dim)
            int_xy = ((math.cos(
                int_point_offset + i * star_point_radian_offset) * half_rectangle_dim) * interior_point_scale + half_rectangle_dim,
                      (math.sin(
                          int_point_offset + i * star_point_radian_offset) * half_rectangle_dim) * interior_point_scale + half_rectangle_dim)
            star_points.append(ext_xy)
            star_points.append(int_xy)

        star_positioning = lambda star_points, gridworld_position: [(self.rectangle_dim * gridworld_position[1] +
                                                                     star_point[0],
                                                                     self.rectangle_dim * gridworld_position[0] +
                                                                     star_point[1]) for star_point in star_points]

        return star_positioning(star_points, gridworld_position)
