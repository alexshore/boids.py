import arcade
from pathlib import Path
import random

from constants import OBSTACLE_SPEED, Direction


class Obstacle(arcade.Sprite):
    def __init__(self, position):
        super(Obstacle, self).__init__(
            filename=str(Path(__file__).parent / "obstacle.png"),
            center_x=position[0],
            center_y=position[1],
            scale=3,
        )


class MovingObstacle(Obstacle):
    def __init__(self, start_position, end_position, direction: Direction):
        super(MovingObstacle, self).__init__(position=start_position)

        self.start_position = start_position
        self.end_position = end_position
        self.direction = direction

        if self.direction == Direction.VERTICAL:
            self.change_y = OBSTACLE_SPEED
        if self.direction == Direction.HORIZONTAL:
            self.change_x = OBSTACLE_SPEED

    def update(self):
        self.set_position(self.center_x + self.change_x, self.center_y + self.change_y)

        if self.direction == Direction.VERTICAL and self.center_y > self.end_position[1]:
            self.center_y = self.start_position[1]
        if self.direction == Direction.HORIZONTAL and self.center_x > self.end_position[0]:
            self.center_x = self.start_position[0]
