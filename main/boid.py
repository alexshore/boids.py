import arcade
import random
import math
import numpy as np
from pathlib import Path

from constants import SCREEN_HEIGHT, SCREEN_WIDTH


class Boid(arcade.Sprite):
    def __init__(self, others=arcade.SpriteList()):
        super(Boid, self).__init__(
            filename=str(Path(__file__).parent / "boid.png"),
            scale=0.5,
            hit_box_algorithm="Detailed",
            flipped_diagonally=True,
            flipped_horizontally=True,
        )

        self.cohesion = 0.1
        self.separation = 0.1
        self.alignment = 0.1

        self.neighbourhood_size = 150
        self.close_neighbourhood_size = 25

        self.speed = 10.0

        self.others = others

        self.set_position(
            center_x=random.randint(0, SCREEN_WIDTH),
            center_y=random.randint(0, SCREEN_HEIGHT),
        )
        self.velocity = [0.0, self.speed]

    def fix_velocity_magnitude(self):
        self.velocity /= np.linalg.norm(self.velocity)
        self.velocity *= self.speed

    def get_neighbours(self):
        neighbours = []
        for other in self.others:
            if arcade.get_distance_between_sprites(self, other) < self.neighbourhood_size and other is not self:
                neighbours.append(other)
        return neighbours

    def get_cohesion(self, neighbours):
        avg_x = avg_y = 0

        for neighbour in neighbours:
            avg_x += neighbour.center_x
            avg_y += neighbour.center_y

        if not neighbours:
            return 0, 0

        return (
            self.cohesion * (avg_x / len(neighbours) - self.center_x),
            self.cohesion * (avg_y / len(neighbours) - self.center_y),
        )

    def get_separation(self, neighbours):
        avg_dx = avg_dy = close_neighbours = 0

        for neighbour in neighbours:
            if arcade.get_distance_between_sprites(self, neighbour) < self.close_neighbourhood_size:
                avg_dx += self.center_x - neighbour.center_x
                avg_dy += self.center_y - neighbour.center_y
                close_neighbours += 1

        if not close_neighbours:
            return 0, 0

        return (
            self.separation * avg_dx / close_neighbours,
            self.separation * avg_dy / close_neighbours,
        )

    def get_alignment(self, neighbours):
        avg_xvel = avg_yvel = 0

        for neighbour in neighbours:
            avg_xvel += neighbour.velocity[0]
            avg_yvel += neighbour.velocity[1]

        if not neighbours:
            return 0, 0

        return (
            self.alignment * (avg_xvel / len(neighbours) - self.velocity[0]),
            self.alignment * (avg_yvel / len(neighbours) - self.velocity[0]),
        )

    def update(self):
        neighbours = self.get_neighbours()
        cohesion = self.get_cohesion(neighbours=neighbours)
        separation = self.get_separation(neighbours=neighbours)
        alignment = self.get_alignment(neighbours=neighbours)

        change_velx = max(cohesion[0] + separation[0] + alignment[0], 0.2)
        change_vely = max(cohesion[1] + separation[1] + alignment[1], 0.2)

        self.velocity[0] += change_velx
        self.velocity[1] += change_vely

        self.fix_velocity_magnitude()

        self.center_x += self.velocity[0]
        self.center_y += self.velocity[1]

        self.angle = math.degrees(math.atan2(self.velocity[1], self.velocity[0]))

        self.wrap_around()

    def wrap_around(self):
        if self.left < 0:
            self.left += SCREEN_WIDTH
        elif self.right > SCREEN_WIDTH - 1:
            self.right -= SCREEN_WIDTH

        if self.bottom < 0:
            self.bottom += SCREEN_HEIGHT
        elif self.top > SCREEN_HEIGHT - 1:
            self.top -= SCREEN_HEIGHT
