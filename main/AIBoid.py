import random
from pathlib import Path

from typing import List, Tuple, Type

import arcade
import numpy as np
from constants import SCREEN_HEIGHT, SCREEN_WIDTH
from network import convert_weights_to_network


class Boid(arcade.Sprite):
    def __init__(self, net_id, weights):
        super(Boid, self).__init__(
            filename=str(Path(__file__).parent / "boid.png"),
            scale=0.5,
            hit_box_algorithm="Detailed",
        )

        self.net_id = net_id
        self.brain = convert_weights_to_network(weights)

        self.set_position(
            center_x=random.randint(0, SCREEN_WIDTH),
            center_y=random.randint(0, SCREEN_HEIGHT),
        )

        self.speed = 5

        self.other_boids = arcade.SpriteList()
        self.obstacles = arcade.SpriteList()

    def get_closest_n(self, n, sprites):
        closest = []

        for boid in sprites:
            if boid is self:
                continue

            closest.append((boid, arcade.get_distance_between_sprites(self, boid)))

        closest.sort(key=lambda boid: boid[1])

        return closest[:n]

    def get_distance_angle_to_and_direction_of_closest(self, closest):
        weights = [
            (
                distance,
                arcade.get_angle_radians(self.center_x, self.center_y, boid.center_x, boid.center_y),
                boid.radians,
            )
            for boid, distance in closest
        ]

        return np.transpose(weights).flatten()

    def update(self):
        closest_boids = self.get_closest_n(n=3, sprites=self.other_boids)
        closest_obstacle = self.get_closest_n(n=1, sprites=self.obstacles)[0]

        closest_boids_weights = self.get_distance_angle_to_and_direction_of_closest(closest_boids)

        network_input = [
            self.center_x,
            self.center_y,
            self.radians,
            *closest_boids_weights,
            closest_obstacle[1],
            arcade.get_angle_radians(
                self.center_x, self.center_y, closest_obstacle[0].center_x, closest_obstacle[0].center_y
            ),
        ]

        movement = self.brain.predict([network_input])[0][0] * 5

        self.angle += movement

        self.set_position(
            center_x=self.center_x + (-self.speed * np.sin(self.radians)),
            center_y=self.center_y + (self.speed * np.cos(self.radians)),
        )

        if self.left < 0 or self.right > SCREEN_WIDTH or self.bottom < 0 or self.top > SCREEN_HEIGHT:
            self.kill()
