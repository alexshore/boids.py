import random
from pathlib import Path
import torch

import arcade
import numpy as np
from constants import SCREEN_HEIGHT, SCREEN_WIDTH
from network import convert_weights_to_torch


class Boid(arcade.Sprite):
    def __init__(self, weights, others=arcade.SpriteList(), obstacles=arcade.SpriteList()):
        super(Boid, self).__init__(
            filename=str(Path(__file__).parent / "boid.png"),
            scale=0.5,
            hit_box_algorithm="Detailed",
        )

        self.brain = convert_weights_to_torch(weights)

        self.set_position(
            center_x=random.randint(0, SCREEN_WIDTH),
            center_y=random.randint(0, SCREEN_HEIGHT),
        )

        self.fitnesses = {
            "movement_variation": 0,
            "wall_avoidance": 0,
            "cohesion": 0,
            "separation": 0,
            "alignment": 0,
        }

        self.speed = 5

        self.last_change_angle = 0
        self.ticks_alive = 1

        self.others = others
        self.obstacles = obstacles

    def get_neighbours(self):
        neighbours = []
        for other in self.others:
            if arcade.get_distance_between_sprites(self, other) < 300 and other is not self:
                neighbours.append(other)
        return neighbours

    def calculate_cohesion_weights(self, neighbours):
        if not neighbours:
            return self.center_x, self.center_y

        avg_x, avg_y = 0, 0
        for neighbour in neighbours:
            avg_x += neighbour.center_x / len(neighbours)
            avg_y += neighbour.center_y / len(neighbours)

        return avg_x, avg_y

    def calculate_separation_weights(self, neighbours):
        if not neighbours:
            return self.center_x, self.center_y

        avg_x, avg_y, close_neighbours = 0, 0, 0
        for neighbour in neighbours:
            if arcade.get_distance_between_sprites(self, neighbour) < 80:
                avg_x += neighbour.center_x
                avg_y += neighbour.center_y
                close_neighbours += 1

        if not close_neighbours:
            return self.center_x, self.center_y

        return avg_x / close_neighbours, avg_y / close_neighbours

    def calculate_alignment_weights(self, neighbours):
        if not neighbours:
            return self.angle

        avg_angle = 0
        for neighbour in neighbours:
            avg_angle += neighbour.angle / len(neighbours)

        return avg_angle

    def update(self):
        neighbours = self.get_neighbours()

        cohesion_weights = self.calculate_cohesion_weights(neighbours)
        separation_weights = self.calculate_separation_weights(neighbours)
        alignment_weights = self.calculate_alignment_weights(neighbours)

        network_input = torch.tensor(
            [
                [
                    self.center_x,
                    self.center_y,
                    self.angle,
                    *cohesion_weights,
                    *separation_weights,
                    alignment_weights,
                ]
            ],
            dtype=torch.float,
            requires_grad=False,
        )

        output = self.brain.forward(network_input)[0]

        if np.argmax(output) == 0:
            self.angle -= output[0].item() * 5
        else:
            self.angle += output[1].item() * 5

        self.ticks_alive += 1

        self.set_position(
            center_x=self.center_x + (-self.speed * np.sin(self.radians)),
            center_y=self.center_y + (self.speed * np.cos(self.radians)),
        )

        self.wrap_around()

        self.update_fitness(cohesion_weights, separation_weights, alignment_weights)

    def update_fitness(self, cohesion_weights, separation_weights, alignment_weights):
        self.fitnesses["cohesion"] += arcade.get_distance(self.center_x, self.center_y, *cohesion_weights)
        self.fitnesses["separation"] += arcade.get_distance(self.center_x, self.center_y, *separation_weights)
        self.fitnesses["alignment"] += abs(self.angle - alignment_weights)

    def wrap_around(self):
        if self.left < 0:
            self.left += SCREEN_WIDTH
        elif self.right > SCREEN_WIDTH - 1:
            self.right -= SCREEN_WIDTH

        if self.bottom < 0:
            self.bottom += SCREEN_HEIGHT
        elif self.top > SCREEN_HEIGHT - 1:
            self.top -= SCREEN_HEIGHT
