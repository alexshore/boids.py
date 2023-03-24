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

    def get_closest_n(self, n, sprites):
        closest = []

        for sprite in sprites:
            closest.append((sprite, arcade.get_distance_between_sprites(self, sprite)))

        closest.sort(key=lambda data: data[1])

        return closest[:n]

    def get_distance_angle_to_and_direction_of_others(self, others):
        weights = [
            (
                distance,
                arcade.get_angle_radians(self.center_x, self.center_y, other.center_x, other.center_y),
                other.radians,
            )
            for other, distance in others
        ]

        return np.array(weights).flatten()

    def calculate_cohesion(self, closest_boids):
        average_x = sum(boid.center_x for boid in closest_boids) / 3
        average_y = sum(boid.center_y for boid in closest_boids) / 3

        return np.sqrt((average_x - self.center_x) ** 2 + (average_y - self.center_y) ** 2)

    def calculate_separation(self, closest_boids):
        separation = 0
        for boid in closest_boids:
            if arcade.get_distance_between_sprites(self, boid) < 10:
                separation += 10
        return separation

    def calculate_alignment(self, closest_boids):
        self_angle = self.angle
        while self_angle < 0:
            self_angle += 360

        average_angle = sum(boid.angle for boid in closest_boids) / 3
        while average_angle < 0:
            average_angle += 360

        return abs(self_angle - average_angle) ** 2

    def get_distance_to_wall(self):
        return min(
            SCREEN_WIDTH - self.center_x,
            self.center_x,
            SCREEN_HEIGHT - self.center_y,
            self.center_y,
        )

    def update_fitness(self, closest_boids):
        self.fitnesses["movement_variation"] += abs(self.change_angle - self.last_change_angle)
        self.fitnesses["wall_avoidance"] += self.get_distance_to_wall()
        self.fitnesses["cohesion"] += self.calculate_cohesion(closest_boids)
        self.fitnesses["separation"] += self.calculate_separation(closest_boids)
        self.fitnesses["alignment"] += self.calculate_alignment(closest_boids)

    def get_neighbours(self):
        neighbours = []
        for other in self.others:
            if arcade.get_distance_between_sprites(self, other) < 150 and other is not self:
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
            return -1, -1

        avg_x, avg_y, close_neighbours = 0, 0, 0
        for neighbour in neighbours:
            if arcade.get_distance_between_sprites(self, neighbour) < 50:
                avg_x += neighbour.center_x
                avg_y += neighbour.center_y
                close_neighbours += 1

        if not close_neighbours:
            return -1, -1

        return avg_x / close_neighbours, avg_y / close_neighbours

    def calculate_alignment_weights(self, neighbours):
        if not neighbours:
            return self.angle

        avg_angle = 0
        for neighbour in neighbours:
            avg_angle += neighbour.angle / len(neighbours)

        return avg_angle

    def update(self):
        closest_boids = self.get_closest_n(n=3, sprites=self.others)
        closest_obstacle = self.get_closest_n(n=1, sprites=self.obstacles)

        closest_boids_weights = self.get_distance_angle_to_and_direction_of_others(closest_boids)
        network_input = torch.tensor(
            [
                [
                    self.center_x,
                    self.center_y,
                    self.radians,
                    *closest_boids_weights,
                    closest_obstacle[0][1],
                    arcade.get_angle_radians(
                        self.center_x, self.center_y, closest_obstacle[0][0].center_x, closest_obstacle[0][0].center_y
                    ),
                ]
            ],
            dtype=torch.float,
            requires_grad=False,
        )

        self.change_angle = self.brain.forward(network_input).item() * 5
        self.angle += self.change_angle

        self.ticks_alive += 1

        self.set_position(
            center_x=self.center_x + (-self.speed * np.sin(self.radians)),
            center_y=self.center_y + (self.speed * np.cos(self.radians)),
        )

        self.update_fitness([item[0] for item in closest_boids])

    def wrap_around(self):
        if self.left < 0:
            self.left += SCREEN_WIDTH
        elif self.right > SCREEN_WIDTH - 1:
            self.right -= SCREEN_WIDTH

        if self.bottom < 0:
            self.bottom += SCREEN_HEIGHT
        elif self.top > SCREEN_HEIGHT - 1:
            self.top -= SCREEN_HEIGHT
