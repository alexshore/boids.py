import random
from pathlib import Path

import arcade
import numpy as np
from AIBoid import Boid
from constants import BOID_ROTATION_SPEED, NUMBER_OF_BOIDS, SCREEN_HEIGHT, SCREEN_WIDTH, Direction
from deap import base, creator, tools
from obstacle import MovingObstacle

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

arcade.enable_timings()


class Simulation(arcade.Window):
    def __init__(self):
        super().__init__(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, update_rate=1 / 60)  # type: ignore

        self.boid_list = arcade.SpriteList()
        self.boid_list.extend([Boid(net_id=1, weights=np.random.normal(0, 0.5, 195)) for _ in range(NUMBER_OF_BOIDS)])

        self.obstacle_list = arcade.SpriteList()
        self.obstacle_list.append(
            MovingObstacle(
                start_position=(SCREEN_WIDTH // 8, SCREEN_HEIGHT // 8),
                end_position=(7 * SCREEN_WIDTH // 8, SCREEN_HEIGHT // 8),
                direction=Direction.HORIZONTAL,
            )
        )
        self.obstacle_list.append(
            MovingObstacle(
                start_position=(SCREEN_WIDTH // 8, 4 * SCREEN_HEIGHT // 8),
                end_position=(7 * SCREEN_WIDTH // 8, 4 * SCREEN_HEIGHT // 8),
                direction=Direction.HORIZONTAL,
            )
        )
        self.obstacle_list.append(
            MovingObstacle(
                start_position=(SCREEN_WIDTH // 8, 7 * SCREEN_HEIGHT // 8),
                end_position=(7 * SCREEN_WIDTH // 8, 7 * SCREEN_HEIGHT // 8),
                direction=Direction.HORIZONTAL,
            )
        )

        for boid in self.boid_list:
            boid.boid_list = self.boid_list
            boid.obstacles = self.obstacle_list

        self.ticks = 0
        self.generation = 0

        self.end_ticks = 100
        self.max_end_ticks = 1000

        arcade.set_background_color(arcade.color.SKY_BLUE)

        self.current_tick = arcade.Text(
            text=f"{self.ticks}",
            start_x=0,
            start_y=0,
            font_size=30,
            anchor_x="left",
        )
        self.current_generation = arcade.Text(
            text=f"{self.generation}",
            start_x=SCREEN_WIDTH,
            start_y=0,
            font_size=30,
            anchor_x="right",
        )
        self.current_boids = arcade.Text(
            text=f"{len(self.boid_list)}",
            start_x=SCREEN_WIDTH // 2,
            start_y=0,
            font_size=30,
            anchor_x="center",
        )
        self.current_fps = arcade.Text(
            text=f"{arcade.get_fps()}",
            start_x=0,
            start_y=SCREEN_HEIGHT,
            font_size=30,
            anchor_x="left",
            anchor_y="top",
        )

        print("hi")

    def spawn_new_generation(self):
        self.boid_list.clear()
        self.boid_list.extend([Boid(net_id=1, weights=np.random.normal(0, 0.5, 195)) for _ in range(NUMBER_OF_BOIDS)])
        for boid in self.boid_list:
            boid.boid_list = self.boid_list
            boid.obstacles = self.obstacle_list

    def on_draw(self):
        self.clear()
        self.boid_list.draw()
        self.obstacle_list.draw()
        self.current_tick.draw()
        self.current_generation.draw()
        self.current_boids.draw()
        self.current_fps.draw()

    def on_update(self, delta_time):
        if (not_inc_ticks := len(self.boid_list) < 6) or self.ticks >= self.end_ticks:
            print("new gen", self.generation)
            self.spawn_new_generation()
            self.generation += 1
            self.ticks = 0
            self.end_ticks += 100 if not not_inc_ticks and self.end_ticks <= self.max_end_ticks else 0

        if not (alive_boids := len(self.boid_list)):
            print("1exit")
            arcade.exit()

        if alive_boids <= 5:
            print("2exit")
            arcade.exit()

        self.ticks += 1
        print(self.ticks)

        for boid in self.boid_list:
            for collision in self.get_boid_collisions(boid):
                self.boid_list.remove(collision)

            if arcade.check_for_collision_with_list(boid, self.obstacle_list) and boid in self.boid_list:
                self.boid_list.remove(boid)

        self.boid_list.update()
        self.obstacle_list.update()

        self.current_tick.text = f"{self.ticks}/{self.end_ticks}"
        self.current_generation.text = f"{self.generation}"
        self.current_boids.text = f"{alive_boids}"

    def get_boid_collisions(self, boid):
        if collisions := arcade.check_for_collision_with_list(boid, self.boid_list):
            return [boid] + collisions
        return []


def main():
    simulation = Simulation()
    arcade.run()


if __name__ == "__main__":
    main()
