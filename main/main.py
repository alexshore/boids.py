import random
from pathlib import Path

import arcade
from boid import Boid

from constants import NUMBER_OF_BOIDS, SCREEN_WIDTH, SCREEN_HEIGHT, BOID_ROTATION_SPEED


class GA:
    def __init__(self, size, mut_prob, cx_prob):
        self.population = [[random.gauss(0.0, 0.5) for _ in range(100)] for _ in range(size)]


class Simulation(arcade.Window):
    def __init__(self):
        super().__init__(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, update_rate=1 / 60)  # type: ignore

        self.GA = GA(NUMBER_OF_BOIDS, 0.01, 0.1)

        self.boid_list = arcade.SpriteList()
        self.boid_list.extend([Boid() for _ in range(NUMBER_OF_BOIDS)])

        self.ticks = 0
        self.generation = 0

        self.end_ticks = 100
        self.max_end_ticks = 1000

        arcade.set_background_color(arcade.color.SKY_BLUE)

        self.current_tick = arcade.Text(text=f"{self.ticks}", start_x=0, start_y=0, font_size=30)
        self.current_generation = arcade.Text(text=f"{self.generation}", start_x=SCREEN_WIDTH, start_y=0, font_size=30, anchor_x="right")
        self.current_boids = arcade.Text(text=f"{len(self.boid_list)}", start_x=SCREEN_WIDTH // 2, start_y=0, font_size=30, anchor_x="center")

    def spawn_new_generation(self):
        self.boid_list.clear()
        self.boid_list.extend([Boid() for _ in range(NUMBER_OF_BOIDS)])

    def on_draw(self):
        self.clear()
        self.boid_list.draw()
        self.current_tick.draw()
        self.current_generation.draw()
        self.current_boids.draw()

    def on_update(self, delta_time):
        self.ticks += 1

        for boid in self.boid_list:
            for collision in self.get_collisions(boid):
                self.boid_list.remove(collision)

        self.boid_list.update()

        self.current_tick.text = f"{self.ticks}/{self.end_ticks}"
        self.current_generation.text = f"{self.generation}"
        self.current_boids.text = f"{len(self.boid_list)}"

        if (not_inc_ticks := len(self.boid_list) < 10) or self.ticks >= self.end_ticks:
            self.spawn_new_generation()
            self.generation += 1
            self.ticks = 0
            self.end_ticks += 100 if not not_inc_ticks and self.end_ticks <= self.max_end_ticks else 0

    def get_collisions(self, boid):
        if collisions := arcade.check_for_collision_with_list(boid, self.boid_list):
            return [boid] + collisions
        return []


def main():
    simulation = Simulation()
    arcade.run()


if __name__ == "__main__":
    main()
