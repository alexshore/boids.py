import random
from pathlib import Path

import arcade
from boid import Boid

from constants import NUMBER_OF_BOIDS, SCREEN_WIDTH, SCREEN_HEIGHT, BOID_ROTATION_SPEED


class Simulation(arcade.Window):
    def __init__(self):
        super().__init__(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, update_rate=1 / 60)  # type: ignore

        self.boid_list = arcade.SpriteList()
        self.boid_list.extend([Boid() for _ in range(NUMBER_OF_BOIDS)])

        arcade.set_background_color(arcade.color.SKY_BLUE)

    def on_draw(self):
        self.clear()
        self.boid_list.draw()

    def on_update(self, delta_time):
        # for boid in self.boid_list:
        #     for collision in self.get_collisions(boid):
        #         self.boid_list.remove(collision)
        self.boid_list.update()

    def get_collisions(self, boid):
        if collisions := arcade.check_for_collision_with_list(boid, self.boid_list):
            return [boid] + collisions
        return []


def main():
    simulation = Simulation()
    arcade.run()


if __name__ == "__main__":
    main()
