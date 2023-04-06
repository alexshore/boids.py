import json
from pathlib import Path
import random
import time
import arcade
import numpy as np
from ai_boid import Boid
from constants import FINAL_TICK_END, FINAL_TICK_INCREMENT, NUMBER_OF_BOIDS, SCREEN_HEIGHT, SCREEN_WIDTH, NUMBER_OF_INDIVIDUALS
from deap import base, creator, tools
from obstacle import HorizontalObstacle, VerticalObstacle


class Simulation(arcade.Window):
    def __init__(self):
        super().__init__(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, update_rate=1 / 240)  # type: ignore
        arcade.set_background_color(arcade.color.SKY_BLUE)

        with open("saved/weights.json", "r") as file:
            self.population = json.load(file)

        self.population_index = 0
        self.individual = self.population[self.population_index]

        self.all_boids = arcade.SpriteList()
        self.alive_boids = arcade.SpriteList()

        self.generate_boids_from_weights()

        self.max_tps = 240
        self.tick = 0

        self.current_species = arcade.Text(
            text=f"species: {self.population_index + 1}/{len(self.population)}",
            start_x=0,
            start_y=SCREEN_HEIGHT - 80,
            font_size=25,
            anchor_x="left",
            anchor_y="top",
        )
        self.current_generation = arcade.Text(
            text=f"generation: {self.individual['generation']}",
            start_x=0,
            start_y=SCREEN_HEIGHT,
            font_size=25,
            anchor_x="left",
            anchor_y="top",
        )
        self.current_individual = arcade.Text(
            text=f"individual: {self.individual['individual']}",
            start_x=0,
            start_y=SCREEN_HEIGHT - 30,
            font_size=25,
            anchor_x="left",
            anchor_y="top",
        )

        self.current_fps = arcade.Text(
            text=f"fps: {arcade.get_fps()}",
            start_x=SCREEN_WIDTH,
            start_y=SCREEN_HEIGHT,
            font_size=25,
            anchor_x="right",
            anchor_y="top",
        )
        self.current_max_tps = arcade.Text(
            text=f"max tps: {self.max_tps}",
            start_x=SCREEN_WIDTH,
            start_y=SCREEN_HEIGHT - 30,
            font_size=25,
            anchor_x="right",
            anchor_y="top",
        )

    def on_draw(self):
        self.clear()

        self.alive_boids.draw()

        self.current_species.draw()
        self.current_generation.draw()
        self.current_individual.draw()

        self.current_fps.draw()
        self.current_max_tps.draw()

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.P:
            self.set_update_rate(1 / 240)
            self.max_tps = 240
        if symbol == arcade.key.O:
            self.set_update_rate(1 / 120)
            self.max_tps = 120
        if symbol == arcade.key.I:
            self.set_update_rate(1 / 60)
            self.max_tps = 60
        if symbol == arcade.key.Q:
            arcade.exit()
        if symbol == arcade.key.S:
            self.save_weights()
        if symbol == arcade.key.D:
            self.clear_saved_weights()

        if symbol == arcade.key.LEFT:
            self.population_index -= 1
            if self.population_index < 0:
                self.population_index = len(self.population)
            self.individual = self.population[self.population_index]
            self.generate_boids_from_weights()

        if symbol == arcade.key.RIGHT:
            self.population_index += 1
            if self.population_index >= len(self.population):
                self.population_index = 0
            self.individual = self.population[self.population_index]
            self.generate_boids_from_weights()

    def generate_boids_from_weights(self):
        self.all_boids.clear()
        self.all_boids.extend([Boid(weights=self.individual["weights"]) for _ in range(NUMBER_OF_BOIDS)])
        self.alive_boids.clear()
        self.alive_boids.extend([boid for boid in self.all_boids])

        for boid in self.all_boids:
            boid.others = self.alive_boids

    def on_update(self, delta_time):
        current_alive_boids = len(self.alive_boids)

        if current_alive_boids <= 4:
            arcade.exit()

        self.alive_boids.update()

        self.tick += 1

        self.current_species.text = f"species: {self.population_index + 1}/{len(self.population)}"
        self.current_generation.text = f"generation: {self.individual['generation']}"
        self.current_individual.text = f"individual: {self.individual['individual']}"

        self.current_fps.text = f"fps: {arcade.get_fps():.0f}"
        self.current_max_tps.text = f"max tps: {self.max_tps}"

    def get_boid_collisions(self, boid):
        if collisions := arcade.check_for_collision_with_list(boid, self.alive_boids):
            return [boid] + collisions
        return []


def evaluate(boids):
    cohesion, separation, alignment = 0, 0, 0

    for boid in boids:
        cohesion += boid.fitnesses["cohesion"] / boid.ticks_alive
        separation += boid.fitnesses["separation"] / boid.ticks_alive
        alignment += boid.fitnesses["alignment"] / boid.ticks_alive

    return (cohesion, separation, alignment)


def main():
    simulation = Simulation()
    arcade.enable_timings()
    arcade.run()


if __name__ == "__main__":
    main()
