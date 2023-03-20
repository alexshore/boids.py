import os
import shutil
from pathlib import Path
import arcade
import numpy as np
import torch
from AIBoid import Boid
from constants import MAX_TICK_END, MAX_TICK_INCREMENT, NUMBER_OF_BOIDS, SCREEN_HEIGHT, SCREEN_WIDTH
from deap import base, creator, tools
from obstacle import HorizontalObstacle


class Simulation(arcade.Window):
    def __init__(self, toolbox):
        super().__init__(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, update_rate=1 / 60)  # type: ignore
        arcade.set_background_color(arcade.color.SKY_BLUE)

        self.toolbox = toolbox

        self.boid_list = arcade.SpriteList()
        self.boid_list.extend([Boid(net_id=1, weights=np.random.normal(0, 0.5, 195)) for _ in range(NUMBER_OF_BOIDS)])

        self.obstacle_list = arcade.SpriteList()
        self.obstacle_list.extend(
            [
                HorizontalObstacle(
                    start_x=SCREEN_WIDTH // 6,
                    end_x=5 * SCREEN_WIDTH // 6,
                    y=SCREEN_HEIGHT // 6,
                ),
                HorizontalObstacle(
                    start_x=SCREEN_WIDTH // 6,
                    end_x=5 * SCREEN_WIDTH // 6,
                    y=SCREEN_HEIGHT // 2,
                ),
                HorizontalObstacle(
                    start_x=SCREEN_WIDTH // 6,
                    end_x=5 * SCREEN_WIDTH // 6,
                    y=5 * SCREEN_HEIGHT // 6,
                ),
            ]
        )

        for boid in self.boid_list:
            boid.boid_list = self.boid_list
            boid.obstacles = self.obstacle_list

        self.tick = 0
        self.final_tick = MAX_TICK_INCREMENT
        self.generation = 0

        self.current_tick = arcade.Text(
            text=f"{self.tick}",
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

    def spawn_new_generation(self):
        self.boid_list.clear()
        self.boid_list.extend([Boid(net_id=1, weights=np.random.normal(0, 0.5, 195)) for _ in range(NUMBER_OF_BOIDS)])

        self.obstacle_list.clear()
        self.obstacle_list.extend(
            [
                HorizontalObstacle(
                    start_x=SCREEN_WIDTH // 8,
                    end_x=7 * SCREEN_WIDTH // 8,
                    y=SCREEN_HEIGHT // 8,
                ),
                HorizontalObstacle(
                    start_x=SCREEN_WIDTH // 8,
                    end_x=7 * SCREEN_WIDTH // 8,
                    y=SCREEN_HEIGHT // 2,
                ),
                HorizontalObstacle(
                    start_x=SCREEN_WIDTH // 8,
                    end_x=7 * SCREEN_WIDTH // 8,
                    y=7 * SCREEN_HEIGHT // 8,
                ),
            ]
        )

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

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.P:
            self.set_update_rate(1 / 240)
        if symbol == arcade.key.O:
            self.set_update_rate(1 / 120)
        if symbol == arcade.key.I:
            self.set_update_rate(1 / 60)
        if symbol == arcade.key.Q:
            arcade.exit()
        if symbol == arcade.key.S:
            self.save_weights()
        if symbol == arcade.key.D:
            self.delete_saved_weights()

    def on_update(self, delta_time):
        if inc_final_tick := (self.tick >= self.final_tick) or len(self.boid_list) < 6:
            self.generation += 1
            self.tick = 0

            if inc_final_tick and self.final_tick < MAX_TICK_END:
                self.final_tick += MAX_TICK_INCREMENT

            self.spawn_new_generation()

            return

        alive_boids = len(self.boid_list)

        if alive_boids <= 4:
            arcade.exit()

        for boid in self.boid_list:
            for collision in self.get_boid_collisions(boid):
                self.boid_list.remove(collision)

            if arcade.check_for_collision_with_list(boid, self.obstacle_list) and boid in self.boid_list:
                self.boid_list.remove(boid)

        self.boid_list.update()
        self.obstacle_list.update()

        self.tick += 1

        self.current_tick.text = f"{self.tick}/{self.final_tick}"
        self.current_generation.text = f"{self.generation}"
        self.current_boids.text = f"{alive_boids}"
        self.current_fps.text = f"{arcade.get_fps():.0f}"

    def get_boid_collisions(self, boid):
        if collisions := arcade.check_for_collision_with_list(boid, self.boid_list):
            return [boid] + collisions
        return []

    def save_weights(self):
        self.delete_saved_weights()

        for i, boid in enumerate(self.boid_list):
            torch.save(boid.brain.state_dict(), f"saved/boid{i}_weights.pt")

    def delete_saved_weights(self):
        weights_folder = Path("saved")
        weights_files = weights_folder.glob("*.pt")
        for weights_file in weights_files:
            weights_file.unlink()


def evaluate(individual):
    ...


def main():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_item", np.random.normal, 0, 0.5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, 195)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.5, indp=0.01)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("evaluate", evaluate)
    toolbox.register("select", tools.selTournament)

    simulation = Simulation(toolbox=toolbox)
    arcade.enable_timings()
    arcade.run()


if __name__ == "__main__":
    main()
