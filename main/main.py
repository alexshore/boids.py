import json
from pathlib import Path
import random
import time
import arcade
import numpy as np
import torch
from AIBoid import Boid
from constants import FINAL_TICK_END, FINAL_TICK_INCREMENT, NUMBER_OF_BOIDS, SCREEN_HEIGHT, SCREEN_WIDTH
from deap import base, creator, tools
from obstacle import HorizontalObstacle, VerticalObstacle


"""

figure out fitness:

- time alive
- distance from obstacles
- variation in movement
- boid traits (cohesion, separation, alignment)

use multi-objective fitness, figure out selection algorithms

what is a pareto front?? do i need it??


"""


class Simulation(arcade.Window):
    def __init__(self, toolbox: base.Toolbox, logbook: tools.Logbook, statistics: tools.Statistics):
        super().__init__(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, update_rate=1 / 60)  # type: ignore
        arcade.set_background_color(arcade.color.SKY_BLUE)

        self.toolbox = toolbox
        self.logbook = logbook
        self.statistics = statistics
        self.population = self.toolbox.population(n=50)
        self.do_evolution = True

        self.all_boids = arcade.SpriteList()
        self.alive_boids = arcade.SpriteList()

        self.obstacles = arcade.SpriteList()
        self.obstacles.extend(
            [
                HorizontalObstacle(
                    start_x=SCREEN_WIDTH // 6,
                    end_x=5 * SCREEN_WIDTH // 6,
                    y=SCREEN_HEIGHT // 6,
                ),
                HorizontalObstacle(
                    start_x=SCREEN_WIDTH // 6,
                    end_x=5 * SCREEN_WIDTH // 6,
                    y=3 * SCREEN_HEIGHT // 6,
                ),
                HorizontalObstacle(
                    start_x=SCREEN_WIDTH // 6,
                    end_x=5 * SCREEN_WIDTH // 6,
                    y=5 * SCREEN_HEIGHT // 6,
                ),
                # VerticalObstacle(
                #     x=SCREEN_WIDTH // 6,
                #     start_y=SCREEN_HEIGHT // 6,
                #     end_y=5 * SCREEN_HEIGHT // 6,
                # ),
                # VerticalObstacle(
                #     x=3 * SCREEN_WIDTH // 6,
                #     start_y=SCREEN_HEIGHT // 6,
                #     end_y=5 * SCREEN_HEIGHT // 6,
                # ),
                # VerticalObstacle(
                #     x=5 * SCREEN_WIDTH // 6,
                #     start_y=SCREEN_HEIGHT // 6,
                #     end_y=5 * SCREEN_HEIGHT // 6,
                # ),
            ]
        )

        self.generate_boids_from_population()

        self.tick = 0
        self.final_tick = FINAL_TICK_INCREMENT
        self.generation = 0

        self.evolving = arcade.Text(
            text=f"evolution: {self.do_evolution}",
            start_x=SCREEN_WIDTH // 2,
            start_y=SCREEN_HEIGHT,
            font_size=30,
            anchor_x="center",
            anchor_y="top",
        )
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
            text=f"{len(self.alive_boids)}",
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

    def on_draw(self):
        self.clear()
        self.alive_boids.draw()
        # self.obstacles.draw()
        self.evolving.draw()
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
            self.clear_saved_weights()
        if symbol == arcade.key.SPACE:
            self.do_evolution = not self.do_evolution

    def generate_boids_from_population(self):
        self.all_boids.clear()
        self.all_boids.extend([Boid(weights=individual) for individual in self.population])
        self.alive_boids.clear()
        self.alive_boids.extend([boid for boid in self.all_boids])

        for boid in self.all_boids:  # give a reference list to each of the boids
            boid.others = self.alive_boids
            boid.obstacles = self.obstacles

    def reset_obstacles(self):
        self.obstacles.clear()
        self.obstacles.extend(
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
                # VerticalObstacle(
                #     x=SCREEN_WIDTH // 6,
                #     start_y=SCREEN_HEIGHT // 6,
                #     end_y=5 * SCREEN_HEIGHT // 6,
                # ),
                # VerticalObstacle(
                #     x=3 * SCREEN_WIDTH // 6,
                #     start_y=SCREEN_HEIGHT // 6,
                #     end_y=5 * SCREEN_HEIGHT // 6,
                # ),
                # VerticalObstacle(
                #     x=5 * SCREEN_WIDTH // 6,
                #     start_y=SCREEN_HEIGHT // 6,
                #     end_y=5 * SCREEN_HEIGHT // 6,
                # ),
            ]
        )

    def new_generation(self, increase_ticks):
        self.save_weights()

        for individual, boid in zip(self.population, self.all_boids):
            individual.fitness.values = self.toolbox.evaluate(boid)

        self.logbook.record(gen=self.generation, **self.statistics.compile(self.population))

        offspring = self.toolbox.select(self.population, len(self.population))
        offspring = list(map(self.toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.02:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for child in offspring:
            self.toolbox.mutate(child)
            del child.fitness.values

        self.population[:] = offspring
        self.generate_boids_from_population()

        self.generation += 1
        self.tick = 0

        if increase_ticks and self.final_tick < FINAL_TICK_END:
            self.final_tick += FINAL_TICK_INCREMENT

    def on_update(self, delta_time):
        if self.do_evolution and (increase_ticks := (self.tick >= self.final_tick) or len(self.alive_boids) <= 10):
            self.new_generation(increase_ticks)
            self.reset_obstacles()
            return

        current_alive_boids = len(self.alive_boids)

        if current_alive_boids <= 4:
            arcade.exit()

        for boid in self.alive_boids:
            for collision in self.get_boid_collisions(boid):
                self.alive_boids.remove(collision)

            # if boid in self.alive_boids and arcade.check_for_collision_with_list(boid, self.obstacles):
            #     self.alive_boids.remove(boid)

            if boid in self.alive_boids and (boid.left < 0 or boid.right > SCREEN_WIDTH or boid.bottom < 0 or boid.top > SCREEN_HEIGHT):
                self.alive_boids.remove(boid)

        self.alive_boids.update()
        self.obstacles.update()

        self.tick += 1

        self.evolving.text = f"evolving: {self.do_evolution}"
        self.current_tick.text = f"tick: {self.tick}/{self.final_tick}"
        self.current_generation.text = f"gen: {self.generation}"
        self.current_boids.text = f"alive: {current_alive_boids}"
        self.current_fps.text = f"fps: {arcade.get_fps():.0f}"

    def get_boid_collisions(self, boid):
        if collisions := arcade.check_for_collision_with_list(boid, self.alive_boids):
            return [boid] + collisions
        return []

    def save_weights(self):
        with open(f"saved/weights_{time.time():.0f}.json", "w") as file:
            json.dump(self.population, file)

    def clear_saved_weights(self):
        weights_files = Path("saved").glob("*.json")
        for file in weights_files:
            file.unlink()


def evaluate(boid: Boid):
    return (
        boid.fitnesses["movement_variation"] / boid.ticks_alive,
        boid.fitnesses["wall_avoidance"] / boid.ticks_alive,
        boid.fitnesses["cohesion"] / boid.ticks_alive,
        boid.fitnesses["separation"] / boid.ticks_alive,
        boid.fitnesses["alignment"] / boid.ticks_alive,
    )


def main():
    creator.create("Fitness", base.Fitness, weights=(1.0, 1.0, -1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register("attr_item", np.random.normal, 0, 0.5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, 535)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.5, indpb=0.005)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate)

    statistics = tools.Statistics(key=lambda ind: ind.fitness.values)
    statistics.register("avg", np.mean)
    statistics.register("std", np.std)
    statistics.register("min", np.min)
    statistics.register("max", np.max)

    logbook = tools.Logbook()

    simulation = Simulation(toolbox=toolbox, statistics=statistics, logbook=logbook)
    # simulation = Simulation(toolbox=toolbox)
    arcade.enable_timings()
    arcade.run()


if __name__ == "__main__":
    main()
