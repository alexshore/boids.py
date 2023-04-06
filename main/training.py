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
    def __init__(self, toolbox: base.Toolbox, logbook: tools.Logbook, statistics: tools.Statistics):
        super().__init__(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, update_rate=1 / 60)  # type: ignore
        arcade.set_background_color(arcade.color.SKY_BLUE)

        self.toolbox = toolbox
        self.logbook = logbook
        self.statistics = statistics
        self.population = self.toolbox.population(n=NUMBER_OF_INDIVIDUALS)
        self.individual = 0
        self.do_evolution = True

        self.all_boids = arcade.SpriteList()
        self.alive_boids = arcade.SpriteList()

        self.generate_boids_from_current_individual()

        self.max_tps = 60
        self.tick = 0
        self.final_tick = FINAL_TICK_INCREMENT
        self.generation = 0

        self.to_save = []

        self.evolving = arcade.Text(
            text=f"evolution: {self.do_evolution}",
            start_x=SCREEN_WIDTH // 2,
            start_y=0,
            font_size=25,
            anchor_x="center",
            anchor_y="bottom",
        )
        self.current_tick = arcade.Text(
            text=f"tick: {self.tick}/{self.final_tick}",
            start_x=0,
            start_y=SCREEN_HEIGHT,
            font_size=25,
            anchor_x="left",
            anchor_y="top",
        )
        self.current_individual = arcade.Text(
            text=f"individual: {self.individual + 1}/{NUMBER_OF_INDIVIDUALS}",
            start_x=0,
            start_y=SCREEN_HEIGHT - 30,
            font_size=25,
            anchor_x="left",
            anchor_y="top",
        )
        self.current_generation = arcade.Text(
            text=f"generation: {self.generation + 1}",
            start_x=0,
            start_y=SCREEN_HEIGHT - 60,
            font_size=25,
            anchor_x="left",
            anchor_y="top",
        )
        self.current_boids = arcade.Text(
            text=f"alive: {len(self.alive_boids)}/{NUMBER_OF_BOIDS}",
            start_x=0,
            start_y=SCREEN_HEIGHT - 120,
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
        self.evolving.draw()
        self.current_tick.draw()
        self.current_individual.draw()
        self.current_generation.draw()
        self.current_boids.draw()
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
        if symbol == arcade.key.SPACE:
            self.do_evolution = not self.do_evolution
        if symbol == arcade.key.RIGHT:
            self.final_tick += FINAL_TICK_INCREMENT
        if symbol == arcade.key.LEFT:
            self.final_tick -= FINAL_TICK_INCREMENT if self.final_tick > 100 else 0
        if symbol == arcade.key.UP:
            self.set_individual(self.individual + 1 if self.individual < 19 else 0)
        if symbol == arcade.key.DOWN:
            self.set_individual(self.individual - 1 if self.individual else 19)

    def set_individual(self, individual):
        self.population[self.individual].fitness.values = self.toolbox.evaluate(self.all_boids)

        self.tick = 0
        self.individual = individual
        self.generate_boids_from_current_individual()

    def generate_boids_from_current_individual(self):
        self.all_boids.clear()
        self.all_boids.extend([Boid(weights=self.population[self.individual]) for _ in range(NUMBER_OF_BOIDS)])
        self.alive_boids.clear()
        self.alive_boids.extend([boid for boid in self.all_boids])

        for boid in self.all_boids:  # give a reference list to each of the boids
            boid.others = self.alive_boids

    def new_species(self):
        self.population[self.individual].fitness.values = self.toolbox.evaluate(self.all_boids)

        self.tick = 0
        self.individual += 1
        self.generate_boids_from_current_individual()

    def new_generation(self, increase_ticks):
        self.population[self.individual].fitness.values = self.toolbox.evaluate(self.all_boids)

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

        self.individual = 0
        self.generation += 1
        self.tick = 0

        if increase_ticks and self.final_tick < FINAL_TICK_END:
            self.final_tick += FINAL_TICK_INCREMENT

        self.generate_boids_from_current_individual()

    def on_update(self, delta_time):
        if self.do_evolution:
            if self.individual < (NUMBER_OF_INDIVIDUALS - 1) and (self.tick >= self.final_tick or len(self.alive_boids) <= 5):
                self.new_species()
                return

            if increase_ticks := (self.tick >= self.final_tick) or len(self.alive_boids) <= 5:
                self.new_generation(increase_ticks)
                return

        current_alive_boids = len(self.alive_boids)

        if current_alive_boids <= 4:
            arcade.exit()

        # for boid in self.alive_boids:
        #     if boid.left < 0 or boid.right > SCREEN_WIDTH or boid.bottom < 0 or boid.top > SCREEN_HEIGHT:
        #         self.alive_boids.remove(boid)

        self.alive_boids.update()

        self.tick += 1

        self.evolving.text = f"evolving: {self.do_evolution}"
        self.current_tick.text = f"tick: {self.tick}/{self.final_tick}"
        self.current_individual.text = f"individual: {self.individual + 1}/{NUMBER_OF_INDIVIDUALS}"
        self.current_generation.text = f"generation: {self.generation + 1}"
        self.current_boids.text = f"alive: {current_alive_boids}/{NUMBER_OF_BOIDS}"
        self.current_fps.text = f"fps: {arcade.get_fps():.0f}"
        self.current_max_tps.text = f"max tps: {self.max_tps}"

    def get_boid_collisions(self, boid):
        if collisions := arcade.check_for_collision_with_list(boid, self.alive_boids):
            return [boid] + collisions
        return []

    def save_weights(self):
        self.to_save.append(
            {
                "generation": self.generation,
                "individual": self.individual,
                "weights": self.population[self.individual],
            }
        )

    def clear_saved_weights(self):
        weights_files = Path("saved").glob("*.json")
        for file in weights_files:
            file.unlink()


def evaluate(boids):
    cohesion, separation, alignment = 0, 0, 0

    for boid in boids:
        cohesion += boid.fitnesses["cohesion"] / boid.ticks_alive
        separation += boid.fitnesses["separation"] / boid.ticks_alive
        alignment += boid.fitnesses["alignment"] / boid.ticks_alive

    return (cohesion, separation, alignment)


def main():
    creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0, -1.0))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register("attr_item", np.random.normal, 0, 0.5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_item, 296)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.5, indpb=0.005)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mate", tools.cxUniform, indpb=0.01)
    # toolbox.register("select", tools.selTournament, tournsize=4)
    # toolbox.register("select", tools.selNSGA2)
    # toolbox.register("select", tools.selNSGA3)
    toolbox.register("select", tools.selSPEA2)
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

    if simulation.to_save:
        with open("saved/weights.json", "w") as file:
            json.dump(simulation.to_save, file)


if __name__ == "__main__":
    main()
