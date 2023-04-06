import arcade
from boid import Boid
from constants import NUMBER_OF_BOIDS, SCREEN_HEIGHT, SCREEN_WIDTH


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
    def __init__(self):
        super().__init__(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, update_rate=1 / 30)  # type: ignore
        arcade.set_background_color(arcade.color.SKY_BLUE)

        self.all_boids = arcade.SpriteList()
        self.all_boids.extend([Boid() for _ in range(NUMBER_OF_BOIDS)])

        self.alive_boids = arcade.SpriteList()
        self.alive_boids.extend([boid for boid in self.all_boids])

        for boid in self.alive_boids:
            boid.others = self.alive_boids

        self.cohesion = 0.1
        self.separation = 0.1
        self.alignment = 0.1

        self.speed = 10.0

        self.neighbourhood_size = 150
        self.close_neighbourhood_size = 25

        self.current_cohesion = arcade.Text(
            text=f"cohesion: {self.cohesion}",
            start_x=0,
            start_y=SCREEN_HEIGHT,
            font_size=25,
            anchor_x="left",
            anchor_y="top",
        )
        self.current_separation = arcade.Text(
            text=f"separation: {self.separation}",
            start_x=0,
            start_y=SCREEN_HEIGHT - 30,
            font_size=25,
            anchor_x="left",
            anchor_y="top",
        )
        self.current_alignment = arcade.Text(
            text=f"alignment: {self.alignment}",
            start_x=0,
            start_y=SCREEN_HEIGHT - 60,
            font_size=25,
            anchor_x="left",
            anchor_y="top",
        )
        self.current_speed = arcade.Text(
            text=f"speed: {self.speed}",
            start_x=0,
            start_y=SCREEN_HEIGHT - 100,
            font_size=25,
            anchor_x="left",
            anchor_y="top",
        )
        self.current_neighbourhood_size = arcade.Text(
            text=f"neighbourhood size: {self.neighbourhood_size}",
            start_x=0,
            start_y=SCREEN_HEIGHT - 140,
            font_size=25,
            anchor_x="left",
            anchor_y="top",
        )
        self.current_close_neighbourhood_size = arcade.Text(
            text=f"close neighbourhood size: {self.close_neighbourhood_size}",
            start_x=0,
            start_y=SCREEN_HEIGHT - 170,
            font_size=25,
            anchor_x="left",
            anchor_y="top",
        )

    def on_draw(self):
        self.clear()
        self.alive_boids.draw()
        self.current_cohesion.draw()
        self.current_separation.draw()
        self.current_alignment.draw()
        self.current_speed.draw()
        self.current_neighbourhood_size.draw()
        self.current_close_neighbourhood_size.draw()

    def on_key_press(self, symbol: int, modifiers: int):
        if symbol == arcade.key.Q:
            arcade.exit()

        if symbol == arcade.key.O:
            self.cohesion += 0.1 if modifiers == arcade.key.MOD_SHIFT else 0.01
            self.cohesion = min(self.cohesion, 1)
            for boid in self.all_boids:
                boid.cohesion = self.cohesion
        if symbol == arcade.key.L:
            self.cohesion -= 0.1 if modifiers == arcade.key.MOD_SHIFT else 0.01
            self.cohesion = max(self.cohesion, 0)
            for boid in self.all_boids:
                boid.cohesion = self.cohesion
        if symbol == arcade.key.I:
            self.separation += 0.1 if modifiers == arcade.key.MOD_SHIFT else 0.01
            self.separation = min(self.separation, 1)
            for boid in self.all_boids:
                boid.separation = self.separation
        if symbol == arcade.key.K:
            self.separation -= 0.1 if modifiers == arcade.key.MOD_SHIFT else 0.01
            self.separation = max(self.separation, 0)
            for boid in self.all_boids:
                boid.separation = self.separation
        if symbol == arcade.key.U:
            self.alignment += 0.1 if modifiers == arcade.key.MOD_SHIFT else 0.01
            self.alignment = min(self.alignment, 1)
            for boid in self.all_boids:
                boid.alignment = self.alignment
        if symbol == arcade.key.J:
            self.alignment -= 0.1 if modifiers == arcade.key.MOD_SHIFT else 0.01
            self.alignment = max(self.alignment, 0)
            for boid in self.all_boids:
                boid.alignment = self.alignment

        if symbol == arcade.key.Y:
            self.speed += 1.0
            for boid in self.all_boids:
                boid.speed = self.speed
        if symbol == arcade.key.H:
            self.speed -= 1.0
            for boid in self.all_boids:
                boid.speed = self.speed

        if symbol == arcade.key.R:
            self.neighbourhood_size += 10
            for boid in self.all_boids:
                boid.neighbourhood_size = self.neighbourhood_size
        if symbol == arcade.key.F:
            self.neighbourhood_size -= 10
            for boid in self.all_boids:
                boid.neighbourhood_size = self.neighbourhood_size
        if symbol == arcade.key.T:
            self.close_neighbourhood_size += 10
            for boid in self.all_boids:
                boid.close_neighbourhood_size = self.close_neighbourhood_size
        if symbol == arcade.key.G:
            self.close_neighbourhood_size -= 10
            for boid in self.all_boids:
                boid.close_neighbourhood_size = self.close_neighbourhood_size

    def on_update(self, delta_time):
        self.alive_boids.update()

        self.current_cohesion.text = f"cohesion: {self.cohesion:.2f}"
        self.current_separation.text = f"separation: {self.separation:.2f}"
        self.current_alignment.text = f"alignment: {self.alignment:.2f}"
        self.current_speed.text = f"speed: {self.speed}"
        self.current_neighbourhood_size.text = f"neighbourhood size: {self.neighbourhood_size}"
        self.current_close_neighbourhood_size.text = f"close neighbourhood size: {self.close_neighbourhood_size}"

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
