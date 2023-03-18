import arcade
import random
import math
from pathlib import Path

from constants import SCREEN_HEIGHT, SCREEN_WIDTH, BOID_MOVEMENT_SPEED, BOID_ROTATION_SPEED

from network import convert_weights_to_network


class Boid(arcade.Sprite):
    def __init__(self):
        super(Boid, self).__init__(
            filename=str(Path(__file__).parent / "boid.png"),
            scale=0.5,
            hit_box_algorithm="Detailed",
        )

        self.boid_list = arcade.SpriteList()

        self.set_position(
            center_x=random.randint(0, SCREEN_WIDTH),
            center_y=random.randint(0, SCREEN_HEIGHT),
        )
        self.speed = random.randint(BOID_MOVEMENT_SPEED - 3, BOID_MOVEMENT_SPEED + 3)

        self.rotation_speed = random.random() * (BOID_ROTATION_SPEED if random.randint(0, 1) else -BOID_ROTATION_SPEED)

    def update(self):
        if random.random() < 0.01:
            self.rotation_speed = -self.rotation_speed

        self.angle += self.rotation_speed

        self.set_position(
            center_x=self.center_x + (-self.speed * math.sin(self.radians)),
            center_y=self.center_y + (self.speed * math.cos(self.radians)),
        )

        if self.left < 0 or self.right > SCREEN_WIDTH or self.bottom < 0 or self.top > SCREEN_HEIGHT:
            self.kill()

    def wrap_around(self):
        if self.left < 0:
            self.left += SCREEN_WIDTH
        elif self.right > SCREEN_WIDTH - 1:
            self.right -= SCREEN_WIDTH

        if self.bottom < 0:
            self.bottom += SCREEN_HEIGHT
        elif self.top > SCREEN_HEIGHT - 1:
            self.top -= SCREEN_HEIGHT
