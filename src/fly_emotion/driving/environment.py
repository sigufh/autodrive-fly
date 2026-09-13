from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius: float


class DrivingEnvironment:
    """Small deterministic 2-D road with a camera-like ray-cast stimulus."""

    road_half_width = 6.0
    road_length = 120.0
    vehicle_radius = 0.55
    max_sensor_distance = 24.0
    dt = 0.12

    def __init__(self, *, width: int = 48, height: int = 24):
        self.image_width, self.image_height = width, height
        self.reset(0)

    def reset(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.x, self.y, self.heading, self.speed = 0.0, 2.0, 0.0, 0.0
        self.steps, self.total_reward, self.done = 0, 0.0, False
        self.trajectory = [(self.x, self.y)]
        ys = np.arange(15.0, 112.0, 11.0) + rng.uniform(-2.0, 2.0, 9)
        xs = rng.uniform(-4.35, 4.35, len(ys))
        # Alternate broad placement to prevent one constant steering bias from succeeding.
        xs[1::2] = np.abs(xs[1::2])
        xs[::2] = -np.abs(xs[::2])
        self.obstacles = [
            Obstacle(float(x), float(y), float(rng.uniform(0.75, 1.25)))
            for x, y in zip(xs, ys, strict=True)
        ]
        self.last_rays = np.full(
            self.image_width, self.max_sensor_distance, dtype=np.float32
        )
        return self.observe()

    def _ray_distance(self, angle: float) -> tuple[float, int]:
        dx, dy = np.sin(angle), np.cos(angle)
        distances: list[tuple[float, int]] = []
        if abs(dx) > 1e-8:
            for wall in (-self.road_half_width, self.road_half_width):
                t = (wall - self.x) / dx
                if t > 0:
                    distances.append((t, 1))
        if dy > 1e-8:
            distances.append(((self.road_length - self.y) / dy, 1))
        for obstacle in self.obstacles:
            ox, oy = self.x - obstacle.x, self.y - obstacle.y
            b = ox * dx + oy * dy
            c = ox * ox + oy * oy - (obstacle.radius + self.vehicle_radius * 0.15) ** 2
            discriminant = b * b - c
            if discriminant >= 0:
                t = -b - np.sqrt(discriminant)
                if t > 0:
                    distances.append((float(t), 2))
        if not distances:
            return self.max_sensor_distance, 0
        distance, kind = min(distances, key=lambda pair: pair[0])
        return min(float(distance), self.max_sensor_distance), kind

    def observe(self) -> np.ndarray:
        image = np.full((self.image_height, self.image_width), 0.035, dtype=np.float32)
        angles = self.heading + np.linspace(-1.25, 1.25, self.image_width)
        rays = [self._ray_distance(float(angle)) for angle in angles]
        self.last_rays = np.asarray([ray[0] for ray in rays], dtype=np.float32)
        horizon = self.image_height // 3
        image[horizon:, :] = 0.12
        for column, (distance, kind) in enumerate(rays):
            if kind == 0 or distance >= self.max_sensor_distance:
                continue
            height = max(
                1,
                int(
                    (1.0 - distance / self.max_sensor_distance)
                    * (self.image_height - horizon)
                ),
            )
            value = 0.98 if kind == 2 else 0.48
            image[-height:, column] = value
        # A dim road centre marker gives the network optic-flow and heading cues.
        centre = self.image_width // 2 + int(np.clip(-self.x * 1.7, -12, 12))
        image[horizon:, max(0, centre - 1) : min(self.image_width, centre + 1)] = 0.3
        return image

    def step(self, steering: float, throttle: float) -> tuple[np.ndarray, float, bool]:
        if self.done:
            return self.observe(), 0.0, True
        steering = float(np.clip(steering, -1.0, 1.0))
        throttle = float(np.clip(throttle, 0.0, 1.0))
        previous_y = self.y
        target_speed = 1.0 + 5.0 * throttle
        self.speed += (target_speed - self.speed) * 0.24
        self.heading = float(np.clip(self.heading + steering * 0.10, -1.15, 1.15))
        self.x += float(np.sin(self.heading) * self.speed * self.dt)
        self.y += float(np.cos(self.heading) * self.speed * self.dt)
        self.steps += 1
        self.trajectory.append((self.x, self.y))
        collision = abs(self.x) + self.vehicle_radius >= self.road_half_width
        collision = collision or any(
            np.hypot(self.x - obstacle.x, self.y - obstacle.y)
            <= self.vehicle_radius + obstacle.radius
            for obstacle in self.obstacles
        )
        success = self.y >= self.road_length
        timeout = self.steps >= 500
        reward = (self.y - previous_y) / 4.0 - 0.006
        if collision:
            reward -= 1.5
        if success:
            reward += 3.0
        self.done = bool(collision or success or timeout)
        self.total_reward += reward
        return self.observe(), float(reward), self.done

    def snapshot(self) -> dict:
        projected = []
        x, y, heading = self.x, self.y, self.heading
        for _ in range(12):
            x += float(np.sin(heading) * self.speed * self.dt)
            y += float(np.cos(heading) * self.speed * self.dt)
            projected.append([x, y])
        return {
            "road_half_width": self.road_half_width,
            "road_length": self.road_length,
            "vehicle": {
                "x": self.x, "y": self.y, "heading": self.heading, "speed": self.speed,
            },
            "obstacles": [obstacle.__dict__ for obstacle in self.obstacles],
            "trajectory": [list(point) for point in self.trajectory[-250:]],
            "projected_trajectory": projected,
            "sensor_rays": self.last_rays.tolist(),
            "step": self.steps, "done": self.done,
            "success": self.y >= self.road_length, "total_reward": self.total_reward,
        }
