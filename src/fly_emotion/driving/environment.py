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
    wheelbase = 2.5
    max_steering_angle = 0.42
    steering_rate_limit = 0.12
    steering_deadband = 0.055

    def __init__(self, *, width: int = 48, height: int = 24):
        self.image_width, self.image_height = width, height
        self.reset(0)

    def reset(self, seed: int = 0) -> np.ndarray:
        # Adjacent seeds form an exact left/right mirror pair. This prevents the
        # first obstacle from teaching one globally preferred turn direction.
        pair_seed = seed // 2
        mirror = 1.0 if seed % 2 == 0 else -1.0
        rng = np.random.default_rng(pair_seed)
        self.seed = int(seed)
        self.pair_seed = int(pair_seed)
        self.mirror = int(mirror)
        self.road_length = 120.0
        self.x, self.y, self.heading, self.speed = 0.0, 2.0, 0.0, 0.0
        self.steering = 0.0
        self.previous_steering = 0.0
        self.steps, self.total_reward, self.done = 0, 0.0, False
        self.trajectory = [(self.x, self.y)]
        ys = np.arange(15.0, 112.0, 11.0) + rng.uniform(-2.0, 2.0, 9)
        magnitudes = rng.uniform(0.35, 4.35, len(ys))
        signs = np.where(np.arange(len(ys)) % 2 == 0, -1.0, 1.0)
        xs = magnitudes * signs * mirror
        self.obstacles = [
            Obstacle(float(x), float(y), float(rng.uniform(0.75, 1.25)))
            for x, y in zip(xs, ys, strict=True)
        ]
        self.last_rays = np.full(self.image_width, self.max_sensor_distance, dtype=np.float32)
        self.last_wall_rays = self.last_rays.copy()
        self.last_obstacle_rays = self.last_rays.copy()
        self.terminal_reason: str | None = None
        self.obstacles_passed = 0
        self.passed_obstacle_indices: set[int] = set()
        self.obstacle_pass_steps: dict[int, int] = {}
        self.pass_reward = 0.35
        self.collision_penalty = 1.5
        self.success_reward = 3.0
        self.progress_reward_scale = 0.25
        self.max_steps = 500
        self.curriculum_stage = "full"
        return self.observe()

    def configure_curriculum(self, stage: str) -> np.ndarray:
        """Configure a reward-only neural curriculum after deterministic reset."""
        rng = np.random.default_rng(70_000 + self.pair_seed)
        if stage == "single":
            side = -1.0 if self.mirror == 1 else 1.0
            obstacle_y = float(rng.uniform(16.0, 20.0))
            self.road_length = obstacle_y + 18.0
            self.obstacles = [
                Obstacle(
                    float(rng.uniform(0.9, 1.7) * side), obstacle_y, float(rng.uniform(1.0, 1.25))
                )
            ]
            self.pass_reward, self.collision_penalty, self.success_reward = 1.5, 8.0, 6.0
            self.progress_reward_scale = 0.02
            self.max_steps = 260
        elif stage == "triple":
            side = -1.0 if self.mirror == 1 else 1.0
            self.road_length = 68.0
            self.obstacles = [
                Obstacle(float(rng.uniform(1.1, 1.8) * side), 18.0, 1.1),
                Obstacle(float(rng.uniform(1.1, 1.8) * -side), 36.0, 1.05),
                Obstacle(float(rng.uniform(1.1, 1.8) * side), 54.0, 1.1),
            ]
            self.pass_reward, self.collision_penalty, self.success_reward = 1.25, 4.0, 5.0
            self.progress_reward_scale = 0.05
            self.max_steps = 420
        elif stage == "nine":
            # Keep the deterministic nine-obstacle geometry created by reset,
            # but use the stronger sparse curriculum rewards.
            self.pass_reward, self.collision_penalty, self.success_reward = 0.8, 8.0, 8.0
            self.progress_reward_scale = 0.03
            self.max_steps = 650
        elif stage != "full":
            raise ValueError(f"unknown neural curriculum stage: {stage}")
        self.curriculum_stage = stage
        self.passed_obstacle_indices.clear()
        self.obstacle_pass_steps.clear()
        self.obstacles_passed = 0
        return self.observe()

    def _ray_distances(self, angle: float) -> tuple[float, float]:
        dx, dy = np.sin(angle), np.cos(angle)
        wall_distances: list[float] = []
        if abs(dx) > 1e-8:
            for wall in (-self.road_half_width, self.road_half_width):
                t = (wall - self.x) / dx
                if t > 0:
                    wall_distances.append(float(t))
        if dy > 1e-8:
            wall_distances.append(float((self.road_length - self.y) / dy))
        obstacle_distances: list[float] = []
        for obstacle in self.obstacles:
            ox, oy = self.x - obstacle.x, self.y - obstacle.y
            b = ox * dx + oy * dy
            c = ox * ox + oy * oy - (obstacle.radius + self.vehicle_radius * 0.15) ** 2
            discriminant = b * b - c
            if discriminant >= 0:
                t = -b - np.sqrt(discriminant)
                if t > 0:
                    obstacle_distances.append(float(t))
        wall = min(wall_distances, default=self.max_sensor_distance)
        obstacle = min(obstacle_distances, default=self.max_sensor_distance)
        return min(wall, self.max_sensor_distance), min(obstacle, self.max_sensor_distance)

    def observe(self) -> np.ndarray:
        image = np.full((self.image_height, self.image_width), 0.035, dtype=np.float32)
        angles = self.heading + np.linspace(-1.25, 1.25, self.image_width)
        components = [self._ray_distances(float(angle)) for angle in angles]
        self.last_wall_rays = np.asarray([ray[0] for ray in components], dtype=np.float32)
        self.last_obstacle_rays = np.asarray([ray[1] for ray in components], dtype=np.float32)
        self.last_rays = np.minimum(self.last_wall_rays, self.last_obstacle_rays)
        horizon = self.image_height // 3
        image[horizon:, :] = 0.12
        for column, (wall, obstacle) in enumerate(components):
            distance = min(wall, obstacle)
            if distance >= self.max_sensor_distance:
                continue
            height = max(
                1,
                int((1.0 - distance / self.max_sensor_distance) * (self.image_height - horizon)),
            )
            value = 0.98 if obstacle <= wall else 0.48
            image[-height:, column] = value
        # A dim road centre marker gives the network optic-flow and heading cues.
        centre = (self.image_width - 1) / 2 + int(np.clip(-self.x * 1.7, -12, 12))
        image[horizon:, np.abs(np.arange(self.image_width) - centre) <= 0.5] = 0.3
        return image

    def step(
        self, steering: float, throttle: float, reverse: float = 0.0
    ) -> tuple[np.ndarray, float, bool]:
        if self.done:
            return self.observe(), 0.0, True
        steering_command = float(np.clip(steering, -1.0, 1.0))
        if abs(steering_command) < self.steering_deadband:
            steering_command = 0.0
        throttle = float(np.clip(throttle, 0.0, 1.0))
        reverse = float(np.clip(reverse, 0.0, 1.0))
        previous_y = self.y
        self.previous_steering = self.steering
        steering_delta = np.clip(
            steering_command - self.steering,
            -self.steering_rate_limit,
            self.steering_rate_limit,
        )
        self.steering = float(np.clip(self.steering + steering_delta, -1.0, 1.0))
        forward_target = 1.0 + 5.0 * throttle
        target_speed = (1.0 - reverse) * forward_target - 2.0 * reverse
        self.speed += (target_speed - self.speed) * 0.24
        yaw_rate = self.speed / self.wheelbase * np.tan(self.max_steering_angle * self.steering)
        self.heading = float(np.clip(self.heading + yaw_rate * self.dt, -1.15, 1.15))
        self.x += float(np.sin(self.heading) * self.speed * self.dt)
        self.y += float(np.cos(self.heading) * self.speed * self.dt)
        self.steps += 1
        self.trajectory.append((self.x, self.y))
        previous_passed = self.obstacles_passed
        previous_indices = self.passed_obstacle_indices.copy()
        collision = abs(self.x) + self.vehicle_radius >= self.road_half_width
        collision = collision or any(
            np.hypot(self.x - obstacle.x, self.y - obstacle.y)
            <= self.vehicle_radius + obstacle.radius
            for obstacle in self.obstacles
        )
        if not collision:
            self.passed_obstacle_indices.update(
                index
                for index, obstacle in enumerate(self.obstacles)
                if self.y - self.vehicle_radius > obstacle.y + obstacle.radius
            )
        self.obstacles_passed = len(self.passed_obstacle_indices)
        newly_passed = self.passed_obstacle_indices - previous_indices
        for index in newly_passed:
            self.obstacle_pass_steps[index] = self.steps
        success = self.y >= self.road_length and not collision
        timeout = self.steps >= self.max_steps
        self.terminal_reason = (
            "road_boundary"
            if abs(self.x) + self.vehicle_radius >= self.road_half_width
            else "obstacle"
            if collision
            else "success"
            if success
            else "timeout"
            if timeout
            else None
        )
        progress_reward = self.progress_reward_scale * (self.y - previous_y)
        pass_reward = self.pass_reward * (self.obstacles_passed - previous_passed)
        reward = progress_reward + pass_reward - 0.006
        if collision:
            reward -= self.collision_penalty
        if success:
            reward += self.success_reward
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
            "pair_seed": self.pair_seed,
            "mirror": self.mirror,
            "vehicle": {
                "x": self.x,
                "y": self.y,
                "heading": self.heading,
                "speed": self.speed,
                "steering": self.steering,
            },
            "obstacles": [obstacle.__dict__ for obstacle in self.obstacles],
            "trajectory": [list(point) for point in self.trajectory[-250:]],
            "projected_trajectory": projected,
            "sensor_rays": self.last_rays.tolist(),
            "obstacle_rays": self.last_obstacle_rays.tolist(),
            "wall_rays": self.last_wall_rays.tolist(),
            "step": self.steps,
            "done": self.done,
            "success": self.terminal_reason == "success",
            "total_reward": self.total_reward,
            "terminal_reason": self.terminal_reason,
            "obstacles_passed": self.obstacles_passed,
            "first_obstacle_passed": 0 in self.passed_obstacle_indices,
            "first_obstacle_side": ("left" if self.obstacles[0].x < 0 else "right")
            if self.obstacles
            else None,
            "curriculum_stage": self.curriculum_stage,
        }
