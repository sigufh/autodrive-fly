"""Deterministic urban-route environment for the next driving stage.

This is deliberately a compact road-graph simulator rather than a claim of a
real-city digital twin.  It provides the interfaces needed to move beyond the
single straight-road benchmark: a multi-road route, lane targets, signalized
intersections, stop lines, speed limits and rule-violation accounting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .environment import Obstacle


@dataclass(frozen=True)
class RoutePiece:
    name: str
    start: float
    end: float
    kind: str
    start_xy: tuple[float, float]
    heading: float
    radius: float = 0.0
    turn: str = "straight"


class CityDrivingEnvironment:
    """Frenet-frame city route with traffic rules and camera-like perception.

    ``x`` is the signed lane offset and ``y`` is route progress.  This preserves
    the local visual-driving interface while ``world_pose`` and ``city`` expose
    the larger road graph to clients.
    """

    scenario = "city_alpha"
    city_name = "Harbor Grid alpha"
    road_half_width = 4.8
    vehicle_radius = 0.55
    max_sensor_distance = 24.0
    dt = 0.12
    wheelbase = 2.5
    max_steering_angle = 0.42
    steering_rate_limit = 0.12
    steering_deadband = 0.055
    max_steps = 900
    lane_width = 3.2
    cruise_throttle = 0.62

    def __init__(self, *, width: int = 48, height: int = 24):
        self.image_width, self.image_height = width, height
        self.route = self._build_route()
        self.road_length = self.route[-1].end
        self.reset(0)

    @staticmethod
    def _build_route() -> list[RoutePiece]:
        first = 62.0
        arc = np.pi * 18.0 / 2
        second = first + arc
        east_end = second + 76.0
        north_start = east_end + arc
        return [
            RoutePiece("Harbor Avenue", 0.0, first, "straight", (0.0, 0.0), 0.0),
            RoutePiece("Harbor / Market", first, second, "arc", (0.0, 62.0), 0.0, 18.0, "right"),
            RoutePiece("Market Street", second, east_end, "straight", (18.0, 80.0), np.pi / 2),
            RoutePiece(
                "Market / Civic",
                east_end,
                north_start,
                "arc",
                (94.0, 80.0),
                np.pi / 2,
                18.0,
                "left",
            ),
            RoutePiece(
                "Civic Boulevard", north_start, north_start + 88.0, "straight", (112.0, 98.0), 0.0
            ),
        ]

    def _piece(self, progress: float) -> RoutePiece:
        return next((piece for piece in self.route if progress <= piece.end), self.route[-1])

    def route_pose(self, progress: float) -> tuple[float, float, float, float]:
        """Return world x/y, tangent heading and signed route curvature."""
        piece = self._piece(float(np.clip(progress, 0.0, self.road_length)))
        local = max(0.0, min(progress, piece.end) - piece.start)
        if piece.kind == "straight":
            x0, y0 = piece.start_xy
            return (
                x0 + np.sin(piece.heading) * local,
                y0 + np.cos(piece.heading) * local,
                piece.heading,
                0.0,
            )
        if piece.turn == "right":
            angle = np.pi - local / piece.radius
            center_x, center_y = 18.0, 62.0
            return (
                center_x + piece.radius * np.cos(angle),
                center_y + piece.radius * np.sin(angle),
                np.pi - angle,
                1.0 / piece.radius,
            )
        angle = -np.pi / 2 + local / piece.radius
        center_x, center_y = 94.0, 98.0
        return (
            center_x + piece.radius * np.cos(angle),
            center_y + piece.radius * np.sin(angle),
            -angle,
            -1.0 / piece.radius,
        )

    def world_pose(
        self, progress: float | None = None, lateral: float | None = None
    ) -> tuple[float, float, float]:
        progress = self.y if progress is None else progress
        lateral = self.x if lateral is None else lateral
        x, y, tangent, _ = self.route_pose(progress)
        return (
            x + np.cos(tangent) * lateral,
            y - np.sin(tangent) * lateral,
            tangent + self.heading,
        )

    def reset(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.pair_seed = int(seed)
        self.mirror = 1
        self.x, self.y, self.heading, self.speed = -self.lane_width / 2, 2.0, 0.0, 0.0
        self.steering = self.previous_steering = 0.0
        self.last_yaw_rate = 0.0
        self.steps, self.total_reward, self.done = 0, 0.0, False
        self.trajectory = [(self.x, self.y)]
        self.world_trajectory = [self.world_pose()[:2]]
        # Route-coordinate actors emulate parked/slow traffic in the active lane.
        self.obstacles = [
            Obstacle(-1.55 + float(rng.uniform(-0.15, 0.15)), 106.0, 0.95),
            Obstacle(-1.45 + float(rng.uniform(-0.15, 0.15)), 194.0, 0.9),
        ]
        self.passed_obstacle_indices: set[int] = set()
        self.obstacles_passed = 0
        self.terminal_reason: str | None = None
        self.violations: list[str] = []
        self.signal_stops = {"Harbor / Market": 50.0, "Market / Civic": 154.0}
        self.signal_crossed: set[str] = set()
        self.stopped_for_signal: set[str] = set()
        self.last_rays = np.full(self.image_width, self.max_sensor_distance, dtype=np.float32)
        self.last_wall_rays = self.last_rays.copy()
        self.last_obstacle_rays = self.last_rays.copy()
        return self.observe()

    def signal_state(self, junction: str) -> str:
        phase = self.steps % 120
        offset = 0 if junction == "Harbor / Market" else 38
        return "red" if (phase + offset) % 120 < 48 else "green"

    def traffic_guidance(self) -> dict:
        piece = self._piece(self.y)
        tangent = self.route_pose(self.y)[3]
        desired_steering = float(np.arctan(tangent * self.wheelbase) / self.max_steering_angle)
        signal_name = next(
            (name for name, stop in self.signal_stops.items() if stop >= self.y - 2.0), None
        )
        stop_line = self.signal_stops.get(signal_name) if signal_name else None
        distance = float(stop_line - self.y) if stop_line is not None else None
        signal = self.signal_state(signal_name) if signal_name else "green"
        speed_cap = self.cruise_throttle
        rule_status = "clear"
        target_lateral = -self.lane_width / 2
        lane_action = "keep_lane"
        # Route-level occupancy gives a legal lane target early enough for a
        # smooth change. Visual control still chooses local clearance.
        blocked = next(
            (
                obstacle
                for obstacle in self.obstacles
                if 2.0 < obstacle.y - self.y < 26.0
                and abs(obstacle.x + self.lane_width / 2) < self.lane_width * 0.55
            ),
            None,
        )
        if blocked is not None:
            target_lateral = self.lane_width / 2
            lane_action = "change_right_for_actor"
        elif any(0.0 < self.y - obstacle.y < 10.0 for obstacle in self.obstacles):
            lane_action = "return_route_lane"
        if distance is not None and signal == "red" and distance >= -0.6:
            # Start controlled braking 14 m before the stop line, while retaining
            # the normal visual controller for obstacles.
            speed_cap = float(self.cruise_throttle * np.clip((distance - 0.7) / 13.0, 0.0, 1.0))
            rule_status = "yield_red"
            if abs(distance) <= 1.2 and abs(self.speed) < 0.35:
                self.stopped_for_signal.add(signal_name)
        return {
            "road": piece.name,
            "next_maneuver": piece.turn
            if piece.kind == "arc"
            else next(
                (
                    candidate.turn
                    for candidate in self.route
                    if candidate.start > piece.start and candidate.kind == "arc"
                ),
                "arrive",
            ),
            "target_lateral_offset": target_lateral,
            "lane_action": lane_action,
            "route_curvature": tangent,
            "desired_steering": desired_steering,
            "speed_limit_mps": 8.0,
            "speed_cap": speed_cap,
            "traffic_light": signal,
            "stop_line_progress": stop_line,
            "distance_to_stop_line": distance,
            "rule_status": rule_status,
        }

    def _ray_distances(self, angle: float) -> tuple[float, float]:
        dx, dy = np.sin(angle), np.cos(angle)
        walls: list[float] = []
        if abs(dx) > 1e-8:
            for wall in (-self.road_half_width, self.road_half_width):
                distance = (wall - self.x) / dx
                if distance > 0:
                    walls.append(float(distance))
        obstacles: list[float] = []
        for obstacle in self.obstacles:
            ox, oy = self.x - obstacle.x, self.y - obstacle.y
            b = ox * dx + oy * dy
            c = ox * ox + oy * oy - (obstacle.radius + self.vehicle_radius * 0.15) ** 2
            discriminant = b * b - c
            if discriminant >= 0:
                distance = -b - np.sqrt(discriminant)
                if distance > 0:
                    obstacles.append(float(distance))
        return (
            min(min(walls, default=self.max_sensor_distance), self.max_sensor_distance),
            min(min(obstacles, default=self.max_sensor_distance), self.max_sensor_distance),
        )

    def observe(self) -> np.ndarray:
        image = np.full((self.image_height, self.image_width), 0.035, dtype=np.float32)
        angles = self.heading + np.linspace(-1.25, 1.25, self.image_width)
        components = [self._ray_distances(float(angle)) for angle in angles]
        self.last_wall_rays = np.asarray([wall for wall, _ in components], dtype=np.float32)
        self.last_obstacle_rays = np.asarray([actor for _, actor in components], dtype=np.float32)
        self.last_rays = np.minimum(self.last_wall_rays, self.last_obstacle_rays)
        horizon = self.image_height // 3
        image[horizon:, :] = 0.12
        for column, (wall, actor) in enumerate(components):
            distance = min(wall, actor)
            if distance >= self.max_sensor_distance:
                continue
            height = max(
                1, int((1.0 - distance / self.max_sensor_distance) * (self.image_height - horizon))
            )
            image[-height:, column] = 0.98 if actor <= wall else 0.48
        centre = (self.image_width - 1) / 2 + int(np.clip(-self.x * 1.7, -12, 12))
        image[horizon:, np.abs(np.arange(self.image_width) - centre) <= 0.5] = 0.3
        return image

    def step(
        self, steering: float, throttle: float, reverse: float = 0.0
    ) -> tuple[np.ndarray, float, bool]:
        if self.done:
            return self.observe(), 0.0, True
        command = float(np.clip(steering, -1.0, 1.0))
        if abs(command) < self.steering_deadband:
            command = 0.0
        throttle, reverse = float(np.clip(throttle, 0.0, 1.0)), float(np.clip(reverse, 0.0, 1.0))
        previous_progress = self.y
        self.previous_steering = self.steering
        self.steering = float(
            np.clip(
                self.steering
                + np.clip(
                    command - self.steering, -self.steering_rate_limit, self.steering_rate_limit
                ),
                -1.0,
                1.0,
            )
        )
        target_speed = (1.0 - reverse) * (6.0 * throttle) - 2.0 * reverse
        self.speed += (target_speed - self.speed) * 0.24
        curvature = self.route_pose(self.y)[3]
        yaw_rate = self.speed / self.wheelbase * np.tan(self.max_steering_angle * self.steering)
        self.last_yaw_rate = float(yaw_rate)
        progress_delta = float(np.cos(self.heading) * self.speed * self.dt)
        self.heading = float(
            np.clip(self.heading + yaw_rate * self.dt - curvature * progress_delta, -1.15, 1.15)
        )
        self.x += float(np.sin(self.heading) * self.speed * self.dt)
        self.y += progress_delta
        self.steps += 1
        self.trajectory.append((self.x, self.y))
        self.world_trajectory.append(self.world_pose()[:2])
        guidance = self.traffic_guidance()
        for junction, stop in self.signal_stops.items():
            if junction not in self.signal_crossed and previous_progress < stop <= self.y:
                self.signal_crossed.add(junction)
                if self.signal_state(junction) == "red" and junction not in self.stopped_for_signal:
                    self.violations.append(f"red_light:{junction}")
        previous_passed = self.obstacles_passed
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
        success = self.y >= self.road_length and not collision and not self.violations
        timeout = self.steps >= self.max_steps
        self.terminal_reason = (
            "road_boundary"
            if abs(self.x) + self.vehicle_radius >= self.road_half_width
            else "obstacle"
            if collision
            else "traffic_violation"
            if self.violations
            else "success"
            if success
            else "timeout"
            if timeout
            else None
        )
        reward = (self.y - previous_progress) / 5.0 - 0.004
        reward += 0.35 * (self.obstacles_passed - previous_passed)
        if guidance["rule_status"] == "yield_red" and abs(self.speed) < 0.35:
            reward += 0.02
        if collision or self.violations:
            reward -= 1.5
        if success:
            reward += 4.0
        self.done = bool(collision or self.violations or success or timeout)
        self.total_reward += reward
        return self.observe(), float(reward), self.done

    def snapshot(self) -> dict:
        world_x, world_y, world_heading = self.world_pose()
        guidance = self.traffic_guidance()
        centerline = [
            list(self.route_pose(value)[:2]) for value in np.linspace(0, self.road_length, 240)
        ]
        actors = []
        for index, obstacle in enumerate(self.obstacles):
            x, y, _ = self.world_pose(obstacle.y, obstacle.x)
            actors.append(
                {
                    "id": index,
                    "x": x,
                    "y": y,
                    "route_progress": obstacle.y,
                    "lane_offset": obstacle.x,
                    "radius": obstacle.radius,
                }
            )
        projected = []
        for ahead in np.linspace(1.5, 18.0, 12):
            projected.append(list(self.world_pose(self.y + ahead, self.x)))
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
                "world_x": world_x,
                "world_y": world_y,
                "world_heading": world_heading,
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
            "first_obstacle_side": "left" if self.obstacles[0].x < 0 else "right",
            "city": {
                "scenario": self.scenario,
                "name": self.city_name,
                "map_bounds": [-28.0, -18.0, 142.0, 205.0],
                "centerline": centerline,
                "world_trajectory": [list(point) for point in self.world_trajectory[-250:]],
                "actors": actors,
                "road": guidance["road"],
                "next_maneuver": guidance["next_maneuver"],
                "speed_limit_mps": guidance["speed_limit_mps"],
                "traffic_light": guidance["traffic_light"],
                "stop_line_progress": guidance["stop_line_progress"],
                "distance_to_stop_line": guidance["distance_to_stop_line"],
                "rule_status": guidance["rule_status"],
                "violations": self.violations,
                "intersections": [
                    {"name": name, "progress": stop, "signal": self.signal_state(name)}
                    for name, stop in self.signal_stops.items()
                ],
            },
        }
