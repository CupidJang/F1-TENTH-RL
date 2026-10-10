from collections import deque
from pathlib import Path

import gym
import numpy as np

from f1tenth_rl.observations import make_observation
from f1tenth_rl.rewards import make_reward
from f1tenth_rl.track import Track


class F110Env(gym.Wrapper):
    def __init__(self, map_path, start_pose, observation_fn, reward_fn, action_type='discrete',
                 steering_angles=(-0.4, 0.0, 0.4), speeds=(3.0,), max_steering=0.4, speed_range=(2.0, 8.0),
                 frame_stack=1, action_repeat=5, time_limit=300.0, laps=2, track=None, spawn_poses=None, seed=0):
        env = gym.make('f110_gym:f110-v0', map=map_path, map_ext='.png', num_agents=1, seed=seed)
        super().__init__(env)
        self.start_pose = np.array([start_pose], dtype=np.float64)
        self.track = track
        self.spawn_poses = spawn_poses
        self.rng = np.random.default_rng(seed)
        if action_type == 'discrete':
            self.controls = np.array([(steer, speed) for speed in speeds for steer in steering_angles])
            self.action_space = gym.spaces.Discrete(len(self.controls))
        elif action_type == 'continuous':
            self.max_steering = max_steering
            self.speed_range = speed_range
            self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        else:
            raise ValueError(f'unknown action_type: {action_type}')
        self.observation_fn = observation_fn
        self.reward_fn = reward_fn
        self.frames = deque(maxlen=frame_stack)
        self.action_repeat = action_repeat
        self.time_limit = time_limit
        self.laps = laps
        self.timestep = env.unwrapped.timestep
        state_dim = self.reset().shape[0]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (state_dim,), np.float32)

    def reset(self):
        self.env.unwrapped.sim.collisions[:] = 0.0
        pose = self.start_pose
        if self.spawn_poses is not None:
            pose = self.spawn_poses[None, self.rng.integers(len(self.spawn_poses))]
        obs, _, _, _ = self.env.reset(poses=pose)
        obs = self._snapshot(obs, np.zeros((1, 2)))
        self.obs = obs
        self.lap_count = 0
        self.path_length = 0.0
        self.elapsed = 0.0
        self.track_progress = 0.0
        if self.track is not None:
            self.track_index, self.track_s = self.track.project(self._position(obs))
        frame = self._observe(obs)
        for _ in range(self.frames.maxlen):
            self.frames.append(frame)
        return np.concatenate(self.frames)

    def step(self, action):
        control = self._control(action)
        for sim_steps in range(1, self.action_repeat + 1):
            obs, _, sim_done, _ = self.env.step(control)
            if obs['collisions'][0] or (sim_done and self.track is None):
                break
        obs = self._snapshot(obs, control)
        self.elapsed += sim_steps * self.timestep
        self.path_length += float(np.hypot(obs['poses_x'][0] - self.obs['poses_x'][0],
                                           obs['poses_y'][0] - self.obs['poses_y'][0]))
        collision = bool(obs['collisions'][0])
        if self.track is None:
            progress = float('nan')
            lap_count = int(obs['lap_counts'][0])
            finished = sim_done and not collision
        else:
            self.track_index, track_s = self.track.project(self._position(obs), self.track_index)
            progress = float(self.track.delta(self.track_s, track_s))
            self.track_s = track_s
            self.track_progress += progress
            lap_count = max(self.lap_count, int(self.track_progress // self.track.length))
            finished = not collision and lap_count >= self.laps
        done = collision or finished

        info = {
            'collision': collision,
            'finished': finished,
            'lap_count': lap_count,
            'new_laps': lap_count - self.lap_count,
            'lap_time': self.elapsed,
            'distance': self.path_length if self.track is None else self.track_progress,
            'progress': progress,
            'dt': sim_steps * self.timestep,
            'truncated': not done and self.elapsed >= self.time_limit,
        }
        reward = self.reward_fn(obs, self.obs, info)
        self.obs = obs
        self.lap_count = lap_count
        self.frames.append(self._observe(obs))
        return np.concatenate(self.frames), reward, done or info['truncated'], info

    def render(self, mode='human_fast'):
        first = self.env.unwrapped.renderer is None
        self.env.render(mode)
        if first:
            self._fit_view(self.env.unwrapped.renderer)

    def close_window(self):
        if self.env.unwrapped.renderer is not None:
            self.env.unwrapped.renderer.close()
            self.env.unwrapped.renderer = None

    @staticmethod
    def _fit_view(renderer, margin=1.05):
        points = renderer.map_points[:, :2]
        (x_min, y_min), (x_max, y_max) = points.min(axis=0), points.max(axis=0)
        zoom = margin * max((x_max - x_min) / renderer.width, (y_max - y_min) / renderer.height)
        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        renderer.zoom_level = zoom
        renderer.zoomed_width, renderer.zoomed_height = zoom * renderer.width, zoom * renderer.height
        renderer.left, renderer.right = center_x - renderer.zoomed_width / 2, center_x + renderer.zoomed_width / 2
        renderer.bottom, renderer.top = center_y - renderer.zoomed_height / 2, center_y + renderer.zoomed_height / 2

    @staticmethod
    def _position(obs):
        return np.array([obs['poses_x'][0], obs['poses_y'][0]])

    @staticmethod
    def _snapshot(obs, control):
        return dict(obs, lap_times=obs['lap_times'].copy(), lap_counts=obs['lap_counts'].copy(),
                    steer_cmd=control[:, 0].copy(), speed_cmd=control[:, 1].copy())

    def _control(self, action):
        if isinstance(self.action_space, gym.spaces.Discrete):
            return self.controls[action][None]
        steer, speed = np.clip(action, -1.0, 1.0)
        low, high = self.speed_range
        return np.array([[steer * self.max_steering, low + (speed + 1.0) / 2.0 * (high - low)]])

    def _observe(self, obs):
        return np.asarray(self.observation_fn(obs), dtype=np.float32).ravel()


def make_env(config, root='.', seed=0, random_start=False):
    env_config = dict(config['env'])
    map_name = env_config.pop('map')
    start_poses = env_config.pop('start_poses')
    if map_name not in start_poses:
        raise ValueError(f"no start pose for map '{map_name}', add it to env.start_poses")
    track_path = Path(root) / 'tracks' / f'{map_name}.csv'
    poses = np.loadtxt(track_path, delimiter=',', skiprows=1) if track_path.exists() else None
    if random_start and poses is None:
        raise FileNotFoundError(f'{track_path} not found, run: python tools/record_track.py {map_name}')
    map_path = (Path(root) / env_config.pop('map_dir')).resolve() / map_name
    return F110Env(str(map_path), start_poses[map_name], make_observation(config['observation']),
                   make_reward(config['reward']), track=None if poses is None else Track(poses[:, :2]),
                   spawn_poses=poses if random_start else None, seed=seed, **env_config)
