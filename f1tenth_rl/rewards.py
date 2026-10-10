import numpy as np

from f1tenth_rl.config import bind

REWARDS = {}


def register(fn):
    REWARDS[fn.__name__] = fn
    return fn


@register
def speed(obs, prev_obs, info, collision_penalty=10.0, lap_bonus=10.0):
    reward = obs['linear_vels_x'][0] * info['dt']
    if info['collision']:
        reward -= collision_penalty
    return reward + lap_bonus * info['new_laps']


@register
def progress(obs, prev_obs, info, collision_penalty=10.0, steer_penalty=0.1, speed_penalty=0.05, max_steering=0.4,
             max_speed=8.0, lap_bonus=0.0):
    if np.isnan(info['progress']):
        raise ValueError('progress reward needs tracks/<map>.csv, run tools/record_track.py')
    reward = info['progress']
    reward -= steer_penalty * abs(obs['steer_cmd'][0] - prev_obs['steer_cmd'][0]) / max_steering
    reward -= speed_penalty * abs(obs['speed_cmd'][0] - prev_obs['speed_cmd'][0]) / max_speed
    if info['collision']:
        reward -= collision_penalty
    return reward + lap_bonus * info['new_laps']


def make_reward(config):
    config = dict(config)
    return bind(REWARDS[config.pop('name')], config)
