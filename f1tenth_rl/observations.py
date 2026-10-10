import numpy as np

from f1tenth_rl.config import bind

OBSERVATIONS = {}


def register(fn):
    OBSERVATIONS[fn.__name__] = fn
    return fn


@register
def lidar(obs, num_beams=108, max_range=10.0, max_speed=8.0):
    scan = np.minimum(obs['scans'][0], max_range).reshape(num_beams, -1).min(axis=1) / max_range
    return np.append(scan, obs['linear_vels_x'][0] / max_speed)


@register
def lidar_control(obs, num_beams=108, max_range=10.0, max_speed=8.0, max_steering=0.4, max_yaw_rate=5.0):
    vehicle = [obs['ang_vels_z'][0] / max_yaw_rate, obs['steer_cmd'][0] / max_steering, obs['speed_cmd'][0] / max_speed]
    return np.concatenate([lidar(obs, num_beams, max_range, max_speed), vehicle])


def make_observation(config):
    config = dict(config)
    return bind(OBSERVATIONS[config.pop('name')], config)
