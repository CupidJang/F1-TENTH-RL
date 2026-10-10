import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'f1tenth-riders-quickstart' / 'pkg' / 'src'))

from f1tenth_rl.config import load_config  # noqa: E402
from f1tenth_rl.env import make_env  # noqa: E402
from pkg.drivers import DisparityExtender  # noqa: E402


def record(map_name, spacing):
    config = load_config(ROOT / 'configs' / 'base.yaml',
                         overrides=[f'env.map={map_name}', 'env.action_type=continuous', 'env.action_repeat=1'])
    env = make_env(config, ROOT)
    driver = DisparityExtender()
    low, high = env.speed_range
    env.reset()
    poses, last = [], None
    while True:
        speed, steer = driver.process_lidar(env.obs['scans'][0])
        _, _, done, info = env.step(np.array([steer / env.max_steering, 2 * (speed - low) / (high - low) - 1]))
        pose = np.array([env.obs['poses_x'][0], env.obs['poses_y'][0], env.obs['poses_theta'][0]])
        if last is None or np.hypot(*(pose[:2] - last[:2])) >= spacing:
            poses.append(pose)
            last = pose
        if info['lap_count'] >= 1:
            break
        if done:
            raise RuntimeError(f'reference driver crashed on {map_name}')
    env.close()
    return np.array(poses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('maps', nargs='+')
    parser.add_argument('--spacing', type=float, default=0.5)
    args = parser.parse_args()

    for map_name in args.maps:
        poses = record(map_name, args.spacing)
        path = ROOT / 'tracks' / f'{map_name}.csv'
        np.savetxt(path, poses, delimiter=',', header='x,y,theta', comments='', fmt='%.5f')
        print(f'{map_name}: {len(poses)} poses -> {path}')


if __name__ == '__main__':
    main()
