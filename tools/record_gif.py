import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from f1tenth_rl.agents import make_agent  # noqa: E402
from f1tenth_rl.config import load_config  # noqa: E402
from f1tenth_rl.env import make_env  # noqa: E402


def drive(run_dir, checkpoint):
    config = load_config(ROOT / 'configs' / 'base.yaml', run_dir / 'config.yaml')
    env = make_env(config, ROOT, config['seed'])
    agent = make_agent(config['agent'], env.observation_space, env.action_space, config['device'])
    agent.load(run_dir / checkpoint)
    state, done, frames = env.reset(), False, []
    while not done:
        state, _, done, info = env.step(agent.act(state))
        obs = env.obs
        frames.append((obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0],
                       info['lap_time'], info['lap_count']))
    env.close()
    return config, np.array(frames)


def load_map(config):
    map_path = (ROOT / config['env']['map_dir']).resolve() / config['env']['map']
    with open(f'{map_path}.yaml') as f:
        meta = yaml.safe_load(f)
    walls = np.array(Image.open(f'{map_path}.png').convert('L')) == 0
    return walls, meta['resolution'], meta['origin'][:2]


PALETTE = [(245, 245, 245), (40, 40, 40), (228, 87, 46), (31, 119, 180), (20, 20, 20), (245, 245, 245)]
FREE, WALL, TRAIL, CAR, TEXT, FREE_ALT = range(len(PALETTE))


def render(config, frames, out, title, fps, speed, width, margin=20, header=32):
    walls, resolution, (origin_x, origin_y) = load_map(config)
    rows, cols = np.nonzero(walls)
    top, bottom = rows.min() - margin, rows.max() + margin
    left, right = cols.min() - margin, cols.max() + margin
    scale = width / (right - left)
    height = int((bottom - top) * scale)
    thickness = max(1, int(round(1.5 / scale)))
    mask = Image.fromarray((walls[top:bottom, left:right] * 255).astype(np.uint8))
    mask = mask.filter(ImageFilter.MaxFilter(2 * thickness + 1)).resize((width, height), Image.BOX)
    background = Image.new('P', (width, height + header), FREE)
    background.putpalette([value for color in PALETTE for value in color] + [0] * (768 - 3 * len(PALETTE)))
    background.paste(WALL, (0, header), mask.point(lambda value: 255 if value > 64 else 0))

    def to_pixel(x, y):
        col = (x - origin_x) / resolution
        row = walls.shape[0] - 1 - (y - origin_y) / resolution
        return (col - left) * scale, (row - top) * scale + header

    font = ImageFont.truetype(str(Path(matplotlib.get_data_path()) / 'fonts' / 'ttf' / 'DejaVuSans.ttf'), 16)
    car_size = 9.0
    stride = max(1, round(100 * speed / fps))
    trail = int(2.0 * 100)
    images = []
    for i in range(0, len(frames), stride):
        image = background.copy()
        draw = ImageDraw.Draw(image)
        path = [to_pixel(x, y) for x, y, *_ in frames[max(0, i - trail):i + 1]]
        if len(path) > 1:
            draw.line(path, fill=TRAIL, width=3)
        x, y, theta, velocity, elapsed, laps = frames[i]
        cx, cy = to_pixel(x, y)
        heading = np.array([np.cos(theta), -np.sin(theta)])
        side = np.array([-heading[1], heading[0]])
        nose = (cx + heading[0] * car_size, cy + heading[1] * car_size)
        rear_left = (cx - heading[0] * car_size * 0.6 + side[0] * car_size * 0.5,
                     cy - heading[1] * car_size * 0.6 + side[1] * car_size * 0.5)
        rear_right = (cx - heading[0] * car_size * 0.6 - side[0] * car_size * 0.5,
                      cy - heading[1] * car_size * 0.6 - side[1] * car_size * 0.5)
        draw.polygon([nose, rear_left, rear_right], fill=CAR)
        corner = FREE_ALT if len(images) % 2 else FREE
        image.putpixel((0, 0), corner)
        image.putpixel((width - 1, height + header - 1), corner)
        draw.text((10, 7), f'{title}    t {elapsed:5.1f} s    {velocity:4.1f} m/s    lap {int(laps) + 1}',
                  fill=TEXT, font=font)
        images.append(image)
    images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
    return len(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--checkpoint', default='best.pt')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--title', default=None)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--speed', type=float, default=6.0)
    parser.add_argument('--width', type=int, default=560)
    args = parser.parse_args()

    torch.set_num_threads(2)
    config, frames = drive(args.run_dir, args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = render(config, frames, args.out, args.title or args.run_dir.name, args.fps, args.speed, args.width)
    print(f'{args.out}: {count} frames, {frames[-1][4]:.1f} s driven')


if __name__ == '__main__':
    main()
