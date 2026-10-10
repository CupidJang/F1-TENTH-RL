import argparse
from pathlib import Path

import torch

from f1tenth_rl.agents import make_agent
from f1tenth_rl.config import load_config
from f1tenth_rl.env import make_env
from f1tenth_rl.evaluation import run_episode

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--checkpoint', default='best.pt')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE')
    parser.add_argument('--no-render', dest='render', action='store_false')
    args = parser.parse_args()

    config = load_config(ROOT / 'configs' / 'base.yaml', args.run_dir / 'config.yaml', overrides=args.set)
    torch.set_num_threads(config['threads'])
    env = make_env(config, ROOT, config['seed'])
    agent = make_agent(config['agent'], env.observation_space, env.action_space, config['device'])
    agent.load(args.run_dir / args.checkpoint)

    print(f"map {config['env']['map']}")
    for episode in range(1, args.episodes + 1):
        episode_return, info = run_episode(env, agent, config['render_every'] if args.render else 0)
        print(f"ep {episode} | return {episode_return:.2f} | laps {info['lap_count']} | time {info['lap_time']:.2f}s "
              f"| distance {info['distance']:.1f}m | finished {info['finished']} | collision {info['collision']} "
              f"| truncated {info['truncated']}")
    env.close()


if __name__ == '__main__':
    main()
