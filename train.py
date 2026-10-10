import argparse
import csv
import random
import shutil
import signal
import sys
import time
from pathlib import Path

import gym
import numpy as np
import torch
import yaml

from f1tenth_rl.agents import make_agent
from f1tenth_rl.config import load_config, merge
from f1tenth_rl.env import make_env
from f1tenth_rl.evaluation import evaluate, score

ROOT = Path(__file__).resolve().parent


class CSVLogger:
    def __init__(self, path, append=False):
        fieldnames = None
        if append and path.exists():
            with open(path) as f:
                fieldnames = next(csv.reader(f), None)
        self.file = open(path, 'a' if append else 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames, extrasaction='ignore') if fieldnames else None

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row), extrasaction='ignore')
            self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_steps(steps):
    if steps >= 1_000_000:
        return f'{steps / 1_000_000:.2f}M'
    if steps >= 1_000:
        return f'{steps / 1_000:.1f}K'
    return str(steps)


def format_value(key, value):
    if key == 'step':
        return format_steps(value)
    return f'{value:.3f}' if isinstance(value, float) else str(value)


def format_row(row):
    return ' | '.join(f'{key} {format_value(key, value)}' for key, value in row.items())


def action_stats(actions, action_space):
    if isinstance(action_space, gym.spaces.Discrete):
        probs = np.bincount(actions, minlength=action_space.n) / len(actions)
        probs = probs[probs > 0]
        return {'action_entropy': float(-(probs * np.log(probs)).sum() / np.log(action_space.n))}
    return {'action_std': float(np.std(actions, axis=0).mean())}


def trim_log(path, max_step):
    if not path.exists():
        return
    with open(path) as f:
        rows = list(csv.reader(f))
    if rows:
        step_column = rows[0].index('step')
        rows = rows[:1] + [row for row in rows[1:] if int(float(row[step_column])) <= max_step]
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerows(rows)


def save_checkpoint(path, agent, step, episode, best_score):
    torch.save({'agent': agent.checkpoint(), 'step': step, 'episode': episode, 'best_score': best_score}, path)


def status(line):
    if sys.stdout.isatty():
        width = shutil.get_terminal_size().columns
        print('\r' + line[:width - 1] + '\033[K', end='', flush=True)
    else:
        print(line)


def log(line):
    print(('\r\033[K' if sys.stdout.isatty() else '') + line, flush=True)


def train(config, run_dir, render=False, resume=False):
    set_seed(config['seed'])
    torch.set_num_threads(config['threads'])
    env = make_env(config, ROOT, config['seed'], config['train']['random_start'])
    eval_map = config['eval']['map'] or config['env']['map']
    eval_env = make_env(merge(config, {'env': {'map': eval_map}}), ROOT, config['seed'])
    render_every = config['render_every']
    if config['eval']['render']:
        eval_env.render()
        eval_env.close_window()
    env.action_space.seed(config['seed'])
    agent = make_agent(config['agent'], env.observation_space, env.action_space, config['device'])
    total_steps, ec = config['train']['total_steps'], config['eval']

    start_step, episode, best_score = 1, 0, None
    if resume:
        checkpoint = torch.load(run_dir / 'checkpoint.pt', map_location=agent.device, weights_only=True)
        agent.restore(checkpoint['agent'])
        start_step, episode, best_score = checkpoint['step'] + 1, checkpoint['episode'], checkpoint['best_score']
        if best_score is not None:
            best_score = tuple(best_score)
        for name in ('progress.csv', 'eval.csv'):
            trim_log(run_dir / name, checkpoint['step'])
        log(f'resumed {run_dir} from step {format_steps(checkpoint["step"])}')

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)
    train_log = CSVLogger(run_dir / 'progress.csv', append=resume)
    eval_log = CSVLogger(run_dir / 'eval.csv', append=resume)

    state = env.reset()
    episode_return, actions = 0.0, []
    step = start_step - 1
    start = time.time()

    try:
        for step in range(start_step, total_steps + 1):
            action = agent.act(state, explore=True)
            next_state, reward, done, info = env.step(action)
            agent.observe(state, action, reward, next_state, done and not info['truncated'], done)
            state = next_state
            episode_return += reward
            actions.append(action)
            if render and step % render_every == 0:
                env.render()

            if done:
                episode += 1
                row = {'episode': episode, 'step': step, 'return': episode_return, 'length': len(actions),
                       'time': info['lap_time'], 'distance': info['distance'],
                       'speed': info['distance'] / max(info['lap_time'], 1e-6), 'lap_count': info['lap_count'],
                       'collision': int(info['collision']), 'finished': int(info['finished']),
                       **action_stats(actions, env.action_space), **agent.episode_metrics()}
                train_log.write(row)
                elapsed = time.time() - start
                eta = elapsed / (step - start_step + 1) * (total_steps - step)
                status(f'[{step / total_steps:6.1%}] elapsed {elapsed / 60:.1f}m | eta {eta / 60:.1f}m | '
                       f'{format_row(row)}')
                state = env.reset()
                episode_return, actions = 0.0, []

            if step % ec['interval'] == 0 or step == total_steps:
                result = evaluate(eval_env, agent, ec['episodes'], render_every if ec['render'] else 0)
                eval_log.write({'step': step, **result})
                agent.save(run_dir / 'last.pt')
                save_checkpoint(run_dir / 'checkpoint.pt', agent, step, episode, best_score)
                is_best = best_score is None or score(result) > best_score
                if is_best:
                    best_score = score(result)
                    agent.save(run_dir / 'best.pt')
                log(f'[eval] step {format_steps(step)} | {format_row(result)} | best {is_best}')
    except KeyboardInterrupt:
        log('interrupted')
    finally:
        agent.save(run_dir / 'last.pt')
        save_checkpoint(run_dir / 'checkpoint.pt', agent, step, episode, best_score)
        log(f'saved {run_dir}')
        train_log.close()
        eval_log.close()
        env.close()
        eval_env.close()


def interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', default='dqn')
    parser.add_argument('--config', default=ROOT / 'configs' / 'base.yaml')
    parser.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE')
    parser.add_argument('--name', default=None)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--resume', type=Path, default=None, metavar='RUN_DIR')
    args = parser.parse_args()

    if args.resume:
        config = load_config(args.config, args.resume / 'config.yaml', overrides=args.set)
        train(config, args.resume, args.render, resume=True)
        return
    config = load_config(args.config, ROOT / 'configs' / 'agents' / f'{args.agent}.yaml', overrides=args.set)
    run_dir = ROOT / 'runs' / (args.name or f"{args.agent}_{time.strftime('%Y%m%d-%H%M%S')}")
    if run_dir.exists():
        parser.error(f'{run_dir} already exists, use --resume {run_dir} or another --name')
    train(config, run_dir, args.render)


if __name__ == '__main__':
    main()
