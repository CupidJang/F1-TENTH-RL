import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PANELS = [
    ('progress.csv', 'return', 'Train return', True),
    ('progress.csv', ('time', 'lap_time'), 'Train survival time (s)', True),
    ('progress.csv', 'speed', 'Train mean speed (m/s)', True),
    ('progress.csv', ('action_entropy', 'action_std'), 'Action diversity', True),
    ('progress.csv', ('loss', 'critic_loss'), 'Loss', True),
    ('progress.csv', 'q', 'Q value', True),
    ('progress.csv', 'grad_norm', 'Grad norm', True),
    ('progress.csv', ('explained_variance', 'alpha', 'epsilon'), 'Explained variance / alpha / epsilon', True),
    ('eval.csv', 'return', 'Eval return', False),
    ('eval.csv', 'time', 'Eval survival time (s)', False),
    ('eval.csv', 'finish_rate', 'Eval finish rate', False),
    ('eval.csv', 'lap_time', 'Eval lap time (s)', False),
]


def load_csv(path):
    if not path.exists():
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def moving_average(values, window):
    window = max(1, min(window, len(values)))
    return np.convolve(values, np.ones(window) / window, mode='valid')


def plot(run_dirs, window, out, show):
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharex=True)
    for color, run_dir in enumerate(run_dirs):
        logs = {name: load_csv(run_dir / name) for name in ('progress.csv', 'eval.csv')}
        for ax, (file, column, title, smooth) in zip(axes.flat, PANELS):
            data = logs[file]
            columns = [name for name in ((column,) if isinstance(column, str) else column) if data and name in data]
            if not columns:
                continue
            steps, values = data['step'], data[columns[0]]
            if smooth:
                mask = ~np.isnan(values)
                steps, values = steps[mask], values[mask]
                if len(values) == 0:
                    continue
                values = moving_average(values, window)
                steps = steps[len(steps) - len(values):]
            ax.plot(steps, values, label=run_dir.name, color=f'C{color}', marker=None if smooth else 'o', markersize=3)

    for ax, (_, _, title, _) in zip(axes.flat, PANELS):
        ax.set_title(title)
        ax.set_xlabel('step')
        ax.grid(alpha=0.3)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f'saved {out}')
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dirs', type=Path, nargs='+')
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    out = args.out or (args.run_dirs[0] / 'plot.png' if len(args.run_dirs) == 1 else Path('runs/compare.png'))
    plot(args.run_dirs, args.window, out, args.show)


if __name__ == '__main__':
    main()
