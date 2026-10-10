import numpy as np


def run_episode(env, agent, render_every=0):
    state, done, episode_return, steps = env.reset(), False, 0.0, 0
    while not done:
        state, reward, done, info = env.step(agent.act(state))
        episode_return += reward
        steps += 1
        if render_every and steps % render_every == 0:
            env.render()
    return episode_return, info


def evaluate(env, agent, episodes, render_every=0):
    results = [run_episode(env, agent, render_every) for _ in range(episodes)]
    if render_every:
        env.close_window()
    infos = [info for _, info in results]
    lap_times = [info['lap_time'] for info in infos if info['finished']]
    return {
        'return': float(np.mean([episode_return for episode_return, _ in results])),
        'time': float(np.mean([info['lap_time'] for info in infos])),
        'distance': float(np.mean([info['distance'] for info in infos])),
        'finish_rate': len(lap_times) / episodes,
        'lap_time': float(np.mean(lap_times)) if lap_times else float('nan'),
    }


def score(result):
    lap_time = result['lap_time'] if result['finish_rate'] > 0 else float('inf')
    return result['finish_rate'], -lap_time, result['return']
