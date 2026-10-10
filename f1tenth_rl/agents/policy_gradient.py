import gym
import numpy as np
import torch

from f1tenth_rl.agents.base import Agent, adam, to_tensor
from f1tenth_rl.networks import PolicyNetwork, ValueNetwork


def compute_gae(rewards, values, next_values, terminals, episode_ends, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (1.0 - terminals[t]) * next_values[t] - values[t]
        last = delta + gamma * gae_lambda * (1.0 - episode_ends[t]) * last
        advantages[t] = last
    return advantages


def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def explained_variance(values, returns):
    variance = returns.var()
    return float('nan') if variance == 0 else (1 - (returns - values).var() / variance).item()


class OnPolicyAgent(Agent):
    metric_names = ('loss', 'policy_loss', 'value_loss', 'entropy', 'grad_norm', 'explained_variance')

    def __init__(self, observation_space, action_space, device='cpu', hidden_sizes=(64, 64), activation='tanh',
                 gamma=0.99, lr=3e-4, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, log_std_init=0.0):
        super().__init__(device)
        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        state_dim = observation_space.shape[0]
        action_dim = action_space.n if self.discrete else action_space.shape[0]
        self.policy = PolicyNetwork(state_dim, action_dim, self.discrete, hidden_sizes, activation,
                                    log_std_init).to(self.device)
        self.value = ValueNetwork(state_dim, hidden_sizes, activation).to(self.device)
        self.params = list(self.policy.parameters()) + list(self.value.parameters())
        self.optimizer = adam(self.params, lr, eps=1e-5)
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.storage = []

    def modules(self):
        return {'policy': self.policy, 'value': self.value}

    def act(self, state, explore=False):
        with torch.no_grad():
            dist = self.policy(to_tensor(state, self.device).unsqueeze(0))
            if explore:
                action = dist.sample()
            else:
                action = dist.probs.argmax(dim=-1) if self.discrete else dist.mean
        return int(action.item()) if self.discrete else action.squeeze(0).cpu().numpy()

    def observe(self, state, action, reward, next_state, terminal, episode_end):
        self.storage.append((state, action, reward, next_state, terminal, episode_end))
        if self._ready(episode_end):
            states, actions, rewards, next_states, terminals, episode_ends = zip(*self.storage)
            self.storage = []
            self._learn(to_tensor(states, self.device),
                        torch.as_tensor(np.asarray(actions), device=self.device),
                        to_tensor(rewards, self.device),
                        to_tensor(next_states, self.device),
                        to_tensor(terminals, self.device),
                        to_tensor(episode_ends, self.device))

    def _ready(self, episode_end):
        raise NotImplementedError

    def _learn(self, states, actions, rewards, next_states, terminals, episode_ends):
        raise NotImplementedError

    def _evaluate_actions(self, states, actions):
        dist = self.policy(states)
        log_prob, entropy = dist.log_prob(actions), dist.entropy()
        if not self.discrete:
            log_prob, entropy = log_prob.sum(dim=-1), entropy.sum(dim=-1)
        return log_prob, entropy

    def _step(self, policy_loss, value_loss, entropy):
        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()
        self.record(loss=loss.item(), policy_loss=policy_loss.item(), value_loss=value_loss.item(),
                    entropy=entropy.item(), grad_norm=grad_norm.item())


class REINFORCEAgent(OnPolicyAgent):
    counter_names = ('episodes',)

    def __init__(self, observation_space, action_space, episodes_per_update=1, baseline=True,
                 normalize_advantages=True, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self.episodes_per_update = episodes_per_update
        self.baseline = baseline
        self.normalize_advantages = normalize_advantages
        self.episodes = 0

    def _ready(self, episode_end):
        self.episodes += int(episode_end)
        return episode_end and self.episodes % self.episodes_per_update == 0

    def _learn(self, states, actions, rewards, next_states, terminals, episode_ends):
        zeros = torch.zeros_like(rewards)
        returns = compute_gae(rewards, zeros, zeros, terminals, episode_ends, self.gamma, 1.0)
        values = self.value(states) if self.baseline else zeros
        advantages = returns - values.detach()
        if self.normalize_advantages:
            advantages = normalize(advantages)
        log_prob, entropy = self._evaluate_actions(states, actions)
        policy_loss = -(log_prob * advantages).mean()
        value_loss = (values - returns).pow(2).mean() if self.baseline else zeros.sum()
        self._step(policy_loss, value_loss, entropy.mean())
        if self.baseline:
            self.record(explained_variance=explained_variance(values.detach(), returns))


class A2CAgent(OnPolicyAgent):
    def __init__(self, observation_space, action_space, rollout_steps=16, gae_lambda=1.0,
                 normalize_advantages=False, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self.rollout_steps = rollout_steps
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def _ready(self, episode_end):
        return len(self.storage) >= self.rollout_steps

    def _learn(self, states, actions, rewards, next_states, terminals, episode_ends):
        values = self.value(states)
        with torch.no_grad():
            next_values = self.value(next_states)
            advantages = compute_gae(rewards, values.detach(), next_values, terminals, episode_ends, self.gamma,
                                     self.gae_lambda)
            returns = advantages + values.detach()
        if self.normalize_advantages:
            advantages = normalize(advantages)
        log_prob, entropy = self._evaluate_actions(states, actions)
        policy_loss = -(log_prob * advantages).mean()
        value_loss = (values - returns).pow(2).mean()
        self._step(policy_loss, value_loss, entropy.mean())
        self.record(explained_variance=explained_variance(values.detach(), returns))


class PPOAgent(OnPolicyAgent):
    metric_names = OnPolicyAgent.metric_names + ('approx_kl', 'clip_fraction')

    def __init__(self, observation_space, action_space, rollout_steps=2048, epochs=10, minibatch_size=64,
                 clip_coef=0.2, gae_lambda=0.95, normalize_advantages=True, target_kl=None, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self.rollout_steps = rollout_steps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.clip_coef = clip_coef
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.target_kl = target_kl

    def _ready(self, episode_end):
        return len(self.storage) >= self.rollout_steps

    def _learn(self, states, actions, rewards, next_states, terminals, episode_ends):
        with torch.no_grad():
            old_log_prob, _ = self._evaluate_actions(states, actions)
            values = self.value(states)
            advantages = compute_gae(rewards, values, self.value(next_states), terminals, episode_ends, self.gamma,
                                     self.gae_lambda)
            returns = advantages + values
        self.record(explained_variance=explained_variance(values, returns))

        for _ in range(self.epochs):
            for batch in torch.randperm(len(states), device=self.device).split(self.minibatch_size):
                log_prob, entropy = self._evaluate_actions(states[batch], actions[batch])
                log_ratio = log_prob - old_log_prob[batch]
                ratio = log_ratio.exp()
                batch_advantages = advantages[batch]
                if self.normalize_advantages:
                    batch_advantages = normalize(batch_advantages)
                clipped = ratio.clamp(1 - self.clip_coef, 1 + self.clip_coef)
                policy_loss = -torch.min(ratio * batch_advantages, clipped * batch_advantages).mean()
                value_loss = 0.5 * (self.value(states[batch]) - returns[batch]).pow(2).mean()
                self._step(policy_loss, value_loss, entropy.mean())
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_fraction = ((ratio - 1).abs() > self.clip_coef).float().mean().item()
                self.record(approx_kl=approx_kl, clip_fraction=clip_fraction)
            if self.target_kl is not None and approx_kl > self.target_kl:
                break
