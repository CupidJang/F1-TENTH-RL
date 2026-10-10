import gym
import numpy as np
import torch
import torch.nn.functional as F

from f1tenth_rl.agents.base import Agent, adam, soft_update, to_tensor
from f1tenth_rl.buffer import NStepCollector, make_buffer
from f1tenth_rl.networks import QNetwork


def linear_schedule(start, end, duration, step):
    return end + (start - end) * max(0.0, 1.0 - step / duration)


class DQNAgent(Agent):
    metric_names = ('loss', 'q', 'grad_norm')
    counter_names = ('steps', 'updates')

    def __init__(self, observation_space, action_space, device='cpu', hidden_sizes=(256, 256), double=False,
                 dueling=False, noisy=False, noisy_std=0.5, distributional=None, num_atoms=51, v_min=-10.0,
                 v_max=10.0, n_step=1, gamma=0.99, lr=3e-4, batch_size=64, learning_starts=5000, train_freq=1,
                 target_update_interval=1000, tau=None, max_grad_norm=10.0, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay_steps=100000, buffer=None):
        super().__init__(device)
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError('DQN requires a discrete action space')
        if distributional not in (None, 'c51', 'qr'):
            raise ValueError(f'unknown distributional type: {distributional}')
        buffer = dict(buffer or {})
        state_dim = observation_space.shape[0]
        self.num_actions = action_space.n
        self.double = double
        self.noisy = noisy
        self.distributional = distributional
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_interval = target_update_interval
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.epsilon_schedule = (epsilon_start, epsilon_end, epsilon_decay_steps)
        self.beta_schedule = (buffer.pop('beta_start', 0.4), buffer.pop('beta_end', 1.0),
                              buffer.pop('beta_steps', 500000))

        num_atoms = num_atoms if distributional else 1
        network = dict(state_dim=state_dim, num_actions=self.num_actions, hidden_sizes=hidden_sizes,
                       dueling=dueling, noisy=noisy, num_atoms=num_atoms, noisy_std=noisy_std)
        self.q = QNetwork(**network).to(self.device)
        self.q_target = QNetwork(**network).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optimizer = adam(self.q.parameters(), lr)
        if distributional == 'c51':
            self.support = torch.linspace(v_min, v_max, num_atoms, device=self.device)
            self.v_min, self.v_max = v_min, v_max
            self.delta_z = (v_max - v_min) / (num_atoms - 1)
        if distributional == 'qr':
            self.taus = (torch.arange(num_atoms, device=self.device, dtype=torch.float32) + 0.5) / num_atoms

        self.buffer = make_buffer(state_dim, **buffer)
        self.n_step = NStepCollector(n_step, gamma)
        self.steps = 0
        self.updates = 0

    @property
    def epsilon(self):
        return 0.0 if self.noisy else linear_schedule(*self.epsilon_schedule, self.steps)

    def modules(self):
        return {'q': self.q, 'q_target': self.q_target}

    def act(self, state, explore=False):
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        self.q.train(explore)
        if explore and self.noisy:
            self.q.reset_noise()
        with torch.no_grad():
            q_values = self._q_values(self.q(to_tensor(state, self.device).unsqueeze(0)))
        self.q.train()
        return int(q_values.argmax(dim=1).item())

    def observe(self, state, action, reward, next_state, terminal, episode_end):
        for transition in self.n_step.push(state, action, reward, next_state, terminal, episode_end):
            self.buffer.add(*transition)
        self.steps += 1
        ready = len(self.buffer) >= max(self.learning_starts, self.batch_size)
        if ready and self.steps % self.train_freq == 0:
            self._update()

    def episode_metrics(self):
        return {'epsilon': self.epsilon, **super().episode_metrics()}

    def _q_values(self, out):
        if self.distributional == 'c51':
            return (out.softmax(dim=-1) * self.support).sum(dim=-1)
        return out.mean(dim=-1)

    def _update(self):
        beta = linear_schedule(*self.beta_schedule, self.steps)
        batch, weights, indices = self.buffer.sample(self.batch_size, beta)
        states, actions, rewards, next_states, discounts = (torch.as_tensor(x, device=self.device) for x in batch)
        if self.noisy:
            self.q.reset_noise()
            self.q_target.reset_noise()

        losses, priorities, q = self._loss(states, actions, rewards, next_states, discounts)
        loss = (torch.as_tensor(weights, device=self.device) * losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.buffer.update_priorities(indices, priorities.cpu().numpy())
        self.record(loss=loss.item(), q=q, grad_norm=grad_norm.item())

        self.updates += 1
        if self.tau is not None:
            soft_update(self.q_target, self.q, self.tau)
        elif self.updates % self.target_update_interval == 0:
            self.q_target.load_state_dict(self.q.state_dict())

    def _loss(self, states, actions, rewards, next_states, discounts):
        rows = torch.arange(len(actions), device=self.device)
        out = self.q(states)[rows, actions]
        with torch.no_grad():
            q_mean = self._q_values(out.unsqueeze(1)).mean().item()
            next_out = self.q_target(next_states)
            selector = self.q(next_states) if self.double else next_out
            next_out = next_out[rows, self._q_values(selector).argmax(dim=1)]

        if self.distributional == 'c51':
            return (*self._categorical_loss(out, rewards, discounts, next_out), q_mean)
        if self.distributional == 'qr':
            return (*self._quantile_loss(out, rewards, discounts, next_out), q_mean)
        q, target = out.squeeze(1), rewards + discounts * next_out.squeeze(1)
        return F.smooth_l1_loss(q, target, reduction='none'), (target - q).detach(), q_mean

    def _categorical_loss(self, logits, rewards, discounts, next_logits):
        with torch.no_grad():
            next_probs = next_logits.softmax(dim=-1)
            projected = rewards.unsqueeze(1) + discounts.unsqueeze(1) * self.support
            b = (projected.clamp(self.v_min, self.v_max) - self.v_min) / self.delta_z
            lower, upper = b.floor().long(), b.ceil().long()
            lower[(upper > 0) & (lower == upper)] -= 1
            upper[(lower < len(self.support) - 1) & (lower == upper)] += 1
            target = torch.zeros_like(next_probs)
            target.scatter_add_(1, lower, next_probs * (upper.float() - b))
            target.scatter_add_(1, upper, next_probs * (b - lower.float()))
        loss = -(target * logits.log_softmax(dim=-1)).sum(dim=-1)
        return loss, loss.detach()

    def _quantile_loss(self, quantiles, rewards, discounts, next_quantiles):
        target = rewards.unsqueeze(1) + discounts.unsqueeze(1) * next_quantiles
        td = target.unsqueeze(1) - quantiles.unsqueeze(2)
        huber = F.smooth_l1_loss(td, torch.zeros_like(td), reduction='none')
        weight = (self.taus.view(1, -1, 1) - (td.detach() < 0).float()).abs()
        loss = (weight * huber).mean(dim=2).sum(dim=1)
        return loss, loss.detach()
