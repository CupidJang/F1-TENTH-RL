import copy

import gym
import numpy as np
import torch
import torch.nn.functional as F

from f1tenth_rl.agents.base import Agent, adam, set_requires_grad, soft_update, to_tensor
from f1tenth_rl.buffer import NStepCollector, make_buffer
from f1tenth_rl.networks import DeterministicActor, QCritic, SquashedGaussianActor


class OffPolicyActorCritic(Agent):
    counter_names = ('steps', 'updates')

    def __init__(self, observation_space, action_space, device='cpu', hidden_sizes=(256, 256), n_step=1, gamma=0.99,
                 batch_size=256, learning_starts=5000, train_freq=1, tau=0.005, buffer=None):
        super().__init__(device)
        if not isinstance(action_space, gym.spaces.Box):
            raise ValueError(f'{type(self).__name__} requires a continuous action space')
        self.action_space = action_space
        self.state_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.action_low = torch.as_tensor(action_space.low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_space.high, dtype=torch.float32, device=self.device)
        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.tau = tau
        self.buffer = make_buffer(self.state_dim, action_dim=self.action_dim, **dict(buffer or {}))
        self.n_step = NStepCollector(n_step, gamma)
        self.steps = 0
        self.updates = 0

    def _make_critics(self):
        self.critic1 = QCritic(self.state_dim, self.action_dim, self.hidden_sizes).to(self.device)
        self.critic2 = QCritic(self.state_dim, self.action_dim, self.hidden_sizes).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

    def act(self, state, explore=False):
        if explore and self.steps < self.learning_starts:
            return self.action_space.sample()
        with torch.no_grad():
            action = self._policy_action(to_tensor(state, self.device).unsqueeze(0), explore)
        return action.squeeze(0).cpu().numpy()

    def observe(self, state, action, reward, next_state, terminal, episode_end):
        for transition in self.n_step.push(state, action, reward, next_state, terminal, episode_end):
            self.buffer.add(*transition)
        self.steps += 1
        ready = len(self.buffer) >= max(self.learning_starts, self.batch_size)
        if ready and self.steps % self.train_freq == 0:
            batch, _, _ = self.buffer.sample(self.batch_size)
            self.updates += 1
            self._update(*(torch.as_tensor(x, device=self.device) for x in batch))

    def _policy_action(self, state, explore):
        raise NotImplementedError

    def _update(self, states, actions, rewards, next_states, discounts):
        raise NotImplementedError


class TD3Agent(OffPolicyActorCritic):
    metric_names = ('critic_loss', 'actor_loss', 'q', 'grad_norm')

    def __init__(self, observation_space, action_space, actor_lr=3e-4, critic_lr=3e-4, exploration_noise=0.1,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2, twin=True, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self.actor = DeterministicActor(self.state_dim, self.action_dim, self.action_low, self.action_high,
                                        self.hidden_sizes).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self._make_critics()
        self.critic_params = list(self.critic1.parameters()) + (list(self.critic2.parameters()) if twin else [])
        self.actor_optimizer = adam(self.actor.parameters(), actor_lr)
        self.critic_optimizer = adam(self.critic_params, critic_lr)
        self.action_scale = (self.action_high - self.action_low) / 2
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.twin = twin

    def modules(self):
        return {'actor': self.actor, 'actor_target': self.actor_target, 'critic1': self.critic1,
                'critic2': self.critic2, 'critic1_target': self.critic1_target, 'critic2_target': self.critic2_target}

    def _policy_action(self, state, explore):
        action = self.actor(state)
        if explore:
            action = action + torch.randn_like(action) * self.exploration_noise * self.action_scale
        return torch.max(torch.min(action, self.action_high), self.action_low)

    def _update(self, states, actions, rewards, next_states, discounts):
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = self.actor_target(next_states) + noise * self.action_scale
            next_actions = torch.max(torch.min(next_actions, self.action_high), self.action_low)
            next_q = self.critic1_target(next_states, next_actions)
            if self.twin:
                next_q = torch.min(next_q, self.critic2_target(next_states, next_actions))
            target = rewards + discounts * next_q

        q1 = self.critic1(states, actions)
        critic_loss = F.mse_loss(q1, target)
        if self.twin:
            critic_loss = critic_loss + F.mse_loss(self.critic2(states, actions), target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_params, float('inf'))
        self.critic_optimizer.step()
        self.record(critic_loss=critic_loss.item(), q=q1.mean().item(), grad_norm=grad_norm.item())

        if self.updates % self.policy_delay == 0:
            set_requires_grad([self.critic1], False)
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            set_requires_grad([self.critic1], True)
            self.record(actor_loss=actor_loss.item())
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic1_target, self.critic1, self.tau)
            soft_update(self.critic2_target, self.critic2, self.tau)


class DDPGAgent(TD3Agent):
    def __init__(self, observation_space, action_space, **kwargs):
        kwargs = dict(kwargs, twin=False, policy_delay=1, policy_noise=0.0)
        super().__init__(observation_space, action_space, **kwargs)


class SACAgent(OffPolicyActorCritic):
    metric_names = ('critic_loss', 'actor_loss', 'alpha', 'q', 'entropy', 'grad_norm')

    def __init__(self, observation_space, action_space, actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, alpha=0.2,
                 autotune=True, target_entropy=None, **kwargs):
        super().__init__(observation_space, action_space, **kwargs)
        self.actor = SquashedGaussianActor(self.state_dim, self.action_dim, self.action_low, self.action_high,
                                           self.hidden_sizes).to(self.device)
        self._make_critics()
        self.actor_optimizer = adam(self.actor.parameters(), actor_lr)
        self.critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_optimizer = adam(self.critic_params, critic_lr)
        self.autotune = autotune
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, device=self.device, requires_grad=autotune)
        if autotune:
            self.target_entropy = -float(self.action_dim) if target_entropy is None else target_entropy
            self.alpha_optimizer = adam([self.log_alpha], alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def modules(self):
        return {'actor': self.actor, 'critic1': self.critic1, 'critic2': self.critic2,
                'critic1_target': self.critic1_target, 'critic2_target': self.critic2_target}

    def _policy_action(self, state, explore):
        action, _ = self.actor(state, deterministic=not explore)
        return action

    def _update(self, states, actions, rewards, next_states, discounts):
        alpha = self.alpha
        with torch.no_grad():
            next_actions, next_log_prob = self.actor(next_states)
            next_q = torch.min(self.critic1_target(next_states, next_actions),
                               self.critic2_target(next_states, next_actions))
            target = rewards + discounts * (next_q - alpha * next_log_prob)

        q1 = self.critic1(states, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(self.critic2(states, actions), target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_params, float('inf'))
        self.critic_optimizer.step()

        set_requires_grad([self.critic1, self.critic2], False)
        new_actions, log_prob = self.actor(states)
        q = torch.min(self.critic1(states, new_actions), self.critic2(states, new_actions))
        actor_loss = (alpha * log_prob - q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        set_requires_grad([self.critic1, self.critic2], True)

        if self.autotune:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        soft_update(self.critic1_target, self.critic1, self.tau)
        soft_update(self.critic2_target, self.critic2, self.tau)
        self.record(critic_loss=critic_loss.item(), actor_loss=actor_loss.item(), alpha=alpha, q=q1.mean().item(),
                    entropy=-log_prob.mean().item(), grad_norm=grad_norm.item())
