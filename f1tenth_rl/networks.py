import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

ACTIVATIONS = {'relu': nn.ReLU, 'tanh': nn.Tanh}


def mlp(in_dim, hidden_sizes, activation='relu', linear=nn.Linear):
    layers = []
    for size in hidden_sizes:
        layers += [linear(in_dim, size), ACTIVATIONS[activation]()]
        in_dim = size
    return nn.Sequential(*layers), in_dim


def orthogonal_init(module, gain=math.sqrt(2)):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain)
            nn.init.zeros_(layer.bias)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.full((out_features, in_features), std_init / math.sqrt(in_features)))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.full((out_features,), std_init / math.sqrt(out_features)))
        self.register_buffer('weight_epsilon', torch.zeros(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.zeros(out_features))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        self.reset_noise()

    def reset_noise(self):
        eps_in = self._scaled_noise(self.in_features)
        eps_out = self._scaled_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(eps_out, eps_in))
        self.bias_epsilon.copy_(eps_out)

    @staticmethod
    def _scaled_noise(size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if not self.training:
            return F.linear(x, self.weight_mu, self.bias_mu)
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(x, weight, bias)


class QNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_sizes=(256, 256), dueling=False, noisy=False, num_atoms=1,
                 noisy_std=0.5):
        super().__init__()
        linear = partial(NoisyLinear, std_init=noisy_std) if noisy else nn.Linear
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.dueling = dueling
        self.body, feature_dim = mlp(state_dim, hidden_sizes, linear=linear)
        self.head = linear(feature_dim, num_actions * num_atoms)
        if dueling:
            self.value = linear(feature_dim, num_atoms)

    def forward(self, x):
        features = self.body(x)
        out = self.head(features).view(-1, self.num_actions, self.num_atoms)
        if self.dueling:
            out = self.value(features).unsqueeze(1) + out - out.mean(dim=1, keepdim=True)
        return out

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, discrete, hidden_sizes=(64, 64), activation='tanh', log_std_init=0.0):
        super().__init__()
        self.discrete = discrete
        self.body, feature_dim = mlp(state_dim, hidden_sizes, activation)
        self.head = nn.Linear(feature_dim, action_dim)
        orthogonal_init(self.body)
        orthogonal_init(self.head, 0.01)
        if not discrete:
            self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def forward(self, x):
        out = self.head(self.body(x))
        if self.discrete:
            return Categorical(logits=out)
        return Normal(out, self.log_std.exp().expand_as(out))


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_sizes=(64, 64), activation='tanh'):
        super().__init__()
        self.body, feature_dim = mlp(state_dim, hidden_sizes, activation)
        self.head = nn.Linear(feature_dim, 1)
        orthogonal_init(self.body)
        orthogonal_init(self.head, 1.0)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)


class QCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.body, feature_dim = mlp(state_dim + action_dim, hidden_sizes)
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, state, action):
        return self.head(self.body(torch.cat([state, action], dim=-1))).squeeze(-1)


class DeterministicActor(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high, hidden_sizes=(256, 256)):
        super().__init__()
        self.body, feature_dim = mlp(state_dim, hidden_sizes)
        self.head = nn.Linear(feature_dim, action_dim)
        self.register_buffer('action_scale', (action_high - action_low) / 2)
        self.register_buffer('action_bias', (action_high + action_low) / 2)

    def forward(self, state):
        return torch.tanh(self.head(self.body(state))) * self.action_scale + self.action_bias


class SquashedGaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high, hidden_sizes=(256, 256),
                 log_std_bounds=(-5.0, 2.0)):
        super().__init__()
        self.body, feature_dim = mlp(state_dim, hidden_sizes)
        self.mean = nn.Linear(feature_dim, action_dim)
        self.log_std = nn.Linear(feature_dim, action_dim)
        self.log_std_bounds = log_std_bounds
        self.register_buffer('action_scale', (action_high - action_low) / 2)
        self.register_buffer('action_bias', (action_high + action_low) / 2)

    def forward(self, state, deterministic=False):
        features = self.body(state)
        mean = self.mean(features)
        if deterministic:
            return torch.tanh(mean) * self.action_scale + self.action_bias, None
        log_std = self.log_std(features).clamp(*self.log_std_bounds)
        normal = Normal(mean, log_std.exp())
        sample = normal.rsample()
        squashed = torch.tanh(sample)
        log_prob = normal.log_prob(sample) - torch.log(self.action_scale * (1 - squashed.pow(2)) + 1e-6)
        return squashed * self.action_scale + self.action_bias, log_prob.sum(dim=-1)
