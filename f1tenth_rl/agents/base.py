import numpy as np
import torch


def to_tensor(x, device):
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)


def adam(params, lr, **kwargs):
    params = list(params)
    try:
        return torch.optim.Adam(params, lr=lr, fused=True, **kwargs)
    except (RuntimeError, TypeError):
        return torch.optim.Adam(params, lr=lr, **kwargs)


def set_requires_grad(modules, flag):
    for module in modules:
        for param in module.parameters():
            param.requires_grad_(flag)


def soft_update(target, source, tau):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.lerp_(param, tau)


class Agent:
    metric_names = ()
    counter_names = ()

    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self._metrics = {name: [] for name in self.metric_names}

    def act(self, state, explore=False):
        raise NotImplementedError

    def observe(self, state, action, reward, next_state, terminal, episode_end):
        raise NotImplementedError

    def modules(self):
        raise NotImplementedError

    def record(self, **values):
        for name, value in values.items():
            self._metrics[name].append(value)

    def episode_metrics(self):
        summary = {name: float(np.mean(values)) if values else float('nan') for name, values in self._metrics.items()}
        for values in self._metrics.values():
            values.clear()
        return summary

    def save(self, path):
        torch.save({name: module.state_dict() for name, module in self.modules().items()}, path)

    def checkpoint(self):
        return {
            'modules': {name: module.state_dict() for name, module in self.modules().items()},
            'optimizers': {name: value.state_dict() for name, value in vars(self).items()
                           if isinstance(value, torch.optim.Optimizer)},
            'parameters': {name: value.detach().clone() for name, value in vars(self).items()
                           if isinstance(value, torch.Tensor) and value.requires_grad},
            'counters': {name: getattr(self, name) for name in self.counter_names},
        }

    def restore(self, checkpoint):
        for name, module in self.modules().items():
            module.load_state_dict(checkpoint['modules'][name])
        for name, state in checkpoint['optimizers'].items():
            getattr(self, name).load_state_dict(state)
        with torch.no_grad():
            for name, value in checkpoint['parameters'].items():
                getattr(self, name).copy_(value)
        for name, value in checkpoint['counters'].items():
            setattr(self, name, value)

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=True)
        for name, module in self.modules().items():
            module.load_state_dict(state[name])
