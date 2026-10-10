import inspect
from functools import partial

import yaml


def bind(fn, params):
    accepted = inspect.signature(fn).parameters
    ignored = sorted(set(params) - set(accepted))
    if ignored:
        print(f"{fn.__name__}: ignoring {', '.join(ignored)}")
    return partial(fn, **{key: value for key, value in params.items() if key in accepted})


def merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(*paths, overrides=()):
    config = {}
    for path in paths:
        with open(path) as f:
            config = merge(config, yaml.safe_load(f) or {})
    for override in overrides:
        key, value = override.split('=', 1)
        *parents, leaf = key.split('.')
        node = config
        for parent in parents:
            node = node.setdefault(parent, {})
        node[leaf] = yaml.safe_load(value)
    return config
