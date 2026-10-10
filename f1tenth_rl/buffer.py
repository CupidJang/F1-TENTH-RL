from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim=None, state_dtype='float16'):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=state_dtype)
        if action_dim is None:
            self.actions = np.zeros(capacity, dtype=np.int64)
        else:
            self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=state_dtype)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, state, action, reward, next_state, discount):
        index = self.pos
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.discounts[index] = discount
        self.pos = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return index

    def sample(self, batch_size, beta=None):
        indices = np.random.randint(0, self.size, size=batch_size)
        return self._gather(indices), np.ones(batch_size, dtype=np.float32), indices

    def update_priorities(self, indices, td_errors):
        pass

    def _gather(self, indices):
        return (self.states[indices].astype(np.float32), self.actions[indices], self.rewards[indices],
                self.next_states[indices].astype(np.float32), self.discounts[indices])


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, state_dim, action_dim=None, state_dtype='float16', alpha=0.6, eps=1e-6):
        super().__init__(capacity, state_dim, action_dim, state_dtype)
        self.alpha = alpha
        self.eps = eps
        self.max_priority = 1.0
        self.tree_size = 1
        while self.tree_size < capacity:
            self.tree_size *= 2
        self.sum_tree = np.zeros(2 * self.tree_size)
        self.min_tree = np.full(2 * self.tree_size, np.inf)

    def add(self, state, action, reward, next_state, discount):
        index = super().add(state, action, reward, next_state, discount)
        self._set(np.array([index]), np.array([self.max_priority ** self.alpha]))
        return index

    def sample(self, batch_size, beta=0.4):
        total = self.sum_tree[1]
        targets = (np.arange(batch_size) + np.random.rand(batch_size)) * total / batch_size
        indices = np.minimum(self._find(targets), self.size - 1)

        probs = self.sum_tree[indices + self.tree_size] / total
        min_prob = self.min_tree[1] / total
        weights = (probs / min_prob) ** -beta
        return self._gather(indices), weights.astype(np.float32), indices

    def update_priorities(self, indices, td_errors):
        priorities = np.abs(td_errors) + self.eps
        self.max_priority = max(self.max_priority, priorities.max())
        self._set(indices, priorities ** self.alpha)

    def _set(self, indices, priorities):
        nodes = indices + self.tree_size
        self.sum_tree[nodes] = priorities
        self.min_tree[nodes] = priorities
        while nodes[0] > 1:
            nodes = nodes // 2
            left, right = 2 * nodes, 2 * nodes + 1
            self.sum_tree[nodes] = self.sum_tree[left] + self.sum_tree[right]
            self.min_tree[nodes] = np.minimum(self.min_tree[left], self.min_tree[right])

    def _find(self, targets):
        nodes = np.ones(len(targets), dtype=np.int64)
        while nodes[0] < self.tree_size:
            left = 2 * nodes
            go_right = targets > self.sum_tree[left]
            targets = np.where(go_right, targets - self.sum_tree[left], targets)
            nodes = np.where(go_right, left + 1, left)
        return nodes - self.tree_size


class NStepCollector:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.queue = deque()

    def push(self, state, action, reward, next_state, terminal, episode_end):
        self.queue.append((state, action, reward))
        self.next_state = next_state
        self.terminal = terminal
        transitions = []
        if len(self.queue) == self.n:
            transitions.append(self._pop())
        if episode_end:
            while self.queue:
                transitions.append(self._pop())
        return transitions

    def _pop(self):
        steps = len(self.queue)
        n_step_return = sum(self.gamma ** i * reward for i, (_, _, reward) in enumerate(self.queue))
        discount = 0.0 if self.terminal else self.gamma ** steps
        state, action, _ = self.queue.popleft()
        return state, action, n_step_return, self.next_state, discount


def make_buffer(state_dim, capacity, action_dim=None, state_dtype='float16', prioritized=False, alpha=0.6):
    if prioritized:
        return PrioritizedReplayBuffer(capacity, state_dim, action_dim, state_dtype, alpha)
    return ReplayBuffer(capacity, state_dim, action_dim, state_dtype)
