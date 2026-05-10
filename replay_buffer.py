from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# -1: slot not yet labeled; 0: episode ended timeout; 1: episode ended with release
_UNLABELED = np.int8(-1)


@dataclass
class Batch:
    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_state: np.ndarray
    done: np.ndarray


@dataclass
class PriorityBatch(Batch):
    indices: np.ndarray
    weights: np.ndarray


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)

        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity,), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity,), dtype=np.float32)
        self.released_episode = np.full((self.capacity,), _UNLABELED, dtype=np.int8)

        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool
    ) -> int:
        i = self._idx
        self.state[i] = state
        self.action[i] = action
        self.reward[i] = float(reward)
        self.next_state[i] = next_state
        self.done[i] = 1.0 if done else 0.0
        self.released_episode[i] = _UNLABELED

        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        return int(i)

    def mark_episode(self, indices: list[int], released: bool) -> None:
        v = np.int8(1 if released else 0)
        for idx in indices:
            self.released_episode[int(idx)] = v

    def _get_batch(self, idx: np.ndarray) -> Batch:
        return Batch(
            state=self.state[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            next_state=self.next_state[idx],
            done=self.done[idx],
        )

    def _stratified_indices(self, batch_size: int, min_release_fraction: float) -> np.ndarray:
        if self._size < batch_size or min_release_fraction <= 0:
            return self.rng.integers(0, self._size, size=(batch_size,))

        rel = np.where(self.released_episode[: self.capacity] == 1)[0]
        tout = np.where(self.released_episode[: self.capacity] == 0)[0]
        n_rel = max(1, int(batch_size * min_release_fraction))
        n_to = batch_size - n_rel

        if len(rel) < n_rel:
            return self.rng.integers(0, self._size, size=(batch_size,))

        r_sample = self.rng.choice(rel, size=n_rel, replace=False)
        parts: list[np.ndarray] = [r_sample]
        if n_to > 0 and len(tout) > 0:
            nt = min(n_to, len(tout))
            t_sample = self.rng.choice(tout, size=nt, replace=len(tout) < nt)
            parts.append(t_sample)

        idxs = np.concatenate(parts)
        if len(idxs) < batch_size:
            pad = self.rng.integers(0, self._size, size=(batch_size - len(idxs),))
            idxs = np.concatenate([idxs, pad])
        return idxs.astype(np.int64)[:batch_size]

    def sample(self, batch_size: int, min_release_fraction: float | None = None) -> Batch:
        if self._size < batch_size:
            raise ValueError(f"Not enough samples: have {self._size}, need {batch_size}.")
        if min_release_fraction is not None and min_release_fraction > 0:
            idx = self._stratified_indices(batch_size, min_release_fraction)
        else:
            idx = self.rng.integers(0, self._size, size=(batch_size,))
        return self._get_batch(idx)


class SumTree:
    """Binary sum tree for O(log n) proportional sampling (prioritized replay)."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)

    def total(self) -> float:
        return float(self.tree[0])

    def update(self, data_idx: int, priority: float) -> None:
        tree_idx = data_idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get(self, s: float) -> int:
        """Leaf data index for cumulative mass s in [0, total)."""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx - (self.capacity - 1)
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right


class PrioritizedReplayBuffer:
    """
    Proportional prioritized replay (Schaul et al. 2015).
    Optional stratified batch (release vs timeout tags): sum-tree not used for index choice that step;
    weights are uniform; priorities still updated from TD errors.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int,
        seed: int = 0,
        *,
        alpha: float = 0.6,
        eps: float = 1e-6,
    ) -> None:
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.max_priority = 1.0

        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity,), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity,), dtype=np.float32)
        self.released_episode = np.full((self.capacity,), _UNLABELED, dtype=np.int8)

        self.tree = SumTree(self.capacity)
        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool
    ) -> int:
        i = self._idx
        self.state[i] = state
        self.action[i] = action
        self.reward[i] = float(reward)
        self.next_state[i] = next_state
        self.done[i] = 1.0 if done else 0.0
        self.released_episode[i] = _UNLABELED

        pr = float(self.max_priority)
        self.tree.update(i, pr)

        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        return int(i)

    def mark_episode(self, indices: list[int], released: bool) -> None:
        v = np.int8(1 if released else 0)
        for idx in indices:
            self.released_episode[int(idx)] = v

    def _get_batch(self, idx: np.ndarray) -> Batch:
        return Batch(
            state=self.state[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            next_state=self.next_state[idx],
            done=self.done[idx],
        )

    def _stratified_indices(self, batch_size: int, min_release_fraction: float) -> np.ndarray:
        if self._size < batch_size or min_release_fraction <= 0:
            return self.rng.integers(0, self._size, size=(batch_size,))

        rel = np.where(self.released_episode[: self.capacity] == 1)[0]
        tout = np.where(self.released_episode[: self.capacity] == 0)[0]
        n_rel = max(1, int(batch_size * min_release_fraction))
        n_to = batch_size - n_rel

        if len(rel) < n_rel:
            return self.rng.integers(0, self._size, size=(batch_size,))

        r_sample = self.rng.choice(rel, size=n_rel, replace=False)
        parts: list[np.ndarray] = [r_sample]
        if n_to > 0 and len(tout) > 0:
            nt = min(n_to, len(tout))
            t_sample = self.rng.choice(tout, size=nt, replace=len(tout) < nt)
            parts.append(t_sample)

        idxs = np.concatenate(parts)
        if len(idxs) < batch_size:
            pad = self.rng.integers(0, self._size, size=(batch_size - len(idxs),))
            idxs = np.concatenate([idxs, pad])
        return idxs.astype(np.int64)[:batch_size]

    def sample(
        self,
        batch_size: int,
        beta: float,
        min_release_fraction: float | None = None,
    ) -> PriorityBatch:
        if self._size < batch_size:
            raise ValueError(f"Not enough samples: have {self._size}, need {batch_size}.")

        if min_release_fraction is not None and min_release_fraction > 0:
            idxs = self._stratified_indices(batch_size, min_release_fraction)
            w = np.ones(batch_size, dtype=np.float32)
            return PriorityBatch(
                state=self.state[idxs],
                action=self.action[idxs],
                reward=self.reward[idxs],
                next_state=self.next_state[idxs],
                done=self.done[idxs],
                indices=idxs.astype(np.int64),
                weights=w,
            )

        total = self.tree.total()
        if total <= 0:
            raise RuntimeError("SumTree total is zero; cannot sample.")

        idxs = np.zeros(batch_size, dtype=np.int64)
        segment = total / batch_size
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            s = float(self.rng.uniform(low, high))
            idxs[i] = self.tree.get(s)

        probs = np.zeros(batch_size, dtype=np.float64)
        for i in range(batch_size):
            ti = int(idxs[i])
            leaf = ti + self.capacity - 1
            probs[i] = self.tree.tree[leaf] / total
        n = float(self._size)
        w = (n * probs) ** (-beta)
        w = w / (w.max() + 1e-8)

        return PriorityBatch(
            state=self.state[idxs],
            action=self.action[idxs],
            reward=self.reward[idxs],
            next_state=self.next_state[idxs],
            done=self.done[idxs],
            indices=idxs.astype(np.int64),
            weights=w.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        td_errors = np.asarray(td_errors, dtype=np.float64).reshape(-1)
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        for idx, td in zip(indices, td_errors, strict=True):
            pr = (float(abs(td)) + self.eps) ** self.alpha
            self.tree.update(int(idx), pr)
            self.max_priority = max(self.max_priority, pr)
