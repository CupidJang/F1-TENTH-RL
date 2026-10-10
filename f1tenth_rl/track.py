import numpy as np


class Track:
    def __init__(self, points, window=20):
        self.points = np.asarray(points, dtype=np.float64)
        self.segments = np.roll(self.points, -1, axis=0) - self.points
        self.lengths = np.linalg.norm(self.segments, axis=1)
        self.offsets = np.concatenate([[0.0], np.cumsum(self.lengths)[:-1]])
        self.length = self.lengths.sum()
        self.window = window

    def project(self, position, index=None):
        if index is None:
            candidates = np.arange(len(self.points))
        else:
            candidates = (index + np.arange(-self.window, self.window + 1)) % len(self.points)
        relative = np.asarray(position) - self.points[candidates]
        segments = self.segments[candidates]
        t = np.clip((relative * segments).sum(axis=1) / self.lengths[candidates] ** 2, 0.0, 1.0)
        best = np.linalg.norm(relative - t[:, None] * segments, axis=1).argmin()
        index = candidates[best]
        return index, self.offsets[index] + t[best] * self.lengths[index]

    def delta(self, start, end):
        return (end - start + self.length / 2) % self.length - self.length / 2
