import numpy as np
from scipy.linalg import expm,norm

class RandomRotation:

    def _M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def __call__(self, coords, feats):
        R = self._M(
            np.random.rand(3) - 0.5, 2 * np.pi * (np.random.rand(1) - 0.5))
        return coords @ R, feats


class RandomScale:

    def __init__(self, min, max):
        self.scale = max - min
        self.bias = min

    def __call__(self, coords, feats):
        s = self.scale * np.random.rand(1) + self.bias
        return coords * s, feats


class RandomShear:

    def __call__(self, coords, feats):
        T = np.eye(3) + np.random.randn(3, 3)
        return coords @ T, feats


class RandomTranslation:

    def __call__(self, coords, feats):
        trans = 0.05 * np.random.randn(1, 3)
        return coords + trans, feats