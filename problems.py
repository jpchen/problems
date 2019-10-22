import numpy as np
import torch
import scipy


class Multinomial(object):
    """
    multinomial sampler
    """
    def __init__(self):
        pass

    def sample(self):
        pass

def test_multinomial():
    sampler = Multinomial()
    for i in range(10000):
        s = sampler.sample()


# ======================================
class SparseVector(object):
    """
    data type to store sparse vectors
    """
    def __init__(self, vector):
        assert len(vector.shape) == 1
        self.length = len(vector)
        self.indices = np.nonzero(vector)[0]
        self.values = vector[np.nonzero(vector)]

    def dot(self, other):
        assert isinstance(other, SparseVector)
        assert other.length == self.length
        common_indices = np.intersect1d(self.indices, other.indices)
        out = 0.
        for i in common_indices:
            # this would be list.index(i)
            idx_1 = np.where(self.indices == i)
            idx_2 = np.where(other.indices == i)
            out += self.values[idx_1] * other.values[idx_2]
        return out


def test_sparse_dot():
    v1 = np.array([1., 2, 3, 4])
    v2 = np.array([1., 0, 2, 0])
    v1 = SparseVector(v1)
    v2 = SparseVector(v2)
    out = v1.dot(v2)
    assert out == 7


# ======================================
"""
sample from a vector without replacement
"""
def sample_vector():
    pass


def test_sample_vector():
    pass
# ======================================
def main():
#     test_multinomial()
    test_sparse_dot()
    test_sample_vector()
    print("everything passed!")


if __name__ == "__main__":
    main()
