import numpy as np
import torch
import scipy


class Multinomial(object):
    """
    multinomial sampler
    """
    def __init__(self):
        pass

    def sample():
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
    def __init__(self):
        pass

    def dot(self. other):
        pass


def test_sparse_dot():
    pass
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
    test_multinomial()
    test_sparse_dot()
    test_sample_vector()
    print("everything passed!")


if __name__ == "__main__":
    main()
