import numpy as np
import torch
import scipy
import math
from collections import defaultdict
from pdb import set_trace as bb


class Multinomial(object):
    """
    multinomial sampler
    """
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def sample(self, n):
        for _ in range(n):
            sample = np.random.random()
        pass

    def categorical_sampler(p, v):
        if len(p) != len(v):
            raise ValueError("mismatch")
        sampled = np.random.random()
        for p_i, v_i in zip(p, v):
            sampled -= p_i
            if sampled <= 0:
                return v_i
        return v[-1]

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
class Graph:
    def __init__(self):
        # {node: edges}
        self.graph = defaultdict(list)

    def addEdge(self, u, v):
        # directed graph u -> v
        self.graph[u].append(v)


def _do_dfs(node, graph, visited, out):
    visited[node] = True
    out.append(node)
    for i in graph[node]:
        if not visited[i]:
            _do_dfs(i, graph, visited, out)

def dfs(node, graph):
    # iterative method
#     s = Stack()
#     s.push(root)
#     out = []
#     while not s.is_empty():
#         curr = s.pop()
#         if curr.right is not None:
#             s.push(curr.right)
#         if curr.left is not None:
#             s.push(curr.left)
#         out.append(curr)
#     return out
    # recursive
    graph = graph.graph
    visited = [False] * len(graph)
    out = []
    _do_dfs(node, graph, visited, out)
    return out


def test_dfs():
    g = Graph()
    g.addEdge(0, 1)
    g.addEdge(0, 2)
    g.addEdge(1, 2)
    g.addEdge(2, 0)
    g.addEdge(2, 3)
    g.addEdge(3, 3)
    out = dfs(2, g)
    assert out == [2, 0, 1, 3], out
    out = dfs(0, g)
    assert out == [0, 1, 2, 3], out


# ======================================
def bfs(root, graph):
    q = []
    q.append(root)
    visited = [False] * len(graph)
    visited[root] = True
    out = []
    while len(q):
        curr = q.pop(0)
        out.append(curr)
        for i in graph[curr]:
            if not visited[i]:
                q.append(i)
                visited[i] = True
    return out


def test_bfs():
    g = Graph()
    g.addEdge(0, 1)
    g.addEdge(0, 2)
    g.addEdge(1, 2)
    g.addEdge(2, 0)
    g.addEdge(2, 3)
    g.addEdge(3, 3)
    out = bfs(2, g.graph)
    assert out == [2, 0, 3, 1], out
    g.addEdge(0, 4)
    g.addEdge(4, 3)
    out = bfs(0, g.graph)
    assert out == [0, 1, 2, 4, 3], out


# ======================================
def convolve(mat, stride):
    # perform a convolution
    pass

def test_convolve():
    print('convolve not implemented')
    assert True


# ======================================
def permute(lst):
    if len(lst) == 0:
        return []
    if len(lst) == 1:
        return [lst]
    out = []
    for i in range(len(lst)):
        # make the ith element first
        m = lst[i]
        # get the remaining list
        remaining = lst[:i] + lst[i+1:]
        for p in permute(remaining):
            # everything in this loop is with ith element first
            out.append([m] + p)
    return out

def test_permute():
    out = permute([1,2,3])
    ans = [
      [1,2,3],
      [1,3,2],
      [2,1,3],
      [2,3,1],
      [3,1,2],
      [3,2,1]
    ]
    for i in ans:
        assert i in out


def mypow(x, n):
    out = x
    if n == 0:
        return 1
    mod2 = math.floor(math.log2(abs(n)))
    for i in range(mod2):
        out *= out
    remaining = abs(n) - 2**mod2
    for i in range(remaining):
        out *= x
    if n < 0:
        return 1. / out
    return out

def test_mypow():
    out = mypow(2,4)
    assert out == 2 ** 4, out
    out = mypow(2,3)
    assert out == 2 ** 3, out
    out = mypow(3,5)
    assert out == 3 ** 5
    out = mypow(2, -3)
    assert out == 2 ** -3, out
    out = mypow(3,-5)
    assert out == 3 ** -5, out
# ======================================
# DYNAMIC PROGRAMMING
def lis(arr):
    # longest increasing subsequence
    lis_arr = [1] * len(arr)
    for i in range(1, len(arr)):
        for j in range(i):
            if arr[i] > arr[j]:
                lis_arr[i] = max(lis_arr[i], 1 + lis_arr[j])
    return max(lis_arr)

def test_lis():
    arr = [50, 3, 10, 7, 40, 80]
    out = lis(arr)
    assert out == len([3, 7, 40, 80])


def l_sum_subarray(arr):
    max_so_far = 0
    max_ending_here = 0
    start, end = 0, 0
    for i in range(len(arr)):
        max_ending_here += arr[i]
        if max_so_far < max_ending_here:
            end = i + 1 # python drops the last element in a[start:end]
            max_so_far = max(max_so_far, max_ending_here)
        if max_ending_here < 0:
            start = i + 1
            max_ending_here = 0
    return max_so_far, arr[start:end]

def test_l_sum_subarray():
    arr = [-2, -3, 4, -1, -2, 1, 5, -3]
    out, sub = l_sum_subarray(arr)
    assert out == 7
    assert sub == [4, -1, -2, 1, 5], sub


def knapsack(val, weight, limit):
    assert len(val) == len(weight)
    a = np.zeros([len(val) + 1, limit + 1])
    for i in range(len(val) + 1):
        for j in range(limit + 1):
            if i == 0 or j == 0:
                continue
            if weight[i - 1] <= j:
                # can fit
                a[i, j] = max(a[i - 1, j - weight[i - 1]] + val[i - 1], a[i - 1, j])
            else:
                # cant fit new item
                a[i, j] = a[i-1, j]
    assert a.max() == a[len(val), limit]
    return a.max()


def test_knapsack():
    val = [50, 80, 80, 90, 150]
    wt = [10, 12, 8, 20, 30]
    W = 50
    score = knapsack(val, wt, W)
    assert score == 310, score


# ======================================
# BITWISE OPERATIONS

def swap(a, b):
    # swap two numbers
    a = a + b
    b = a - b
    a = a - b
    return a, b

def bit_swap(a, b):
    # swap two numbers
    a = a ^ b
    b = b ^ a
    a = a ^ b
    return a, b

def test_swap():
    x = 2
    y = 64
    x, y = swap(x, y)
    assert x == 64
    assert y == 2
    x, y = bit_swap(x, y)
    assert x == 2
    assert y == 64


# ======================================
def main():
#     test_multinomial()
    test_sparse_dot()
    test_sample_vector()
    test_bfs()
    test_dfs()
    test_convolve()
    test_permute()
    test_swap()
    test_lis()
    test_l_sum_subarray()
    test_knapsack()
    test_mypow()
    print("ALL TESTS PASSED!")


if __name__ == "__main__":
    main()
