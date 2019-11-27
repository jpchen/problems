import numpy as np
import torch
import scipy
import math
from collections import defaultdict, Counter
from pdb import set_trace as bb
from termcolor import cprint


def test(x):
    cprint("Running {}".format(x.__name__), 'magenta', end='...')
    try:
        o = x()
        if o == -1:
            cprint("NOT IMPLEMENTED", 'yellow')
        else:
            cprint('PASSED', 'green')
    except AssertionError as e:
        cprint('FAILED', 'red')
        print(e)


# twitter
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

#test
def test_multinomial():
    sampler = Multinomial()
    for i in range(10000):
        s = sampler.sample()

# ======================================
# facebook

def _check_if_valid(string):
    left = [0, 0]
    right = [0, 0]
    for i in string:
        if i == "[":
            left[1] += 1
        elif i == "(":
            left[0] += 1
        elif i == "]":
            if left[1]:
                left[1] -= 1
            else:
                right[1] += 1
        elif i == ")":
            if left[0]:
                left[0] -= 1
            else:
                right[0] += 1
    # return how many need to be removed
    return left, right


def rm_bad_parens(string):
    left, right = _check_if_valid(string)
    copy = extra.copy()
    str_copy = string.copy()
    out = set()
    for i in extra:
        for l in str_copy:
            if i == l:
                str_copy[:i] + str_copy[i+1:]
    return out


@test
def test_rm_bad_parens():
    x = "[([)])"
#     assert not _check_if_valid(x) == []
    x = "[(([]))]"
    x = "[(([]]))]"


def merge_intervals(arr):
    out = []
    arr.sort(key = lambda x: x[0])
    for i in arr:
        if not out or i[0] > out[-1][1]:
            out.append(i)
        else:
            out[-1][1] = max(out[-1][1], i[1])
    return out


@test
def test_merge_intervals():
    tst = [[1,3],[2,6],[8,10],[15,18]]
    out = merge_intervals(tst)
    expected = [[1,6],[8,10],[15,18]]
    assert expected == out


def num_rooms(arr):
    arr.sort(key = lambda x: x[0])
    starts = [[x[0], 0] for x in arr]
    ends = [[x[1], 1] for x in arr]
    times = starts + ends
    times.sort(key = lambda x: x[0])
    rooms = 0
    max_rooms = 0
    for i in times:
        if i[1] == 0:
            rooms += 1
            max_rooms = max(max_rooms, rooms)
        else:
            assert i[1] == 1
            rooms -= 1
            assert rooms >= 0
    return max_rooms


@test
def test_num_rooms():
    # number of meeting rooms needed given schedule
    tst = [[0, 30],[5, 10],[15, 20]]
    out = num_rooms(tst)
    assert 2 == out
    tst = [[7,10],[2,4]]
    out = num_rooms(tst)
    assert 1 == out


def add_binary(a, b):
    # add two binary strings together
    if len(a) == 0: return b
    if len(b) == 0: return a
    if a[-1] == '1' and b[-1] == '1':
        return add_binary(add_binary(a[0:-1],b[0:-1]),'1')+'0'
    if a[-1] == '0' and b[-1] == '0':
        return add_binary(a[0:-1],b[0:-1])+'0'
    else:
        return add_binary(a[0:-1],b[0:-1])+'1'


@test
def test_add_binary():
    pass


def task_scheduler(tasks, N):
    # do this greedily adding most to least
    # todo figure out max heap impl

    # just counting it
    # one of two cases, either every task is fittable in the
    # n intervals or it's not.
    # if n < num tasks, you will always only need length of list
    # adding them greedily
    # if n > num tasks, at most you need the task with most count
    # times the interval (since thats the min needed to schedule that task)
    # then you have to add the other tasks that also have that many at the end
    # everything else is guaranteed to fit in the middle
    task_counts = list(Counter(tasks).values())
    M = max(task_counts)
    Mct = task_counts.count(M)
    return max(len(tasks), (M - 1) * (N + 1) + Mct)


@test
def test_task_scheduler():
    tasks = ['a', 'a', 'a', 'b', 'b', 'c']
    assert task_scheduler(tasks, 1) == 6
    tasks = ['a', 'a', 'a', 'b', 'b', 'c']
    assert task_scheduler(tasks, 2) == 7
    tasks = ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd']
    assert task_scheduler(tasks, 1) == 9
    tasks = ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'd']
    assert task_scheduler(tasks, 1) == 10
    tasks = ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'e']
    assert task_scheduler(tasks, 1) == 10
    tasks = ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd', 'e']
    assert task_scheduler(tasks, 1) == 11


def three_sum(arr, n):
    # find all sets of three numbers that add to n
    arr.sort()
    res = []
    for i in range(len(arr) - 2):
        l, r = i + 1, len(arr) - 1
        while l < r:
            sum_ = arr[i] + arr[l] + arr[r]
            if sum_ < n:
                l += 1
            elif sum_ > n:
                r -= 1
            else:
                res.append([arr[i], arr[l], arr[r]])
                break
    return res


@test
def test_three_sum():
    test = [-1, 0, 1, 2, -1, -4]
    exp = [[-1, 0, 1], [-1, -1, 2]]
    three_sum(test, 0) == exp


def min_window_substring(s, t):
    need = Counter(t)            #hash table to store char frequency
    missing = len(t)                         #total number of chars we care
    start, end = 0, 0
    i = 0
    for j, char in enumerate(s, 1):          #index j from 1
        if need[char] > 0:
            missing -= 1
        need[char] -= 1
        if missing == 0:                     #match all chars
            while i < j and need[s[i]] < 0:  #remove chars to find the real start
                need[s[i]] += 1
                i += 1
            if end == 0 or j-i < end-start:  #update window
                start, end = i, j
            need[s[i]] += 1
            missing += 1
            i += 1
    return s[start:end]


@test
def test_min_window_substring():
    string = "ADOBECODEBANC"
    sub = "ABC"
    min_window_substring(string, sub) == "BANC"
    string = "AEBECEEEEEBCA"
    sub = "ABC"
    min_window_substring(string, sub) == "BCA"


def leftmost_ones(x):
    # find the 1 indexed leftmost ones in a matrix
    i, j = 0, len(x[0]) - 1
    while j > -1 and i < len(x):
        if x[i][j]:
            j -= 1
        else:
            i += 1
    return j + 2


@test
def test_leftmost_ones():
    mat = [[0, 0, 0, 1],
           [0, 0, 1, 1],
           [0, 0, 1, 1],
           [0, 1, 1, 1],
           [0, 0, 0, 1],
          ]
    assert leftmost_ones(mat) == 2
    mat = [[0, 1],
           [1, 1],
           [1, 1],
           [0, 0],
          ]
    assert leftmost_ones(mat) == 1


def lca(x):
    # lca of a binary tree
    pass


@test
def test_lca():
    return -1
# ======================================
# FAIR
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


@test
def test_sparse_dot():
    v1 = np.array([1., 2, 3, 4])
    v2 = np.array([1., 0, 2, 0])
    v1 = SparseVector(v1)
    v2 = SparseVector(v2)
    out = v1.dot(v2)
    assert out == 7


"""
sample from a vector without replacement
"""
def sample_vector():
    pass


@test
def test_sample_vector():
    pass


def num_trees(trees, fov):
    #           /
    #  *       /  *
    #         /   *                             \
    #       0        *                          \   x
    #  **    \                                  -----
    #        \    *
    #        \
    # trees, fov (radians)
    # -> 4
    best, angles = [], []
    for t in trees:
        # compute angles
        x = math.atan2(t[1], t[0])  # returns [-math.pi, math.pi]
        if x < 0:
            x += math.pi * 2
        angles.append(x)
    for a in angles:
        assert 0. <= a <= math.pi * 2
    # discretized search. can be made arbitrarily precise
    for i in range(360):
        curr = []
        min_ang = i * math.pi / 180.
        this_fov = fov + min_ang
        for a, t in zip(angles, trees):
            # check if point is within fov
            if min_ang <= a <= this_fov:
                curr.append(t)
        if len(curr) > len(best):
            best = curr.copy()
    return len(best)


@test
def test_num_trees():
    trees = [[1,1], [2, 0], [3, -1], [-3, -1]]
    angle = math.pi / 2
    assert num_trees(trees, angle) == 2
# ======================================
def find_pivot(arr):
    sum_ = sum(arr)
    accum = 0
    for i in range(len(arr) - 1):
        accum += arr[i]
        sum_ -= arr[i]
        if accum == sum_ - arr[i+1]:
            return i+1
    return -1


@test
def test_find_pivot():
    out = find_pivot([1, 7, 3, 6, 5, 6])
    assert out == 3


def find_deadlock(arr):
    graph = defaultdict(list)
    graph_mutex = defaultdict(list)
    for idx, i in enumerate(arr):
        id, status, mutex = map(int, i.split(' '))
        if status == 0:
            if id in graph_mutex[mutex]:
                list_of_threads = graph_mutex[mutex].copy()
                while list_of_threads:
                    thread = list_of_thread.pop(0)
                return i + 1
            graph[id].append(mutex)
            graph_mutex[mutex].append(id)
        if status == 1:
            graph[id].remove(mutex)
            graph_mutex[mutex].remove(id)
    return 0


def test_find_deadlock():
    lines = ['1 0 1',
             '2 0 2',
             '2 0 1',
             '1 0 2',]
    out = find_deadlock(lines)
    assert out == 4
    lines = ['1 0 1',
             '2 0 2',
             '2 1 1',
             '1 1 1',
             '2 0 1',]
    out = find_deadlock(lines)
    assert out == 0


def cpu_process_intervals(arr):
    stack = []
    out = defaultdict(list)
    for event in arr:
        x = event.split(' ')
        id = x[0]
        start = x[1] == 'True'
        time = int(x[2])  # can be float
        if len(stack) == 0:
            assert start  # cant stop nonexistent event
            stack.append([id, time])
        elif start:
            # preempt different job
            if stack[-1][0] != id:
                out[stack[-1][0]].append([stack[-1][1], time])
            # same job
            stack.append([id, time])
        elif not start:
            old = stack.pop()
            assert old[0] == id
            out[id].append([old[1], time])
            if stack and stack[-1][0] != id:
                # resume the last job
                stack[-1][1] = time
    return out


@test
def test_cpu_process_intervals():
    input = ['f1 True 0',
        'f2 True 2',
        'f1 True 5',
        'f1 False 7',
        'f2 False 10',
        'f3 True 11',
        'f3 False 12',
        'f1 False 15',
        'f4 True 16',
        'f4 False 19',
        ]
    out = cpu_process_intervals(input)
    assert out['f1'] == [[0,2], [5, 7], [10, 11], [12, 15]]
    assert out['f2'] == [[2,5], [7, 10]]
    assert out['f3'] == [[11, 12]]
    assert out['f4'] == [[16, 19]]

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
            # check condition here
            # ....
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


@test
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
        # check condition here
        # ....
        out.append(curr)
        for i in graph[curr]:
            if not visited[i]:
                q.append(i)
                visited[i] = True
    return out


@test
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
# cruise
def linear_interp(points, point):
    i = 0
    while point > points[i][0]:
        i += 1
    slope = float(points[i][1] - points[i - 1][1]) / (points[i][0] - points[i - 1][0])
    intercept = points[i][1] - slope * points[i][0]
    return point * slope + intercept


@test
def test_linear_interp():
    points = [[0, 2], [2, 1], [4, 3], [5, 1]]
    point = 3
    assert linear_interp(points, point) == 2


def k_means(points):
    pass


@test
def test_k_means():
    return -1


# ======================================
# snapchat cv and tesla autopilot
# overview: http://cs231n.github.io/convolutional-networks/
# code implementation: https://victorzhou.com/blog/intro-to-cnns-part-1/

def iterate_regions(image):
    '''
    Generates all possible 3x3 image regions using valid padding.
    - image is a 2d numpy array
    '''
    h, w = image.shape

    for i in range(h - 2):
      for j in range(w - 2):
        im_region = image[i:(i + 3), j:(j + 3)]
        yield im_region, i, j


def forward(input, num_filters):
    '''
    Performs a forward pass of the conv layer using the given input.
    Returns a 3d numpy array with dimensions (h, w, num_filters).
    - input is a 2d numpy array
    '''
    filters = np.random.randn(num_filters, 3, 3)
    h, w = input.shape
    output = np.zeros((h - 2, w - 2, num_filters))

    for im_region, i, j in iterate_regions(input):
      output[i, j] = np.sum(im_region * filters, axis=(1, 2))

    return output


def conv_2d(input_tensor, weight, kernel_size):
    # FIXME this doesnt work
    h_out = input_tensor.shape[2] - weight.shape[0] + 1
    w_out = input_tensor.shape[3] - weight.shape[1] + 1

    out = [[], [], []]
    for i in range(input_tensor.shape[0]):
        for j in range(w_out):
            for k in range(h_out):
                # drag kernel through image with stride of 1
                output = weight[:, : , i].reshape(weight.shape[-1], weight.shape[0], -1) @ input_tensor[0, i, j:j+3, k:k+3]
                # accumulate result
                out[i].append(output)
    # convert into numpy array
    # output = np.array(map(np.concatenate, out))
    output = np.array(out)
    print(output.shape)
    # assert output.shape == (input_tensor.shape[0], h_out, w_out, weight.shape[-1]), output.shape
    return output.reshape(input_tensor.shape[0], weight.shape[-1], h_out, w_out)



@test
def test_convolve():
    input_tensor =  np.random.rand(1, 3, 10, 10) # np.array()  # shape = (1, 3, 10, 10)  NCHW
    weight = np.random.rand(3, 3, 3, 10)  # np.array()  # shape = (3, 3, 3, 10) H x W x C_i x C_o
    # output = conv_2d(input_tensor, weight, kernel_size=3)  # shape = (1, 10, 8, 8)
    return -1


# ======================================
# RECURSION
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

@test
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


def subarray(arr, start, end, out):
    if end == len(arr):
        return out
    if start > end:
        return subarray(arr, 0, end + 1, out)
    else:
        out.append(arr[start:end + 1])
        return subarray(arr, start + 1, end, out)

@test
def test_subarray():
    arr = [1,2,3]
    out = subarray(arr, 0, 0, [])
    exp = [[1],
           [1, 2],
           [2],
           [1, 2, 3],
           [2, 3],
           [3]]
    assert len(out) == len(exp)
    for i in out:
        assert i in exp


def rev_stack(stack, new_stack):
    if len(stack) == 0:
        return new_stack
    new_stack.append(stack.pop())
    return rev_stack(stack, new_stack)

@test
def test_rev_stack():
    stack = [1,2,3, 4]
    out = rev_stack(stack, [])
    assert out == [4, 3, 2, 1]

# ======================================
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

@test
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
# TREES
def same_bsts(a, b):
    # this isnt quite right
    # check if bst equal without building trees
    # recursively check if elements greater and less appear
    # in same order
    if len(a) == 0:
        return True
    for i in a:
        greater_a = set(x for x in a if x > i)
        greater_b = set(x for x in b if x > i)
        if greater_a != greater_b:
            return False
        less_a = set(x for x in a if x < i)
        less_b = set(x for x in b if x < i)
        if less_a != less_b:
            return False
        return same_bsts(a[1:], b)


@test
def test_same_bsts():
    a = [8, 3, 6, 1, 4, 7, 10, 14, 13]
    b = [8, 10, 14, 3, 6, 1, 4, 7, 13]
    c = [8, 10, 14, 3, 7, 6, 4, 1, 13]
    # assert same_bsts(a, b)
    assert not same_bsts(a, c), c


class Node:
    def __init__(self, val):
        self.val = val
        self.children = []

    def add(self, x):
        if not isinstance(x, Node):
            x = Node(x)
        self.children.append(x)


def largest_two_leaves(root):
    # tesla question:
    # return the largest two leaves by iteratively
    # beam search
    # by ONLY looking at the largest two nodes at every step
    def get_max_2(x):
        x.sort(key = lambda z: z.val)
        return [x[-2], x[-1]]
    x = get_max_2(root.children)
    while(x):
        assert len(x) == 2
        i = []
        for z in x:
            if z.children:
                i += z.children
            else:
                return get_max_2(x)
        x = get_max_2(i)


@test
def test_largest_two_leaves():
    a = Node(1.)
    b = Node(0.8)
    b.children = [Node(0.3), Node(0.4), Node(0.7)]
    c = Node(0.7)
    c.children = [Node(0.1), Node(2.4), Node(0.6)]
    d = Node(0.6)
    d.children = [Node(200.), Node(2.), Node(3.4)]
    a.children = [b, c, d]
    fst, snd = largest_two_leaves(a)
    assert fst.val == 0.7
    assert snd.val == 2.4

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

@test
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

@test
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


@test
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

@test
def test_swap():
    x = 2
    y = 64
    x, y = swap(x, y)
    assert x == 64
    assert y == 2
    x, y = bit_swap(x, y)
    assert x == 2
    assert y == 64
