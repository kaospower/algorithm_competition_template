from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd
from collections import defaultdict

# from sortedcontainers import SortedList
# from itertools import pairwise

fmax = lambda x, y: x if x > y else y
fmin = lambda x, y: x if x < y else y


class LazySegmentTree:
    # 懒标记初始值,根据实际需要修改
    _TODO_INIT = 0

    def __init__(self, nums):
        self.n = len(nums)
        self._tree = [0] * (self.n << 2)
        self._todo = [0] * (self.n << 2)
        self._build(1, self.n, 1, nums)

    def _apply(self, l, r, o, v):
        self._tree[o] += v * (r - l + 1)
        self._todo[o] += v

    def _spread(self, l, r, o):
        v = self._todo[o]
        mid = (l + r) >> 1
        if v == self._TODO_INIT:
            return
        self._apply(l, mid, o << 1, v)
        self._apply(mid + 1, r, o << 1 | 1, v)
        self._todo[o] = self._TODO_INIT

    def _maintain(self, o):
        self._tree[o] = self._tree[o << 1] + self._tree[o << 1 | 1]

    def _build(self, l, r, o, nums):
        if l == r:
            self._tree[o] = nums[l - 1]
            return
        mid = (l + r) >> 1
        self._build(l, mid, o << 1, nums)
        self._build(mid + 1, r, o << 1 | 1, nums)
        self._maintain(o)

    def _update(self, l, r, o, L, R, v):
        if L <= l and r <= R:
            self._apply(l, r, o, v)
            return
        self._spread(l, r, o)
        mid = (l + r) >> 1
        if mid >= L:
            self._update(l, mid, o << 1, L, R, v)
        if mid < R:
            self._update(mid + 1, r, o << 1 | 1, L, R, v)
        self._maintain(o)

    def _query(self, l, r, o, L, R):
        if L <= l and r <= R:
            return self._tree[o]
        self._spread(l, r, o)
        mid = (l + r) >> 1
        if mid >= R:
            return self._query(l, mid, o << 1, L, R)
        if mid < L:
            return self._query(mid + 1, r, o << 1 | 1, L, R)
        return self._query(l, mid, o << 1, L, R) + self._query(mid + 1, r, o << 1 | 1, L, R)

    def update(self, L, R, v):
        self._update(1, self.n, 1, L + 1, R + 1, v)

    def query(self, L, R):
        return self._query(1, self.n, 1, L + 1, R + 1)
