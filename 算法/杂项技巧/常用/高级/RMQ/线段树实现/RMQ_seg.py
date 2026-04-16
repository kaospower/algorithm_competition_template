from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd
from collections import defaultdict, deque

# from sortedcontainers import SortedList
# from itertools import pairwise

Max = lambda x, y: x if x > y else y
Min = lambda x, y: x if x < y else y

# RMQ求区间最小,使用线段树实现,去掉update函数,只保留query函数
class RMQ:
    def __init__(self, nums):
        self.sg=SegmentTree(nums)
    def query(self, L, R):
        return self.sg.query(L, R)

# 维护区间最小值的线段树
class SegmentTree:
    def __init__(self, nums):
        self.n = len(nums)
        self._tree = [0] * (self.n << 2)  # 4n空间
        self._build(1, self.n, 1, nums)

    def _merge(self, a, b):
        return Min(a, b)

    def _maintain(self, o):
        self._tree[o] = self._merge(self._tree[o << 1], self._tree[o << 1 | 1])

    def _build(self, l, r, o, nums):
        if l == r:
            self._tree[o] = nums[l - 1]
            return
        mid = (l + r) >> 1
        self._build(l, mid, o << 1, nums)
        self._build(mid + 1, r, o << 1 | 1, nums)
        self._maintain(o)

    def _query(self, l, r, o, L, R):
        if r < L or l > R:
            return inf
        if L <= l and r <= R:
            return self._tree[o]
        mid = (l + r) >> 1
        if mid >= R:
            return self._query(l, mid, o << 1, L, R)
        if mid < L:
            return self._query(mid + 1, r, o << 1 | 1, L, R)
        l_res = self._query(l, mid, o << 1, L, R)
        r_res = self._query(mid + 1, r, o << 1 | 1, L, R)
        return self._merge(l_res, r_res)

    def query(self, L, R):
        return self._query(1, self.n, 1, L, R)

    def get(self, i):
        return self._query(1, self.n, 1, i, i)


