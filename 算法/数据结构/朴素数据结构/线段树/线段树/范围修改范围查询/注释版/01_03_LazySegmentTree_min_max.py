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


# Lazy线段树,维护区间最大值和最小值
class LazySegmentTree:
    # 懒标记初始值,根据实际需要修改
    _TODO_INIT = 0

    # 初始化
    def __init__(self, n):
        self.n = n
        self._min = [0] * (self.n << 2)
        self._max = [0] * (self.n << 2)
        self._todo = [0] * (self.n << 2)

    # 把懒标记作用于子树
    def _apply(self, o, todo):
        self._min[o] += todo
        self._max[o] += todo
        self._todo[o] += todo

    # 将当前节点懒标记下传给左右儿子
    def _spread(self, o):
        todo = self._todo[o]
        if todo == self._TODO_INIT:
            return
        self._apply(o << 1, todo)
        self._apply(o << 1 | 1, todo)
        self._todo[o] = self._TODO_INIT

    # 合并左右儿子信息
    def _maintain(self, o):
        self._min[o] = fmin(self._min[o << 1], self._min[o << 1 | 1])
        self._max[o] = fmax(self._max[o << 1], self._max[o << 1 | 1])

    # 区间修改函数
    def _update(self, l, r, o, L, R, v):
        if L <= l and r <= R:
            self._apply(o, v)
            return
        self._spread(o)
        mid = (l + r) >> 1
        if mid >= L:
            self._update(l, mid, o << 1, L, R, v)
        if mid < R:
            self._update(mid + 1, r, o << 1 | 1, L, R, v)
        self._maintain(o)

    # 区间查询函数
    def _query(self, l, r, o, L, R):
        if L <= l and r <= R:
            return self._min[o], self._max[o]
        self._spread(o)
        mid = (l + r) >> 1
        if mid >= R:
            return self._query(l, mid, o << 1, L, R)
        if mid < L:
            return self._query(mid + 1, r, o << 1 | 1, L, R)
        l_res = self._query(l, mid, o << 1, L, R)
        r_res = self._query(mid + 1, r, o << 1 | 1, L, R)
        return fmin(l_res[0], r_res[0]), fmax(l_res[1], r_res[1])

    # 实际区间更新函数
    def update(self, L, R, v):
        self._update(1, self.n, 1, L + 1, R + 1, v)

    # 实际区间查询函数
    def query(self, L, R):
        return self._query(1, self.n, 1, L + 1, R + 1)
