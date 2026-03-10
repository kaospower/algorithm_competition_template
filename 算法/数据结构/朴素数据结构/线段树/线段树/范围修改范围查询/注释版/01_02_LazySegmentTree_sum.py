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

Max = lambda x, y: x if x > y else y
Min = lambda x, y: x if x < y else y


# 范围修改范围查询线段树,不包含建树,维护的信息是区间累加和
class LazySegmentTree:
    # 懒标记初始值,根据实际需要修改
    _TODO_INIT = 0

    # 初始化
    def __init__(self, n):
        self.n = n
        self._tree = [0] * (self.n << 2)
        self._todo = [0] * (self.n << 2)

    # 把懒标记作用于子树
    def _apply(self, l, r, o, v):
        self._tree[o] += v * (r - l + 1)
        self._todo[o] += v

    # 将当前节点懒标记下传给左右儿子
    def _spread(self, l, r, o):
        v = self._todo[o]
        mid = (l + r) >> 1
        if v == self._TODO_INIT:
            return
        self._apply(l, mid, o << 1, v)
        self._apply(mid + 1, r, o << 1 | 1, v)
        self._todo[o] = self._TODO_INIT

    # 合并左右儿子信息
    def _maintain(self, o):
        self._tree[o] = self._tree[o << 1] + self._tree[o << 1 | 1]

    # 区间修改函数
    def _update(self, l, r, o, L, R, v):
        # 当前区间[l,r]被待查询区间[L,R]全包
        if L <= l and r <= R:
            self._apply(l, r, o, v)
            return
        self._spread(l, r, o)
        mid = (l + r) >> 1
        # 如果当前区间中点>=L,说明[l,mid]与待查询区间有交集,更新左子树
        if mid >= L:
            self._update(l, mid, o << 1, L, R, v)
        # 如果当前区间中点<R,说明[mid,r]与待查询区间有交集,更新右子树
        if mid < R:
            self._update(mid + 1, r, o << 1 | 1, L, R, v)
        self._maintain(o)

    # 区间查询函数
    def _query(self, l, r, o, L, R):
        # 当前区间被待查询区间全包
        if L <= l and r <= R:
            return self._tree[o]
        self._spread(l, r, o)
        mid = (l + r) >> 1
        # 说明右子树和[L,R]无交集,只查询左子树
        if mid >= R:
            return self._query(l, mid, o << 1, L, R)
        # 说明左子树和[L,R]无交集,只查询右子树
        if mid < L:
            return self._query(mid + 1, r, o << 1 | 1, L, R)
        return self._query(l, mid, o << 1, L, R) + self._query(mid + 1, r, o << 1 | 1, L, R)

    # 实际区间更新函数
    def update(self, L, R, v):
        self._update(1, self.n, 1, L + 1, R + 1, v)

    # 实际区间查询函数
    def query(self, L, R):
        return self._query(1, self.n, 1, L + 1, R + 1)

# 使用案例
# https://leetcode.cn/problems/coin-bonus/description/
