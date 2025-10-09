from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd
from collections import defaultdict, deque, Counter

# from sortedcontainers import SortedList
# from itertools import pairwise

Max = lambda x, y: x if x > y else y
Min = lambda x, y: x if x < y else y

#线段树套哈希表,每个节点是一个哈希表
class SegmentTree:
    def __init__(self, nums):
        self.n = len(nums)
        self._tree = [defaultdict(int) for _ in range(self.n << 2)]
        self._build(1, self.n, 1, nums)


    def _build(self, l, r, o, nums):
        for i in range(l,r+1):
            self._tree[o][nums[i-1]]+=1
        if l == r:
            return
        mid = (l + r) >> 1
        self._build(l, mid, o << 1, nums)
        self._build(mid + 1, r, o << 1 | 1, nums)

    def _query(self, l, r, o, L, R, val):
        if R < L:
            return 0
        if L <= l and r <= R:
            return self._tree[o][val]

        mid = (l + r) >> 1

        if mid >= R:
            return self._query(l, mid, o << 1, L, R, val)

        if mid < L:
            return self._query(mid + 1, r, o << 1 | 1, L, R, val)
        l_res = self._query(l, mid, o << 1, L, R, val)
        r_res = self._query(mid + 1, r, o << 1 | 1, L, R, val)
        return l_res + r_res

    def query(self, L, R, val):
        return self._query(1, self.n, 1, L, R, val)

#示例题:2080:https://leetcode.cn/problems/range-frequency-queries/