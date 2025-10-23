from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd
from collections import defaultdict
#这两行头文件在老的oj可能不支持,因为属于新版本特性
# from sortedcontainers import SortedList
# from itertools import pairwise

fmax = lambda x, y: x if x > y else y
fmin = lambda x, y: x if x < y else y

class SegmentTree:
    def __init__(self, nums):
        self.n = len(nums)
        self._tree = [0] * (self.n << 2)  # 4n空间
        self._build(1, self.n, 1, nums)

    def _alter(self, o, val):
        self._tree[o] = val

    def _merge(self, a, b):
        return fmax(a, b)

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

    def _update(self, l, r, o, idx, val):
        if l == r:
            self._alter(o, val)
            return
        mid = (l + r) >> 1
        if idx <= mid:
            self._update(l, mid, o << 1, idx, val)
        else:
            self._update(mid + 1, r, o << 1 | 1, idx, val)
        self._maintain(o)

    def _query(self, l, r, o, L, R):
        if R < L:
            return 0
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

    def update(self, idx, val):
        return self._update(1, self.n, 1, idx+1, val)

    def query(self, L, R):
        return self._query(1, self.n, 1, L+1, R+1)

    #线段树二分函数,同时负责更新
    def find_first_and_update(self,l,r,o,val):
        #区间内不存在>=x的数
        if self._tree[o]<val:
            return -1
        #找到目标值,将线段树对应值更新为-1,表示不能放水果
        if l==r:
            self._tree[o]=-1
            return l
        m=(l+r)>>1
        #先递归左子树
        i=self.find_first_and_update(l,m,o<<1,val)
        if i<0: #左子树没找到
            i=self.find_first_and_update(m+1,r,o<<1|1,val)
        self._maintain(o)
        return i

    def get(self, i):
        return self._query(1, self.n, 1, i+1, i+1)