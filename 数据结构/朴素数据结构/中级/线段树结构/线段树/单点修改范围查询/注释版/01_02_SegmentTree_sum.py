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

Max = lambda x, y: x if x > y else y
Min = lambda x, y: x if x < y else y

# 单点修改范围查询线段树,不包含建树,维护的信息是区间累加和
class SegmentTree:
    # 初始化线段树,nums为输入数组,线段树下标从1开始,nums下标从0开始
    # 如果原数组有n个元素,那么线段树需要容纳4n-1个元素,因为线段树下标从1开始,因此开长度为4n的数组刚好够
    def __init__(self, n):
        self.n = n
        self._tree = [0] * (self.n << 2)  # 4n空间

    # 单点修改时的情况,常见的有=(直接修改),+=(增量修改),max/min(和原来值比较求最大/最小),也可能是更复杂的逻辑,需要根据实际情况修改
    def _alter(self, o, val):
        self._tree[o] = val

    # 合并答案,常见的有+,-,max,min,也可能是更复杂的逻辑,需要根据实际情况修改
    def _merge(self, a, b):
        return a + b

    # 合并左右子树
    def _maintain(self, o):
        self._tree[o] = self._merge(self._tree[o << 1], self._tree[o << 1 | 1])

    # 单点修改函数,将下标为idx的元素的值修改为val
    def _update(self, l, r, o, idx, val):  # o表示当前节点
        # 递归边界
        if l == r:
            self._alter(o, val)
            return
        mid = (l + r) >> 1
        if idx <= mid:
            # 在左子树
            self._update(l, mid, o << 1, idx, val)  # o<<1表示左孩子
        else:
            # 在右子树
            self._update(mid + 1, r, o << 1 | 1, idx, val)
        self._maintain(o)

    # 区间查询函数,L,R为待查询区间,l,r为当前区间
    def _query(self, l, r, o, L, R):
        # 区间越界的情况,如果
        if R < L:
            return 0
        # 待查询区间全包含当前区间,即当前区间[l,r]是待查询区间[l,r]的子集
        if L <= l and r <= R:
            return self._tree[o]

        mid = (l + r) >> 1

        # 三分支写法,注意和下面双分支写法的区别,为了方便维护,推荐使用三分支写法

        # 中点在待查询区间右侧,待查询区间落在左子树
        if mid >= R:
            return self._query(l, mid, o << 1, L, R)
        # 中点在待查询区间左侧, 待查询区间落在右子树
        if mid < L:
            return self._query(mid + 1, r, o << 1 | 1, L, R)
        l_res = self._query(l, mid, o << 1, L, R)
        r_res = self._query(mid + 1, r, o << 1 | 1, L, R)
        return self._merge(l_res, r_res)

        # 双分支写法
        # s = 0
        # if mid >= L:
        #     s += self._query(l, mid, o << 1, L, R)
        # if mid < R:
        #     s += self._query(mid + 1, r, o << 1 | 1, L, R)
        # return s

    # 真正的范围修改函数,减少参数
    def update(self, idx, val):
        return self._update(1, self.n, 1, idx+1, val)

    # 真正的范围查询函数,减少参数
    def query(self, L, R):
        return self._query(1, self.n, 1, L+1, R+1)

    # 真正的单点查询函数,减少参数
    def get(self, i):
        return self._query(1, self.n, 1, i+1, i+1)