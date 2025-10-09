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

# 无权树lca及求树上任意两点间距离
class LCA:
    # 如果下标从0开始,n初始化成边数+1,如果下标从1开始,n初始化成边数+2,root代表根节点编号
    def __init__(self, edges, n, root):
        m = n.bit_length()
        g = [[] for _ in range(n)]
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)
        depth = [0] * n
        pa = [[-1] * m for _ in range(n)]

        def f(u, fa):
            pa[u][0] = fa
            for v in g[u]:
                if v == fa: continue
                depth[v] = depth[u] + 1
                f(v, u)

        f(root, -1)
        for i in range(m - 1):
            for x in range(n):
                if (t := pa[x][i]) != -1:
                    pa[x][i + 1] = pa[t][i]
        self.depth = depth
        self.pa = pa
        self.m = m

    def get_kth_ancestor(self, node, k):
        for i in range(k.bit_length()):
            if k >> i & 1:
                node = self.pa[node][i]
        return node

    def get_lca(self, x, y):
        if self.depth[x] > self.depth[y]:
            x, y = y, x
        y = self.get_kth_ancestor(y, self.depth[y] - self.depth[x])
        if y == x:
            return x
        for i in range(self.m - 1, -1, -1):
            px, py = self.pa[x][i], self.pa[y][i]
            if px != py:
                x, y = px, py
        return self.pa[x][0]

    def get_dis(self, x, y):
        # 同时返回lca,距离
        # t=self.get_lca(x, y)
        # return t,self.depth[x] + self.depth[y] - 2 * self.depth[t]
        # 只返回距离
        return self.depth[x] + self.depth[y] - 2 * self.depth[self.get_lca(x, y)]


