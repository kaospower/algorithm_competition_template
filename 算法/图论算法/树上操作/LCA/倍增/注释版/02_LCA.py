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


# 带权树lca及求树上任意两点间距离
class LCA:
    # 如果下标从0开始,n初始化成边数+1,如果下标从1开始,n初始化成边数+2,root代表根节点编号
    def __init__(self, edges, n, root):
        m = n.bit_length()  # 可以跳2^0,2^1,...,2^(m-1),因此pa数组列长度为m
        g = [[] for _ in range(n)]
        for x, y, w in edges:
            g[x].append((y,w))
            g[y].append((x,w))
        depth = [0] * n
        dis = [0] * n
        pa = [[-1] * m for _ in range(n)]

        def f(u, fa):
            pa[u][0] = fa
            for v,w in g[u]:
                if v == fa: continue
                depth[v] = depth[u] + 1
                dis[v] = dis[u] + w
                f(v, u)

        f(root, -1)
        for i in range(m - 1):
            for x in range(n):
                if (t := pa[x][i]) != -1:
                    pa[x][i + 1] = pa[t][i]
        self.depth = depth
        self.dis = dis
        self.pa = pa
        self.m = m

    # 求node的第k个祖先
    def get_kth_ancestor(self, node, k):
        for i in range(k.bit_length()):
            if k >> i & 1:
                node = self.pa[node][i]
        return node

    # 求树上任意两点的lca
    def get_lca(self, x, y):
        # y为x,y中深度更大的节点,如果y的深度小于x,则交换两个节点
        if self.depth[x] > self.depth[y]:
            x, y = y, x
        # y向上跳到和x深度一样
        y = self.get_kth_ancestor(y, self.depth[y] - self.depth[x])
        # 如果此时y和x重合,说明x就是最近公共祖先
        if y == x:
            return x
        # px,py一起往上跳2^i步,从大到小依次尝试
        # 如果步子大了,跳过了lca,那么px和py会跳到相等的位置(可能是-1也可能是其他点)
        # 如果步子小了，则i-1,下一轮重复上述过程
        # 最终会到达离lca差一步的位置
        # 返回self.pa[x][0]即可,即从x再向上跳2^0=1步
        # 当px==py时,认为步子跳大了,选择更小的i,即在循环中,px,py永远不会等于lca,最多只会跳到离lca差一步的位置
        # 设x,y到lca距离为d,到lca下面一步的距离为d-1,由于任意数一定可以转化成二进制,而倍增每步跳的都对应二进制某位的值
        # 因此最终x,y一定可以跳到lca下面一步的位置
        # 而当x,y跳到>=lca的位置(即跳过了),px一定会等于py,这时不更新px,py,尝试更小的i,只有px!=py时,才更新x,y
        for i in range(self.m - 1, -1, -1):
            px, py = self.pa[x][i], self.pa[y][i]
            if px != py:
                x, y = px, py
        return self.pa[x][0]

    # #求树上任意两点间距离
    # x,y之间距离=根到x的距离+根到y的距离-2*根到lca(x,y)的距离
    def get_dis(self, x, y):
        return self.dis[x] + self.dis[y] - 2 * self.dis[self.get_lca(x, y)]
