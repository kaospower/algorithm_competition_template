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


# 拓扑排序,注意拓扑排序只能用于有向图
def topo(edges, n):
    # 入度表
    indegree = [0] * n
    # 建图
    g = [[] for _ in range(n)]
    for x, y in edges:
        g[x].append(y)
        # 对于边有向边x->y,y的入度+1
        indegree[y] += 1
    # 找到所有入度为0的点,加入队列
    arr = [i for i, x in enumerate(indegree) if x == 0]
    q = deque(arr)
    # 保存点的拓扑序排列(注意可能不为1)
    res = []
    while q:
        # 弹出节点
        cur = q.popleft()
        # 加入答案数组
        res.append(cur)
        # 遍历点的子节点
        for x in g[cur]:
            # 去除点cur所连的所有边,即子节点入度-1
            indegree[x] -= 1
            # 如果子节点入度因此变成0,加入队列
            if indegree[x] == 0:
                q.append(x)
    # 如果答案数组长度不为n,说明无法找到拓扑序,即原有向图图有环
    if len(res) != n:
        return -1
