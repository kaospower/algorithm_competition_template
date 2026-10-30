from typing import List, Optional
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd, isqrt
from collections import defaultdict, deque, Counter

# from sortedcontainers import SortedList
# from itertools import pairwise

fmax = lambda x, y: x if x > y else y
fmin = lambda x, y: x if x < y else y

#leetcode 3970(https://leetcode.cn/problems/shortest-path-with-at-most-k-consecutive-identical-characters/)
#分层图Dijkstra
class Solution:
    def shortestPath(self, n: int, edges: list[list[int]], labels: str, k: int) -> int:
        g = [[] for _ in range(n)]
        for x, y, w in edges:
            g[x].append((y, w))

        dis = [[inf] * (k + 1) for _ in range(n)]
        h = [(0, 0, 1)]  # (最短路长度, 节点编号, 最后连续相同字母个数)

        while h:
            d, x, cnt = heappop(h)
            if x == n - 1:
                return d
            if d > dis[x][cnt]:
                continue
            for y, w in g[x]:
                new_cnt = 1 if labels[y] != labels[x] else cnt + 1
                new_dis = d + w
                if new_cnt <= k and new_dis < dis[y][new_cnt]:
                    dis[y][new_cnt] = new_dis
                    heappush(h, (new_dis, y, new_cnt))

        return -1
