from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd, isqrt
from collections import defaultdict, deque, Counter

# from sortedcontainers import SortedList
# from itertools import pairwise

Max = lambda x, y: x if x > y else y
Min = lambda x, y: x if x < y else y

#无权最短路
def bfs(start,edges,n):
    g=[[] for _ in range(n)]
    for x,y in edges:
        g[x].append(y)
        g[y].append(x)
    dis=[inf]*n
    dis[start]=0
    q=deque([start])
    level=1
    while q:
        size=len(q)
        for _ in range(size):
            u=q.popleft()
            for v in g[u]:
                if level<dis[v]:
                    q.append(v)
                    dis[v]=level
        level+=1
    return dis

# 例:leetcode 1345(https://leetcode.cn/problems/jump-game-iv/description/)
# 本题状态转移不具有无后效性,不能用dp,本质是图上bfs求最短路
class Solution:
    def minJumps(self, arr: List[int]) -> int:
        n = len(arr)
        d = defaultdict(list)
        for i, x in enumerate(arr):
            d[x].append(i)
        vis = [False] * n
        vis[0] = True
        q = deque([0])
        dis = 0
        while q:
            size = len(q)
            for _ in range(size):
                cur = q.popleft()
                if cur == n - 1:
                    return dis
                if cur > 0 and not vis[cur - 1]:
                    q.append(cur - 1)
                    vis[cur - 1] = True
                if cur < n - 1 and not vis[cur + 1]:
                    q.append(cur + 1)
                    vis[cur + 1] = True
                v = arr[cur]
                if v not in d:
                    continue
                for i in d[v]:
                    if not vis[i]:
                        q.append(i)
                        vis[i] = True
                del d[v]

            dis += 1

