from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache, reduce
from math import gcd, isqrt
from operator import and_, or_, xor, add, mul
from collections import defaultdict, deque, Counter

# from sortedcontainers import SortedList
# from itertools import pairwise

fmin = lambda x, y: x if x < y else y
fmax = lambda x, y: x if x > y else y

# 本文件主要收录一些看似像dp但是绝对不能用dp的题目
# 这些题目主要是违背了dp的前提条件,如无后效性,需要牢记
# 无后效性有两层含义:
# 1.某阶段的状态一旦确定,不再受此前各状态及决策的影响,即"未来与过去无关"
# 2.某阶段的状态一旦确定,不再受后续各状态及决策的影响,即"当前与未来无关"

# leetcode 1345(https://leetcode.cn/problems/jump-game-iv/description/)
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