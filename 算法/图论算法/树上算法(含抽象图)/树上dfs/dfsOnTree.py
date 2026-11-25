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

#3331(https://leetcode.cn/problems/find-subtree-sizes-after-changes/)
#树上dfs
class Solution:
    def findSubtreeSizes(self, parent: List[int], s: str) -> List[int]:
        n = len(parent)
        g = [[] for _ in range(n)]
        for i,p in enumerate(parent):
            if p!=-1:
                g[p].append(i)

        size = [1] * n
        ancestor = defaultdict(lambda: -1)
        def dfs(x: int) -> None:
            old = ancestor[s[x]]
            ancestor[s[x]] = x
            for y in g[x]:
                dfs(y)
                anc = ancestor[s[y]]
                size[x if anc < 0 else anc] += size[y]
            ancestor[s[x]] = old  # 恢复现场
        dfs(0)
        return size