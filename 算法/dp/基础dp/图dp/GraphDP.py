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


# 图dp(拓扑排序+dp),核心思想是按拓扑顺序处理节点，把当前节点的结果传给它的后继。

#leetcode 2050(https://leetcode.cn/problems/parallel-courses-iii/description/)
#这道题的dp递推关系较为简单,更复杂的题目的dp递推关系会更复杂,需要先执行拓扑排序,再执行dp
class Solution:
    def minimumTime(self, n: int, relations: List[List[int]], time: List[int]) -> int:
        # 题目可以抽象成DAG(有向无环图),先做拓扑排序
        # 然后对于课程i,其完成所需时间可以用dp表示
        # 即fi=time[i]+max(fj1,fj2,...fjn),fi表示完成课程i所需最短时间,fjk表示i的前置课程
        # 即课程i时间加上完成它所有前置课程时间的最大值

        # 最终答案即为所有fi的最大值
        # 入度表
        deg = [0] * n
        # 建图
        g = [[] for _ in range(n)]
        for a, b in relations:
            deg[b - 1] += 1
            g[a - 1].append(b - 1)
        # 所有入度为0的点入队
        q = deque([i for i, x in enumerate(deg) if x == 0])
        # f表示第i门课程完成所需要的最短时间
        f = [0] * n
        while q:
            u = q.popleft()
            f[u] += time[u]
            for v in g[u]:
                f[v] = max(f[v], f[u])
                deg[v] -= 1
                if deg[v] == 0:
                    q.append(v)
        return max(f)