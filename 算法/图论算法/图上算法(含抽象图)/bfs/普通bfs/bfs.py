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

#对于求最短路且在循环中直接return的,也可以使用无限迭代器写法,从而节省一个距离变量
class Solution:
    def minJumps(self, arr: List[int]) -> int:
        n=len(arr)
        d=defaultdict(list)
        for i,x in enumerate(arr):
            d[x].append(i)
        vis=[False]*n
        vis[0]=True
        q=deque([0])
        for dis in count(0):
            size=len(q)
            for _ in range(size):
                cur=q.popleft()
                if cur==n-1:
                    return dis
                if cur>0 and not vis[cur-1]:
                    q.append(cur-1)
                    vis[cur-1]=True
                if cur<n-1 and not vis[cur+1]:
                    q.append(cur+1)
                    vis[cur+1]=True
                v=arr[cur]
                if v not in d:
                    continue
                for i in d[v]:
                    if not vis[i]:
                        q.append(i)
                        vis[i]=True
                del d[v]

# bfs在数组上的应用
# 例:leetcode 1871(https://leetcode.cn/problems/jump-game-vii/description/)
class Solution:
    def canReach(self, s: str, minJump: int, maxJump: int) -> bool:
        if s[-1]=='1':
            return False
        n=len(s)
        farthest=0 #历史到达的最远位置
        q=deque([0])
        while q:
            size=len(q)
            for _ in range(size):
                cur=q.popleft()
                if cur==n-1:
                    return True
                for i in range(max(farthest+1,cur+minJump),min(cur+maxJump+1,len(s))):
                    if s[i]=='0':
                        q.append(i)
                farthest=i
        return False