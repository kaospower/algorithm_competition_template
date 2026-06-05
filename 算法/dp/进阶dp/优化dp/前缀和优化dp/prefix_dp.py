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

# leetcode 1871(https://leetcode.cn/problems/jump-game-vii/description/)
#前缀和优化dp
class Solution:
    def canReach(self, s: str, minJump: int, maxJump: int) -> bool:
        n=len(s)
        pre=[0]*(n+1)
        pre[1]=1
        for i in range(1,n):
            f=s[i]=='0' and i-minJump>=0 and pre[i-minJump+1]-pre[max(0,i-maxJump)]>0
            pre[i+1]=pre[i]+f
        return f