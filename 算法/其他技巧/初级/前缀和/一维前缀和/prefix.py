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
# 差分数组
# 利用差分数组标记可以访问到的位置,然后同时求前缀和,前缀和表示实际访问次数,如果某位置次数>0就说明可以访问到
class Solution:
    def canReach(self, s: str, minJump: int, maxJump: int) -> bool:
        n=len(s)
        diff=[0]*(n+1)
        diff[0]=1
        diff[1]=-1
        sum_=0
        for i,x in enumerate(s):
            sum_+=diff[i]
            if sum_>0 and x=='0':
                diff[min(i+minJump,n)]+=1
                diff[min(i+maxJump+1,n)]-=1
        return sum_>0 and s[-1]=='0'
