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

# leetcode 2540(https://leetcode.cn/problems/minimum-common-value/description/)
# 双指针
class Solution:
    def getCommon(self, nums1: List[int], nums2: List[int]) -> int:
        i=j=0
        while i<len(nums1) and j<len(nums2):
            if nums1[i]==nums2[j]:
                return nums1[i]
            elif nums1[i]<nums2[j]:
                i+=1
            else:
                j+=1
        return -1

# leetcode 1871(https://leetcode.cn/problems/jump-game-vii/description/)
# 同向双指针
class Solution:
    def canReach(self, s: str, minJump: int, maxJump: int) -> bool:
        n=len(s)
        canReach=[False]*n #canReach记录可以访问到的下标
        canReach[0]=True
        j=1 #j记录还没访问的下标
        for i,x in enumerate(s):
            if x=='0' and canReach[i]:
                mx=min(i+maxJump,n-1)
                j=max(j,i+minJump)
                while j<=mx:
                    canReach[j]=True
                    j+=1
                if j==n:
                    break #终点可以抵达,只需验证其是否为0即可
        return s[-1]=='0' and canReach[-1]
