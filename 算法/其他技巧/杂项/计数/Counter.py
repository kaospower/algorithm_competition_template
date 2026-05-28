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

# leetcode 2784(https://leetcode.cn/problems/check-if-array-is-good/description/)
#计数
class Solution:
    def isGood(self, nums: List[int]) -> bool:
        #注意题目中的n和数组长度的关系
        n=len(nums)-1
        cnt=[0]*(n+1)
        for x in nums:
            if x>n or x==n and cnt[x]>1 or x<n and cnt[x]>0:
                return False
            cnt[x]+=1
        return True