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

# Leetcode 3043(https://leetcode.cn/problems/find-the-length-of-the-longest-common-prefix/description/)
# 转成字符串后枚举长度从而实现遍历前缀
class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        s=set()
        for x in map(str,arr1):
            for i in range(1,len(x)+1):
                s.add(x[:i])
        ans=0
        for x in map(str,arr2):
            for i in range(1,len(x)+1):
                if x[:i] not in s:
                    break
                ans=max(ans,i)
        return ans

# 不转字符串的写法
class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        s=set()
        for x in arr1:
            while x and x not in s:
                s.add(x)
                x//=10
        ans=0
        for x in arr2:
            while x and x not in s:
                x//=10
            ans=max(ans,x)
        return len(str(ans)) if ans else 0