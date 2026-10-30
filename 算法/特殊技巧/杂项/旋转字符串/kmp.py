from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd, isqrt
from collections import defaultdict, deque, Counter

# from sortedcontainers import SortedList
# from itertools import pairwise,chain

fmin = lambda x, y: x if x < y else y
fmax = lambda x, y: x if x > y else y

# leetcode 796(https://leetcode.cn/problems/rotate-string/description/)
# 旋转字符串本质是s的后缀和前缀拼接而成,因此其一定在s+s中
# 由此转化成字符串匹配问题,可以用暴力或者kmp解决

#解法1:暴力
class Solution:
    def rotateString(self, s: str, goal: str) -> bool:
        return len(s)==len(goal) and goal in s+s

#解法2:kmp
class Solution:
    def rotateString(self, s: str, goal: str) -> bool:
        def kmp(s, p):
            n, m = len(s), len(p)
            ne = [0] * m
            j = 0
            for i in range(1, m):
                while j and p[j] != p[i]:
                    j = ne[j - 1]
                if p[j] == p[i]:
                    j += 1
                ne[i] = j

            j = 0
            for i, v in enumerate(s):
                while j and p[j] != v:
                    j = ne[j - 1]
                if p[j] == v:
                    j += 1
                if j == m:
                    return i - m + 1
            return -1

        return len(s) == len(goal) and kmp(s + s, goal) != -1


