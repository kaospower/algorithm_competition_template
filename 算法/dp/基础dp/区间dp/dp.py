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

#模版题:leetcode 516(https://leetcode.cn/problems/longest-palindromic-subsequence/description/)
#记忆化搜索
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        @cache
        def f(i,j):
            if i==j:
                return 1
            if i>j:
                return 0
            if s[i]==s[j]:
                return f(i+1,j-1)+2
            return max(f(i+1,j),f(i,j-1))
        return f(0,len(s)-1)

#迭代
#区间dp经典写法,外层循环倒序枚举,内层循环正序枚举
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        f = [[0] * n for _ in range(n)]
        for i in range(n - 1, -1, -1):
            f[i][i] = 1
            for j in range(i + 1, n):  # i=n-1时不会走第二层循环,因此不存在下标越界问题
                if s[i] == s[j]:
                    f[i][j] = f[i + 1][j - 1] + 2
                else:
                    f[i][j] = max(f[i + 1][j], f[i][j - 1])
        return f[0][-1]

