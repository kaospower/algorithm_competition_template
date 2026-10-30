from string import digits
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

#leetcode 3723(https://leetcode.cn/problems/maximize-sum-of-squares-of-digits/)
#字符串贪心
class Solution:
    def maxSumOfSquares(self, n: int, s: int) -> str:
        if n*9<s:
            return ''
        ans='9'*(s//9)
        if s%9:
            ans+=digits[s%9]
        return ans+'0'*(n-len(ans))
