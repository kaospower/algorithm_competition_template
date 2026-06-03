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

# leetcode 1340(https://leetcode.cn/problems/jump-game-v/description/)
# 单调栈优化dp
class Solution:
    def maxJumps(self, arr: List[int], d: int) -> int:
        n = len(arr)
        right = [inf] * n
        st = []
        for i, t in enumerate(arr):
            while st and t > arr[st[-1]]:
                right[st.pop()] = i
            st.append(i)

        left = [-inf] * n
        st = []
        for i, t in enumerate(arr):
            while st and t >= arr[st[-1]]:
                st.pop()
            if st:
                left[i] = st[-1]
            st.append(i)

        @cache
        def f(i):
            ans = 1
            if i - left[i] <= d:
                ans = max(ans, f(left[i]) + 1)
            if right[i] - i <= d:
                ans = max(ans, f(right[i]) + 1)
            return ans

        return max(f(i) for i in range(n))