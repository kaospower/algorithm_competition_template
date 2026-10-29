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


# leetcode 1024(https://leetcode.cn/problems/video-stitching/)
class Solution:
    def videoStitching(self, clips: List[List[int]], time: int) -> int:
        mx = [0] * time
        for l, r in clips:
            if l < time:
                mx[l] = max(mx[l], r)
        last = pre = ans = 0
        for i in range(time):
            last = max(last, mx[i])
            if i == last:
                return -1
            if i == pre:
                pre = last
                ans += 1
        return ans


