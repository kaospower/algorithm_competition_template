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

# leetcode 1456(https://leetcode.cn/problems/maximum-number-of-vowels-in-a-substring-of-given-length/)
# 定长滑动窗口
class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        ans=l=cnt=0
        for r,x in enumerate(s):
            #进
            cnt+=x in 'aeiou'
            if r-l+1<k:
                continue

            #更新答案
            ans=max(ans,cnt)

            #出
            out=s[l]
            cnt-=out in 'aeiou'
            l+=1
        return ans
