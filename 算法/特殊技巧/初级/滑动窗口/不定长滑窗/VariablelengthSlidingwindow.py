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

# leetcode 3(https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/)
#不定长滑动窗口
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        l=ans=0
        cnt=defaultdict(int)
        for r,x in enumerate(s):
            cnt[x]+=1
            while cnt[x]>1:
                cnt[s[l]]-=1
                l+=1
            ans=max(ans,r-l+1)
        return ans