from typing import List, Optional
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd, isqrt
from collections import defaultdict, deque, Counter

from sortedcontainers import SortedList
# from itertools import pairwise

fmax = lambda x, y: x if x > y else y
fmin = lambda x, y: x if x < y else y


# leetcode 3739(https://leetcode.cn/problems/count-subarrays-with-majority-element-ii/)

# 把target视作1,其他数视为-1,本题转化成计算元素和严格大于0的子数组个数,即525题
# 子段和用前缀和之差维护,枚举右,维护左
class Solution:
    def countMajoritySubarrays(self, nums: List[int], target: int) -> int:
        d=SortedList([0]) #先将空数组前缀和0加入有序表
        ans=s=0
        for x in nums:
            s+=1 if x==target else -1
            ans+=d.bisect_left(s)
            d.add(s)
        return ans


# 本题相邻前缀和变化只有1,可以根据此性质进行优化
class Solution:
    def countMajoritySubarrays(self, nums: List[int], target: int) -> int:
        cnt = defaultdict(int)
        cnt[0] = 1
        # f[j]表示满足i<j且s[i]<s[j]的i的个数
        ans = s = f = 0
        for x in nums:
            if x == target:
                f += cnt[s]
                s += 1
            else:
                s -= 1
                f -= cnt[s]
            ans += f
            cnt[s] += 1
        return ans





