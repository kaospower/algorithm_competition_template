from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd, isqrt
from collections import defaultdict, deque, Counter

# from sortedcontainers import SortedList
# from itertools import pairwise

#约束划分个数为k,记忆化搜索模版
#模版题:3599(https://leetcode.cn/problems/partition-array-to-minimize-xor/description/)
fmax = lambda x, y: x if x > y else y
fmin = lambda x, y: x if x < y else y
def solve(nums, k):
    @cache
    def f(i, k):
        #当k==0时,如果i来到越界位置,返回0,否则返回inf代表不合法
        if k == 0:
            return 0 if i == -1 else inf
        s = 0
        ans = inf
        # j表示当前划分的开头位置
        # j左边必须留出k-1个位置给剩下k-1个划分,下标为0~k-2
        # 因此剩下划分的结尾位置为k-2,而j的最左下标刚好是k-2+1=k-1
        # 由于python是左闭右开区间,因此范围写成range(i,k-2,-1)
        # 当k>=1时,k-1>=0,j的下标不会越界
        # 注意到0~i总共有i+1个数,最多划分成i+1段
        # 如果k>i+1,即i<k-1时,range(i, k - 2, -1)为空区间,会返回ans默认值inf,即不合法
        for j in range(i, k - 2, -1):
            s ^= nums[j]
            ans = fmin(ans, fmax(s, f(j - 1, k - 1)))
        return ans

    ans = f(len(nums) - 1, k)
    f.cache_clear()
    return ans



