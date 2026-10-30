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

# 数学写法
class Solution:
    def separateDigits(self, nums: List[int]) -> List[int]:
        ans=[]
        # 由于数学提取只能从低位到高位,因此需要将数组反转,从最后一个开始遍历
        for x in reversed(nums):
            while x:
                ans.append(x%10)
                x//=10
        ans.reverse()
        return ans