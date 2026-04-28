from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache
from math import gcd, isqrt
from collections import defaultdict, deque, Counter

# from sortedcontainers import SortedList
from itertools import pairwise,chain

fmin = lambda x, y: x if x < y else y
fmax = lambda x, y: x if x > y else y

#中位数贪心
#中位数贪心用于求解以下问题:
#给定长为n的数组a,找到数x使其和a中每个数的绝对差之和最小
#结论是x为a的中位数,如果a长度为偶数,则两个中位数均可作为x,其最小距离是一样的
#证明链接(https://zhuanlan.zhihu.com/p/1922938031687595039)

#leetcode 2033(https://leetcode.cn/problems/minimum-operations-to-make-a-uni-value-grid/description/)
fmin=lambda x,y:x if x<y else y
class Solution:
    def minOperations(self, grid: List[List[int]], x: int) -> int:
        arr=sorted(chain.from_iterable(grid))
        n=len(arr)
        ans=0
        mid=arr[n//2]
        for y in arr:
            v=abs(y-mid)
            if v%x:
                return -1
            ans+=v//x
        return ans
