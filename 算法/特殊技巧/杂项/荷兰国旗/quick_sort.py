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

# leetcode 75(https://leetcode.cn/problems/sort-colors/description/)
# 荷兰国旗算法(三路快排),注意三路快排是不稳定排序,会交换元素位置
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        l,r,pivot=0,len(nums)-1,1
        i=l
        while i<=r:
            if nums[i]==pivot:
                i+=1
            elif nums[i]<pivot:
                nums[l],nums[i]=nums[i],nums[l]
                l+=1
                i+=1
            else:
                nums[r],nums[i]=nums[i],nums[r]
                r-=1

