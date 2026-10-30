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


#博弈dp
# leetcode 486(https://leetcode.cn/problems/predict-the-winner/)
#直接考虑谁胜(谁得分多)不好处理,但是考虑两人分数之差的正负就简单了
class Solution:
    def predictTheWinner(self, nums: list[int]) -> bool:
        #当前剩余数组为nums[l...r]时,先手得分-后手得分的最大值
        @cache
        def f(l,r):
            #注意玩家1取了一个数之后,玩家2变成了先手
            #比如玩家1取了nums[l],f(l+1,r)表示的是玩家2-玩家1的最大值
            #对于玩家1来说,需要将这个差值取反
            if l==r:
                return nums[l]
            return max(nums[l]-f(l+1,r),nums[r]-f(l,r-1))
        return f(0,len(nums)-1)>=0
