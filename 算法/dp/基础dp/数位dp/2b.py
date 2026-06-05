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

"""
上下界数位dp精简模版,只有isLimit参数,无论是否含有前导0,都可以处理,是更加通用的模版
"""
def g(num1, num2):
    low_s = list(map(int, str(num1)))
    high_s = list(map(int, str(num2)))
    n = len(high_s)
    diff_lh = n - len(low_s)

    @cache
    def f(i, limit_low, limit_high):
        if i == n:
            return 1
        lo = low_s[i - diff_lh] if limit_low and i >= diff_lh else 0
        hi = high_s[i] if limit_high else 9
        res = 0
        # is_num现在可以通过其他参数推断出来,如果不使用可以去掉
        is_num = not limit_low or i > diff_lh
        # 如果前导零不影响答案,该if分支可以去掉
        if not is_num:
            res = f(i + 1, True, True)
            lo = 1

        for d in range(lo, hi + 1):
            res += f(i + 1, limit_low and d == lo, limit_high and d == hi)
        return res

    return f(0, 0, 0, 0, True, True)



# 例:leetcode 3753(https://leetcode.cn/problems/total-waviness-of-numbers-in-range-ii/solutions/3839571/shang-xia-jie-shu-wei-dppythonjavacgo-by-74vp/)
# 数位dp2.1版本,时间复杂度O(D^2n^2),D=10
class Solution:
    def totalWaviness(self, num1: int, num2: int) -> int:
        low_s=list(map(int,str(num1)))
        high_s=list(map(int,str(num2)))
        n=len(high_s)
        diff_lh=n-len(low_s)
        # last_cmp表示i-1位和i-2位的关系,-1,0,1分别表示小于,等于,大于
        # last_digit表示i-1位填的数字
        @cache
        def f(i,s,last_cmp,last_digit,limit_low,limit_high):
            if i==n:
                return s
            lo=low_s[i-diff_lh] if limit_low and i>=diff_lh else 0
            hi=high_s[i] if limit_high else 9
            res=0
            is_num= not limit_low or i>diff_lh #之前填了数字或者现在在填次高位
            for d in range(lo,hi+1):
                c=(d>last_digit)-(d<last_digit) if is_num else 0
                #c*last_cmp<0表示形成了一个峰或者谷
                res+=f(i+1,s+(c*last_cmp<0),c,d,limit_low and d==lo,limit_high and d==hi)
            return res
        return f(0,0,0,0,True,True)
