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

fmax = lambda x, y: x if x > y else y
fmin = lambda x, y: x if x < y else y

#位图优化逻辑运算型01背包
#返回值表示能否恰后装满体积为s的背包
#模版题:416(https://leetcode.cn/problems/partition-equal-subset-sum/)
def f(nums):
    s=sum(nums)
    f = 1  # 用一个二进制数代替原来的布尔数组,同时优化空间和时间
    u = (1 << (s + 1)) - 1
    # 逻辑运算型背包问题本质是看能否恰好装满容积为j的背包
    # 在这些背包中增加体积为x的物品,就会可以恰好装满体积更大的背包
    # 因此将原来的f左移x位,然后和原来的f做或运算,就可以实现迭代
    for x in nums:
        # 背包容量最多为s,和长度为s+1的全1二进制数做与运算可以限制物品体积在s之内
        # 因为如果背包物品体积超过s是无意义的,且过长的二进制数会降低运算效率
        f |= f << x & u
    return f >> s & 1 == 1

