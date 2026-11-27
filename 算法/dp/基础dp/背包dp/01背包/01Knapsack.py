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


#01背包空间优化
#返回值表示能否恰后装满体积为s的背包
#模版题:416(https://leetcode.cn/problems/partition-equal-subset-sum/)
def f(nums):
    s=sum(nums)
    n=len(nums)
    #长度为s+1的数组,代表背包容量从0~s
    f=[True]+[False]*s
    for i,x in enumerate(nums):
        #01背包空间优化时,需要倒序枚举体积
        for j in range(s,x-1,-1):
            f[j]|=f[j-x]
        if f[s]:
            return True
    return False


# 01背包+贪心
# 先只考虑每个物品只购买一件的情况,应用01背包求出可以购买的最多物品数,然后将剩下的钱全买最便宜的物品
# 模版题:3946(https://leetcode.cn/problems/maximum-number-of-items-from-sale-i/)
class Solution:
    def maximumSaleItems(self, items: list[list[int]], budget: int) -> int:
        f=[0]*(budget+1)
        mn=inf
        for fi,p in items:
            mn=min(mn,p)
            cnt=0
            for fj,_ in items:
                if fj%fi==0:
                    cnt+=1
            for j in range(budget,p-1,-1):
                f[j]=max(f[j],f[j-p]+cnt)
        return max(fi+(budget-i)//mn for i,fi in enumerate(f))