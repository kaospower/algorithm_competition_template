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
from random import randint

fmax = lambda x, y: x if x > y else y
fmin = lambda x, y: x if x < y else y

#二分答案+滚动哈希
class Solution:
    def smallestUniqueSubarray(self, nums: list[int]) -> int:
        n = len(nums)
        # 多项式哈希（方便计算子数组哈希值）
        # 哈希函数 hash(nums) = nums[0] * base^(n-1) + nums[1] * base^(n-2) + ... + nums[n-2] * base + nums[n-1]
        MOD = 1_070_777_777
        BASE = randint(8 * 10 ** 8, 9 * 10 ** 8)  # 随机 BASE，防止 hack
        pow_base = [1] + [0] * n  # pow_base[i] = BASE^i
        pre_hash = [0] * (n + 1)  # 前缀哈希值 pre_hash[i] = hash(nums[:i])
        for i, x in enumerate(nums):
            pow_base[i + 1] = pow_base[i] * BASE % MOD
            pre_hash[i + 1] = (pre_hash[i] * BASE + x) % MOD  # 秦九韶算法计算多项式哈希

        # 计算子数组 nums[l:r] 的哈希值，注意这是左闭右开区间 [l,r)
        # 计算方法类似前缀和
        def sub_hash(l: int, r: int) -> int:
            return (pre_hash[r] - pre_hash[l] * pow_base[r - l]) % MOD

        def check(length: int) -> bool:
            cnt = defaultdict(int)
            for i in range(length, n + 1):
                h = sub_hash(i - length, i)
                cnt[h] += 1
            return 1 in cnt.values()

        # 开区间二分
        left, right = 0, n
        while left + 1 < right:
            mid = (left + right) // 2
            if check(mid):
                right = mid
            else:
                left = mid
        return right
