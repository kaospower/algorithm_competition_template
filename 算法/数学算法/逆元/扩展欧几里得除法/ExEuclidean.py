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

fmin = lambda x, y: x if x < y else y
fmax = lambda x, y: x if x > y else y

#leetcode2906题解(https://leetcode.cn/problems/construct-product-matrix/solutions/2501258/qian-zhui-hou-zhui-ni-yuan-zou-zui-nan-d-8ukw)
# 扩展欧几里得算法
# 质数过滤，剔除 num 中的指定质因数，并返回数据：(剔除的质数的数量列表, 最后剩余数值)
def primeFilter(primes: List[int], num: int) -> List[int]:
    res = []
    for prime in primes:
        cnt = 0
        while num % prime == 0:
            num //= prime
            cnt += 1
        res.append(cnt)
    res.append(num)
    return res


# 欧几里得逆元，线性同余，需要保证 gcd(num, mod) == 1才能使用，不然将进入错误分支环节
def mulInverse(num: int, mod: int) -> int:
    res = ex_gcd(num, mod)
    # if res[0] != 1: return -1 # 判断错误分支
    return (res[1] + mod) % mod


def ex_gcd(a: int, b: int) -> List[int]:
    if b == 0: return [a, 1, 0]
    res = ex_gcd(b, a % b)
    return [res[0], res[2], res[1] - (a // b) * res[2]]


class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        MOD = 12345
        n, m = len(grid), len(grid[0])

        primes = [3, 5, 823]
        mulAllPFCnt = [0, 0, 0]

        mulAll = 1

        for y in range(n):
            for x in range(m):
                num = grid[y][x]
                numPFCnt = primeFilter(primes, num)
                mulAll = (mulAll * numPFCnt[-1]) % MOD
                for i in range(len(primes)): mulAllPFCnt[i] += numPFCnt[i]

        for y in range(n):
            for x in range(m):
                num = grid[y][x]
                numPFCnt = primeFilter(primes, num)
                if 0 in [mulAllPFCnt[i] - numPFCnt[i] for i in range(len(primes))]:
                    res = mulAll * mulInverse(numPFCnt[-1], MOD) % MOD
                    for i, p in enumerate(primes):
                        res = res * pow(p, mulAllPFCnt[i] - numPFCnt[i], MOD) % MOD
                    grid[y][x] = res
                else:
                    grid[y][x] = 0

        return grid

