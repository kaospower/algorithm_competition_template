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

class Solution:
    def maxSumDivThree(self, nums: List[int]) -> int:
        #这里直接采用空间压缩的写法
        #f[0],f[1],f[2]分别表示模3余0,1,2的最大和
        #当和为0时,天然满足模3余0,初始化成0,余1,余2则不存在,初始化成-inf
        f=[0,-inf,-inf]
        #在更新f[0],f[1],f[2]时,直观想法是考虑其转移来源,但是这种写法需要讨论每一种余数的转移来源
        #如果题目改成模一个较大的数,详细讨论就会造成类似穷举的操作,因此这种方法不具有一般性
        #而换过来思考,对于一个刚遍历的数x,将其与0,1,2依次相加再模3,可以转移到未来的位置
        #因此如果用当前状态更新未来状态,即使本题改为模10这样较大的数,仍然可以用循环轻易解决,这便是刷表法
        for x in nums:
            #注意转移方程需要用到上一轮的状态,因此需要将上一轮状态拷贝一份,记作g
            g=f[:]
            for i in range(3):
                #由当前状态更新未来状态,刷表
                j=(i+x)%3
                f[j]=max(f[j],g[i]+x)
        return f[0]

#模版题:1262(https://leetcode.cn/problems/greatest-sum-divisible-by-three/description/)