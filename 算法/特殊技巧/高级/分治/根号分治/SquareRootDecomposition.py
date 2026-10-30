from typing import List
from bisect import bisect_left, bisect_right
from heapq import heappush, heappop, heapreplace
from itertools import permutations, accumulate
from math import inf, comb, sqrt, ceil, floor, log, log2, log10
from functools import cache, reduce
from math import gcd, isqrt
from operator import and_,or_,xor,add,mul
from collections import defaultdict, deque, Counter

# from sortedcontainers import SortedList
# from itertools import pairwise

fmin = lambda x, y: x if x < y else y
fmax = lambda x, y: x if x > y else y

# leetcode 3655(https://leetcode.cn/problems/xor-after-range-multiplication-queries-ii/description/)
# 注意本题是根号分治和商分数组模版题

#根号分治,对于阈值两侧,采用不同算法,从而降低整体时间复杂度
#暴力求解的复杂度为O(qn/k),k越大,时间复杂度越低
#使用分组商分数组的复杂度为O(Kn+qlogM),K越小,时间复杂度越低(K为所有k的最大值)
#总计复杂度为两者之和O(kn+qn/k+qlogM),根据基本不等式,当kn=qn/k时,k=isqrt(q),此时时间复杂度取得最小值O(n*isqrt(q)+qlogM)
#因此当k小于isqrt(q)时,使用分组商分数组,当k>=isqrt(q)时,使用暴力求解
mod=1_000_000_007
class Solution:
    def xorAfterQueries(self, nums: List[int], queries: List[List[int]]) -> int:
        n=len(nums)
        B=isqrt(len(queries))
        #定义B个商分数组
        diff=[None]*B
        for l,r,k,v in queries:
            #k小于阈值,采用分组商分数组求解
            if k<B:
                #初始化商分数组
                if not diff[k]:
                    diff[k]=[1]*(n+k)
                #类似差分,商分将加法换成乘法,减法换成除法,由于本体需要取模,因此除法换成了乘上逆元
                diff[k][l]=diff[k][l]*v%mod
                #间隔k的商分右端点
                r=r-(r-l)%k+k
                #python独有的逆元写法,这种写法和费马小定理相比,即使mod不是素数也可以用
                diff[k][r]=diff[k][r]*pow(v,-1,mod)%mod
            else:
                #k大于阈值,暴力
                for i in range(l,r+1,k):
                    nums[i]=nums[i]*v%mod
        for k,d in enumerate(diff):
            if not d:
                continue
            #从商分数组中还原原来的数组,即求前缀积
            for i in range(k):
                mul=1
                for j in range(i,n,k):
                    mul=mul*d[j]%mod
                    nums[j]=nums[j]*mul%mod
        #求数组异或和
        return reduce(xor,nums)


