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

Max = lambda x, y: x if x > y else y
Min = lambda x, y: x if x < y else y

# 倍增
# 倍增本质是求<=x的最大位置,即至多是x的位置
# 当求>=x的最小位置时,可以等价转化成求<=x-1的最大位置,然后再+1,这里的+1是指再跳一步
# 二分本质是求>=x的最小位置,即至少是x的位置
# 因为只能通过跳的距离<=目标距离来判定是否越界,>目标距离时无法准确确定越界了多少,只能是缩小跳的步子
# 注意倍增和二分的内在联系,不要混淆

# 生成倍增数组,长度为n.bit_length()
# 由于倍增数组pa[x][i]表示x跳2^i可以到达的位置
# 因此上界即为log(n)下取整,即为n.bit_length()-1
# 从0~n.bit_length()-1,总计有n.bit_length()个数,因此二维数组内层长度是n.bit_length()

n, m, k = map(int, input().split())
nums = list(map(int, input().split()))
mx = n.bit_length()
pa = [[0] * mx for _ in range(n + 1)]
pa[n] = [n] * mx
# 前缀和
# pre = [0] * (n + 1)
# for i in range(n):
#     pre[i + 1] = pre[i] + nums[i]
pre = list(accumulate(nums, initial=0))
# 用二分初始化每个点向右跳不超过k的距离最多到哪
# 即从当前点延伸长度k(包含当前点)能到达的位置右边的位置
# 即我们实际用的是左闭右开区间[l,r)
for x in range(n):
    # pa[x][0]=bisect_left(pre,pre[x-1]+k+1)
    pa[x][0] = bisect_right(pre, pre[x] + k) - 1
# 初始化
for i in range(mx - 1):
    for x in range(n):
        p = pa[x][i]
        pa[x][i + 1] = pa[p][i]

for _ in range(m):
    l, r = map(int, input().split())
    # 题目下标从1开始,减1与实际下标对齐
    l -= 1
    r -= 1
    ans = 0
    # 我们目标是跳到r+1,即>=r+1的第一个位置,但是由于倍增只能找到<=x的第一个位置
    # 所以可以转化成查找<=r的第一个位置,然后再跳一步
    for i in range(mx - 1, -1, -1):
        if pa[l][i] <= r:
            ans += 1 << i
            l = pa[l][i]
    # 再跳一步如果能到r+1,就输出答案,+1代表最后跳的那步
    if pa[l][0] > r:
        print(ans + 1)
    else:
        print('Chtholly')

# 模版题:https://ac.nowcoder.com/acm/problem/15429
