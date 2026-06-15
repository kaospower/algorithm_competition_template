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

# 重要结论列表

# 1.从n条边中取奇数条边,有2^(n-1)种取法
# 证明:去除一个边,剩下的边每个可以选或不选,如果剩下边数量是奇数,那么当前这个边就不要了,否则,就要这条边
# 从而保证边数总数为奇数个,两种情况都是2^(n-1)种取法,从而证明了结论
