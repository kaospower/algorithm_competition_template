from typing import List,Optional
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

# leetcode 61(https://leetcode.cn/problems/rotate-list/)
fmin = lambda x, y: x if x < y else y
fmax = lambda x, y: x if x > y else y

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

#首尾相连再断开
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if head is None:
            return head
        p=head
        length=1
        while p.next:
            p=p.next
            length+=1
        k%=length

        p.next=head #首尾相连
        p=head
        for i in range(length-k-1):
            p=p.next
        head=p.next
        p.next=None
        return head
