"""
基于defaultdict实现,这种写法最简洁
"""
from collections import  defaultdict
from typing import List
trie=lambda:defaultdict(trie)
class Trie:
    def __init__(self):
        self.root=trie()

    def insert(self,word):
        p=self.root
        for c in word:
            p=p[c]
        p['#']=True

    def search(self, word: str) -> bool:
        p=self.root
        for c in word:
            if c not in p:
                return False
            p=p[c]
        return p.get('#')


# 例:Leetcode 3043()
# 引入了一个变量cnt记录最长前缀(https://leetcode.cn/problems/find-the-length-of-the-longest-common-prefix/description/)
class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        t=Trie()
        for x in map(str,arr1):
            t.insert(x)
        ans=0
        for x in map(str,arr2):
            ans=max(ans,t.search(x))
        return ans

trie=lambda:defaultdict(trie)
class Trie:
    def __init__(self):
        self.root=trie()

    def insert(self,word):
        p=self.root
        for c in word:
            p=p[c]
        p['#']=True

    def search(self, word: str) -> bool:
        p=self.root
        cnt=0
        for c in word:
            if c not in p:
                break
            p=p[c]
            cnt+=1
        return cnt