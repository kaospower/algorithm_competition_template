"""
基于结构体+defaultdict实现,可以适应一些特殊情况
"""
from collections import defaultdict
class Trie:
    __slots__='son','end'

    def __init__(self):
        self.son=defaultdict(Trie)
        self.end=False
    def insert(self,word):
        p=self
        for c in word:
            p=p.son[c]
        p.end=True
    def search(self,word):
        p=self
        for c in word:
            if c not in p.son:
                return False
            p=p.son[c]
        return p.end