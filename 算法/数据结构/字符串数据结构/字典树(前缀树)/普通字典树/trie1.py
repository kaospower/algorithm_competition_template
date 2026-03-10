"""
基于defaultdict实现,这种写法最简洁
"""
from collections import  defaultdict
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