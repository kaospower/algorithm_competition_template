"""
基于数组实现,这种构建的字典树原始就满足字典序,处理字典序相关题目,更简单
"""
class Node:
    __slots__='son','end'

    def __init__(self):
        self.son=[None] * 26
        self.end=False

class Trie:
    def __init__(self):
        self.root = Node()

    def insert(self,word):
        p=self.root
        for c in word:
            c=ord(c)-ord('a')
            if not p.son[c]:
                p.son[c]=Node()
            p=p.son[c]
        p.end=True

    def search(self,word):
        p=self.root
        for c in word:
            p=p.son[ord(c)-ord('a')]
            if not p.end:
                return False
        return True