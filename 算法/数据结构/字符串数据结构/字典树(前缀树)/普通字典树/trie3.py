"""
基于结构体+哈希表实现,这种可以在结构体中记录额外信息,更灵活
"""
class Node:
    __slots__ = 'son', 'end'

    def __init__(self):
        self.son = {}
        self.end = False


class Trie:
    def __init__(self):
        self.root = Node()

    def insert(self, word):
        p = self.root
        for c in word:
            if c not in p.son:
                p.son[c] = Node()
            p = p.son[c]
        p.end = True

    def search(self, word):
        p = self.root
        for c in word:
            p = p.son[c]
            if not p.end:
                return False
        return p.end