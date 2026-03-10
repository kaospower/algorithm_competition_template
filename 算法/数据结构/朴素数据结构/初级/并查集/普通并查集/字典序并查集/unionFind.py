# 字典序并查集(按秩合并时把字典序充当秩)
# 按秩合并时把字典序充当秩
from string import ascii_lowercase

class UnionFind:
    def __init__(self):
        self.father = {x: x for x in ascii_lowercase}

    def find(self, i):
        if i != self.father[i]:
            self.father[i] = self.find(self.father[i])
        return self.father[i]

    def union(self, x, y):
        fx, fy = self.find(x), self.find(y)
        if fx == fy:
            return False
        if fx > fy:
            self.father[fx] = fy
        else:
            self.father[fy] = fx
        return True
