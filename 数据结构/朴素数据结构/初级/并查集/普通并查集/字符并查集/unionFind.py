# 字符并查集(用于处理字符类型)
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
        self.father[fx] = fy
        return True
