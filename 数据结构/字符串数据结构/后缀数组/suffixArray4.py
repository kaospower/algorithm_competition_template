# 倍增法+基数排序求后缀数组,时间复杂度O(nlogn)
class SuffixArray:
    def __init__(self, s):
        self.s = [0] + list(map(ord, s))
        self.n = len(s)
        self.m = 122
        self.N = max(self.n + 5, self.m + 5)
        self.x = [0] * self.N
        self.y = [0] * self.N
        self.c = [0] * self.N
        self.sa = [0] * self.N
        self.rk = [0] * self.N
        self.height = [0] * self.N

    # 下标从1开始
    def get_sa(self):
        for i in range(1, self.n + 1):
            self.x[i] = self.s[i];
            self.c[self.x[i]] += 1
        for i in range(2, self.m + 1):
            self.c[i] += self.c[i - 1]
        for i in range(self.n, 0, -1):
            self.sa[self.c[self.x[i]]] = i;
            self.c[self.x[i]] -= 1

        k = 1
        while k <= self.n:
            num = 0
            for i in range(self.n - k + 1, self.n + 1):
                num += 1;
                self.y[num] = i
            for i in range(1, self.n + 1):
                if self.sa[i] > k:
                    num += 1;
                    self.y[num] = self.sa[i] - k
            for i in range(1, self.m + 1):
                self.c[i] = 0
            for i in range(1, self.n + 1): self.c[self.x[i]] += 1
            for i in range(2, self.m + 1): self.c[i] += self.c[i - 1]
            for i in range(self.n, 0, -1): self.sa[self.c[self.x[self.y[i]]]] = self.y[i];self.c[
                self.x[self.y[i]]] -= 1;self.y[i] = 0
            self.x, self.y = self.y, self.x
            self.x[self.sa[1]] = 1;
            num = 1

            for i in range(2, self.n + 1):
                if self.y[self.sa[i]] == self.y[self.sa[i - 1]] and self.y[self.sa[i] + k] == self.y[
                    self.sa[i - 1] + k]:
                    self.x[self.sa[i]] = num
                else:
                    num += 1
                    self.x[self.sa[i]] = num

            if num == self.n: break
            self.m = num

            k <<= 1
        return self.sa

    def get_height(self):
        for i in range(1, self.n + 1):
            self.rk[self.sa[i]] = i
        k = 0
        for i in range(1, self.n + 1):
            if self.rk[i] == 1:
                continue
            if k: k -= 1
            j = self.sa[self.rk[i] - 1]
            while i + k <= self.n and j + k <= self.n and self.s[i + k] == self.s[j + k]:
                k += 1
            self.height[self.rk[i]] = k
        return self.height
