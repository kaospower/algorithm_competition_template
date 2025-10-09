#利用后缀数组求最长公共子串LCS
class Solution:
    def LCS(self, s1, s2):
        sf = SuffixArray()
        combined = s1 + '#' + s2  # 分隔符的ascii码值只要大于哨兵字符即可
        sa, rk, h = sf.SAIS_sa_rk_h(combined)  # 这里的sa数组和高度数组h大小一样,h[i]表示sa[i]和sa[i-1]的LCP(最长公共前缀)
        mx = start = 0
        m, n = len(s1), len(sa)
        for i in range(1, n):
            # 确保两个后缀来自不同的字符串
            if (sa[i] < m) != (sa[i - 1] < m):  # 这里必须加括号,因为<和!=优先级相同
                if h[i] > mx:
                    mx = h[i]
                    start = sa[i - 1]
        return combined[start:start + mx]



# 模版:sa-is后缀数组求最长公共子串(LCS),时间复杂度O(n)
from array import array

L_TYPE = ord('L')
S_TYPE = ord('S')


class SuffixArray:
    def is_lms(self, i, t):
        return t[i] == S_TYPE and t[i - 1] == L_TYPE

    def classify(self, text, n):
        t = bytearray(n)
        t[-1] = S_TYPE
        for i in range(n - 2, -1, -1):
            if text[i] == text[i + 1]:
                t[i] = t[i + 1]
            else:
                if text[i] > text[i + 1]:
                    t[i] = L_TYPE
                else:
                    t[i] = S_TYPE
        return t

    def find_lms_suffixes(self, t, n):
        pos = array('l')
        for i in range(n):
            if t[i] == S_TYPE and t[i - 1] == L_TYPE:
                pos.append(i)
        return pos

    def buckets(self, text, sigma):
        alpha = []
        bucket_sizes = array('L', [0] * sigma)
        for c in text:
            bucket_sizes[c] += 1
        for i in range(sigma):
            if bucket_sizes[i] != 0:
                alpha.append(i)

        return alpha, bucket_sizes

    def bucket_intervals(self, alpha, bucket_sizes, sigma):
        heads = array('l', [0] * sigma)
        tails = array('l', [0] * sigma)
        j = 0
        for i in range(len(alpha)):
            heads[alpha[i]] = j
            j += bucket_sizes[alpha[i]]
            tails[alpha[i]] = j - 1

        return heads, tails

    def induced_sorting(self, lms, tails, heads, SA, type_suffix, text, n, m, alpha, bucket_sizes, sigma):
        for i in range(m - 1, -1, -1):
            nfs = tails[text[lms[i]]]
            SA[nfs] = lms[i]
            tails[text[lms[i]]] -= 1

        for i in range(n):
            if SA[i] > 0 and type_suffix[SA[i] - 1] == L_TYPE:
                nfs = heads[text[SA[i] - 1]]
                SA[nfs] = SA[i] - 1
                heads[text[SA[i] - 1]] += 1

        heads, tails = self.bucket_intervals(alpha, bucket_sizes, sigma)

        for i in range(n - 1, -1, -1):
            if SA[i] > 0 and type_suffix[SA[i] - 1] == S_TYPE:
                nfs = tails[text[SA[i] - 1]]
                SA[nfs] = SA[i] - 1
                tails[text[SA[i] - 1]] -= 1

    def blocks_are_equal(self, i, j, types, text, n):
        while i < n and j < n:
            if text[i] == text[j]:
                if self.is_lms(i, types) and self.is_lms(j, types):
                    return True
                else:
                    i += 1
                    j += 1
            else:
                return False
        return False

    def get_reduced_substring(self, types, SA, lms, ordered_lms, text, n, m):
        j = 0
        for i in range(n):
            if self.is_lms(SA[i], types):
                ordered_lms[j] = SA[i]
                j += 1

        pIS = array('l', [0] * m)
        k, i = 1, 1
        pIS[0] = 0
        for i in range(1, m):
            if text[ordered_lms[i]] == text[ordered_lms[i - 1]] and \
                    self.blocks_are_equal(ordered_lms[i] + 1, ordered_lms[i - 1] + 1, types, text, n):
                pIS[i] = pIS[i - 1]
            else:
                pIS[i] = k
                k += 1

        inverse_lms = array('l', [0] * n)
        for i in range(m):
            inverse_lms[ordered_lms[i]] = pIS[i]
        for i in range(m):
            pIS[i] = inverse_lms[lms[i]]

        return pIS, k == m, k + 1

    def construct_suffix_array(self, T, SA, n, sigma):
        if len(T) == 1:
            SA[0] = 0
            return SA

        t = self.classify(T, n)
        lms = self.find_lms_suffixes(t, n)
        m = len(lms)

        alpha, sizes = self.buckets(T, sigma)
        heads, tails = self.bucket_intervals(alpha, sizes, sigma)
        self.induced_sorting(lms, tails, heads, SA, t, T, n, m, alpha, sizes, sigma)  # first induced sort

        ordered_lms = array('L', [0] * len(lms))

        reduced_text, blocks_unique, sigma_reduced = self.get_reduced_substring(t, SA, lms, ordered_lms, T, n, m)
        reduced_SA = array('l', [-1] * m)
        if blocks_unique:
            for i in range(m):
                reduced_SA[reduced_text[i]] = i
        else:
            self.construct_suffix_array(reduced_text, reduced_SA, m, sigma_reduced)

        for i in range(m):
            ordered_lms[i] = lms[reduced_SA[i]]

        heads, tails = self.bucket_intervals(alpha, sizes, sigma)  # reset bucket tails and heads
        for i in range(n):  SA[i] = 0
        self.induced_sorting(ordered_lms, tails, heads, SA, t, T, n, m, alpha, sizes, sigma)

    def bwt(self, T, SA: array, BWT: bytearray, n: int):
        for i in range(n):
            BWT[i] = T[SA[i] - 1]

    def SAIS_sa(self, text):
        text += ' '  # 哨兵字符必须小于字符串中所有字符,这里我们用空字符(ascii码值32)作为哨兵字符
        text = [ord(c) for c in text]
        sigma = max(text) + 1
        n = len(text)
        SA = array('l', [-1] * n)
        self.construct_suffix_array(text, SA, n, sigma)
        bt = bytearray(n)
        self.bwt(text, SA, bt, n)
        return SA.tolist()[1:]

    def SAIS_sa_rk_h(self, text):
        sa = self.SAIS_sa(text)
        n, k = len(sa), 0
        rk, h = [0] * n, [0] * n # 这里的sa数组和高度数组h大小一样,h[i]表示sa[i]和sa[i-1]的LCP(最长公共前缀)
        for i, sa_i in enumerate(sa):
            rk[sa_i] = i

        for i in range(n):
            if k > 0: k -= 1
            while i + k < n and rk[i] - 1 >= 0 and sa[rk[i] - 1] + k < n and text[i + k] == text[sa[rk[i] - 1] + k]:
                k += 1
            h[rk[i]] = k
        return sa, rk, h
