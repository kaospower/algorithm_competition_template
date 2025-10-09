# 后缀数组+rmq
import math
from array import array

L_TYPE = ord('L')
S_TYPE = ord('S')


def is_lms(i, t):
    """Returns whether the suffix/ character at index i is a leftmost S-type"""
    return t[i] == S_TYPE and t[i - 1] == L_TYPE


def print_types(data: bytearray):
    """Simple method to the types of the characters of T"""
    print(data.decode('ascii'))
    print("".join(
        "^" if is_lms(i, data) else " "
        for i in range(len(data))
    ))


def classify(text, n) -> bytearray:
    """Classifies the suffixes in text as either S-type, or L-type
    This method can be merged with find_lms_suffixes but I have not done so for readability
    Args:
        text: the input string/array to be classified
        n: the length of text
    Returns:
        t: a bytearray object, where t[i] contains the type of text[i]
    """
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


def find_lms_suffixes(t, n):
    """Finds the positions of all lms_suffixes
    Args:
        t: the type array
        n: the length of text and t
    """
    pos = array('l')
    for i in range(n):
        if t[i] == S_TYPE and t[i - 1] == L_TYPE:
            pos.append(i)
    return pos


def print_buckets(bucks):
    """Simple method to print bucket sizes"""
    res = '[ '
    for b in bucks:
        if b != 0:
            res += str(b)
            res += ' '
    res += ']'
    print(res)


def buckets(text, sigma):
    """Find the alphabet and the sizes of the bucket for each character in the text"""
    alpha = []
    bucket_sizes = array('L', [0] * sigma)
    for c in text:
        bucket_sizes[c] += 1
    for i in range(sigma):
        if bucket_sizes[i] != 0:
            alpha.append(i)

    # print_buckets(bucket_sizes)
    return alpha, bucket_sizes


def bucket_intervals(alpha, bucket_sizes, sigma):
    """Computes the bucket intervals, i.e heads and tails"""
    heads = array('l', [0] * sigma)
    tails = array('l', [0] * sigma)
    j = 0
    for i in range(len(alpha)):
        heads[alpha[i]] = j
        j += bucket_sizes[alpha[i]]
        tails[alpha[i]] = j - 1

    # print_buckets(heads)
    # print_buckets(tails)
    return heads, tails


def induced_sorting(lms, tails, heads, SA, type_suffix, text, n, m, alpha, bucket_sizes, sigma):
    """Inductively creates the suffix array based on LMS
    Args:
        lms: an array indicating the positions of LMS Blocks/Suffixes in text
        tails: an array indexed by the characters in T which tells the ends of the buckets
        heads: an array indexed by the characters in T which tells the fronts of the buckets of those characters
        SA: an empty array to be filled during the creation of the suffix array
        type_suffix: an array in which type_suffix[i] tells the type of text[i]
        text: the input whose suffix array is to be created
        n: the length of the input 'text'
        alpha: an array of the alphabet of T in sorted order
        bucket_sizes: an array containing the sizes of each bucket: Used in resetting heads, tails
        """
    for i in range(m - 1, -1, -1):  # place LMS suffixes at the end of their buckets
        nfs = tails[text[lms[i]]]
        SA[nfs] = lms[i]
        tails[text[lms[i]]] -= 1

    for i in range(n):  # place the L-type suffixes at the fronts of their buckets
        if SA[i] > 0 and type_suffix[SA[i] - 1] == L_TYPE:
            nfs = heads[text[SA[i] - 1]]
            SA[nfs] = SA[i] - 1
            heads[text[SA[i] - 1]] += 1

    # reset bucket counters
    heads, tails = bucket_intervals(alpha, bucket_sizes, sigma)

    for i in range(n - 1, -1, -1):  # place the S-type suffixes at the ends of their buckets
        if SA[i] > 0 and type_suffix[SA[i] - 1] == S_TYPE:
            nfs = tails[text[SA[i] - 1]]
            SA[nfs] = SA[i] - 1
            tails[text[SA[i] - 1]] -= 1


def blocks_are_equal(i, j, types, text, n):
    """Testing for the equality of two blocks"""
    while i < n and j < n:
        if text[i] == text[j]:
            if is_lms(i, types) and is_lms(j, types):
                return True
            else:
                i += 1
                j += 1
        else:
            return False
    return False


def get_reduced_substring(types, SA, lms, ordered_lms, text, n, m):
    """Finds the reduced substring"""
    j = 0
    for i in range(n):
        if is_lms(SA[i], types):
            ordered_lms[j] = SA[i]
            j += 1

    # number the lms blocks and form the reduced substring
    pIS = array('l', [0] * m)
    k, i = 1, 1
    pIS[0] = 0
    for i in range(1, m):
        if text[ordered_lms[i]] == text[ordered_lms[i - 1]] and \
                blocks_are_equal(ordered_lms[i] + 1, ordered_lms[i - 1] + 1, types, text, n):
            pIS[i] = pIS[i - 1]
        else:
            pIS[i] = k
            k += 1

    # form the reduced substring

    inverse_lms = array('l', [0] * n)
    for i in range(m):
        inverse_lms[ordered_lms[i]] = pIS[i]
    for i in range(m):
        pIS[i] = inverse_lms[lms[i]]

    return pIS, k == m, k + 1


def construct_suffix_array(T, SA, n, sigma):
    """Constructs the suffix array of T and stores it in SA
    Args:
        T: the text whose suffix array is to be built
        SA: the array to be filled
        n: the length of T and SA
        sigma: the size of the alphabet of T, i.e the largest value in T
        """
    if len(T) == 1:  # special case
        SA[0] = 0
        return SA

    t = classify(T, n)  # step 1: classification
    lms = find_lms_suffixes(t, n)  # step 2: finding the indices of LMS suffixes
    m = len(lms)

    # print_types(t)

    alpha, sizes = buckets(T, sigma)  # finding the bucket sizes and alphabet of T
    heads, tails = bucket_intervals(alpha, sizes, sigma)
    induced_sorting(lms, tails, heads, SA, t, T, n, m, alpha, sizes, sigma)  # first induced sort

    ordered_lms = array('L', [0] * len(lms))

    reduced_text, blocks_unique, sigma_reduced = get_reduced_substring(t, SA, lms, ordered_lms, T, n, m)
    reduced_SA = array('l', [-1] * m)  # reduced SA
    if blocks_unique:  # base case
        # compute suffix array manually
        for i in range(m):
            reduced_SA[reduced_text[i]] = i
    else:
        construct_suffix_array(reduced_text, reduced_SA, m, sigma_reduced)

    # use the suffix array to sort the LMS suffixes
    for i in range(m):
        ordered_lms[i] = lms[reduced_SA[i]]

    heads, tails = bucket_intervals(alpha, sizes, sigma)  # reset bucket tails and heads
    for i in range(n):  SA[i] = 0  # clear suffix array
    induced_sorting(ordered_lms, tails, heads, SA, t, T, n, m, alpha, sizes, sigma)


def bwt(T, SA: array, BWT: bytearray, n: int):
    """If SA[i] = 0 then T[SA[i] - 1] = T[0 - 1] = T[-1] = '$"""
    for i in range(n):
        BWT[i] = T[SA[i] - 1]


def isa(SA, ISA, n):
    """Constructs an inverse suffix array"""
    for i in range(n):
        ISA[SA[i]] = i


def fm_index(SA, ISA, LF, n):
    """Constructs a last-to-first column mapping in linear time"""
    for i in range(n):
        if SA[i] == 0:
            LF[i] = 0
        else:
            LF[i] = ISA[SA[i] - 1]


def naive_suffix_array(s, n):
    """Naive suffix array implementation, just as a sanity check"""
    sa_tuple = sorted([(s[i:], i) for i in range(n)])
    return array('l', map(lambda x: x[1], sa_tuple))


'''
text: str:要处理的字符串
return: sa 
'''


def SAIS_sa(text):
    text += '$'
    text = [ord(c) for c in text]
    sigma = max(text) + 1
    n = len(text)
    SA = array('l', [-1] * n)
    construct_suffix_array(text, SA, n, sigma)
    bt = bytearray(n)
    bwt(text, SA, bt, n)
    return SA.tolist()[1:]


'''
text: str:要处理的字符串
return: sa,rk
'''


def SAIS_sa_rk(text):
    sa = SAIS_sa(text)
    rk = [0] * len(sa)
    for i, sa_i in enumerate(sa):
        rk[sa_i] = i
    return sa, rk


'''
text: str:要处理的字符串
return: sa,rk,h
'''


def SAIS_sa_rk_h(text):
    sa = SAIS_sa(text)
    n, k = len(sa), 0
    rk, h = [0] * n, [0] * n
    for i, sa_i in enumerate(sa):
        rk[sa_i] = i

    for i in range(n):
        if k > 0: k -= 1
        while i + k < n and rk[i] - 1 >= 0 and sa[rk[i] - 1] + k < n and text[i + k] == text[sa[rk[i] - 1] + k]:
            k += 1
        h[rk[i]] = k
    return sa, rk, h


class RMQ:
    def __init__(self, a):
        n = len(a)
        maxn, maxs, maxb = len(a), len(a) + 20, 20
        a = a + [0] * maxb  # 扩充a

        hightbit = [0] * maxs

        stmax = [[0] * maxb for _ in range(maxs)]
        premax = [[0] * maxb for _ in range(maxs)]
        sufmax = [[0] * maxb for _ in range(maxs)]
        quemax = [[0] * maxb for _ in range(maxs)]
        stackmax = [0] * maxb

        stmin = [[0] * maxb for _ in range(maxs)]
        premin = [[0] * maxb for _ in range(maxs)]
        sufmin = [[0] * maxb for _ in range(maxs)]
        quemin = [[0] * maxb for _ in range(maxs)]
        stackmin = [0] * maxb

        B = int(math.log2(n))
        S = (n - 1) // B + 1
        for b in range(S):
            stmin[b][0] = int(1e6)
        for i in range(n):
            stmin[i // B][0] = min(stmin[i // B][0], a[i])
            stmax[i // B][0] = max(stmin[i // B][0], a[i])
        for b in range(S - 1, -1, -1):
            k = 1
            while b + (1 << k) - 1 < S:
                stmin[b][k] = min(stmin[b][k - 1], stmin[b + (1 << (k - 1))][k - 1])
                stmax[b][k] = max(stmax[b][k - 1], stmax[b + (1 << (k - 1))][k - 1])
                k += 1
        for b in range(S):
            be = b * B
            premin[b][0], premax[b][0] = a[be], a[be]
            for k in range(1, B):
                premin[b][k] = min(premin[b][k - 1], a[be + k])
                premax[b][k] = max(premax[b][k - 1], a[be + k])
            sufmin[b][B - 1], sufmax[b][B - 1] = a[be + B - 1], a[be + B - 1]
            for k in range(B - 2, -1, -1):
                sufmin[b][k] = min(sufmin[b][k + 1], a[be + k])
                sufmax[b][k] = max(sufmax[b][k + 1], a[be + k])
        for b in range(S):
            be = b * B
            spmin, nowmin, spmax, nowmax = 0, 0, 0, 0
            for i in range(B):
                while spmin and a[be + stackmin[spmin]] > a[be + i]:
                    nowmin ^= 1 << stackmin[spmin]
                    spmin -= 1
                while spmax and a[be + stackmax[spmax]] < a[be + i]:
                    nowmax ^= 1 << stackmax[spmax]
                    spmax -= 1
                spmin += 1
                stackmin[spmin] = i
                nowmin ^= 1 << i
                quemin[b][i] = nowmin
                spmax += 1
                stackmax[spmax] = i
                nowmax ^= 1 << i
                quemax[b][i] = nowmax
        for i in range(2, S + 1):
            hightbit[i] = hightbit[i >> 1] + 1

        self.B = B
        self.a = a
        self.quemin = quemin
        self.sufmin = sufmin
        self.premin = premin
        self.sufmax = sufmax
        self.premax = premax
        self.quemax = quemax
        self.hightbit = hightbit
        self.stmin = stmin
        self.stmax = stmax

    def transferLR(self, l, r, rk):
        l, r = rk[l], rk[r]
        if l > r: l, r = r, l
        l += 1
        return l, r

    # 求[L,R]内的最值
    def query_min_max(self, l, r, rk):
        a = self.a
        l, r = self.transferLR(l, r, rk)
        L, R = l // self.B, r // self.B
        li, ri = l % self.B, r % self.B
        min_, max_ = 1e6, 0

        multiply_De_Bruijn_position = [
            0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
            31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9]

        def tail_zero_count(num):
            return multiply_De_Bruijn_position[(((num & (-num)) * 0x077CB531) >> 27) & 31]

        if L == R:
            min_ = min(min_, a[l + tail_zero_count(self.quemin[R][ri] >> li)])
            max_ = max(max_, a[l + tail_zero_count(self.quemax[R][ri] >> li)])
        else:
            min_ = min(min_, self.sufmin[L][li], self.premin[R][ri])
            max_ = max(max_, self.sufmax[L][li], self.premax[R][ri])
            len_ = R - L - 1
            k = self.hightbit[len_]
            if len_ > 0:
                min_ = min(min_, self.stmin[L + 1][k], self.stmin[R - (1 << k)][k])
                max_ = max(max_, self.stmax[L + 1][k], self.stmax[R - (1 << k)][k])

        return min_, max_


class Solution:
    def longestPalindrome(self, S: str) -> str:
        S = S + '|' + S[::-1]
        sa, rk, h = SAIS_sa_rk_h(S)

        R = RMQ(h)

        n = len(S)
        s = 0  # 起始位置
        l = 0  # 子串长度:较小的数
        for i in range(n // 2):
            ht, _ = R.query_min_max(i, n - 1 - i, rk)  # h数组值
            lt = 2 * ht - 1  # 奇数 真实子串长度
            if lt > l: s, l = i - ht + 1, lt
            if i != 0:
                ht, _ = R.query_min_max(i, n - i, rk)
                lt = 2 * ht  # 偶数
                if lt > l: s, l = i - ht, lt

        return S[s:s + l]