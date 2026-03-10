"""
中心扩展求最长回文子串
"""
class Solution:
    def longestPalindrome(self, s: str) -> str:
        mx=L=R=0
        n=len(s)
        for i in range(2*n-1):
            l,r=i//2,(i+1)//2
            while l>=0 and r<n and s[l]==s[r]:
                l-=1
                r+=1
            if r-l-1>mx:
                mx=r-l-1
                L,R=l,r
        return s[L+1:R]