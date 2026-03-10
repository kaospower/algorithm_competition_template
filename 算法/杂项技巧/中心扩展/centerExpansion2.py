"""
中心扩展求最长回文子串长度
"""
def f(s):
    ans=0
    n=len(s)
    for i in range(2*n-1):
        l,r=i//2,(i+1)//2
        while l>=0 and r<n and s[l]==s[r]:
            l-=1
            r+=1
        ans=max(ans,r-l-1)
    return ans