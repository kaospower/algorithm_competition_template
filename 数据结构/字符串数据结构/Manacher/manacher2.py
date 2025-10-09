"""
manacher算法求解具体最长回文回文串
"""
class Solution:
    def longestPalindrome(self, s: str) -> str:
        def f(s):
            #扩展串2n+1长度
            n=len(s)<<1|1
            #p是回文半径数组
            p=[0]*n
            #ss是Manacher扩展串
            ss='#'+'#'.join(s)+'#'
            mx=end=0 #end标记最长回文半径在扩展串中对应的结束位置
            #i代表当前来到的中心,c代表回文中心,r代表回文覆盖右边界
            c=r=0
            for i in range(n):
                #至少的回文半径长度
                #这行代码涵盖了4种情况,r>i时走min(),r==i时回文串为单个字符,回文半径长度为1
                lens=min(p[2*c-i],r-i) if r>i else 1
                #考虑能不能往外扩,2和3情况会直接不执行while,只有1和4情况会执行while
                while i+lens<n and i-lens>=0 and ss[i+lens]==ss[i-lens]:
                    lens+=1
                #更新回文覆盖右边界
                if i+lens>r:
                    r=i+lens
                    c=i
                if lens>mx:
                    mx=lens
                    end=i+lens-1 #end记录最长回文半径的字符串在扩展串中的结束位置
                p[i]=lens
            #end//2即为原串中的结束位置+1
            #mx是扩展串最长回文半径,等于原串中最长回文串长度
            return end//2,mx-1
        end,size=f(s)
        return s[end-size:end]