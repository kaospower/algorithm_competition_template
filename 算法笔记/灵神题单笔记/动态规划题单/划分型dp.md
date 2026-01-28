# 1.判定能否划分

# 2.最优划分

**3579.字符串转换需要的最小操作数**

划分型dp
题目中提到了三种情况
情况2可以看成执行两次情况1,实际实现时利用哈希表可以将其转化成情况1
对于情况3,分别计算反转和不反转的情况,然后取最小值即可

这种解法时间复杂度$O(n^3)$

```python
fmin=lambda x,y:x if x<y else y
class Solution:
    def minOperations(self, s: str, t: str) -> int:
        n=len(s)
        #f[i+1]表示下标0...i范围的最少操作次数
        f=[0]*(n+1)
        for i in range(n):
            res=inf
            cnt=defaultdict(int)
            op=0 #op代表段j...i不执行情况3的最少操作次数
            #划分型dp,割点为j
            for j in range(i,-1,-1):
                #不反转
                #当下标为j时,s,t中对应的字符分别为x,y
                #如果可以通过交换实现,那么(y,x)之前应该出现过
                x,y=s[j],t[j]
                #当x!=y时,如果发现(y,x)之前出现过,哈希表对应次数-1
                #否则,说明需要替换,哈希表次数+1,同时操作次数op+1
                if x!=y:
                    if cnt[(y,x)]:
                        cnt[(y,x)]-=1
                    else:
                        cnt[(x,y)]+=1
                        op+=1
                #反转
                #对于反转的情况,需要重新遍历段j...i
                #分别检验s中段的第1个字符,t中段的倒数第1个字符是否相等,
                #s中段的第2个字符,t中段的倒数第2个字符是否相等,...等等
                #如果不等,看成需要1次替换即可(情况1)
                #由于反转消耗一个次数,因此rev_op初始为1
                #反转的情况需要重新开一个哈希表记录
                rev_cnt=defaultdict(int)
                rev_op=1
                for p in range(j,i+1):
                    #p在下标i...j范围中的对称点为i+j-p
                    x,y=s[p],t[i+j-p]
                    #相等时无需替换
                    if x==y:
                        continue
                    #不等时,如果(y,x)之前在哈希表中,次数-1
                    #否则,次数+1,反转情况操作次数rev_op+1
                    if rev_cnt[(y,x)]:
                        rev_cnt[(y,x)]-=1
                    else:
                        rev_cnt[(x,y)]+=1
                        rev_op+=1
                #段j..i的最少操作次数为min(op,rev_op)
                #段0...i的最少操作次数为f[j]+min(op,rev_op)(即划分型dp的转移方程)
                res=fmin(res,f[j]+fmin(op,rev_op))
            #更新f[i+1]
            f[i+1]=res
        return f[-1]
```

思维扩展,利用中心扩展法将时间复杂度优化为$O(n^2)$

```python
#划分型dp+中心扩展优化+重构消除重复代码
fmin=lambda x,y:x if x<y else y
class Solution:
    def minOperations(self, s: str, t: str) -> int:
        def update(x,y):
            if x==y:
                return
            if cnt[(y,x)]:
                cnt[(y,x)]-=1
            else:
                cnt[(x,y)]+=1
                nonlocal op
                op+=1
        n=len(s)
        #预处理rev_op
        rev_op=[[0]*n for _ in range(n)]
        #中心扩展模版
        #i为偶数表示奇长度子串,i为奇数表示偶长度子串
        for i in range(2*n-1):
            cnt=defaultdict(int)
            op=1
            #从[l,r]开始向左右扩展
            l,r=i//2,(i+1)//2
            while l>=0 and r<n:
                #两种扩展情况
                #1.s[l],t[r]
                update(s[l],t[r])
                #2.s[r],t[l]
                #l!=r时讨论情况2才有意义
                if l!=r:
                    update(s[r],t[l])
                rev_op[l][r]=op
                l-=1
                r+=1

        f=[0]*(n+1)
        for i in range(n):
            res=inf
            cnt=defaultdict(int)
            op=0
            for j in range(i,-1,-1):
                update(s[j],t[j])
                x,y=s[j],t[j]
                res=fmin(res,f[j]+fmin(op,rev_op[j][i]))
            #更新f[i+1]
            f[i+1]=res
        return f[-1]
```

# 3.约束划分个数

**410.分割数组的最大值**

注意本题也可以用二分答案,最小化最大解决

```python
#410(https://leetcode.cn/problems/split-array-largest-sum/)
fmax = lambda x, y: x if x > y else y
fmin = lambda x, y: x if x < y else y
class Solution:
    def splitArray(self, nums: List[int], k: int) -> int:
        def solve(nums,K):
            n=len(nums)
            f=[[inf]*(n+1) for _ in range(K+1)]
            f[0][0]=0
            for k in range(1,K+1):
                for i in range(k,n-(K-k)+1):
                    s=0
                    ans=inf
                    for j in range(i-1,k-2,-1):
                        s+=nums[j]
                        ans=fmin(ans,fmax(s,f[k-1][j]))
                    f[k][i]=ans
            return f[-1][-1]
        return solve(nums,k)
```

