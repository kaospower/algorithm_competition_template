# 3660.跳跃游戏9

```python
#3660(https://leetcode.cn/problems/jump-game-ix/description/)
#倒序思考+状态机dp
#最后一个数只能向左跳,它所能到的最大值就是max(nums)
#leftmax表示前缀最大值,rightmin表示后缀最小值
#对于nums[n-2],可以选择向左跳到其左侧最大值,即leftmax[i]
#如果leftmax[i]>rightmin[i+1],也可以选择先跳到leftmax[i],再跳到rightmin[i+1],然后跳到i+1位置
#因此当leftmax[i]>rightmin[i+1]时,ans[i]=ans[i+1]
#否则如果leftmax[i]<=rightmin[i+1],只能向左跳,ans[i]=leftmax[i]
fmax=lambda x,y:x if x>y else y
fmin=lambda x,y:x if x<y else y
class Solution:
    def maxValue(self, nums: List[int]) -> List[int]:
        n=len(nums)
        leftmax=[0]*n
        for i,x in enumerate(nums):
            leftmax[i]=fmax(leftmax[i-1] if i else 0,x)
        f=[0]*n
        f[-1]=max(nums)
        rightmin=inf
        for i in range(n-1,-1,-1):
            f[i]=leftmax[i] if leftmax[i]<=rightmin else f[i+1]
            rightmin=fmin(rightmin,nums[i])
        return f
```

# 3661.可以被机器人摧毁的最大墙壁数目

```python
#3661(https://leetcode.cn/problems/maximum-walls-destroyed-by-robots/)
#状态机dp
fmax=lambda x,y:x if x>y else y
fmin=lambda x,y:x if x<y else y
class Solution:
    def maxWalls(self, robots: List[int], distance: List[int], walls: List[int]) -> int:
        n=len(robots)
        #首尾添加哨兵
        a=[(0,0)]+sorted(zip(robots,distance))+[(inf,0)]
        walls.sort()
        #f(i,j)表示在第i+1个机器人向左/向右射击的情况下,前i个机器人可以摧毁墙的最大数量
        #j=0表示向左射,j=1表示向右射
        @cache
        def f(i,j):
            if i==0:
                return 0
            x,d=a[i]
            #往左射
            left_x=fmax(x-d,a[i-1][0]+1) #即不能射到第i-1个机器人
            left=bisect_left(walls,left_x)
            cur=bisect_right(walls,x)
            res1=f(i-1,0)+cur-left
            #往右射
            x2,d2=a[i+1]
            #右边机器人往左射
            if j==0:
                x2-=d2
            right_x=fmin(x+d,x2-1) #即第i个机器人不能射到第i+1个机器人(或是其向左射到的墙)
            right=bisect_right(walls,right_x)
            cur=bisect_left(walls,x)
            res2=f(i-1,1)+right-cur
            return fmax(res1,res2)
        return f(n,1)
```

