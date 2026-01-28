**LogTrick使用条件**

如果随着运算的数越多,运算的结果在减半(或者说倍增),运算结果只有logU种,则可以用LogTrick

常见可以使用LogTrick的运算:gcd,lcm,AND,OR,乘法(有乘积上限时)

特别地,对于或运算,从左到右非递增,对于与运算,从左到右非递减

注意:LogTrick通常可被效率更高的栈+滑动窗口方法替代

**3171.找到按位或最接近K的子数组**
把二进制数视作集合,两个数的或运算,等价于两个集合的并集,把ai对应的集合记作Ai
i=1时,把A0到A1的并集记录在A0中,即把A1并入A0,合并后,A1是A0的子集,即A0>A1(后面的a>b表示a包含b)
i=2时,把A2并入A1和A0,即A0>A1>A2
i=3时,把A3并入A2,A1,A0,即A0>A1>A2>A3
一般情况,A0>A1>A2>...>Ai,因此如果在某一步合并集合时,发现合并后集合不变,说明待并入的集合是原集合的子集,无需执行后续并入操作,break

```python
#3171(https://leetcode.cn/problems/find-subarray-with-bitwise-or-closest-to-k/)
fmin=lambda x,y:x if x<y else y
class Solution:
    def minimumDifference(self, nums: List[int], k: int) -> int:
        ans=inf
        for i,x in enumerate(nums):
            ans=min(ans,abs(x-k))
            for j in range(i-1,-1,-1):
                if nums[j]|x==nums[j]:
                    break
                nums[j]|=x
                ans=fmin(ans,abs(nums[j]-k))
        return ans     
```

**2411.按位或最大的最小子数组长度**

此题本质是对LogTrick底层原理的直接应用
对于每个右端点i,子数组nums\[j:i\](左闭右开)的OR值保存在nums[j]处
比如i=3时,nums[0]保存的是nums[0]|nums[1]|nums[2],nums[1]保存的是nums[1]|nums[2],
nums[2]保存的是nums[2],我们现在想知道的是能否延长上述子数组,即在上一轮的基础上扩充一个nums[i]
而如果nums[j]|x==nums[j],就说明nums[i]是上一轮结果的子集,或上nums[i]不会增加OR值
因此在本轮中,所有左端点<=j的子数组,都无法通过在结尾添加nums[i]这个数来实现长度的扩张,直接break即可
否则,可以扩张,nums[j]保存的OR值从nums[j:i]变成了nums[j:i+1],更新ans[j]=i-j+1

```python
#2411(https://leetcode.cn/problems/smallest-subarrays-with-maximum-bitwise-or/)
class Solution:
    def smallestSubarrays(self, nums: List[int]) -> List[int]:
        ans=[0]*len(nums)
        for i,x in enumerate(nums):
            ans[i]=1
            for j in range(i-1,-1,-1):
                if nums[j]|x==nums[j]:
                    break
                nums[j]|=x
                ans[j]=i-j+1
        return ans
```

**3209.子数组按位与值为K的数目**

LogTrick与二分结合
每个元素都是其右侧元素的子集,从nums[0]到nums[i]是非递减的,因此可以用二分

```python
#3209(https://leetcode.cn/problems/number-of-subarrays-with-and-value-of-k/)
class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        ans=0
        for i,x in enumerate(nums):
            for j in range(i-1,-1,-1):
                if nums[j]&x==nums[j]:
                    break
                nums[j]&=x
            ans+=bisect_left(nums,k+1,0,i+1)-bisect_left(nums,k,0,i+1)
        return ans
```

