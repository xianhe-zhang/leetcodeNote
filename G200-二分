leetcode-69
#两种方法：二分，库函数
class Solution:
    def mySqrt(self, x: int) -> int:
        y = math.sqrt(x)
        return int(y)        
2-二分——太臃肿了
class Solution:
    def mySqrt(self, x: int) -> int:
        left = 0
        right = x
        if x == 0:
            return 0

        while right > left: 
            mid = left + (right - left)//2
            mid2 = mid * mid
            
            if mid2 == x:
                return mid
            if mid2 > x:
                right = mid
            if mid2 < x:
                left = mid
            if right == left+1:
                print(right*right)
                return left if right*right > x else right
        
@题解-二分
class Solution:
    def mySqrt(self, x: int) -> int:
        l, r, ans = 0, x, -1
        while l <= r:
            mid = (l + r) // 2  #//取整，都是取靠左的数字
            if mid * mid <= x:
                ans = mid    
                l = mid + 1
            else:
                r = mid - 1
        return ans
#题解代码简洁的原因是利用了双指针碰撞得到答案这一思想，并且l=mid+1, r=mid-1起到了很重要的作用
#二分三要素：指针碰撞，状态判断，指针转移

leetcode-744
class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        left = 0
        right = len(letters) - 1

        if target > letters[-1]:
            return letters[0]

        while left <= right:
            mid = (left + right) // 2
            if letters[mid] > target:
                right = mid - 1
            else: #mid <= target
                left = mid + 1
        return letters[left]
#进一步思考：二分mid+-1的套路是比较固定的，最后一次的循环会收敛到两个数字，i，和i+1；你的target在i～i+1之间，最终取什么值，要看你的需求，如果是i+1，那么就返回left——这是这个算法tricky的点

leetcode-540
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        left = 0 
        right = len(nums) - 1 

        while left < right: #这样最终经过运算，left与right将会是一个值，相碰，是我们要的值； 但是究竟是否加等于号，要看求的值是你找的值，还是你找的值的临近值。
 
            mid = (left + right) // 2

            #开始调整mid
            if nums[mid] == nums[mid + 1]:
                if (right - (mid + 2) + 1) % 2 == 1:
                    left = mid + 2 
                else:
                    right = mid - 1

            elif nums[mid] == nums[mid - 1]:
                if ((mid - 2) - left + 1) % 2 == 1:
                    right = mid - 2 
                else: 
                    left = mid + 1
            else:
                return nums[mid]

        return nums[right]
#这题最重要的就是每次二分时，需要操作一对数据，而非一个数据，这个点抓住后整个算法就清楚了。


leetcode-278
class Solution:
    def firstBadVersion(self, n):
        left = 0
        right = n
        
        while left < right:
            mid = (left + right) // 2

            if not isBadVersion(mid):
                left = mid + 1
            else: 
                right = mid
            
        return right
#take-away：根据题意和需求，灵活更改边界条件

leetcode-153
#1.函数 2.排序 3.二分
1-
class Solution:
    def findMin(self, nums: List[int]) -> int:
        return min(nums)
2-
class Solution:
    def findMin(self, nums: List[int]) -> int:
        nums.sort()
        return nums[0]
3-Binary Search #这一题有意思
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left = 0 
        right = len(nums) - 1 

        while left <= right: #这里必须有等号，解释如下。
            mid = (left + right) // 2

            if nums[mid] > nums[-1]:
                left = mid + 1
            else:   #nums[mid] <= nums[-1]
                right = mid - 1
        return nums[left] #必须是left
        
# 这里因为下面right的转换公式，所以要添加等号。#本体正确解法为下：
"""
本题left = mid + 1有两种情况：升序/或突然降序，但无论如何都不会错过最小值；
right = mid 而非 mid - 1：因为mid - 1草率了，可能本来就是mid，在判断条件的判断下，可能会导致mid为min而right（mid - 1）就因此错过最小值了。
这一题的nums[-1] 也可以用 nums[right]一样的效果
总结：核心是根据运算情况灵活运用套路，确保缩小取值空间时不能将值排出在外
--------------------------------------------------------------------------------------------------------------------
上面循环判断条件如果是<=，那么mid + 1 和 mid - 1；如果是<， 那么是mid + 1 和 mid。

<= :
1. 循环退出后right在左，left在右；🌟
2. 没有<的思考路径直观
3. 果要找的问题性质复杂，可以在循环外进行判断返回
4. 最后left = target, right = target - 1 

<:
1. 值的区间为1，跳出后直接可以返回值，就是我们要的值
2. 思考路径简单
3. 可以直接在循环体里跳出循环
4. left = right = target

注意：两者不是100%互换的，需要根据题意及时进行调整。
"""
leetcode-34
class Solution:
    def searchRange(self,nums, target):
        def left_func(nums,target): #找到target的起始点/或者大于target的第一位数起始点
            n = len(nums)-1
            left = 0
            right = n
            while(left<=right):
                mid = (left+right)//2
                if nums[mid] >= target:
                    right = mid-1
                if nums[mid] < target:
                    left = mid+1
            return left

        tar_cur =  left_func(nums,target)
        tar_next = left_func(nums,target+1)
        if  tar_cur == len(nums) or nums[tar_cur] != target:    #🌟 太讲究了，下面解释。
            return [-1,-1]      
        else:
            return [tar_cur,tar_next - 1]
"""
两个表达式，涵盖所有可能的情况：
~ tar_cur == len(nums)
    1. target 大于 所有值
    2. None 空集
~ nums[tar_cur] != target
    1. target不存在
    2. target 小于 所有值
"""

