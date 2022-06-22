
# 69. Sqrt(x)
# time: O(logN)
from bisect import bisect_left


class Solution:
    def mySqrt(self, x: int) -> int:       
        if x < 2:
            return x
        left, right = 2, x//2
        while left <= right:
            pivot = left + (right-left)//2
            num = pivot ** 2
            
            if num > x:
                right = pivot - 1
            elif num < x:
                left = pivot + 1
            else:
                return pivot
        
        return right


# 704. Binary Search
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)
        while left < right:
            pivot = left + (right - left) // 2
            num = nums[pivot]
            if num < target:
                left = pivot + 1
            elif num > target:
                right = pivot
            else:
                return pivot
        
        return -1

# 35. Search Insert Position
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)
        while left < right:
            mid = left + (right - left) // 2
            n = nums[mid]
            if n < target:
                left = mid + 1
            elif n > target:
                right = mid
            else:
                return mid
                
        return right

# 349. Intersection of Two Arrays
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # return list(set(nums1).intersection(set(nums2)))
        # return list(set1 & set2)
 
        nums1.sort() 
        nums2.sort()
        result = []

        while nums1 and nums2:
            if nums2[-1] > nums1[-1]:
                nums2.pop()
            elif nums2[-1] < nums1[-1]:
                nums1.pop()
            else:
                # to avoid duplicates
                if not result or nums1[-1] != result[-1]:
                    result.append(nums1[-1])
                nums1.pop()
                nums2.pop()

        return result


# 167. Two Sum II - Input Array Is Sorted
# 这是双指针不是binary
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        left, right = 0, len(nums)-1
        
        while left < right:
            tt = nums[left] + nums[right]
            if tt == target:
                return [left+1, right+1]
            elif tt < target:
                left += 1
            else:
                right -= 1
                
                

# 300. Longest Increasing Subsequence
# 这一题两个解法，DP是N^2和N；Binary Search是N*logN和N 为什么是N*logN？LogN是binary，N是针对每一个num
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            # 去遍历之前的所有可能性，因为之前遇到奇大无比的数字
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] =  max(dp[i], dp[j] + 1)
        
        return max(dp)
    
# 思路很新奇，机制很创新。我们有个sub，遍历每个num，利用bisect_left寻找目标元素返回index，
# 如果index和len一致我们直接在结尾append并且扩充len
# 如果我们能够在文中找到相应位置，我们就进行替换，主要是将最后一位替换为较小位，有点类似greedy

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        sub = []
        for num in nums:
            i = bisect_left(sub, num)

            # If num is greater than any element in sub
            if i == len(sub):
                sub.append(num)
            
            # Otherwise, replace the first element in sub greater than or equal to num
            else:
                sub[i] = num
        
        return len(sub)

# 74. Search a 2D Matrix
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        i, j = 0, len(matrix[0]) - 1
        while i < len(matrix) and j > -1:
            num = matrix[i][j]
            
            if num > target:
                j -= 1
            elif num < target:
                i += 1
            else:
                return True            
        return False

# 上面是自己写的解法，复杂度为O(MN)； 下面binary seach只用O(logMN)
# 🌟binary search的本质就是将有序的matrix转化为有序的一列，利用数学关系确定坐标。
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:   
        m = len(matrix)
        if m == 0:
            return False
        n = len(matrix[0])
        
        # binary search
        left, right = 0, m * n - 1
        while left <= right:
                pivot_idx = (left + right) // 2
                pivot_element = matrix[pivot_idx // n][pivot_idx % n]
                if target == pivot_element:
                    return True
                else:
                    if target < pivot_element:
                        right = pivot_idx - 1
                    else:
                        left = pivot_idx + 1
        return False

# 34. Find First and Last Position of Element in Sorted Array
# 这一题很妙呀！通过一个变量进行传旨，从而使得helper function有轻微的变动！👍赞！
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        low = self.findBoundery(nums, target, True)
        if low == -1: return [-1, -1]
        high = self.findBoundery(nums, target, False)
        return [low, high]
    
    def findBoundery(self, nums, target, flag):
        left, right = 0, len(nums)-1
        while left <= right:
            mid = left + (right - left) // 2
            n = nums[mid]
            
            if n == target:
                if flag:
                    if mid == left or nums[mid-1] < target:
                        return mid
                    right = mid - 1
                else:
                    if mid == right or nums[mid+1] > target:
                        return mid
                    left = mid + 1
                
            elif n > target:
                right -= 1
            else:
                left += 1
        
        return -1
        
    
# 33. Search in Rotated Sorted Array
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums: return - 1
        lo, hi = 0, len(nums)-1
        
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            

            if nums[mid] == target:
                return mid
            
            # 🌟理解nested if condition关键点在于我们这道题是如何减小区间的？(思考两个问题：1.区间内是否包含“变化”点？ 2.target是否则判断后的区间内？ 谨记我们只能处理确认过单调的区间，针对有变化的区间我们处理不了。)
            # A-1: 首先决定“变化”是否是在左边区间 （不在的话，有可能在右边，也有可能不存在变化）
            # A-2: 如果不在左边区间，意味着左边lo～mid是单调的，可以判断target是否在该区间。在/不在都可以缩小范围
            elif nums[mid] >= nums[lo]:
                if nums[lo] <= target < nums[mid]:
                    hi = mid - 1
                else:
                    lo = mid + 1
            # A-3: 如果“变化”那个点在左边，那么我们看右边，判断逻辑同A-2.
            else:
                if nums[mid] < target <= nums[hi]:
                    lo = mid + 1
                else:
                    hi = mid - 1
            
        return -1

# 454. 4Sum II
# 这题不是binary search
class Solution:
    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        cnt = 0
        m = collections.defaultdict(int)
        for a in nums1:
            for b in nums2:
                m[a+b] += 1
        
        for c in nums3:
            for d in nums4:
                cnt += m[-(c+d)]
        
        return cnt



# 875. Koko Eating Bananas
# 🌟/周赛/OA/VO 这一题最精妙的地方在于我们的mid不是index，而是猩猩的eating rate！
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        left, right = 1, max(piles)
        
        while left < right: 
            mid = (left + right) // 2
            hours = 0
            
            for p in piles:
                hours += math.ceil(p / mid)
                
            if hours <= h:
                right = mid
            else:
                left = mid + 1
        
        return right

# 240. Search a 2D Matrix II
# 这道题也是非常不错的，亮点是区间是什么？一般来讲我们的二分实在一维数据上进行查找。
# 这道题是二维的数据，但是找到pivot之后，判断结束大小之后我们有两个方向可以继续search！ 然后利用top-down recursion也是没有想到过的。
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:

        # an empty matrix obviously does not contain `target`
        if not matrix:
            return False

        # 每次缩小区间其实是缩小查找的rectangle，缩小的图形画出来是很有意思的，mid和row可以将matrix分为四个部分，排除掉左上和右下，为什么？
        # 左上无论横竖都满足小于target，右下无论横竖都满足大于target，但是剩下的两个区域就不一定了。
        def search_rec(left, up, right, down):
            if left > right or up > down or target < matrix[up][left] or target > matrix[down][right]:
                return False

            mid = left + (right-left) // 2

            # 该题的遍历方法也很好。通过二分确定一维，然后通过遍历找到另一维！
            row = up
            while row <= down and matrix[row][mid] <= target:
                if matrix[row][mid] == target:
                    return True
                row += 1
            
            return search_rec(left, row, mid - 1, down) or \
                   search_rec(mid + 1, up, right, row - 1)

        # 这道题利用helper就是因为开头是一种情况，而helper return的是两个recursion
        return search_rec(0, 0, len(matrix[0]) - 1, len(matrix) - 1)



# 718. Maximum Length of Repeated Subarray
class Solution:
    def findLength1(self, A, B):
        memo = [[0] * (len(B) + 1) for _ in range(len(A) + 1)]
        for i in range(1, len(A)+1):
            for j in range(1, len(B)+1):
                if A[i-1] == B[j-1]:
                    memo[i][j] = memo[i - 1][j - 1] + 1
        return max(max(row) for row in memo)

    def findLength2(self, A, B):
        memo = [[0] * (len(B) + 1) for _ in range(len(A) + 1)]
        for i in range(len(A) - 1, -1, -1):
            for j in range(len(B) - 1, -1, -1):
                if A[i] == B[j]:
                    memo[i][j] = memo[i + 1][j + 1] + 1
        return max(max(row) for row in memo)
# 第二种解法为什么要用倒序？为了确保初始化顺利，我们的dp记忆要比原有index多一位，如果倒序我们meory里面的index就可以和原始数据的index保持一致
# 我们遍历用正序也可以，但是index也要随之调整。
# 不用binary 太扯淡了


# 50. Pow(x, n)
# 为赋新词强说愁，强行二分，时间复杂度logN
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return(1.0)
        res = 1 
        t = abs(n)
        while t != 0: 
            # t是我们的项，如果是基数，先乘一下消消项
            if t%2 == 1: 
                res *= x
            t >>= 1 # right shifting t so it will divide t by 2.
            x = x*x # 为什么t除以2，这里是x=x*x，有关运算法则，(2^x)^y = 2^(xy)
        return 1/res if n<0 else res


# 287. Find the Duplicate Number
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        nums.sort()
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                return nums[i]
# 二分的话，我想从index和数据的关系上下手吧，行不通
# 好难理解
# 因为这题input比较随意，不是紧凑的，因此不能使用index与数据之间的关系
# 下面题解非常优秀，牛，真是被玩出花了...
# 二分的对象是区间内按值mid(其实为中位数/平均值)，小于mid的值+1计数下来cnt，比较mid与cnt：
# 1. 如果cnt比较小，意味着一定是
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # 'low' and 'high' represent the range of values of the target
        low = 1
        high = len(nums) - 1
        
        # 通过我们不断缩减
        while low <= high:
            cur = (low + high) // 2
            count = 0

            # Count how many numbers are less than or equal to 'cur'
            # 这里有趣的是，我们并不narrow nums的scope，我们narrow的只是取值范围。
            count = sum(num <= cur for num in nums)
            # 如果count大于cur，意味着重复的值一定在左侧(包括cur本身)
            if count > cur:
                duplicate = cur
                high = cur - 1
            # 如果count <= cur，意味着重复的数字一定在cur的右侧！
            else:
                low = cur + 1
                
        return duplicate


# 209. Minimum Size Subarray Sum
# 时间复杂度为O(n)，这题的解法其实是sliding window
class Solution:
    def minSubArrayLen(self, target:int, nums) -> int:
        if not target or not nums: return 0
        total, left, result = 0, 0, float('inf')
        for i in range(len(nums)):
            total += nums[i]
            while total >= target:
                result = min(result, i-left+1)
                total -= nums[left]
                left += 1
        return result if result != float('inf') else 0



# 153. Find Minimum in Rotated Sorted Array
class Solution:
    def findMin(self, nums: List[int]) -> int:
        lo, hi = 0, len(nums) - 1
        while lo < hi:
            mid = lo + (hi - lo) // 2
            # 左侧比右侧最大的都大，因此这种情况表明mid~hi都在右侧区间，可以直接缩小
            if nums[mid] < nums[hi]:
                hi = mid
            # 表明mid在左侧区间，因此lo = mid + 1就可以。
            else:
                lo = mid + 1

# 162. Find Peak Element
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        left, right = 0, len(nums)-1
        while left < right:
            mid = (left+right) // 2
            # 🌟这个判断条件是精华！如果mid大，意味着一定有peak在是mid或者左边，可以直接缩小右边范围
            if nums[mid] > nums[mid + 1]:
                right = mid
            # 否则mid一定不是peak，右侧有可能是！
            else:
                left = mid + 1
        return left
"""
public class Solution {
    public int findPeakElement(int[] nums) {
        return search(nums, 0, nums.length - 1);
    }
    public int search(int[] nums, int l, int r) {
        if (l == r)
            return l;
        int mid = (l + r) / 2;
        if (nums[mid] > nums[mid + 1])
            return search(nums, l, mid);
        return search(nums, mid + 1, r);
    }
}
如何用recusion写二分
"""