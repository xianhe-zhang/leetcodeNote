# 704. Binary Search
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2
            num = nums[mid]
            if num == target:
                return mid
            elif num > target:
                right = mid - 1
            else:
                left = mid + 1
        return left if nums[left] == target else -1



# 34. Find First and Last Position of Element in Sorted Array
# 这道题不推荐使用找到一位数，然后去扩展边界，因为在处理边界比较难搞，会有很多判断，比如你最后while扩张后，返回应该是left还是left+1？
# 因此还是推荐使用一个helper function去找上下边界，上下边界的处理是你需要学习的。
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        
        lower_bound = self.findBound(nums, target, True)
        if (lower_bound == -1):
            return [-1, -1]
        
        upper_bound = self.findBound(nums, target, False)
        return [lower_bound, upper_bound]
        
    def findBound(self, nums: List[int], target: int, isFirst: bool) -> int:
        
        N = len(nums)
        begin, end = 0, N - 1
        while begin <= end:
            mid = int((begin + end) / 2)    
            
            if nums[mid] == target: # 首先我们改动的代码仅仅限于发现我们的target了。
                if isFirst: 
                    # 如何判断是不是首位？以及如何移动获得首位？
                    # 1. mid是否==begin，因为我们的数字一定是在begin和end中间的；并且判断mid-1是否小于target
                    # ✨这里的or可以精妙地避开index out of range的问题，因为其那面已经判断过不是首位了。
                    # 2. 如果发现不是首位，那么我只用移动end = mid - 1就可以了。
                    if mid == begin or nums[mid - 1] < target: 
                        return mid
                    end = mid - 1
                else:
                    if mid == end or nums[mid + 1] > target:
                        return mid
                    begin = mid + 1
            
            elif nums[mid] > target:
                end = mid - 1
            else:
                begin = mid + 1
        
        return -1


# 702. Search in a Sorted Array of Unknown Size
class Solution:
    def search(self, reader: 'ArrayReader', target: int) -> int:
        left, right = 0, 1
        # 其实就新增了找index的这一步。
        while reader.get(right) < target:
            left = right
            right <<= 1
            
        while left < right:
            mid = left + (right - left) // 2
            val = reader.get(mid)
            if val == target: 
                return mid
            elif val > target:
                right = mid
            else:
                left = mid + 1
        mid = (left+right)//2
        return mid if reader.get(mid) == target else -1
# 🌟注意🌟 用<有一个问题，那就是我们最后找到的left/right可能满足或者不满足题意，反正他们俩碰在一起了。
# 因此我们需要在最后结尾再次进行判断。
# 如果用 left <= right就不用。 为什么🌟？？？ 因为left <= right默认是把当前碰到的index也再次循环，
# 因此在while中进行if == target判断仿佛是与生俱来的。


# 33. Search in Rotated Sorted Array
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums: return -1
        left, right = 0, len(nums)-1
        
        while left <= right: # 你可以永远相<=
            mid = left + (right - left) // 2
            val = nums[mid]            
            
            if val == target:
                return mid

            # 确定区间(left, right)一半区间的单调性。 
            # 确定了(left,mid)是单调递增的，右侧可能是单调/可能有断崖
            if val >= nums[left]: 
                
                # 确定target在左侧单调时，缩小范围
                if nums[left] <= target <= val:
                    right = mid - 1
                # 消极处理：如果在右侧的话，val有可能在崖上，也有可能在崖下。
                else:            
                    left = mid + 1  
            # 确定(left,mid)有断崖，断崖无法处理，因此我们向右看(mid, right)，我们最终的目标是向断崖收敛，直至找到单调的小区间。
            else:
                if val <= target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1 
# 二分上述采用的不是主动寻找，而是被动缩小区间，那么最后留下来的一定是我们要找的数！
# 第一个if是判断我们应该看左区间，还是右区间是单调的。第二个nested-if是在确定单调后，
# 我们给出不同的condition clause判断val与target的关系，最终缩短left/right


# 81. Search in Rotated Sorted Array II
# 这一题是上一题的升级版:主要矛盾点在于在区间内存在重复元素，试想如果nums[0] == nums[-1] == nums[mid]，那么我们应该向哪个方向进行缩小？
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        if not nums:
            return False
        n = len(nums)
        if n == 1:
            return nums[0] == target
        l, r = 0, n - 1

        
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return True
            # 碰到相同的情况，两侧减小；
            if nums[l] == nums[mid] and nums[mid] == nums[r]:
                l += 1
                r -= 1
            # 进到这里其实就是和正常的33题一样了。
            elif nums[l] <= nums[mid]:
                if nums[l] <= target and target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[mid] < target and target <= nums[n - 1]:
                    l = mid + 1
                else:
                    r = mid - 1

        return False


# 4. Median of Two Sorted Arrays
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        n1 = len(nums1)
        n2 = len(nums2)
        
        if n1 > n2: return self.findMedianSortedArrays(nums2, nums1)
        
        # 我们需要在num1和num2中找k个数。
        k = (n1+n2+1) // 2 # 这里+1的话偶数没有影响，奇数的话会进1，因为是找中位数 
        
        l, r = 0, n1
        
        # 我们只针对一个nums进行二分，然后另一个nums利用两者之间的关系辅助二分。
        while l < r:
            # m1,m2分别是需要的元素的个数, 而非
            m1 = l + (r-l) // 2 # 我们的r是从n1开始的，因此只用找floor()个就成了，比如7个数只找3个
            # 但是这里m1是要当index的，因此[3]刚好是0～6的中位数
            m2 = k - m1 # m2-1要相对应的更新了
            # 如果小的话表示第一个数组中被选中前往最终合并数组的前面的个数不够多，所以要扩大
            # 记住这里while二分是找nums1中有多少数组能够组成我们的合并数组的前半部分
            if nums1[m1] < nums2[m2-1]: 
                l = m1+1
            else: # 太大的话右侧就缩进来。
                r = m1
        # l==r的时候上述循环结束，意味着nums1中区间遍历结束。
        # 开始分配m1和m2
        m1 = l
        m2 = k - l
        
        # m1-1和m2-1是排序后
        c1 = max(
        nums1[m1-1] if m1 > 0 else float('-inf'),
        nums2[m2-1] if m2 > 0 else float('-inf')
        )
        if (n1+n2)%2 == 1:
            return c1
        c2 = min(
        nums1[m1] if m1 < n1 else float('inf'),
        nums2[m2] if m2 < n2 else float('inf')
        )


        return (c1+c2) * 0.5
            
        
# 74. Search a 2D Matrix
# 这一题比较简单，就2个点注意到就好：1. 从哪里开始traverse 2. index与图形的转换与对应关系。
class Solution:
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
        
# 162. Find Peak Element
# 难点只有一个如何用二分解决这道题...二分的if条件
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

# 1283. Find the Smallest Divisor Given a Threshold
class Solution:
    def smallestDivisor(self, nums: List[int], threshold: int) -> int:
        # 因为有 nums.length <= threshold 这个条件
        left, right = 1, max(nums)
        while left < right:
            divisor = (left + right)//2
            total = sum([ceil(n/divisor) for n in nums]) # 这个ceil可把我坑惨了
            if total <= threshold: # 这里只要想明白是如何缩小区间的就没问题。
                right = divisor
            else: 
                left = divisor + 1
        return left


