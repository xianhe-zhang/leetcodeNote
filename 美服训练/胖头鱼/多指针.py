# 75. Sort Colors
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        red, white, blue = 0, 0, len(nums)-1
        # 这个tranverse有点意思，白色作为中间颜色，和红色一起进行，然后通过+1进行区分。
        # 红index意味着下一个1应该插入的位置；blue index意味着下一个2应该插入那里，不过此时blue的元素没有遍历过！
        while white<=blue:
            if nums[white] == 0:
                nums[red], nums[white] = nums[white], nums[red]
                red += 1
                white += 1
            elif nums[white] == 1:
                white += 1
            else:
                nums[blue], nums[white] = nums[white], nums[blue]
                blue -= 1
                
            
            
# 26. Remove Duplicates from Sorted Array 太绕了
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        left, right = 0, 1
        while right < len(nums):
            if nums[right] == nums[left]:
                right += 1
                continue
            left += 1
            nums[left] = nums[right]
            right += 1
        return left+1
            

# 80. Remove Duplicates from Sorted Array II
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        for i in nums:
            count_num = nums.count(i)           # 多靠
            if count_num > 2:
                for j in range(count_num - 2):
                    nums.remove(i)
        return len(nums)


# 347. Top K Frequent Elements
from collections import Counter
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]: 
        # O(1) time 
        if k == len(nums):
            return nums
        
        # 1. build hash map : character and how often it appears
        # O(N) time
        count = Counter(nums)   
        # 2-3. build heap of top k frequent elements and
        # convert it into an output array
        # O(N log k) time
        return heapq.nlargest(k, count.keys(), key=count.get) 
        
# 349. Intersection of Two Arrays
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # return list(set(nums1).intersection(set(nums2)))
        # return list(set1 & set2)
        nums1.sort() # assume sorted
        nums2.sort() # assume sorted
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


# 43. Multiply Strings
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        n1, n2 = 0, 0
        for i in num1:
            n1 = n1 * 10 + (ord(i) - 48)
        for i in num2:
            n2 = n2 * 10 + (ord(i) - 48)
        return str(n1*n2)



# 845. Longest Mountain in Array
# 我没考虑edge case: 
#   1. 如果都是一样的数怎么办？我的代码会返回1，但是不构成mountain，应该返回0
#   2. 如果只有一侧满足题意是无法构成mountain的
# 本题思路：顺序遍历，先上山，再下山，找到一个山峰是一个while loop。所以你看需要用到nested if和while
# 并且更新ans 和 base
class Solution(object):
    def longestMountain(self, A):
        N = len(A)
        ans = base = 0

        while base < N:
            end = base
            if end + 1 < N and A[end] < A[end + 1]: #if base is a left-boundary
                #set end to the peak of this potential mountain
                while end+1 < N and A[end] < A[end+1]:
                    end += 1

                if end + 1 < N and A[end] > A[end + 1]: #if end is really a peak..
                    #set 'end' to right-boundary of mountain
                    while end+1 < N and A[end] > A[end+1]:
                        end += 1
                    #record candidate answer
                    ans = max(ans, end - base + 1)

            base = max(end, base + 1)

        return ans



# 215. Kth Largest Element in an Array
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        n = len(nums)
        self.divide(nums, 0, n-1, k)
        return nums[n-k]
    
    # divide作用就是利用position决定在哪区间内寻找position，如果找到就返回，没有的话继续divide/dive-deep去寻找(排序)
    def divide(self, nums, left, right, k):
        if left >= right: return 
        position = self.conquer(nums, left, right)
        
        if position == len(nums) - k: return    # position这个点已经求出，就是我们要得n-k，可以直接return
        elif position < len(nums) - k: self.divide(nums, position+1, right, k) # n-k在position右侧，右侧需要继续divide去寻找找到position；题外话，每排好一个position意味着，这个点就已经完全排好了。
        else: self.divide(nums, left, position - 1, k) # n-k我们找的点在左侧，因此我们需要在左侧找position。
    
    # conquer的作用就是将divide中指定的区间进行粗糙排序，并且返回pivot/position
    def conquer(self, nums, left, right):
        pivot, wall = nums[right], left
        for i in range(left, right):
            if nums[i] < pivot:
                # 碰到小的/应放左边的，就swap；碰到大的/应放右边的，continue，i继续增加，wall指向应该swap的大数；这样在下一轮循环的时候可以swap掉。
                # 而且在i与wall有间隔的情况，没碰到一个小的都要放在前面，这样更容易理解为什么每一次都需要swap
                nums[i], nums[wall] = nums[wall], nums[i]
                wall += 1
        # wall在loop结束后时+1的，因此肯定是较小区间的右侧，较大区间的第一位，因此将wall与pivot swap
        nums[wall], nums[right] = nums[right], nums[wall]
        # 我们返回的为什么是wall？因为返回的是分界点pi
        return wall


# 42. Trapping Rain Water 单调栈的题
class Solution:
    def trap(self, height: List[int]) -> int:
        res, stack = 0, []
        for i in range(len(height)):
            # 单调递减栈，如果发现i的值大于栈顶，意味着可以形成沟槽用来trapping water
            while stack and height[i] > height[stack[-1]]:
                # 由于单调栈和进入while的特性，bottom一定是目前已知的最小值。
                bottom = stack.pop()
                # 为什么not stack就break呢？因为“沟槽存在”的条件不成立了。
                # 理解：我们发现较大值i的话，前面小值是bottom，然后stack[-1]将会是另一个“墙/bank/边界”，因为pop后的stack[-1]一定大于bottom
                if not stack:
                    break
                distance = i - stack[-1] - 1
                # 每次只根据bottom决定两点之间的trapping water，并且随时update bottom
                # 这里min也很有意思，下面会解释。
                b_h = min(height[i], height[stack[-1]]) - height[bottom]
                res += distance * b_h
            stack.append(i)
        return res
# 解释min与思路：
# 试想出现一个i对应的值最大，那么stack中可能存在一系列满足沟槽的值[...3,2,1]
# 那么我们在极大值与1之间取最小值，意味着1-i之间可以存储水量的高低是多少，然后接着往下遍历2，然后是3，更新bottom然后依次填满
# 为什么我们在使用stack中的数据一次后就将其抛弃pop？因为每一个沟槽只能使用一次。一层层地填，这满足于题意，哪怕之后出现更大的，也没关系。


# 1004. Max Consecutive Ones III
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        left = 0
        for right in range(len(nums)):
            k -= 1 - nums[right] 
            # 这个太有灵性了，sliding window详解见下面。
            if k < 0: 
                k += 1 - nums[left]
                left += 1
        return right - left + 1
# 首先我们没必要keep k>0，因为我们找的是最大gap，只要sliding window达到一个最大值，我们不积极缩小window
# 缩小window和扩展window可以是async的。