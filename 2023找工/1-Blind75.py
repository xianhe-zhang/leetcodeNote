# Array	
# 1 - two sum / O(n) O(n) 宝刀未老
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hm = {}
        for i in range(len(nums)):
            r = target - nums[i]
            if r in hm:
                return [hm[r], i]
            hm[nums[i]] = i

# 121. Best Time to Buy and Sell Stock
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        bl, mp = prices[0], 0
        for p in prices[1:]:
            mp = max(mp, p-bl)
            bl = min(bl, p)
        
        return mp

        
# 217 Contains Duplicate
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        return len(nums) != len(set(nums))

# 238. Product of Array Except Self
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        ans = [1] * len(nums)
        L = R = 1 # 这个没想起来，想到用一个list解释了，但是没想到两个变量。
        # “L没有用”的理解很关键，第一遍是在原ans上直接进行遍历；此时ans均为‘prefix’的乘积
        # 之后只需要在用一个R变量模拟右侧的乘积就可以了。这种一个变量与一个list的配合是关键。
        for i in range(1, len(nums)):
            ans[i] = ans[i-1] * nums[i-1]
        for i in range(len(nums)-1, -1, -1):  # 可以用reversed()
            ans[i] *= R
            R *= nums[i]
        return ans


# 53
# 152
# 153
# 33
# 15
# 11
# Binary	
# 371
# 191
# 338
# 268
# 190
# DP	
# 70
# 322
# 300
# 1143
# 139
# 377
# 198
# 213
# 91
# 62
# 55
# Graph	
# 133
# 207
# 417
# 200
# 128
# 269
# 261
# 323
# Interval	
# 57
# 56
# 435
# 252
# 253
# LinkedList	
# 206
# 141
# 21
# 23
# 19
# 143
# Matrix	
# 73
# 54
# 48
# 79
# String	
# 3
# 424
# 76
# 242
# 49
# 20
# 125
# 5
# 647
# 271
# Tree	
# 104
# 100
# 226
# 124
# 102
# 297
# 572
# 105
# 98
# 230
# 235
# 208
# 211
# 212
# Heap	
# 23
# 347
# 295