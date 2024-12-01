# Array	
# 1 - two sum / O(n) O(n) 宝刀未老 ✅ 用complement
# 121. Best Time to Buy and Sell Stock ✅  用两个变量 max_profit=max; lowest_pric=min
# 217 Contains Duplicate ✅
# 238. Product of Array Except Self 🌟 ✅ 如何利用ans，也需要L/R单独的变量类似与prefix_product
# 53. Maximum Subarray 🌟 ✅
    # cur_sum = max(cur_sum + n, 0)
    # max_sum = max(max_sum, cur_sum)

# 152. Maximum Product Subarray - Neetcode ㊗️ / ㊗️
class Solution: 
    def maxProduct(self, nums: List[int]) -> int:
        if len(nums) < 2: return nums[0]
        max_prod, min_prod = 0, 0
        res = 0
        for n in nums:
            prev_max_prod = max_prod
            prev_min_prod = min_prod
            max_prod = max(prev_max_prod * n, prev_min_prod * n, n) # 你没有办法判断之前的min/max再乘当前的num之后会变成最大/最小，n是舍弃之前的不要。
            min_prod = min(prev_max_prod * n, prev_min_prod * n, n)
            res = max(res, max_prod)
        return res

# 153. Find Minimum in Rotated Sorted Array - ㊗️
class Solution:
    def findMin(self, nums: List[int]) -> int:
        if len(nums) == 1: return nums[0]
        l, r = 0, len(nums)-1
        while l < r:
            mid = (l+r) // 2
            l_val, m_val, r_val = nums[l], nums[mid], nums[r] # 精彩之地
            
            if m_val > r_val: # 意味着有rotate # 为什么不能拿m与l比较，大方向是一样的，但是边界会出错，因为我们的mid是floor()
                l = mid + 1
            else:
                r = mid
        return nums[l]


# 33. Search in Rotated Sorted Array ✅
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        l, r = 0, n-1
        while l < r:
            mid = (l + r) // 2
            mid_val = nums[mid]
            
            if target == mid_val: return mid # cut edge

            if mid_val >= nums[l]:
                if nums[l] <= target < mid_val:
                    r = mid
                else: 
                    l = mid + 1
            else:
                if mid_val < target <= nums[r]:
                    l = mid + 1
                else: 
                    r = mid
       
        # 如果是l<=r, 在最后一遍的循环中，我们其实可以检测当前值，因此如果跳出了，就意味着target不在区间中，因此直接返回-1.
        # return -1
        return l if nums[l] == target else -1

# 15. 3Sum - ㊗️
class Solution:
    def threeSum_outter_noOptimization(self, nums: List[int]) -> List[List[int]]:
        res, seen = set(), set()
        for i, v1 in enumerate(nums):
            for v2 in nums[i+1:]:
                complement = -(v1+v2)
                if complement in seen:
                    res.add(tuple(sorted([v1, v2, complement]))) 
            seen.add(v1)
        return list(res)
    
    def threeSum_outter_withOptimization(self, nums):
        res, dups = set(), set()
        for i, val1 in enumerate(nums):
            if val1 in dups:  # 跳过重复的第一个数
                continue
            dups.add(val1)
            # 这里就变成two sum了，只不过第一个数称为target一样的存在。
            seen = set()  
            for val2 in nums[i + 1:]:
                complement = -val1 - val2
                if complement in seen:
                    res.add(tuple(sorted((val1, val2, complement))))
                seen.add(val2)  # 标记当前值为已访问
        return list(res)
    
    def threeSum_inner_withOptimization(self, nums):
        res, dups = set(), set()
        seen = {} 
        for i, val1 in enumerate(nums):
            if val1 not in dups: 
                dups.add(val1)
                for j, val2 in enumerate(nums[i+1:]):
                    complement = -val1 - val2
                    if complement in seen and seen[complement] == i:  # 我们需要确保complement是当前val1下可以取到的值。如果不添加这个设置条件，那么有一个例外v1=2, v2=-4, 此时我们需要2，但是只有一个2是v1，我们的seen会被之前的循环更新，误以为我们存在2，其实不存在。
                        res.add(tuple(sorted((val1, val2, complement))))
                    seen[val2] = i
        return list(res)
# 11. Container With Most Water - ✅ 双指针 贪心


# Bit manipulation - neetcode 都有 跳过
# 371
# 191
# 338
# 268
# 190
# DP	
# 70. Climbing Stairs - Neetcode 👍
# 322. Coin Change - Neetcode 👍
# 300. Longest Increasing Subsequence  - Neetcode ㊗️
# 1143. Longest Common Subsequence - Neetcode 👍
# 139. Word Break - Neetcode ㊗️
# 377. Combination Sum IV
# 198. House Robber - Neetcode 👍
# 213. House Robber II - Neetcode ㊗️
# 55. Jump Game 
# 62. Unique Paths - Neetcode 👍
# 91. Decode Ways - Neetcode ㊗️
# Graph	
# 133. Clone Graph
# 207. Course Schedule
# 417. Pacific Atlantic Water Flow
# 200. Number of Islands 经典题 - 没必要再刷了。
# 128. Longest Consecutive Sequence
# 261. Graph Valid Tree
# 323. Number of Connected Components in an Undirected Graph - Neetcode ㊗️
# 269. Alien Dictionary 
# # Interval	
# 57. Insert Interval
# 56. Merge Intervals
# 435. Non-overlapping Intervals
# 252 - meeting room - 没啥难的，排序就成，只需要记录end
# 253. Meeting Rooms II
# LinkedList	
# 206 Reverse LinkedList
# 141. Linked List Cycle 
# 21. Merge Two Sorted Lists 
# 23. Merge k Sorted Lists
# 19 Remove Nth Node From End of List
# 143. Reorder List
# Matrix	
# 73. Set Matrix Zeroes
# 79. Word Search - Neetcode ㊗️
# 54. Spiral Matrix
# 48. Rotate Image - Neetcode
# String	
# 3. Longest Substring Without Repeating Characters - 滑动窗口 - 简单
# 424. Longest Repeating Character Replacement
# 76. Minimum Window Substring  
# 242. Valid Anagram - 简单秒杀
# 49。Group Anagrams
# 20. Valid Parentheses
# 5. Longest Palindromic Substring
# 125. Valid Palindrome / 两种方法：1-比较相反的， 2-双指针
# 647. Palindromic Substrings 
# 271. Encode and Decode Strings
# Tree	
# 104. Maximum Depth of Binary Tree
# 100. Same Tree    
# 226. Invert Binary Tree    
# 124. Binary Tree Maximum Path Sum
# 102. binary tree level order traversal
# 297. Serialize and Deserialize Binary Tree
# 572. Subtree of Another Tree
# 105. Construct Binary Tree from Preorder and Inorder Traversal
# 98. Validate Binary Search Tree
# 230. Kth Smallest Element in a BST （三种order方式）
# 235. Lowest Common Ancestor of a Binary Search Tree
# 236题目是关于没有BST这么强力的设定的。那一题返回的就是True/False，因此需要一个全局的self.node去取recursion中满足条件的值
# 208. Implement Trie (Prefix Tree) Trie树也是属于固定套路的东西。
# 211. Design Add and Search Words Data Structure
# 212. Word Search II
# 347. Top K Frequent Elements
# 295. Find Median from Data Stream 


# 一共80道题 - 10道 = 70道待刷 一周搞定。