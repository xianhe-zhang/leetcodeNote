# 46. Permutations
# 这一题的难点就是给你一个list如何找到所有的排序可能性
# 我自己思路是每一次递归找一个可能的数字进行排列。
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums, path):
            if not nums:
                res.append(path)
            for i in range(len(nums)): 
                dfs(nums[:i]+nums[i+1:], path+[nums[i]])
        res = []
        dfs(nums, [])
        return res