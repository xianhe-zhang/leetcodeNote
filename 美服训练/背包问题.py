# 322 Coin Change
# 完全背包问题 先看东西，再看容量，记得初始化的条件。
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float("inf")] * (amount+1)
        dp[0] = 0
        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x-coin] + 1)
        return dp[amount] if dp[amount] != float("inf") else -1
# 518 Coin Change II
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1
        for coin in coins: 
            for i in range(coin, amount+1):
                dp[i] += dp[i - coin]
        return dp[amount]


# 416 
# 0-1 问题
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total%2 != 0: return False
        target = total // 2
        dp = [False] * (target + 1)
        dp[0] = True
        
        # 其实1D list没有那么容易理解背包的含义，这都属于bottom up的方法
        for n in nums:
            # 注意这里需要用到倒序，为什么？
            # 当我们使用一维数组的时候，针对每个元素和背包，我们要去看各个背包的容量。如果我们正序，那么当我们在满足前者的时候，比如1，如果1满足后，之后会依次满足，这个时候不满足0/1条件
            # 如果我们使用倒序，因为条件限制，一直为False，只有到之前满足过的点的时候才可以满足。
            for j in range(target, n - 1, -1):
                dp[j] = dp[j] | dp[j-n]
        return dp[target]
                
# 474 Count Zeroes and Ones
# 类似top-down的做法
class Solution:
    def findMaxForm(self, strs, m, n):
        xy = [[s.count("0"), s.count("1")] for s in strs]

        # 这个还蛮关键的，否则会TLE
        # @lru_cache(None)
        
        def dp(mm, nn, kk):
            # edge case，不满足效果
            if mm < 0 or nn < 0: return -float("inf")
            # 遍历结束了
            if kk == len(strs): return 0
            # kk对应的这一项有多少
            x, y = xy[kk]
            # 选或者不选。
            return max(1 + dp(mm-x, nn-y, kk + 1), dp(mm, nn, kk + 1))
        
        return dp(m, n, 0)

# 494
# 879
# 279
# 377
# 1049
# 1155
