# 3. Longest Substring Without Repeating Characters
from ast import List, Tuple
import collections
from functools import cache, lru_cache


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # 这一题的seen[s[j]] = j+ 1, 包括j-i+1难以理解！
        # 如果要理解，就先要想用中文给自己解释清楚应该怎么处理！
        # 从开头的edge case开始想，我们的seen存的是什么，我们遇到了seen之后更新的逻辑是什么？我们求result的逻辑是什么？
        # 当我们遇到一个字母，肯定要用双指针求当前的res，然后比较存值，那么该窗口内一定没有重复的char，那么seen里面存的就是j+1
        # 然后随之而来的j-i+1就能说通了。
        # 但是与此同时我们不能够将左指针指向已经出现过的数字，因为试想从头到尾都没有出现重复的数字怎么办。
        i = 0
        res = 0
        seen = {}
        for j in range(len(s)):
            if s[j] in seen:
                i = max(seen[s[j]], i)
            res = max(res, j - i + 1)
            seen[s[j]] = j + 1
        return res
    
    
    

# 70. Climbing Stairs
class Solution:
    def climbStairs(self, n: int) -> int:
        a = b = 1
        # (0,1) (1,2) (2,3) (3,4) ...
        # a和b是一级级跳的，不是2个一起跳。a是main function，b是helper
        for _ in range(n):
            a, b = b, a + b
        return a


# 53. Maximum Subarray
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        cur = res = nums[0]
        for num in nums[1:]:
            cur = max(num, cur+num)
            res = max(res, cur)
        return res
        
        
# 121. Best Time to Buy and Sell Stock
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_price = float('inf')
        max_profit = 0
        for i in range(len(prices)):
            if prices[i] < min_price:
                min_price = prices[i]
            elif prices[i] - min_price > max_profit:
                max_profit = prices[i] - min_price
                
        return max_profit

# 746. Min Cost Climbing Stairs
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        for i in range(2, len(cost)):
            cost[i] += min(cost[i-1], cost[i-2])
        return min(cost[-1], cost[-2])



# 337. House Robber III
# 自己写的，但是非常不完美
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        def dp(root):
            if not root:
                return [0,0] 
            
            # 这里不需要用if root.left: 因为我们的限制条件是if not root: return [0,0]
            left = dp(root.left)
            right = dp(root.right)
            
            rob = left[1]+right[1]+root.val
            # 首先我没有想到notrob是用max，因为既然决定不rob了，那么子树rob不rob都可以
            notrob = max(left)+max(right)
             
            return [rob, notrob]
        
        
        money = dp(root)
        return max(money)

# 322. Coin Change
# 想清楚怎么遍历就成了
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        if not amount: return 0
#       sys.maxsize
        dp = [float('inf') ]* (amount+1)
        dp[0] = 0
        for i in range(amount+1):
            for c in coins:
                if i >= c:
                    dp[i] = min(dp[i], dp[i-c]+1)
                
        return dp[amount] if dp[amount] != float('inf') else -1
        

# 300. Longest Increasing Subsequence
# 这一题主要学习的是第二个solution，第一个方法很简单。
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
            # 主要针对最后一项
            else:
                sub[i] = num
        
        return len(sub)


# 139. Word Break
# 利用dp的难点就在于状态转换，确定状态+确定转换标准

"""
这一题其实是dfs，又有点像树的结构思维。很厉害。
难点在于理解dfs是如何通过递归思维解决这道题，其实都是针对某一段字符串看其能否满足一个特性。最终可以确保每一个ch都能满足。
"""
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        word_set = set(wordDict)
        dp = [False] * (len(s) + 1)
        dp[0] = True
    
        # dp为什么要len(s) + 1，一般是初始化需要。index可能也需要
        # 这里两个for是，i是end_index，j是start_index；
        for i in range(1, len(s) + 1):
            for j in range(i):
                # 如果dp[j]能组成，s[j:i]也能组成，那么我们就可以说dp[i]是没问题的！
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        return dp[len(s)]
    
    
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        def dfs(s, wordDict, start):
            # base case就是我们看的start index已经==len(s)，如果能遍历到这一步，那肯定为True
            if start == len(s):
                return True
            # 这里很有趣，针对每一个开始的start，都去遍历所有，就是看所有的可能性。只要其中有一个可能性成功了，那就是return True
            # 如果没有成功，不会做什么，如果所有end循环都没有成功才会返回false
            # 这里的end要取到到len(s)，因为是要给切片器使用。
            for end in range(start + 1, len(s) + 1):
                if s[start:end] in wordDict and dfs(s, wordDict, end):
                    return True
            return False
        return dfs(s, set(wordDict), 0)

# 152. Maximum Product Subarray
# 首先分析题很重要，only meet 0/positive/negative, 遇到postive怎么都是大的。遇到0时候必须跳过，无论正负。遇到负的时候可以先存起来。
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0

        # ma是用来存放所有的max值的
        # mi是用来存放所有的min值的，主要用来对付negative值...
        ma = nums[0]
        mi = nums[0]
        result = ma    

        # 利用两个list来处理，反而避免掉了那些不连续的值。
        for i in range(1, len(nums)):
            curr = nums[i]
            temp_max = max(curr, ma * curr, mi * curr)
            mi       = min(curr, ma * curr, mi * curr)
            
            ma = temp_max
            
            result = max(ma, result)

        return result


# 338. Counting Bits
# 利用API很简单，用BIT的话要与DP联系起来
class Solution:
    def countBits(self, n: int) -> List[int]:
        res = []
        for i in range(n+1):
            res.append(str(bin(i)).count('1'))
        return res
# ans[x]与ans[x//2]的关系其实就是前者和后者大数部分的1肯定相同，小数部分的就是余数部分有没有相等的。
class Solution:
    def countBits(self, n: int) -> List[int]:
        ans = [0] * (n + 1)
        for x in range(1, n + 1):
            # x // 2 is x >> 1 and x % 2 is x & 1
            ans[x] = ans[x >> 1] + (x & 1) 
        return ans 
    

class Solution:
    def countBits(self, n: int) -> List[int]:
        ans = [0] * (n + 1)
        for x in range(1, n + 1):
            ans[x] = ans[x & (x - 1)] + 1
        return ans 


# 264. Ugly Number II
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        # 初始化
        ugly = [1]
        i2, i3, i5 = 0, 0, 0

        while n > 1:
            # 找出u2,u3,u5中的最小值；u数字是目前已知的最紧凑的一批ugly number
            u2, u3, u5 = 2 * ugly[i2], 3 * ugly[i3], 5 * ugly[i5]
            umin = min(u2, u3, u5)
            # 如果碰见umin与我们的u数字相同的话，意味着该位要跳过。
            if umin == u2:
                i2 += 1
            if umin == u3:
                i3 += 1
            if umin == u5:
                i5 += 1
            ugly.append(umin)
            n -= 1
        return ugly[-1]


# 279. Perfect Squares
class Solution(object):
    def numSquares(self, n):
        square_nums = [i**2 for i in range(0, int(math.sqrt(n))+1)] # int是floor所以要+1
        
        dp = [float('inf')] * (n+1)
        # bottom case
        dp[0] = 0
        
        for i in range(1, n+1):
            for square in square_nums:
                # 如果i小于square没必要看了，直接跳过。
                if i < square:
                    break
                dp[i] = min(dp[i], dp[i-square] + 1)
        
        return dp[-1]


# 309. Best Time to Buy and Sell Stock with Cooldown
# 这道题很难，其实牵扯到了状态机！状态机三个状态之间的配合，让我们每一次的计算都可以同步，刚好也存了多种状态/情况。比如我们每日最大收益只会来自reset/sold；我们的held其实就是起了过渡帮助的作用，reset同理。
class Solution(object):
    def maxProfit(self, prices):
        sold, held, reset = float('-inf'), float('-inf'), 0

        for price in prices:
            # Alternative: the calculation is done in parallel.
            # Therefore no need to keep temporary variables
            #sold, held, reset = held + price, max(held, reset-price), max(reset, sold)

            pre_sold = sold
            sold = held + price
            held = max(held, reset - price)
            reset = max(reset, pre_sold) # 今天休息
            
        return max(sold, reset)


# 5. Longest Palindromic Substring
class Solution:
    def longestPalindrome(self, s):
        res = ""
        for i in range(len(s)):
            res = max(self.helper(s, i, i), self.helper(s,i,i+1), res, key=len)
        return res
    
    def helper(self, s, l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return s[l+1:r]
    
    

# 1143. Longest Common Subsequence
class Solution:
    def longestCommonSubsequence(self, t1: str, t2: str) -> int:
        dp = [[0]*(len(t2)+1) for _ in range(len(t1)+1)]
        
        # 为什么我们需要dp.size = len()+1? 因为针对首行首列我们也需要进行状态转移。
        for i in range(len(t1)):
            for j in range(len(t2)):
                # print(dp)
                if t1[i] == t2[j]:
                    dp[i+1][j+1] = 1 + dp[i][j]
                else:
                    dp[i+1][j+1] = max(dp[i+1][j],dp[i][j+1])
        return dp[-1][-1]

               
    

# 131. Palindrome Partitioning
# 很经典呀，针对线性数据结构如何进行dfs/dp/backtracking!
# 最终结果是所有元素都要返回的！第一层循环的话，我们就可以从0遍历到n
# 第二层就可以从1或者从之后遍历，dfs出来了，就可以遍历所有情况。
# 时间N*2^N，因为n个节点；每个简单最多是2^n种组合情况。
class Solution(object):
    @cache  # the memory trick can save some time
    def partition(self, s):
        if not s: return [[]]
        ans = []
        for i in range(1, len(s) + 1):
            if s[:i] == s[:i][::-1]:  
                # 这个是灵魂！🌟
                # 要搞清楚的话首先要明白这个function做了什么工作/返回了什么value
                # base case是没有s了；切分逻辑的实现是用🌟🌟🌟for循环+递归🌟🌟🌟
                # 直接写进for省事。
                # 遍历的情况也是我没想到的，是DFS不是BFS，这一点记清，是第一层的一种情况穷尽后，才是第一层的其他情况
                # 我们return得ans是list(list)所以不要每一个suf就是后缀的一个完全组合可能。
                for suf in self.partition(s[i:]):  
                    ans.append([s[:i]] + suf)
        return ans

class Solution:
    # main function，把变量init好，使得helper去生成我们的result
    def partition(self, s):
        if not s: return []
        result = []
        self.helper(s, [], result)
        return result 
    
    
    def helper(self, s, step, result):
        if not s:
            # step是一个list不要直接添加到另一个list里面去，而是套层list的外壳。
            result.append(list(step))
            return 
        for i in range(1, len(s)+1):
            if s[:i] != s[:i][::-1]: continue
            step.append(s[:i])
            self.helper(s[i:], step, result)
            step.pop()
        return 



# 62. Unique Paths
# 想到用DP的话，说实在是比较困难的点。特别是当前格子的value是由其旁边两个格子决定的。
# 想来想去，难点还是在于如何意识到状态转移方程。方程很简单，想明白却没有那么简单
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        d = [[1] * n for _ in range(m)]
        for col in range(1, m):
            for row in range(1, n):
                d[col][row] = d[col - 1][row] + d[col][row - 1]
        return d[m - 1][n - 1]


# 64. Minimum Path Sum
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if i == 0 and j == 0:
                    continue
                elif i == 0:
                    grid[i][j] = grid[i][j] + grid[i][j-1]
                elif j == 0:
                    grid[i][j] = grid[i][j] + grid[i-1][j]
                else:
                    grid[i][j] = grid[i][j] + min(grid[i][j-1], grid[i-1][j])
        return grid[-1][-1]


# 221. Maximal Square
# 这一题的难点就在于怎么解决这个数学题哈哈哈。利用边的关系，很巧妙，不用一定掌握。
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if not matrix: return 0
        row, col = len(matrix), len(matrix[0])
        
        dp = [[0] * (col + 1) for _ in range(row + 1)]
        max_side = 0
        
        for i in range(row):
            for j in range(col):
                if matrix[i][j] == '1':
                    dp[i+1][j+1] = min(dp[i][j], dp[i+1][j], dp[i][j+1]) + 1
                    max_side = max(max_side, dp[i+1][j+1])
                    
        return max_side**2
        
        

# 416. Partition Equal Subset Sum
# Top-Down的dp方法。
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        @lru_cache(maxsize=None)
        def dfs(nums: Tuple[int], n: int, subset_sum: int) -> bool:
            # Base cases
            if subset_sum == 0:
                return True
            if n == 0 or subset_sum < 0:
                return False
            # 两种情况，选VS不选
            result = (dfs(nums, n - 1, subset_sum - nums[n - 1])
                    or dfs(nums, n - 1, subset_sum))
            return result

        # find sum of array elements
        total_sum = sum(nums)

        # if total_sum is odd, it cannot be partitioned into equal sum subsets
        if total_sum % 2 != 0:
            return False

        subset_sum = total_sum // 2
        n = len(nums)
        return dfs(tuple(nums), n - 1, subset_sum)
# 🌟经典0-1背包问题
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total % 2 != 0: return False
        capacity = total // 2
        n = len(nums)
        # 因为我们要考虑为0的情况，也是照顾edge case，所以这里要+1
        dp = [[False] * (capacity+1) for _ in range(n+1)]
        dp[0][0] = True
        
        # 这里是否+1归属于index的游戏，没有对错，只有是否容易理解/方便
        # 首先明确：dp的长度都+1了；capacity+1没有关系因为可以考虑到0-capacity的情况了；num+1也是有意义的意味着我们不放任何coin。
        # 这里还必须要从1开始，curr=nums[i-1]还必须这么写，为什么？因为i不仅仅指向了nums，还和我们的dp有关。i=0的话没有办法更新dp，所以要从1走，然后处理curr
        
        for i in range(1, n+1):
            curr = nums[i-1]
            for j in range(capacity+1):
                if curr > j:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j-curr]                
        return dp[n][capacity]



# 718. Maximum Length of Repeated Subarray
# index与nums存在的差别也可以用倒序进行解决！秒哇
class Solution:
    def findLength(self, A, B):
        memo = [[0] * (len(B) + 1) for _ in range(len(A) + 1)]
        for i in range(len(A) - 1, -1, -1):
            for j in range(len(B) - 1, -1, -1):
                if A[i] == B[j]:
                    memo[i][j] = memo[i + 1][j + 1] + 1
        return max(max(row) for row in memo)



# 494. Target Sum
"""
Java的DP思路: 其实挺雷同就是把capacity换成所有的能取值范围。
然后每次遇到main都会考虑两种情况 +/- 而不是 选/不选
public class Solution {
    public int findTargetSumWays(int[] nums, int S) {
        int total = Arrays.stream(nums).sum();
        int[][] dp = new int[nums.length][2 * total + 1];
        dp[0][nums[0] + total] = 1;
        dp[0][-nums[0] + total] += 1;
        
        for (int i = 1; i < nums.length; i++) {
            for (int sum = -total; sum <= total; sum++) {
                if (dp[i - 1][sum + total] > 0) {
                    dp[i][sum + nums[i] + total] += dp[i - 1][sum + total];
                    dp[i][sum - nums[i] + total] += dp[i - 1][sum + total];
                }
            }
        }
        
        return Math.abs(S) > total ? 0 : dp[nums.length - 1][S + total];
    }
}"""
class Solution(object):
    def findTargetSumWays(self, nums, S):
        if not nums:
            return 0
        dic = {nums[0]: 1, -nums[0]: 1} if nums[0] != 0 else {0: 2}
        for i in range(1, len(nums)):
            tdic = {}
            for d in dic:
                tdic[d + nums[i]] = tdic.get(d + nums[i], 0) + dic.get(d, 0)
                tdic[d - nums[i]] = tdic.get(d - nums[i], 0) + dic.get(d, 0)
            dic = tdic
        return dic.get(S, 0)
class Solution(object):
    def findTargetSumWays(self, A, S):
        count = collections.Counter({0: 1})
        for x in A:
            step = collections.Counter()
            for y in count:
                step[y + x] += count[y]
                step[y - x] += count[y]
            count = step
        return count[S]