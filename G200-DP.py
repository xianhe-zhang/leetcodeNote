"""
DP解题四步走：
    1.定义子问题
    2.写出子问题的递推关系
    3.确定 DP 数组的计算顺序
    4.空间优化（可选
"""

from _typeshed import IdentityFunction
from typing import ItemsView, List


leetcode-70 爬楼梯
#自己写的，省略了几个循环
class Solution:
    def climbStairs(self, n: int) -> int:
        if n < 3:
            return n 
        i = 2
        dp = [0] * n  #🌟这里必须有0，否则bug成山
        dp[0] = 1
        dp[1] = 2
        while i < n:
            dp[i] = dp[i-2] + dp[i-1]
        return dp[n-1]  
#题解-利用了dp[0]更容易理解鞋
class Solution:
    def climbStairs(self, n: int) -> int:
        dp = [] * (n+1)
        dp[0] = 1
        dp[1] = 1
        for i in range(2, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]
#空间优化
class Solution:
    def climbStairs(self, n: int) -> int:
        pre, cur = 1, 1
        for i in range(n-1):
            pre, cur = cur, pre + cur
        return cur



leetcode-198 打家劫舍
def rob(self, nums: List[int]) -> int:
    prev = 0
    curr = 0
    
    # 每次循环，计算“偷到当前房子为止的最大金额”
    for i in nums:
        # 循环开始时，curr 表示 dp[k-1]，prev 表示 dp[k-2]
        # dp[k] = max{ dp[k-1], dp[k-2] + i }   #表示偷第k个房子时的最大值
        prev, curr = curr, max(curr, prev + i)
        # 循环结束时，curr 表示 dp[k]，prev 表示 dp[k-1]

    return curr
#利用dp[]是没有进行空间优化的解法；最后结果只与k-1 和 k-2有关，因此用两个变量表示就可以了。

----------------  @尚未优化题解
def rob(self, nums: List[int]) -> int:
    if len(nums) == 0:
        return 0
    N = len(nums)
    dp = [0] * (N+1)
    dp[0] = 0
    dp[1] = nums[0]
    for k in range(2, N+1):
        dp[k] = max(dp[k-1], nums[k-1] + dp[k-2])
    return dp[N]
----------------  @自己写的
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        elif len(nums) == 1:
            return nums[0]
        n = len(nums)
        dp = [[] for _ in range(n)]
        dp[0] = nums[0]
        dp[1] = max(nums[0],nums[1])    #队首的处理十分细节。巧妙地回避了边界的陷阱
        i = 2
        while i < n:
            dp[i] = max(dp[i-2]+nums[i], dp[i-1])
            i += 1
        return dp[n-1]
#可以看出来自己写的代码长，原因是因为自己没有思考完整，[k]的最大值需要k-1和k-2，那么就要考虑队首的元素应该怎么处理。优化的方法借助了dp[0]=0


leetcode-213 打家劫舍II
#环状房屋，那么能把环切成两个列
class Solution:
    def rob(self, nums: List[int]) -> int:
        def iRob(nums):
            pre , cur = 0, 0
            for num in nums:
                cur, pre = max(cur, num + pre), cur #别忘储存的两个值同时变换
            return cur
        return max(iRob(nums[1:]), iRob(nums[:-1])) if len(nums) != 1 else nums[0] 


leetcode-64
class Solution:
    def minPathSum(self, grid: [[int]]) -> int:
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if i == j == 0: continue
                elif i == 0:  grid[i][j] = grid[i][j - 1] + grid[i][j]
                elif j == 0:  grid[i][j] = grid[i - 1][j] + grid[i][j]
                else: grid[i][j] = min(grid[i - 1][j], grid[i][j - 1]) + grid[i][j]
        return grid[-1][-1]
#还是相当于列出了所有的可能性，因为边界实在无法最小值，但是中间的值可以选择最小值。

leetcode-62
#自己写的没有优化
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        if n == 0 or m == 0:
            return 1
        steps = [[0] * m for _ in range(n)] #搞错了，m那里应该为n，应该是col。好在这一题不影响结果。
        print(steps[0])
        for i in range(n):
            for j in range(m):
                if i == 0 or j == 0: steps[i][j] = 1
                else: steps[i][j] = steps[i][j-1] + steps[i-1][j]
        return steps[-1][-1]
@优化的题解
#这一题优化后，每一次判断也是只用储存前两个值；这个解法考虑了数学因素，无论走上走下每个单元格都有意义，每个单元格都应该加起来。
#这一题只用了一列的空间，然后每一次遍历的时候进行优化，太极限了...
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        cur = [1] * n
        for i in range(1, m):   #🌟这里的1是关键，意味着不考虑两个边界。
            for j in range(1, n):
                cur[j] += cur[j-1]
        return cur[-1]

leetcode-303(easy)
@官方题解
#利用前缀和的思想
class NumArray:

    def __init__(self, nums: List[int]):
        self.sums = [0]  #用self声明可以变为全局变量，其他方法也可以调用了。
        _sums = self.sums

        for num in nums:
            _sums.append(_sums[-1] + num)

    def sumRange(self, i: int, j: int) -> int:
        _sums = self.sums
        return _sums[j + 1] - _sums[i]
# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(left,right)

#这一题就是说，我们有nums，那么如何用这两个方法配合，从而更快地实现我们的sumRange功能；本题的算法是在初始化数据的时候利用前缀和的技巧，然后sumrange直接执行运算便可。
#复杂度均为O(n)


leetcode-413
#DP解法
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        tmp_cnt = 0
        cnt = 0
        n = len(nums)
        for i in range(n-2):
            if nums[i+1] - nums[i] == nums[i+2] - nums[i+1]:
                tmp_cnt += 1
                cnt += tmp_cnt #这个地方不要放错，否则会多写几行代码，放在这里没有影响。
            else:
                tmp_cnt = 0
        return cnt
#复杂度为n，空间为1
@不适应的感觉就是在提高的时候

#双指针解法-复杂度稍高
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        n = len(nums)
        self.res = 0
        self.helper(nums, n-1)
        return self.res
    
    def helper(self, nums, end): #helper利用了递归，
        if end < 2:
            return 0
        cnt = 0
        if nums[end-1] - nums[end] == nums[end-2] - nums[end-1]:
            cnt = 1 + self.helper(nums, end-1)  #这一行非常难理解，测例[1,2,3,4]；end = 4的时候其实 cnt = 1 + 1; end = 3 的时候 cnt = 1
            """
            如何理解呢？因为cnt在每一个循环中都会初始化，那么self.helper这里究竟等于几其实是从底层积累起来的。比如432,4321，那么在end = 4的时候，cnt =2
            这样因为每一层都会累积，都进入res，这样就相当于end为各个元素的所有子递增列都进入了，不管长度多少。非常巧妙
            """
            self.res += cnt
        else:
            self.helper(nums, end-1)
        return cnt
#时间复杂度为n2，空间复杂度为N（调用栈最大深度为N）

#滑动窗口
#思路：从队首开始，如果满足，则右指针右移；如果不满足，左指针与指针接轨。
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        if not nums or len(nums) < 3:
            return 0
        res = 0
        L = 2
        predif = nums[1] - nums[0]
        n = len(nums)
        for i in range(1, n-1):
            dif = nums[i+1] - nums[i]
            if dif == predif:
                L += 1              #L是用来判断该等差数列多长，然后利用数学的方法求出有多少种组合
            else:
                res += (L-1) * (L-2) // 2 #L=3的话，只有一种组合；
                L = 2
                predif = dif
        res += (L-1) * (L-2) // 2   #最后如果有的话还要存入
        return res


leetcode-343 整数拆分
#这一题主要涉及数学归纳
class Solution:
    def integerBreak(self, n: int) -> int:
        dp = [0] * (n + 1)
        dp[2] = 1
        for i in range(3, n + 1):
            # 假设对正整数 i 拆分出的第一个正整数是 j（1 <= j < i），则有以下两种方案：
            # 1) 将 i 拆分成 j 和 i−j 的和，且 i−j 不再拆分成多个正整数，此时的乘积是 j * (i-j)
            # 2) 将 i 拆分成 j 和 i−j 的和，且 i−j 继续拆分成多个正整数，此时的乘积是 j * dp[i-j]
            for j in range(1, i - 1):
                dp[i] = max(dp[i], max(j * (i - j), j * dp[i - j]))
        return dp[n]
#复杂度n2
#自己找到灵感了，但是就放过去了。

@数学归纳
class Solution:
    def integerBreak(self, n: int) -> int:
        if n <= 3: return n - 1
        a, b = n // 3, n % 3
        if b == 0: return int(math.pow(3, a))
        if b == 1: return int(math.pow(3, a - 1) * 4)
        return int(math.pow(3, a) * 2)


leetcode-279 完全平方数
1-DP
class Solution:
    def numSquares(self, n: int) -> int:
        dp = [0] * (n+1)            
        for i in range(1, n+1):
            j = 1
            dp[i] = i               #按照最多的数字，每一个平方数都拆成1表示
            while j * j <= i:
                dp[i] = min(dp[i-j*j] + 1, dp[i])         #按照item看每一个背包，这里dp[ℹ️]表示数字i时，能拆成平方数最小的组合。
                j += 1
        return dp[n]

2-四平方和定理-任意一个正整数都可以被表示为至多四个正整数的平方和
#只用判定1，2，4就行；1和4情况特殊，直接判断；2循环判断；3直接用排除法
3-完全背包
class Solution:
    import math
    def numSquares(self, n: int) -> int:
        #dp[i] 表示和为 i 的 nums 组合中完全平方数最少有 dp[i] 个。
        dp = [float('inf')]*(n+1)
        dp[0] = 0
        #可供选则的nums列表 [1,4,9,...,sqrt(n)]  从中任意挑选数字组成和为n
        #完全背包
        for i in range(1, int(math.sqrt(n))+1):
            for j in range(n+1):
                if(j>=i*i):
                    dp[j] = min(dp[j], dp[j-i*i]+1)
        return dp[n]

"""
3-完全背包DP（学习一下  
nums = [i*i for i in range(1, int(n**0.5)+1)]               #把符合目标的所有平方数放进nums背包里
f = [0] + [float('inf')]*n                                  #f[i]是最少需要多少个平方数应该。 制造出13位数字list，刚好对应index        
for num in nums:                                            
    for j in range(num, n+1):                               
        f[j] = min(f[j], f[j-num]+1)                        #这个转化有意思，要么是f[j]不动，要么就是
return f[-1]

4-贪心
ps = set([i * i for i in range(1, int(n**0.5)+1)])      #**是表示幂的意思
def divisible(n, count):
    if count == 1: return n in ps
    for p in ps:
        if divisible(n-p, count-1):
            return True
    return False

for count in range(1, n+1):
    if divisible(n, count):
        return count
"""
------------------------------------背包问题专辑----------------------------------------------------
"""
首先是背包分类的模板：
1、0/1背包：外循环nums,内循环target,target倒序且target>=nums[i];
2、完全背包：外循环nums,内循环target,target正序且target>=nums[i];
3、组合背包：外循环target,内循环nums,target正序且target>=nums[i];
4、分组背包：这个比较特殊，需要三重循环：外循环背包bags,内部两层循环根据题目的要求转化为1,2,3三种背包类型的模板

然后是问题分类的模板：
1、最值问题: dp[i] = max/min(dp[i], dp[i-nums]+1)或dp[i] = max/min(dp[i], dp[i-num]+nums);
2、存在问题(bool)：dp[i]=dp[i]||dp[i-num];
3、组合问题：dp[i]+=dp[i-num];

自己的笔记：
    1. 正序/倒序主要是因为进行了空间优化，完全组合背包问题（重复利用一个元素）就是需要利用到已经更新的数据，因此正序没关系；倒序不能用到更新到当前行的数据，要利用上一行的数据，因此需要倒序。
    2. 二维比一维容易理解。
    3. 记住状态转移方程前的条件判断有时候有边界问题。
    4. 状态转移的确认有几个ex：1. 最后一位数往前面的组合加上去 2. 是否选择当前元素  3.前面看后面的元素（难理解...自己胡说的
"""
1-零钱兑换
#零钱兑换：给定amount,求用任意数量不同面值的零钱换到amount所用的最少数量
#完全背包最值问题：外循环coins,内循环amount正序,应用状态方程1
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        #dp[i] 表示换到面值i所需要的最小数量
        dp = [float('inf')]*(amount+1)
        dp[0] = 0 #dp[i]:换到面值i所用的最小数量

        for i in range(len(coins)):
            for j in range(coins[i], amount+1): #最小面值从coins[i]开始计算
                dp[j] = min(dp[j], dp[j-coins[i]]+1)
        
        if dp[amount]!=float('inf'):
            return dp[amount]
        else:
            return -1
2-分割等和子集
#分割等和子集：判断是否能将一个数组分割为两个子集,其和相等
#0-1背包存在性问题：是否存在一个子集,其和为target=sum/2,外循环nums,内循环target倒序,应用状态方程2
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        s_sum = sum(nums)
        if  s_sum%2==1:
            return False
        target =    s_sum//2
        dp = [False]*(target+1) #dp[i]是否存在子集和为i
        dp[0] = True #初始化：target=0不需要选择任何元素，所以是可以实现的
        for i in range(len(nums)):
            for j in range(target, nums[i]-1, -1): #nums[i]-1 最小取到num[i]
                dp[j] = dp[j] or dp[j-nums[i]]
        return dp[target]
    
3-目标和
#目标和：给数组里的每个数字添加正负号得到target
#数组和sum,目标和s, 正数和x,负数和y,则x+y=sum,x-y=s,那么x=(s+sum)/2=target
#0-1背包不考虑元素顺序的组合问题:选nums里的数得到target的种数,外循环nums,内循环target倒序,应用状态方程3
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        num_sum =sum(nums)
        if (num_sum<target or (num_sum+target)%2!=0):
            return 0 
        target = (num_sum+target)//2
        dp = [0]*(target+1)  #dp[i]:和为i的不同表达式的数目
        dp[0] = 1
        for i in range(len(nums)):
            for j in range(target, nums[i]-1, -1):
                dp[j] += dp[j-nums[i]]
        return dp[target]

4-完全平方数
#完全平方数：对于一个正整数n,找出若干个完全平方数使其和为n,返回完全平方数最少数量
#完全背包的最值问题：完全平方数最小为1,最大为sqrt(n),故题目转换为在nums=[1,2.....sqrt(n)]中选任意数平方和为target=n
#外循环nums,内循环target正序,应用转移方程1
class Solution:
    import math
    def numSquares(self, n: int) -> int:
        #dp[i] 表示和为 i 的 nums 组合中完全平方数最少有 dp[i] 个。
        dp = [float('inf')]*(n+1)
        dp[0] = 0
        #可供选则的nums列表 [1,4,9,...,sqrt(n)]  从中任意挑选数字组成和为n
        #完全背包
        for i in range(1, int(math.sqrt(n))+1):
            for j in range(n+1):
                if(j>=i*i):
                    dp[j] = min(dp[j], dp[j-i*i]+1)
        return dp[n]

5-组合总和V
#在nums中任选一些数,和为target
#考虑顺序的组合问题：外循环target,内循环nums,应用状态方程3
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        #dp[i]表示总和为i的元素组合的个数
        dp = [0]*(target+1)
        dp[0] = 1
        for i in range(target+1):
            for j in range(len(nums)):
                if i>=nums[j]:
                    dp[i] += dp[i-nums[j]]
        return dp[target]
    
6-零钱兑换II
#零钱兑换2：任选硬币凑成指定金额,求组合总数
#完全背包不考虑顺序的组合问题：外循环coins,内循环target正序,应用转移方程3
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        #dp[i]表示凑成总金额为i的硬币组合数
        dp = [0]*(amount+1)
        dp[0] = 1
        for i in range(len(coins)):
            for j in range(amount+1):
                if j>=coins[i]:
                    dp[j]+= dp[j-coins[i]]
        return dp[amount]

7-最后一块石头的重量
"""
这道题看出是背包问题比较有难度
最后一块石头的重量：从一堆石头中,每次拿两块重量分别为x,y的石头,若x=y,则两块石头均粉碎;若x<y,两块石头变为一块重量为y-x的石头求最后剩下石头的最小重量(若没有剩下返回0)
问题转化为：把一堆石头分成两堆,求两堆石头重量差最小值
进一步分析：要让差值小,两堆石头的重量都要接近sum/2;我们假设两堆分别为A,B,A<sum/2,B>sum/2,若A更接近sum/2,B也相应更接近sum/2
进一步转化：将一堆stone放进最大容量为sum/2的背包,求放进去的石头的最大重量MaxWeight,最终答案即为sum-2*MaxWeight;、
0/1背包最值问题：外循环stones,内循环target=sum/2倒叙,应用转移方程1
"""
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        target = sum(stones)//2
        #dp[i]容量为i的背包放进去的石头的最大重量
        dp = [0]*(target+1)
        #01背包 最值问题
        for i in range(len(stones)):
            for j in range(target, stones[i]-1, -1):
                    dp[j] = max(dp[j], dp[j-stones[i]]+stones[i])
        return sum(stones)-2*dp[target]

8-掷骰子的N种方法
#投掷骰子的方法数：d个骰子,每个有f个面(点数为1,2,...f),求骰子点数和为target的方法
#分组0/1背包的组合问题：dp[i][j]表示投掷i个骰子点数和为j的方法数;三层循环：最外层为背包d,然后先遍历target后遍历点数f
#应用二维拓展的转移方程3：dp[i][j]+=dp[i-1][j-f]
class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        #dp[i][j] 投i个骰子，点数之和为j的方法数
        dp = [[0]*(target+1) for i in range(d+1)]
        dp[0][0] = 1
        for i in range(1, d+1):
            for j in range(1, target+1):
                for k in range(1, f+1):
                    if j>=k: #点数之和大于当前面上的点数
                        dp[i][j] += dp[i-1][j-k]
        return dp[d][target]

------------------------------------背包问题结束--------------------------------------------

leetcode91-解码方法
@自己写的
class Solution:
    def numDecodings(self, s: str) -> int:
        num_list = []
        for i in list(s):   #将list中的字符串转化为list中的int
            num_list.append(int(i))
        print(num_list)
        dp = [0] * (len(num_list)t+1)
        dp[0] = 0
        for i in num_list:  
            for j in range(len(dp)):
                if i == 0:
                    dp[i] = dp[i-1]
                else: return 0
#❌写到这里卡住了，因为我的判断方法同时需要当前遍历的i和上一轮遍历的i，因此该方法行不通。

@题解
#思路：不难发现，边际最惨的没有组合，比如全是9. 如果有组合，这一题当中就是特殊情况，特殊处理便可；通过题意不难发现，题目当前答案与已遍历的元素有关，因此提出概念：滚动数组
#同时学到了字符串的简易转化方式
#这一题没有想到的是：组合成功不仅仅是+1，而是翻一倍，意味着之前的组合数量，都可以有一个全新的变化
class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        s = ' ' + s 
        f = [0] * (n + 1)
        f[0] = 1
        for i in range(1,n + 1):
            a = ord(s[i]) - ord('0')
            b = ( ord(s[i - 1]) - ord('0') ) * 10 + ord(s[i]) - ord('0') #列表中的字符数字处理方式 ord(s[i]) - ord('0') 
            if 1 <= a <= 9:
                f[i] = f[i - 1]
            if 10 <= b <= 26:
#这个很妙，为什么不是f[i] = f[i-2]*2? #如果是*2，意味着后两位数的两种变化叠加到之前的数上，那么这样遍历的基本元素是两位数而非一位数，因此思路不通。
#除此之外，*2忽略了一种情况，就是i-1跟i-2不是分割开的，他们也有可能组成一位2位数，因此不成立。
#f[i] + f[i-2] = f[i-1] +f[i-2] 如果两者相同，就是跟*2一致；如果不同，f[i-1]就是两个元素分割，而f[i-2]就是两个元素一起叠加到列表中去。
                f[i] += f[i - 2]                            
        return f[n]

@空间优化
#根据思路发现，不需要n+1个f，因为每次循环只跟上两个数字有关系，那么我们可以用一个长度为3的滚动数组来替代。
class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        s = ' ' + s
        f = [0] * 3
        f[0] = 1
        for i in range(1,n + 1):
            f[i % 3] = 0        #定位到余数
            a = ord(s[i]) - ord('0')
            b = ( ord(s[i - 1]) - ord('0') ) * 10 + ord(s[i]) - ord('0')
            if 1 <= a <= 9:     
                f[i % 3] = f[(i - 1) % 3]       #这里重复利用了长度为3的空间，顺序一致，但是位置不一定一致罢了。
            if 10 <= b <= 26:
                f[i % 3] += f[(i - 2) % 3]
        return f[n % 3]
#优化的核心就是: i % 3, (i-1) % 3, (i-2) % 3，顺序一致就行，位置不用管。
#DP对于我的核心难点就是：1.将问题抽象成我熟悉的解题模型。2.发现状态转移的存在 3.状态转移方程的书写

leetcode-300 最长递增子序列
#解法一：暴力 Big O = n^2。但暴力的话就没意思了;
    #同时暴力解法有两种思路：dp[i]为首位的最大递增子序列，dp[i]为末位的最大递增子序列；
    #前者为遍历循环，后者为DP，虽然都是暴力解法，但是后者会快一些，因为利用了i与i-1之间的状态转移
# Dynamic programming.
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums: return 0
        dp = [1] * len(nums)
        for i in range(len(nums)):
            for j in range(i):
                if nums[j] < nums[i]: # 如果要求非严格递增，将此行 '<' 改为 '<=' 即可。 #这一步也可以遍历i之前的所有元素，选择最大的子序列加入
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
#Take-Away: 这题的状态转化思路没有搞懂：dp[i]是以i元素为结尾的最长递增子序列的元素数，在遍历的时候子问题是如果nums[i]> nums[j]，那么就可以讲nums[i]添加到以nums[j]结尾的最大递增子序列队尾
#以当前
#解法二：Big O = n * logn
    #有了对解法一的理解，那么如果将二分查找嵌入到解题思路中，复杂度就会变为Logn；
# Dynamic programming + Dichotomy.
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        size = len(nums)
        # 特判
        if size < 2:
            return size

        # 为了防止后序逻辑发生数组索引越界，先把第 1 个数放进去
        tail = [nums[0]] #注意tail是个子序列数组，tail[i]记录的是长度为(i+1)的递增子序列的最小末尾值
        """
        Q: 为什么要末尾最小  A：因为这样可以避免漏掉某些序列
        Q：贪心的点在哪？ A：遇到最小的，进行末尾替换
        Q：如何确定遍历了所有？基础逻辑是？ A：
        Q：二分优化在哪？ A：在以num为基础的情况下，寻找它在tail中的位置
        """
        for i in range(1, size):
            # 【逻辑 1】比 tail 数组实际有效的末尾的那个元素还大
            # 先尝试是否可以接在末尾
            if nums[i] > tail[-1]:
                tail.append(nums[i])
                continue

            # 使用二分查找法，在有序数组 tail 中
            # 找到第 1 个大于等于 nums[i] 的元素，尝试让那个元素更小
            left = 0
            right = len(tail) - 1
            while left < right:
                # 选左中位数不是偶然，而是有原因的，原因请见 LeetCode 第 35 题题解
                # mid = left + (right - left) // 2
                mid = (left + right) >> 1
                if tail[mid] < nums[i]:
                    # 中位数肯定不是要找的数，把它写在分支的前面
                    left = mid + 1
                else:
                    right = mid
            # 走到这里是因为【逻辑 1】的反面，因此一定能找到第 1 个大于等于 nums[i] 的元素，因此无需再单独判断
            tail[left] = nums[i]
        return len(tail)
#但是tail = [i] 虽然每个i值虽然储藏的是长度为(i+1)的最小末尾值，但是在与原本tail中的值替换的时候还是会造成一些浪费，因为我们只需要替换最后一位就可以，前面的都是锦上添花。
#❌上句后半部分错了！只判断最后一位没有办法确认这个数是否应该放在最后一位还是放在其他位。

    
#“回溯”的思想十分重要，即i 和 i-1，而非i 和 i+1；因为计算机只知道你遍历过的元素，不清楚它还没遍历过的元素。


leetcode-646 最长数对链
#自己的思路是先排序，然后去看首位，从而决定有几个可以满足； 但这个写法算是贪心，不是dp
1- DP
class Solution(object): #Time Limit Exceeded
    def findLongestChain(self, pairs):
        pairs.sort()
        dp = [1] * len(pairs)

        for j in range(len(pairs)):
            for i in range(j):
                if pairs[i][1] < pairs[j][0]:
                    dp[j] = max(dp[j], dp[i] + 1)

        return max(dp)

#for x, y in sorted(pairs, key = operator.itemgetter(1)) 
#for x, y in sorted(pairs, key = lambda x: x[1]) // lambda x, y: x[1], -x[0]

2-greed #自己不完善的思路
class Solution:
    def findLongestChain(self, pairs):
        pairs.sort(key = lambda x: x[1]) #按照末尾排序，跟串气球一样。
        cur, res = float('-inf'), 0
        for x, y in pairs:
            if x > cur:
                cur = y
                res += 1
        return res

leetcode-376 摆动序列
#❌我的思路：每次存在上升/下降的判断，因此需要flag变量去帮助循环判断，但是flag的开始成立适用于非重复集合，因此我的算法必定会有一些特殊情况顾虑不到；
# 当然也可以多加机制去控制这种特殊情况，但复杂度就稍微高了。因此我的思路不太合适，但是大方向是对的！✅
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        size = len(nums)
        if size < 2 : return size
        if size == 2 and nums[0] == nums[1]: return 1
        if nums[0] > nums[1]: 

@题解
#大佬思路果然牛！新建了两个变量，妙处：1. 题目关于ans是有两套判断逻辑的，两个变量各负责一个； 2. 两个变量虽然循环都操作，碰到合适的情况下，操作才有效！🌟这两点很重要，记下！
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        if not nums: return 0
        up, down = 1, 1
        for i in range(1, len(nums)):
            if nums[i] > nums[i-1]:
                up = down + 1
            if nums[i] < nums[i-1]:
                down = up + 1
        return max(up, down)
        #return 0 if len(nums) == 0 else max(up, down)  #用这个更快！
#Big O = n

leetcode-1143 最长公共子序列
#自己的思路是对的。但没有考虑到所有情况，而且不知道如何写出来。
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        n, m = len(text1) ,len(text2)
        dp = [[0] * (m+1) for _ in range(n+1)]
        for i in range(1, n+1):
            for j in range(1, m+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1 #为什么要用[i-1][j-1]? 首先[j-1]容易理解，即从上一个遍历的元素继承过来；[i-1]可以避免掉上个j元素已经使用过的场景（比如两个e对应1个e），因此如果我们发现有该列有满足的情况应该从上一行上一列的角度出发考虑（回溯）

                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1]) #首先明白一点，不满足的时候是如何继承的？ 二维数组总结如下⬇️
        "[i][j]只能通过[i-1][j]/[i][j-1]中来，为什么要max呢？1.如果该行前面有对应上的而本元素没有对应，那么本元素继承从[i-1]处继承 2.如果处理的元素刚好是这个，那么从上一行该行元素继承"
        #比较难理解，希望多做题改善。
        return dp[-1][-1]
# bug写作小能手，dp二维数组的list弄混了。下面的i和j写错位置了；创建二维数组的时候，[0]*n 是一个底层数组，而for _ in range(m)是最长层数组，也就是dp[m][n]
# 子序列的细节处理问题，已经在文中了。

@滚动数组优化
#为什么滚动数组优化？从上面可以发现参与状态转移的变量只有3个，且相互不会影响，这三个变量分别为：dp[i][j], dp[i-1][j], dp[i][j-1]

3-dfs的写法 #复杂度太高了，没必要
class Solution:
    def longestCommonSubsequence(self, t1: str, t2: str) -> int:
        @lru_cache(None)
        def dfs(i, j):
            if not i or not j: return 0
            if t1[i-1] == t2[j-1]: return dfs(i-1, j-1) + 1
            return max(dfs(i-1,j), dfs(i, j-1))
        return dfs(len(t1), len(t2))


leetcode-416 分割等和子集
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        sum_num = sum(nums)
        #如果sum为奇数则肯定无法满足。
        if sum_num & 1:
            return False
        target = sum_num // 2 #一个/出来的是float，两个//出来的是int
        n = len(nums)
        dp = [[False] * (target+1) for _ in range(n)] #bug小能手：这里写错了，中括号围住的只有False

        if nums[0] <= target:
            dp[0][nums[0]] = True

        for i in range(1, n):
            for j in range(target+1):
                #先抄下来，再修改
                dp[i][j] = dp[i-1][j]

                if nums[i] == j:
                    dp[i][j] = True
                    continue
                if nums[i] < j:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i]]
        return dp[n-1][target]
"""主要卡在了思路上：
        1. 我们的目标就是在dp[i][target]上能够取道True，无论i为多少，只要满足在target容量的背包里取到True就行；
            为什么？因为True意味着序列中存在元素组合能够组合成target，而target就是集合的中数，因此只要能取到True就意味着可以分割成两个相同的子序列。
        2. 该题解的遍历顺序和之前的也略有不同，先遍历数字，然后去看他们能放在那个包里不。
        3. 很精妙的一步骤是dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i]]；
            前者意味着这个元素不选的话，元素列表能否形成该背包target；后者意味着，如果选择该元素列表，剩下的的元素能否组成剩下的target（状态转移）
        4. 最重要的思想，如何将这个问题构成01背包问题。一个一个元素去看，容量也一个个增加；
"""
2- 优化的方法「状态数组从二维降低至一维」
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        sum = 0
        for num in nums:
            sum += num
        if sum & 1:
            return False
        target = sum // 2
        n = len(nums)

        dp = [False for _ in range(target + 1)]

        # 依据状态定义做判断:
        # 因为下标[0,0]中nums[0]凑不出0所以设置成False
        # 如果依据状态转移则可以理解为:
        # [j - nums[i]] == 0 表示nums[i]恰好为一组,其余为一组,刚才凑成,所以True没问题
        dp[0] = True

        # 先填表格第 0 行，第 1 个数只能让容积为它自己的背包恰好装满
        if nums[0] <= target:
            dp[nums[0]] = True
        
        for i in range(1, n):
            for j in range(target, -1, -1):
                #「从后向前」 写的过程中，一旦 nums[i] <= j 不满足，可以马上退出当前循环
                # 因为后面的 j 的值肯定越来越小，没有必要继续做判断，直接进入外层循环的下一层。
                # 相当于也是一个剪枝，这一点是「从前向后」填表所不具备的。
                if nums[i] <= j:
                    dp[j] = dp[j] or dp[j - nums[i]]
                else:
                    break

        return dp[-1]
"""
1. 这里可能会有人困惑为什么压缩到一维时，要采用逆序。
    因为在一维情况下，是根据 dp[j] || dp[j - nums[i]]来推d[j]的值，
    如不逆序，就无法保证在外循环 i 值保持不变 j 值递增的情况下，dp[j - num[i]]的值不会被当前所放入的nums[i]所修改，
    当j值未到达临界条件前，会一直被nums[i]影响，也即是可能重复的放入了多次nums[i]，为了避免前面对后面产生影响，故用逆序。 
    举个例子，数组为[2,2,3,5]，要找和为6的组合，i = 0时，dp[2]为真，当i自增到1，j = 4时，nums[i] = 2,dp[4] = dp[4] || dp[4 - 2]为true，
    当i不变，j = 6时,dp[6] = dp [6] || dp [6 - 2],而dp[4]为true，所以dp[6] = true,显然是错误的。 故必须得纠正在正序情况下，i值不变时多次放入nums[i]的情况。
2. 如果是正序的话，后面dp访问前面的dp时得到的是已经更新的内容，此时求的是完全背包问题。
3. 自己的理解：一维时采用倒序，那么我们遍历的j元素在利用[j-nums[i]]时，j元素之前的数据都是i-1行的，是上一个节点的数据，而没有收到该j元素的影响，因此是我们需要的value
"""
3- dfs 记忆化递归
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        self.res = False
        s = sum(nums)
        if  s & 1:
            return False
        memo = {}
        target =    s // 2
        def dfs(i, cur):
            if (cur, i) in memo:        #如果cur,i 在之前的分支中已经遍历过了，那么这里就可以做兼枝处理。
                return memo[(cur, i)]       
            if cur == target:       #确实存在一个集合的和为target
                return True 
            if cur > target or i == len(nums): #cur集合超过了target/探索到头了仍没满足，return False
                return False
            res = dfs(i + 1, cur + nums[i]) or dfs(i + 1, cur)  #只要满足就返回True, 并且dfs分支是两条
            memo[(cur, i)] = res
            return res 
        return dfs(0, 0)


leetcode-494 目标和
1-dfs
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        def dfs(nums, target, index, cur):
            #终止条件
            if index == len(nums):
                return 1 if cur == target else 0
            
            left  = dfs(nums, target, index+1, cur + nums[index])
            right = dfs(nums, target, index+1, cur - nums[index])
            return left + right 
        return dfs(nums, target, 0, 0)
#超时；复杂度2^n

2-记忆化搜素
#优化方法：记录当前解cur与节点i，并且在之后的循环中遇到的话，做剪枝处理。因为无论i节点之前是什么排列组合，之后的所有可能都是一样的，因此可以做剪枝处理。
"""优化代码
memo = ()
if (cur, i) in memo:
    continue
....
memo.append((cur, i))
"""
3-动态规划
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        size  = len(nums)
        s = sum(nums)
        #如果想要的结果大于和，那么无论如何都是取不到的。
        if target > s: return 0 
        dp = [[0] * (2 * s + 1) for _ in range(size + 1)] #因为元素可以是相减，也可以是相加，因此我们并不清楚究竟是什么情况
        dp[0][0 + s] = 1
        
        for i in range(1, size + 1):
            x = nums[i - 1]
            for j in range(-s, s + 1):
                #加上x
                if (j - x) + s >= 0:    #看加上的操作是否满足需求，就是限制操作前的背包容量在这个范围内，下面的if同理
                    dp[i][j + s] += dp[i - 1][(j - x) + s]
                #减去x
                if (j + x) + s <= 2 * s:    
                    dp[i][j + s] += dp[i - 1][(j + x) + s]
                #两个if，代表了两个路径。也就是说该节点如果两个都满足，那么就是两条路；如果只满足一条子条件，那么就是一条路；如果没有路可以走，那么自动忽略，进行下一个遍历。
        return dp[size][target + s]

#复杂度还挺够高的...
#🌟这次背包不一样的地方在其左边界并不是0，而是  s，因此表格是2n+1，而非n。
#优化方向：将无法触达到的区域避免掉，具体操作步骤没有学。

leetcode-474 一和零
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        dp = [[0] * (n + 1) for _ in range(m + 1)]	# 默认初始化0
        # 遍历物品
        for str in strs:
            ones = str.count('1')
            zeros = str.count('0')
            # 遍历背包容量且从后向前遍历！
            for i in range(m, zeros - 1, -1):       #这里利用了zeros - 1巧妙地避开了边界问题，同时实现了剪枝的操作。🌟
                for j in range(n, ones - 1, -1):
                    dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
        return dp[m][n]
#这题按理说不难，状态转移就是是否选择这个元素，如果选择就减去i和j，然后加一；如果不选择，那么直接继承。
#这里涉及到倒序搜索。因为每一行新的数据需要基于上一版本/上一行数据，因此需要特别注意一下，这里需要倒序搜索。




@看答案自己写的跑不出来
#创新点在于这里引进了dp[i][j][k]三维数组，不容易理解。
class Solution:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        size = len(strs)
        dp = [[[0] * (n + 1) for _ in range(m + 1)] for _ in range(size + 1)]
        for i in range(1, size + 1):
            ones  = strs[i - 1].count('1')
            zeros = strs[i - 1].count('0')
            for j in range(m + 1):
                for k in range(n + 1):
                    dp[i][j][k] = dp[i - 1][j][k]

                    if j >= ones and k >= zeros:
                        dp[i][j][k] = max(dp[i - 1][j][k], dp[i - 1][j - zeros][k - ones] + 1)

        return dp[size][m][n]

leetcode-322 零钱兑换
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        #dp[i] 表示换到面值i所需要的最小数量
        dp = [float('inf')]*(amount+1)
        dp[0] = 0 #dp[i]:换到面值i所用的最小数量

        for i in range(len(coins)):
            for j in range(coins[i], amount+1): #最小面值从coins[i]开始计算
                dp[j] = min(dp[j], dp[j-coins[i]]+1)
        
        if dp[amount]!=float('inf'):
            return dp[amount]
        else:
            return -1

#本题的Take-away: 1. 如果不能拼成应怎么办？ 2. 硬币有价值大小？每次j遍历的时候遍历的边界问题。 3.最后应该怎么判断？
#1. 最后的dp[amount] 没拼成的话就算了。
#2. j每次都要从coins[i]开始，如果能拼成就进行状态转移，最后的转移路线应该是从0到amount的一条路径。
#3. 这样下来最后amount就可以判断了。

leetcdoe-518 零钱兑换II
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        #建造矩阵
        size = len(coins)
        dp = [[0] * (amount + 1) for _ in range(size + 1)]
        
        #dp初始化
        dp[0][0] = 1

        for i in range(1, size + 1):
            for j in range(amount + 1):
                dp[i][j] = dp[i - 1][j]
                k = 1
     
                while (j - k * coins[i - 1]) >= 0:
                    dp[i][j] += dp[i - 1][j - k * coins[i - 1]]
                    k += 1
    
        return dp[size][amount]
#这里往前做个复盘，具体的笔记记在了上面的背包问题汇总

leetcode-139 单词拆分
#又是一道模版题，dp的写法自己想出来的；
"""
1. DFS
2. 记忆化搜索
3. BFS
4. 优化BFS
5. 动态规划
6. 优化动态规划
"""
1- DFS
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        
        memo = [None]*len(s)                #记忆化搜索如何理解？比如第一个子树遍历过j元素，j元素的分支可能为True/False，在之后的其他大类子树下再遍历到j节点可以直接跳过，返回j节点的值就行。

        # 以索引i为起始到末尾的字符串能否由字典组成
        def dfs(i):
            # 长度超过s,返回True(空字符能组成)
            if i >= len(s): 
                return True
            # 存在以i为起始的递归结果
            if memo[i] != None:             #这里就是如果该节点记录过的话，直接使用该节点的值就好了，不用再往下遍历了。
                return memo[i]
            # 递归
            for j in range(i,len(s)):
                if s[i:j+1] in wordDict and dfs(j+1): #其中[i:j+1]就是我们当前遍历的字符串，如果满足要求，并且剩下的j+1子树也满足要求，那么就可以进来。
                    memo[i] = True                    #进来后，证明了i节点是可以成功的。并且返回True
                    return True                       #这个返回可以是根节点的True，也可以是各个子节点判断的返回。
            memo[i] = False                           #当遍历完一个子树后，没有成功返回True的话，才会回溯到这里。把这个子树下的各子树判断的memo回溯掉。
            return False
        
        return dfs(0)

2- BFS  #这道题还挺容易理解的。
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        
        visited = [False]*len(s)
        q = collections.deque()         #双向队列
        q.append(0)

        while q:
            i = q.popleft()
            # 节点若访问则跳过
            if visited[i]: 
                continue
            else:
                visited[i] = True
            # 扫描从索引i开始的字符串
            for j in range(i,len(s)):
                # 子字符串在字典中
                if s[i:j+1] in wordDict:
                    # 并且到达结尾，返回True
                    if j == len(s) - 1:
                        return True
                    # 未到达结尾，则添加j+1起始索引到队列
                    else:
                        q.append(j+1)
        
        return False

3- DP 
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        
        # 前n个字符是否能由字典组成
        dp = [False]*(len(s)+1)
        
        # 初始状态
        dp[0] = True

        for i in range(1,len(s)+1): 
            for j in range(i,-1,-1):        
                # 转移公式
                if dp[j] == True and s[j:i] in wordDict:
                    dp[i] = True
                    break           #因为不管j在哪，只要j满足条件，我们就可以说我们的i满足条件。666
        
        return dp[-1]
# dp[i]是[0，i]的字符串能否满足题意，这一段字符串切割为[0, j]和[j, i]。

#当然这个也可以当作一个背包处理问题。

leetcode-377 组合总和IV
#本题与完全背包不同的地方在于：元素的组合顺序不同算是不同的答案。
#🌟这一题有几个全新知识点，类似分组背包

class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        '''
        #二维dp，超时
        dp=[[0]*(target+1) for i in range(target+1)]
        dp[0][0]=1
        res=0
        for i in range(1,target+1):             #表明长度为i的背包
            for j in range(target+1):           #这里容量为j的背包
                for k in nums:                  #遍历nums中的k
                    if j-k>=0:                  #确保容量大于数字，因此才能放下呀。
                        dp[i][j]+=dp[i-1][j-k]  #放入了k这个数组，那么这一组就是上一组相对应的方案，上一组可能+1，可能加+2，因此两款都满足的话都用+=囊括进来
            res+=dp[i][target]
        return res
        ####
        '''
        #一维
        dp=[0]*(target+1)
        dp[0]=1
        for i in range(1,target+1):
            for j in nums:
                if i-j>=0:           #当前num小于背包容量，意味着可以由低纬背包进化而来。
                    dp[i]+=dp[i-j]   #这里的循环相当于bfs，每一次i的循环，就把当前所有的可能性遍历完了。
        return dp[target]
#因为是有序排列，状态转移时考虑的不是是否选择，而是选择哪个数作为该list的结尾。dp[i][j]是i长度的和为j的方案数，因此dp[1]+dp[2]+dp[3]是最终的答案，这个思路可以学习。
#这里最新颖的地方在于针对j的处理，这里的j不再是一个index，而是一个数字了，但是我们的i还是“index”是背包的容量。

leetcode-309 最佳买卖股票事件含冷冻期
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        size = len(prices)
        if size <= 1: return 0

        #dp是指第i天可以取得的最大利润
        dp = [[0] * 3 for _ in range(size)]
        dp[0][0] = 0                #0表示不持有，且当天不卖
        dp[0][1] = -1 * prices[0]   #1表示当天持有（不管买不买）
        dp[0][2] = 0                #2表示不持有，当天卖了

        for i in range(1, size):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][2])              #当天不持有  = 昨天也不持有/昨天卖了
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])  #当天持有   = 昨天持有/昨天买了
            dp[i][2] = dp[i-1][1] + prices[i]                       #今天卖了   = 昨天持有今天卖了。
        
        return max(dp[size - 1][0], dp[size - 1][2])                #最后只比较不持有状态下的最值。

#Take-away：1. 题目中的一些数据/背景可能是迷雾弹，对状态转换没有影响， 针对次不用考虑就行，比如这题的冻结期。
#           2. 状态转移可能又上一个多个状态转移，具体见题目中的for循环。

leetcode-714
#自己写的
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        size = len(prices)
        if size <= 1: return 0

        #dp是指第i天可以取得的最大利润
        dp = [[0] * 2 for _ in range(size)]
        dp[0][0] = 0               
        dp[0][1] = -1 * prices[0]   

        for i in range(1, size):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee)
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i]) 
            
        return dp[size - 1][0]
#复杂度为On
@空间优化
#空间优化为常数
class Solution:
    def maxProfit(self, prices: List[int], fee: int) -> int:
        size = len(prices)
        if size <= 1: return 0

        #dp是指第i天可以取得的最大利润
        dp =[0] * 2
        dp[0] = 0               
        dp[1] = -1 * prices[0]   

        for i in range(1, size):
            dp[0] = max(dp[0], dp[1] + prices[i] - fee)
            dp[1] = max(dp[1], dp[0] - prices[i]) 
            
        return dp[0]
#复杂度为On

leetcode-123
class Solution:
    def maxProfit(self, prices):
        n=len(prices)
        if n==[]:
            return 0

        #初始化，但是有两个组合其实不会参与状态转移，
        dp=[[0,0,0],[0,0,0]] 
        dp[0][0], dp[0][1], dp[0][2] = 0, float('-inf'), float('-inf')
        dp[1][0], dp[1][1], dp[1][2] = -prices[0], float('-inf'), float('-inf')
       
        for i in range(1,n):
            dp[0][1]=max(dp[1][0]+prices[i],dp[0][1])
            dp[0][2]=max(dp[1][1]+prices[i],dp[0][2])
            dp[1][0]=max(dp[0][0]-prices[i],dp[1][0])
            dp[1][1]=max(dp[0][1]-prices[i],dp[1][1])
        return max(dp[0][1],dp[0][2],dp[0][0])
#🌟Take-away,初始化不能全写成0，因为全写成0的话，会遗漏一些情况。因为如果下边某些变量需要暂时小于0，但是一max就会遗漏这些情况，所以会出错！
#因此，在我们取不到的边界，我们可以用float('-inf')

leetcode-188 买卖股票的最佳时机IV
#看下面的题解吧
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        if len(prices) == 0:
            return 0
        dp = [[0] * (2*k+1) for _ in range(len(prices))]        # 这个题解把买和卖拆分成奇偶数而且这个边界没有做限制不太妙。
        for j in range(1, 2*k, 2):
            dp[0][j] = -prices[0]
        for i in range(1, len(prices)):
            for j in range(0, 2*k-1, 2):
                dp[i][j+1] = max(dp[i-1][j+1], dp[i-1][j] - prices[i])
                dp[i][j+2] = max(dp[i-1][j+2], dp[i-1][j+1] + prices[i])
        return dp[-1][2*k]

#！！！看这个题解
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        k = min(k, len(prices) // 2)        #k和len肯定有一个是边界

        buy = [-float("inf")] * (k+1)       #如果是买入动作的话，无法避免小于0的存在
        sell = [0] * (k+1)                  #卖出至少不能亏钱
    
        for p in prices:                                    #遍历每一天的股价
            for i in range(1, k+1):                         #最有趣的点：类似bfs，把每一天的所有状态都更新
                buy[i] = max(buy[i], sell[i-1] - p)         #第i天买了，2种可能：1. 之前买的，今天持股，则为前者； 2.昨天不持股了，今天买的，则为后者
                sell[i] = max(sell[i], buy[i] + p)          #第i天卖了，2种可能：1. 之前卖了，不持股，为前者;     2.昨天买了，今天卖了，为后者。
                # 这里注意神奇的机制，因为sell[i]更新时，利用了更新后的buy[i]的数据，这个时候buy[i]的意思不是昨天买了，而是昨天持股！ 把buy和sell理解为状态而非动作就成。
        return sell[-1]     #不需要注意buy的变化。

leetcode-583 两个字符串的删除操作
#这题自己想的思路——转化成公共子序列还挺对的！
#使用二维数组
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        dp = [[0] * (len(word2)+1) for _ in range(len(word1)+1)]
        for i in range(len(word1)+1):   #这里的定义都非常有趣，因为最快结果要写出来。所以初始化的时候把边界的最坏情况先写出来。
            dp[i][0] = i
        for j in range(len(word2)+1):
            dp[0][j] = j
        for i in range(1, len(word1)+1):
            for j in range(1, len(word2)+1):
                if word1[i-1] == word2[j-1]:        #注意字符串的index和我们矩阵中的index是有偏差的。
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1] + 2, dp[i-1][j] + 1, dp[i][j-1] + 1)
        return dp[-1][-1]
#这题中间的状态转移有点难以理解。
#   1. dp[i][j]是走到i，j时候使两个字符串相同删除的最小元素，最坏答案就是i+j，完全没有相同的。 记住处理后的dp状态为子序列满足题意状态下的最小步数
#   2. 状态转移：dp[i][j]跟三个变量有关系。1. i和j都不选，因此答案为XXX+2; 2. 只选i或只选j，那么只用处理一步就可以了，即 +1。

leetcode-72  编辑距离
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n1 = len(word1)
        n2 = len(word2)                 #把长度命名出来后会快很多！
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
        #初始化
        for j in range(1, n2 + 1):
            dp[0][j] = dp[0][j-1] + 1
        for i in range(1, n1 + 1):
            dp[i][0] = dp[i-1][0] + 1
        #循环
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1] ) + 1
        #print(dp)      
        return dp[-1][-1]
#这一题同583题不同的地点在于状态的转换。
#因为支持添加、删除、替换，所以只要有不一样的，无论是单个不一样，还是两个不一样，都只用一步操作就可以解决。


leetcode-650 只有两个键的键盘
class Solution:
    def minSteps(self, n: int) -> int:
        
        dp = [0] * (n + 1)

        for i in range(2, n+1): #因为如果只有一个A的话，不需要进行操作，所以从2开始
            dp[i] = i           #初始化可以放在这里，就不用多写一个循环了
            nb = int(i ** (0.5)) + 1
            for j in range(2, nb):
                if i % j == 0:
                    dp[i] = dp[j] + dp[int(i/j)] 
        return dp[n]
#这题的思路比较奇妙：copyall的存在证明了我们的可以优化的对象是能够用n组相同数字相加得到的。像素数就不行，因为除了1和它自己没有其他因数。
#状态转移非常奇妙！明白转移过程，我们就可以得到dp[i]的优化方式就找到一个可以组成i的x，然后看有多少个x可以组成。
#那为什么不直接用n/x,而用i/j呢？ 因为i/j可能继续被优化！这个时候可以利用已经更新过的i/j；
#在i的一个循环里dp[i]可以多次被操纵。j越大，那么操作次数越少！所以这个顺序没有什么影响。 自己试过了！
#这题妙蛙蛙！标记一下。🌟