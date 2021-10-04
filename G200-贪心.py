leetcode-455
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        ans = 0
        g.sort()
        s.sort()
        n = len(g)
        m = len(s)
        p0 = n - 1
        p1 = m - 1
        while p0 >= 0 and p1 >= 0:
            if s[p1] < g[p0]:
                p0 -= 1
            elif s[p1] >= g[p0]:
                ans += 1
                p0 -= 1
                p1 -= 1
        return ans
#看过答案自己写的，先把大块的饼干分享给胃口大的。
#模范答案的时间稍好
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        n, m = len(g), len(s)
        i = j = count = 0

        while i < n and j < m:
            while j < m and g[i] > s[j]:
                j += 1
            if j < m:
                count += 1
            i += 1
            j += 1
        
        return count
#官解思路不太一样，从小的走，小的先喂饱，喂饱后再喂大的。因为题目设置，所以两种思路都可以，可能自己的思路会造成一丢丢资源浪费。

leetcode-435
#自己写的——错了，只适用于从头开始的情况，没想到如果第一个元素是[1,100]怎么办？
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        ans = 0 
        p0, p1 = 0, 1
        n = len(intervals)  #包含max_index
        while p1 < n and p0 < n:
            if intervals[p0][1] > intervals[p1][0]:
                p1 += 1
                ans += 1
            else:
                p0 += 1
                p1 += 1
        return ans

1- 贪心
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        
        intervals.sort(key=lambda x: x[1]) #按照尾元素排列
        #这个按照尾巴排序是整个贪心的核心。
        n = len(intervals)                 #长度
        right = intervals[0][1]             
        ans = 1

        for i in range(1, n):
            if intervals[i][0] >= right:
                ans += 1    #ans是有几个区间可以连在一起
                right = intervals[i][1]
        
        return n - ans

leetcode-452
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if not points:
            return 0 

        points.sort()
        print(points)
        n = len(points)
        i = 0
        ans = j = 1 

        while True: #这里再写什么其他指针的边界条件就没有意义了，因为你看while循环外面是没有的return的，另一方面这也是个hint说明你的代码写的垃圾。

            if j == n :
                return ans

            while points[i][1] >= points[j][0] and j < n :
                if j == n - 1:
                    return ans
                else:
                    j += 1
            
            i = j
            j += 1
            ans += 1       
 
#上面是自己写的，但是超时...我写的思想是，如果最靠左的气球的右边界都能囊括到其他气球左边界的话，也就是这个气球的右边界的箭能够射穿这些气球，则可以计算出来。
#这代码写的好垃圾，一直在改bug
"""
几个笔记可以做：
1. 因为这里不是同向双指针，因此while跳出条件者利用大小不太好用，我写的跳出条件直接为return
2. 上面算法致命的问题，没有考虑到一种情况：如果所有第一个值为[1，100]，其他值都是1～10，只有一个为11～20，那么这个时候只能用2个指针
3. 没有把实际问题抽象出来形成模型。
"""
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if not points:
            return 0
        
        points.sort(key=lambda balloon: balloon[1])
        pos = points[0][1]
        ans = 1
        for balloon in points:
            if balloon[0] > pos:
                pos = balloon[1]
                ans += 1
        
        return ans
#贪心思想有趣：首先发现，如果想要弓箭尽量少的话，那么我们一定是要使得它穿越更多的气球。同时因为气球是固定的，所以我们只要从尾部在最左侧的开始着手（下面会解释
#如果有气球的首部能和我们找到的气球的最右侧重叠，那么一个弓箭可以解决这么多；（这就是贪心，每一发弓箭，我们尽量带走气球
#所以往右走，如果有气球x没有达到上面那个气球的右侧，也就是说原来的那根弓箭无法穿破气球x，那么就以这个气球x为基准，新增一根弓箭，重复上面的操作即可，直到最后。
#我的答案错的主要原因是因为思路不对！

leetcode-406
@总体思路是先排序再插入
1-从高到低
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people.sort(key=lambda x: (-x[0], x[1]))
        n = len(people)
        ans = list()
        for person in people:
            ans[person[1]:person[1]] = [person] #这个切片技术666，可以将[person]中加入到ans，并且自动扩张长度。
        return ans

2- 从低到高
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people.sort(key=lambda x: (x[0], -x[1]))    #二维数组先按照x[0]顺序，再按照x[1]的倒序排雷
        n = len(people)
        ans = [[] for _ in range(n)]
        for person in people:
            spaces = person[1] + 1  #这里+1因为自己本身还占着一位数
            for i in range(n):
                if not ans[i]:      #如果ans[i]有值，那么这个地方不能计数，因为我们先插的是小的值，目前有值的是不应该被纳入考虑的
                    spaces -= 1
                    if spaces == 0:  #如果等于0，那么就是这个数的位置
                        ans[i] = person
                        break
        return ans

leetcode-121
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        res = 0 
        low = float("inf")  #用来最开始设定边界用
        for price in prices:
            low = min(low, price)   
            res = max(res, price - low)
        return res
#这一题第二次做了，没做出来，原因是没有理解核心思想
#low是为了每日最小值，res的是找最小值与当前price的最大价差。
@我自己会考虑所有情况-这样想不明白，因为维度太多而且相互交叉。
#答案的思路：只在乎最小值，和当前price与最小值的价差，这两个要素，就可以应对所有情况，cool！


leetcode-122
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        low = 0
        high = 1 
        n = len(prices)
        ans = 0

        while high < n:
            if prices[high] > prices[low]:
                ans += prices[high] - prices[low]
            high += 1
            low += 1

        return ans 
#手撕-简单

leetcode-605
#防御式编程，如何处理数列两端的数据，可以在首尾加上0
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        flowerbed = [0] + flowerbed + [0]  #也可以用insert()
        i = 1
        ans = 0
        while i < len(flowerbed) - 1:
            if flowerbed[i] == 0 and flowerbed[i-1] == 0 and flowerbed[i+1] ==0:
                i += 2
                ans += 1
            else:
                i += 1
        return n <= ans
#时间可以是快的，但是由于添加了两个元素，因此空间可能牺牲一些
#还有一种解法，就是数连续的0，然后利用数学归纳法，决定可以种多少花

leetcode-392
#用双指针很容易就做出来饿了
#这道题也可以用DP矩阵去做，= 最短路径问题
@边界问题
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        dp = [[0] * len(s) for _ in range(len(t))] #dp的index应该和i契合，这里需要加一
        for i in range(len(s)- 1):
            for j in range(len(t) - 1):
                if s[i] == t[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1 #最开始值都为0，如果发现相同字母，就计数
                else: 
                    dp[i][j + 1] = dp[i][j]
        return dp[-1][-1] == len(s)
#这个思路是对的，但是list的边界有问题，当遍历最后一个字母的时候，dp已经放不下了。下方为优化版本，主要改动在index上


@初始化问题
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        dp = [[0] * (len(t) + 1) for _ in range(len(s) + 1)]  #这里如果错了会导致index assignment error
        for i in range(len(s)): #不减一是因为要保证遍历所有
            for j in range(len(t)):
                if s[i] == t[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1 #因为index的存在，所以上面矩阵加1
                else: 
                    dp[i][j + 1] = dp[i][j]
        return dp[-1][-1] == len(s)
#笑死我了，我为什么犯这么困难的错误
"""
遍历顺序是以i/s为基准，遍历j然后将之后的全部+1，比如遇到了字母a，就算之后t中再有字母a，这个算法也可以保证dp数字一样，从而达到跳过之后相同字母的影响
模版答案为下，我写的逻辑是没有逻辑，因为是按照答案改的！🌟🌟🌟重点 这是这一题的take-away
我考虑的是将结果存在下一part中，但是没有考虑之后的所有逻辑，所以出错。
"""
@模版答案
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        dp = [[0] * (len(t)+1) for _ in range(len(s)+1)]
        for i in range(1, len(s)+1):
            for j in range(1, len(t)+1):
                if s[i-1] == t[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = dp[i][j-1]
        if dp[-1][-1] == len(s):
            return True
        return False

#TIP：遍历数组时，判断元素i与元素i+1时，最好不要修改i+1 而是修改i，以免接下里的判断

#禁止面向案例编程：在没有很好的办法一次性处理问题时，即无法处理边际/条件等，要重新思考思路，看有没有更好的处理方法去解决问题，否则就会导致“面向案例编程”，也就是代码又臭又长，这个时候倒不如直接看题解爽快。

#index使用i和i+1时会出现尴尬的情景，判断逻辑里每一次循环判断应该只针对一个新的元素，但是此时无法考虑到队首，因为刚入循环的时候，是进入了两个元素，如果要删除，无法判断是删除i还是删除i+1
#这个时候需要判断，这种尴尬的场景跟nums的size是否有关，如果3个以内是这种特例，那么通过遍历的开始或者特殊值处理掉就好。

#能用for就不用while，用while是因为对指针可以进行处理； 可以多用i和i-1，因为这样只用限制范围，比较容易理解，针对边界处理也比较简单。


leetcode-665
@难点_在于所有情况没有充分考虑到并且不清楚在各个情况下对应的的操作方式
class Solution:
    def checkPossibility(self, nums: List[int]) -> bool:
        p = 1 
        count = 0
        n = len(nums)
        while p < n:
            if nums[p - 1] > nums[p]:
                count += 1
                if p == 1 or nums[p] >= nums[p-2]:
                    nums[p - 1] = nums[p]
                else:
                    nums[p] = nums[p - 1]      
            p += 1
        return count <= 1
#递增序列三种情况：1.最大值在队首 2.突出值在两元素的第一位 3.突出值在两元素的第二位
#宏观来看：除了边界情况，在队列中遇到异常情况会有两种情形，两种情形的处理方式不同。

leetcode-53
@最大子序和
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        temp = nums[0]                      #这个操作十分重要，如果不是直接操作temp，而是以0入场，那么如果没有满足的条件的话，返回的值为0，尽管答案为负数
        ans = temp

        for i in range(1,len(nums)):
            temp = max(0, temp) + nums[i]    #temp = 子序列最后一位为nums[i]的最大值
            ans = max(ans, temp)             #ans  = 1.保存最大值 2.比较不同子序列的和
        return ans
#这种也算分治思想
#下面是大神DP的算法，牛。
class Solution(object):
    def maxSubArray(self, nums):
        for i in range(1, len(nums)):
            nums[i]= nums[i] + max(nums[i-1], 0) 
        return max(nums)

leetcode-763
#这一题题目理解起来有难度，贪心算法的体现是从头针对单个字母进行处理。
class Solution:
    def partitionLabels(self, S: str) -> List[int]:
        dic = {s: index for index, s in enumerate(S)}   #dic储存字母 和 最后一次出现的index，实现by for的遍历循环 #注意enuerate的用法

        num = 0  #直接计数
        result = []
        j = dic[S[0]]  #第一个字符的最后位置 #S[0]是列表


        for i in range(len(S)):  #逐个遍历
            num += 1  #找到一个就加1个长度
            if dic[S[i]] > j:  #思路一样，如果最后位置比刚才的大，就更新最后位置
                j = dic[S[i]]
            if i == j:  #思路一样，形式不同，这里就是找到这一段的结束了，就说明当前位置的index和这个字母在字典里的最后位置应该是相同的。 -- 精彩
                result.append(num)  # 加入result
                num = 0 # 归0  -- 精彩
        return result 
