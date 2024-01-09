# 1339. Maximum Product of Splitted Binary Tree
class Solution:
    def maxProduct(self, root: Optional[TreeNode]) -> int:
        all_sum = [] # to store the sum of each sub-tree 这个是关键，存了n个node的值。
        def dfs(node):
            if not node: return 0
            left = dfs(node.left)
            right = dfs(node.right)
            tt = left+right+node.val
            all_sum.append(tt)
            return tt
        total = dfs(root)
        best = 0
        for s in all_sum: 
            best = max(best, s*(total-s))
        return best % (10**9 + 7)

# 710. Random Pick with Blacklist
# 核心思想是：将位于 [0, n-len(blacklist)-1] 范围内的黑名单数字映射到不在该范围内的非黑名单数字上。
import random
class Solution:
    def __init__(self, n, blacklist):
        bLen = len(blacklist)
        self.blackListMap = {}
        blacklistSet = set(blacklist)

        # init mapping dict
        for b in blacklist:
            if b >= (n-bLen): # n-blen是目前可以选的，如果b大于这个值，不在设置
                continue
            self.blackListMap[b] = None # 我们只把小于n-blen的黑名单的值初始化出来，因为它们需要重新映射

        self.numElements = ptr = n - bLen # 需要映射的range开端。我们需要关注的是[0, n-bLen]
        
        for b in self.blackListMap.keys():
            while ptr < n and ptr in blacklistSet:
                ptr += 1
            self.blackListMap[b] = ptr
            ptr += 1
    
    def pick(self):
        randIndex = int(random.random() * self.numElements)
        return self.blackListMap[randIndex] if randIndex in self.blackListMap else randIndex
# 好聪明这个写法。


# 329. Longest Increasing Path in a Matrix
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        m, n = len(matrix), len(matrix[0])
        visited = [[0]* n for _ in range(m)] 

        # 如果不是DAG，既不是有向图，是不能够用memorization的
        def dfs(x, y):
            if visited[x][y]: return visited[x][y] # 如果是0就是没有经历过！
            for nx, ny in ((x+1, y),(x-1, y),(x, y-1),(x, y+1)):
                if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] > matrix[x][y]:
                    visited[x][y] = max(visited[x][y], dfs(nx, ny)) # 不能在max()里存放+1 这样多次相同层的遍历会将+1重复计算
            visited[x][y] += 1 # 这里有点意思哦～
            return visited[x][y]    

        ans = 0
        for i in range(m):
            for j in range(n):
                ans = max(ans, dfs(i, j))

        return ans
    

# 43. Multiply Strings
class Solution:
    def multiply(self, num1, num2):

        m, n = len(num1), len(num2)
        product = [0] * (m + n)

        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                mul = int(num1[i]) * int(num2[j])
                # 这里的index操作是精髓，product[]里的存放顺序是对的，我只需要最终join起来就成了。
                p1, p2 = i + j, i + j + 1 # i+j+1是低位，i+j是高位，这里需要想明白。
                total = mul + product[p2]

                product[p1] += total // 10
                product[p2] = total % 10

        result = ''.join(map(str, product))
        return result.lstrip('0') or '0'
    

# 1013 Partition an array into three parts with equal sum
class Solution:
    def canThreePartsEqualSum(self, arr: List[int]) -> bool:
        tt = sum(arr)
        if tt % 3: return False
        st = tt / 3
        tempt = 0
        cnt = 0
        for a in arr:
            tempt += a
            if tempt == st:
                cnt += 1
                tempt = 0

        return True if cnt >= 3 else False # 这里是数学的关系！精髓是这个>= 3


# 692. Top K Frequent Words
import heapq
class Solution:
    # def topKFrequent(self, words: List[str], k: int) -> List[str]:
    #     items = collections.Counter(words).items()
    #     return [k for k, v in sorted(items, key = lambda x: (-x[1], x[0]))[:k]]
        
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        cnt = Counter(words)
        heap = [(-freq, word) for word, freq in cnt.items()]
        heapify(heap)
        return [heappop(heap)[1] for _ in range(k)]




# 528. Random Pick with Weight
# 这一题的精华在于如何能够实现按照weight的权重，随机选取值。
# -> 我们利用prefix，这样n个值，每两个值之间的prefix不一样，就看作total_sum的相对应的权重。
import bisect
class Solution:
    def __init__(self, w: List[int]):
        self.prefix = []
        self.total_sum = 0
        for n in w:
            self.total_sum += n
            self.prefix.append(self.total_sum)
        
    
    def pickIndex(self) -> int:
        target = self.total_sum * random.random()
        left, right = 0, len(self.prefix) - 1
        return bisect.bisect_left(self.prefix, target, left, right)
        while left < right:
            mid = (left+right) // 2
            if target > self.prefix[mid]:
                left = mid + 1
            else:
                right = mid 
        # 为什么要return left，你对二分的理解不够！
        # 你要找的是什么值？比target大的第一个值！
        # why什么找这个？假定prefix1, prefix2，它们的差值是x(p2-p1), 那么x在整个total_sum的比重就是p1~p2/total_sum，如果target落在了p1~p2,那么右侧第一个值就是x，也就是我们要找的index/value
        return left
        
# 44. Wildcard Matching
class Solution:
    def isMatch(self, s, p):
        m, n = len(s), len(p)
        dp = [[False] * (n+1) for _ in range(m+1)]
        dp[0][0] = True
        
        # 如果p中有*，就把第一行的给init掉。
        # 初始化没必要把首行/首列全部初始化掉。
        for j in range(1, n+1):
            if p[j-1] == '*':
                dp[0][j] = dp[0][j-1]
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                if p[j-1] == s[i-1] or p[j-1] == '?': # 不需要判断dp[i-1, j-1]只需要直接集成dp[i-1, j-1]的值就好了！trick！
                    dp[i][j] = dp[i-1][j-1]
                elif p[j-1] == '*':
                    # 意味着什么? 匹配一个/多个字符 or 不匹配字符
                    dp[i][j] = dp[i-1][j] or dp[i][j-1]
        
        return dp[m][n]
"""
这道题dp对于我来讲好难理解。
1. 首先是两个for循环彼此的意义
2. 状态转移方程中，状态为什么这么转移/dp[i-1,j],dp[i,j-1]到dp[i,j]分别什么意思
这两个问题是Independent的。
首先理解整体for循环的思路 -> 在初始化的时候，我们把能用的首行的*都给初始化了。
1. 每一个i循环就意味着一个完整的for-j循环: 针对i这一位，我们看了所有j的可能性，并且尝试了在只到当前i的情况下，j能走多远。
那么接下来，我们来看for循环里面的detail
2. 如果两个相同(p[j-1]==s[i-1])或者遇到了?，直接继承，这很好理解; 如果遇到了'*'，会有两种情况
    2.1 我们去看dp[i, j-1], 当我们遍历到当前j层的时候，在j-1层i的所有情况都遍历了，因此我们可以从dp[i, j-1]从去继承，这意味着我们尝试不匹配j(*)这一位，如果前面能匹配成功，那么dp[i,j]理所应当是True;
    2.2 dp[i-1, j]呢？这个时候我们知道j位(*)一定能匹配到i位(s[i])，但是dp[i,j]存放什么我们不清楚，因此我们去看dp[i-1, j]。dp[i-1，j]也有点意思，因为j这一行都是用*进行匹配的，所以我们可以看做往前找最早的i能够满足匹配上的条件。那为什么不往后看，因为for循环单向性，我们只用顾着前面就行了。
"""


# 289. Game of Life
# 这一题如果不用额外的数组的话，精髓在于使用信号量！
# 原来是live/dead + 遍历过后是live/dead = 4 case, 因此四个信号量就可以解决了。
class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        
        neighbors = [(1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1), (0,1), (1,1)]

        rows = len(board)
        cols = len(board[0])

        for row in range(rows):
            for col in range(cols):
                live_neighbors = 0
                for neighbor in neighbors:
                    r = (row + neighbor[0])
                    c = (col + neighbor[1])
                    if (r < rows and r >= 0) and (c < cols and c >= 0) and abs(board[r][c]) == 1:
                        live_neighbors += 1
                if board[row][col] == 1 and (live_neighbors < 2 or live_neighbors > 3):
                    board[row][col] = -1
                if board[row][col] == 0 and live_neighbors == 3:
                    board[row][col] = 2

        for row in range(rows):
            for col in range(cols):
                if board[row][col] > 0:
                    board[row][col] = 1
                else:
                    board[row][col] = 0



# 271 huffman encoding
# What to use as a delimiter? Each string may contain any possible characters out of 256 valid ascii characters.
class Codec:
    def encode(self, strs: List[str]) -> str:
        """Encodes a list of strings to a single string.
        """
        if len(strs) == 0: return chr(258)
        return chr(257).join(x for x in strs)        

    def decode(self, s: str) -> List[str]:
        if s == chr(258): return []
        return s.split(chr(257))

# 342. power of 4
from math import log2
class Solution:
    def isPowerOfFour(self, num: int) -> bool:
        return num > 0 and log2(num) % 2 == 0
# 349 简单 直接过

# 547. Number of Provinces
class Solution:
    def findCircleNum(self, M):
        n = len(M)
        seen = set()
        
        
        def dfs(curIndex):
            for index, neighbor in enumerate(M[curIndex]):
                if neighbor and index not in seen:
                    seen.add(index)
                    dfs(index)
        
        res = 0
        for i in range(n):
            if i not in seen:
                dfs(i)
                res += 1
        return res
    
"""
MD 53和918都是找最大substring -> 用Kadane/类似DP
但是918却不能够利用 (延长list到2倍+常规的套路去处理Kadane) why? -> 会跳过最大数组。
在53-Kadane中 我们每次是如何缩小边界的? cur_sum = max(cur_sum+num, num) 来跳过前面有negative impact的组合
比如[5, 6, -2, -7, 8, ....]这个数组 因为for循环遍历的原因 我们是不会跳过任何可能的值的
但是在918中 我们不得不可能要缩小左边边界从而导致window=[6,-2,-7, 8] 这样的话, 是不能满足贪心的思想的, 哪怕left此刻指向6, which is a postive num
而是要找到此刻left~right的已right为右边界的substring的最大可能情况, 在上面这个例子中应该为[8], 这样才是比较的套路
Kadane的话不会遇到这种情况, 因为它在左边的选取范围是固定的就是0~right，无论长短；而918是需要还要看看最短的有没有有更大的，这样才是贪心，因为还要circular，还要去看队首。
"""

# 918. Maximum Sum Circular Subarray
# 方法1 两次遍历 第一次住要看
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        n = len(nums)
        leftMax = [0] * n
        # 对坐标为 0 处的元素单独处理，避免考虑子数组为空的情况
        leftMax[0], leftSum = nums[0], nums[0] 
        pre, res = nums[0], nums[0]
        for i in range(1, n):
            pre = max(pre + nums[i], nums[i]) # 正常如53题，只考虑nums[i:j]
            res = max(res, pre) # Kadane
            leftSum += nums[i] # prefix_sum
            # leftMax存的是当前最大的prefix_sum 在下一遍中向前遍历时可以与rightfix_sum结合。
            leftMax[i] = max(leftMax[i - 1], leftSum) 
        # 从右到左枚举后缀，固定后缀，选择最大前缀
        rightSum = 0
        for i in range(n - 1, 0, -1):
            rightSum += nums[i]
            res = max(res, rightSum + leftMax[i - 1])
        return res


# 找到max_substrin, min_substring然后看哪个大。聪明的trick
class Solution:
    def maxSubarraySumCircular(self, nums: List[int]) -> int:
        n = len(nums)
        preMax, maxRes = nums[0], nums[0]
        preMin, minRes = nums[0], nums[0]
        sum = nums[0]
        for i in range(1, n):
            preMax = max(preMax + nums[i], nums[i])
            maxRes = max(maxRes, preMax)
            preMin = min(preMin + nums[i], nums[i])
            minRes = min(minRes, preMin)
            sum += nums[i]
        if maxRes < 0: # 如果maxRes<0，意味着全员negative，这样的sum-minRes一定为0, 但是0不是substring
            return maxRes
        else:
            return max(maxRes, sum - minRes)

# 1048. Longest String Chain
# 切片器的妙用
class Solution:
    def longestStrChain(self, words):
        dp = collections.defaultdict(int)
        for w in sorted(words, key=len):
            dp[w] = max(dp[w[:i]+w[i+1:]] + 1 for i in range(len(w)))
        return max(dp.values())
        

class Solution:
    def longestStrChain(self, words: List[str]) -> int:
        words = list(set(words))
        words.sort(key=len)
        n = len(words)
        dp = [1] * n
        
        def isChain(s, t):
            if len(s) + 1 != len(t):
                return False
            cnt = 0
            s_ptr = 0
            for ch in t:
                # 如果碰到相同的就移动s_ptr, s_ptr已经移动完了，如果想要成立就要s完全匹配，因此移动完的话，cnt+=1 我们期望cnt至少为1.
                if s_ptr < len(s) and ch == s[s_ptr]:
                    s_ptr += 1
                else:
                    cnt += 1
                if cnt == 2:
                    return False
            return True
        
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                if isChain(words[i], words[j]):
                    dp[j] = max(dp[j], dp[i] + 1)

        return max(dp)
    

# 698. Partition to K Equal Sum Subsets
class Solution:
    def canPartitionKSubsets(self, arr: List[int], k: int) -> bool:
        n = len(arr)
        total_array_sum = sum(arr)
        if total_array_sum % k != 0: return False
        target_sum = total_array_sum // k
        arr.sort(reverse=True) # 优化的地方
        taken = ['0'] * n
        
        # 唯一一个优化的地方
        memo = {}
        def backtrack(index, count, curr_sum):
            n = len(arr)
            
            taken_str = ''.join(taken)
            if count == k - 1: return True
            if curr_sum > target_sum: return False

            # 之所以不用t/f，而用1/0，就是因为我们想达到减枝
            if taken_str in memo: return memo[taken_str]
            
            if curr_sum == target_sum:
                memo[taken_str] = backtrack(0, count + 1, 0)
                return memo[taken_str]
        
            for j in range(index, n):
                if taken[j] == '0':
                    taken[j] = '1'
                    if backtrack(j + 1, count, curr_sum + arr[j]): return True
                    taken[j] = '0'
    
            memo[taken_str] = False
            return False
        
        return backtrack(0, 0, 0)
    
# 1186. Maximum Subarray Sum with One Deletion
# 这一题好nb。注意here_1队列中，从index的操作可以看出我们只考虑了删除[1......n-2]的index的情况
# 那么为什么没有考虑删除0/(n-1)index的情况呢？因为在我们的here_0中使用的是kadane，已经考虑过了，就是53的变种，完整队列找substring。
class Solution:
    def maximumSum(self, arr: List[int]) -> int:
        n = len(arr)
        max_ending_here0 = n * [arr[0]]  # no deletion
        max_ending_here1 = n * [arr[0]]  # at most 1 deletion
        for i in range(1, n):
            max_ending_here0[i] = max(max_ending_here0[i-1] + arr[i], arr[i])
            max_ending_here1[i] = max(max_ending_here1[i-1] + arr[i], arr[i])
            if i >= 2:
                max_ending_here1[i] = max(max_ending_here1[i], max_ending_here0[i-2] + arr[i])
        return max(max_ending_here1)
    

        
class Solution:
    def isValid(self, s: str) -> bool:
        m = {
            "]":"[",
            "}":"{",
            ")":"("
        }

        stack = []

        for ch in s:
            if stack and ch in m and stack[-1] == m[ch]:
                stack.pop()
            else:
                stack.append(ch)

        return not stack



# 706
class Bucket:
    def __init__(self):
        self.bucket = []

    def get(self, key):
        for (k, v) in self.bucket:
            if k == key:
                return v
        return -1

    def update(self, key, value):
        found = False
        for i, kv in enumerate(self.bucket):
            if key == kv[0]:
                self.bucket[i] = (key, value)
                found = True
                break

        if not found:
            self.bucket.append((key, value))

    def remove(self, key):
        for i, kv in enumerate(self.bucket):
            if key == kv[0]:
                del self.bucket[i]


class MyHashMap(object):
    def __init__(self):
        # better to be a prime number, less collision
        self.key_space = 2069
        self.hash_table = [Bucket() for i in range(self.key_space)]


    def put(self, key, value):
        hash_key = key % self.key_space
        self.hash_table[hash_key].update(key, value)


    def get(self, key):
        hash_key = key % self.key_space
        return self.hash_table[hash_key].get(key)


    def remove(self, key):
        hash_key = key % self.key_space
        self.hash_table[hash_key].remove(key)





# 252. meeting room -> sort之后，扫描线做法，比较last_end和new_start -> 返回t/f

# 253. Meeting Rooms II
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        # if not intervals: return 0
        # intervals.sort()
        # room = 1
        # pq = []
        # heapq.heappush(pq, intervals[0][1])
        # for s, e in intervals[1:]:
        #     if pq and s >= pq[0]:
        #         heapq.heappop(pq)
        #     heapq.heappush(pq,e)
        #     room = max(room, len(pq))
        # return room


        if not intervals: return 0
        intervals.sort()
        room = 1 
        pq = []
        heapq.heappush(pq, intervals[0][1])
        for s, e in intervals[1:]:
            if s >= pq[0]:
                heapq.heappop(pq) 
                room -= 1
            heapq.heappush(pq, e)
            room += 1
        return room
    
# 2402. Meeting Rooms III
# 这题如果你每次找最小的endtime是不可以的，因为有些endTime大，roomNumber小，但仍然满足题意，你会忽略这种情况
# 因此每次遇到新的会议的时候，你需要得到所有可用的meeting room，因此需要一个数据结构来帮助你。
class Solution:
    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
        meetings.sort()
        roomInUse = []
        roomSpare = [i for i in range(n)]
        record = collections.defaultdict(int)

        for s, e in meetings:
            # 1/看看有没有用完的会议室
            while roomInUse and s >= roomInUse[0][0]:
                time, room = heapq.heappop(roomInUse)
                heapq.heappush(roomSpare, room)

            # 1/有空房
            if roomSpare:
                room = heapq.heappop(roomSpare)
                heapq.heappush(roomInUse, [e, room])
            # 2/没空房
            else:
                nextTime, room = heappop(roomInUse)
                heapq.heappush(roomInUse, [nextTime+e-s, room])
            record[room] += 1
        print(f"record: {record}")
        return sorted(record.items(), key=lambda x: (-x[1], x[0]))[0][0]
        # 最后找使用最多room的也可以指使用一个单一的list
        # res = [0] * n           # 每个room用过多少次
        # return res.index(max(res)) # 
        


# 1.线程和进程有什么区别？
# 2. 解释 TCP 和 UDP 的 3 个区别。
# 3. 解释带宽和吞吐量。它们有何不同？
# process vs thread, latency vs throughput

# Process vs Thread:
# Process: Independent unit of execution with its own memory space.
# Thread: Smallest unit of execution within a process, shares process's memory.


# TCP vs UDP:
# Reliability: TCP is connection-oriented and ensures data delivery, while UDP is connectionless and doesn't guarantee delivery.
# Header Size: TCP has a larger header than UDP, thus more overhead.
# Use cases: TCP is used for applications requiring reliability like web browsing, and UDP for streaming and real-time applications.


# Bandwidth vs Throughput:
# Bandwidth: Maximum rate of data transfer over a network.
# Throughput: Actual rate at which data is successfully transmitted.


# Latency vs Throughput:
# Latency: Time taken for a single data packet to travel from source to destination.
# Throughput: Amount of data transferred from source to destination in a given period of time.





# LC-200/ Grid1, Grid2 / Totally same Island. 如何判断两个岛屿形状相同？在dfs的时候添加一个路径参数，
# 每次往path中添加direction，根据方向的固定性，在每次遍历完成后都会获得一个字符串，如果字符串相同，就意味着两个变量完全相同。


# 给几个点的坐标，每个点有一个名字。 给某一个点，要你找到距离最小的点，而且这个点要和给定的点有相同x 或者相同 y。如果距离相同，则按照字母顺序排，小的字母优先。 好简单
    


# 前人总结大全 # https://sammy-sheng.gitbook.io/two-sigma/


# 第二轮面试 给一个棋盘，要你implement connect6 这个游戏， 需要写几个function: 1) reset 棋盘， 2）告诉你下一个player 是谁 3） 第一个player 摆棋子 4） 第二个player 摆棋子。 follow up 是要在constant time 判断是否当前player赢
# https://en.wikipedia.org/wiki/Connect6


# random number generater






# 在这个预测游戏中，第一个玩家给第二个玩家一些股票市场数据
# 连续几天。数据包含公司每天的股票价格。游戏规则是
# - 玩家 1 会告诉玩家 2 一个天数
# - 玩家 2 必须找到最近的股票价格小于给定日的日子
# - 如果有两个结果，则玩家 2 找到较小的天数
# - 如果不存在这样的日子，则返回 -1。
# 示例 n=10
# 库存数据=[5,6,8,4,9,10,8,3,6,4]
# 查询=[6,5,4]
# 该函数应返回一个 INTEGER_ARRAY。
# 公共静态列表&lt;整数&gt; predictAnswer(List<Integer> stockData, List<Integer> 查询)

# 面试 2 - 问题 2：
# 类似于：https://leetcode.com/problems/multiply-strings/
# 但是您必须将两个数字相除。
# 给定两个表示为字符串的非负整数 n1 和 n1，返回 num1 和 num2 的乘积，也表示为字符串。
# 您不允许直接在输入上使用您的语言中的内置 Integer 函数。




# 問了下what's your best/ideal work environment
# 題目Huffman decode...
# Input Data : AAAAAABCCCCCCDDEEEEEFrequencies : A: 6, B: 1, C: 6, D: 2, E: 5 (you are given binary of the freq number)Encoded Data : 0000000000001100101010101011111111010101010
# Key point: prefix code cannot be reused in more than one encoding. 最傻做法是用HashMap 做<code, encodedWord> lookup; scan the stream for encoding,
# 經了解 面官想要TrieTree，我就想到了帶著TrieNode 搜的做法。。。 的確高大上一點， optimised 點，但Practically，我覺得小題大做。。。
# 下邊叫Huffman tree，其實一毛樣，不用一個Internal Map 而已，trie tree in this case is just binary tree....
# https://www.geeksforgeeks.org/huffman-decoding/




# 1. Autoscale policy. 给一个系统每秒平均利用率的数组util list和当前系统运行instance的数量. 如果当前利用率 > 60%， 则double 系统中instance的数量（一次扩容操作），最高不超过2* 10^8。 如果<25%， 则Halve 系统中instance的数量（一次缩容操作），并向上取整。 如果当前 25% <=利用率 <=60% ，不进行任何操作。 如果有操作，系统接下来的10秒中不允许有任何操作。求最后遍历完数组后instance的个数。


# 2. Team Formation 给一个队员score list，一个k，和 一个team_size.  从score list前后各k个队员里，选择最高的一个score， 将这个被选择的人移除score list ，然后加入到team中去，直至选择的成员个数到达team_size。如果在前后k个队员中有多个队员的成绩相同，选择index最小的那一个。 如果score list少于k个队员，则从整个score list 中选择。 最后返回team 中的成员的score得分。


# Similar to largest substring with maximum character frequency less than k


# / 一个朋友关系和company，找最大的set，类似于island那题。输入是{[1,2,1], [2,3,1]}前两个是朋友id，最后一个是companyID，meaning 在
# company1下，1和2是朋友，2和3也是朋友，所以这个集合就是1，2，3在company1下。同一个人可以在多个company下，求最大的这个集合
# input是一组数字每组三个数，分别代表friendId, friendId, companyId，
from collections import defaultdict, deque

def largest_group_in_same_company(paired_friends):
    company_groups = defaultdict(list)
    all_groups = []

    for friends in paired_friends:
        company_groups[friends[2]].append([friends[0], friends[1]])

    for company, friend_pairs in company_groups.items():
        friendships = defaultdict(list)
        # Populate friendships with friend_id: direct friends
        for friendship in friend_pairs:
            friendships[friendship[0]].append(friendship[1])
            friendships[friendship[1]].append(friendship[0])

        # BFS on each ungrouped friend
        visited = set()
        for f, direct_friends in friendships.items():
            if f not in visited:
                visited.add(f)
                friends_group = [f]
                friends_of_friend = deque(direct_friends)
                
                while friends_of_friend:
                    current_friend = friends_of_friend.popleft()
                    if current_friend not in visited:
                        visited.add(current_friend)
                        friends_group.append(current_friend)
                        for indirect_friend in friendships[current_friend]:
                            if indirect_friend not in visited:
                                friends_of_friend.append(indirect_friend)

                all_groups.append(friends_group)
    # 上面找到了
    all_groups.sort(key=len, reverse=True)
    max_product = 0
    last_group_size = len(all_groups[0])

    for group in all_groups:
        if len(group) == last_group_size:
            group.sort(reverse=True)
            if len(group) >= 2:
                max_product = max(group[0] * group[1], max_product)
        else:
            break

    return max_product

