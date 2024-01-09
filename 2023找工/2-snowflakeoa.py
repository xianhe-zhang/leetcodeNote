# Task Scheduling - https://leetcode.com/discuss/interview-question/2775415

# Server Selection- https://leetcode.com/discuss/interview-question/2794537

# Max Order Value / Max Profit in Job Scheduling - https://leetcode.com/discuss/interview-question/1028649

# Non-Overlapping Intervals - https://leetcode.com/discuss/interview-question/2115993

# Grid Land / Kth Smallest Instructions - https://leetcode.com/discuss/interview-question/527769
 
# Lexicographically largest array (MEX) - https://leetcode.com/discuss/interview-question/2550995

# Minimize Array Value - https://leetcode.com/discuss/interview-question/2146013

# Maximum Array Value - https://leetcode.com/discuss/interview-question/2551033

# Largest Sub Grid - https://leetcode.com/discuss/interview-question/850974

# String Patterns - https://leetcode.com/discuss/interview-question/2825744

# Same Bit Pair - https://leetcode.com/discuss/interview-question/2835233

# 1851. Minimum Interval to Include Each Query
class Solution:
    def minInterval(self, intervals, queries):
        intervals = sorted(intervals)[::-1] # intervals倒序，又大到小。
        h = [] # h存放的是pair, duration-end pair
        res = {}

        for q in sorted(queries): # 先查小的。
            
            # 这一步是常识放入当前q满足的intervals
            # intervals[-1][0]也是小的left，要是小于q的话，才有机会进入到loop里面
            while intervals and intervals[-1][0] <= q: 
                i, j = intervals.pop()
                
                # 如果end>q，意味着当前的interval真正的满足q了，因此入h。
                # 如果不满足呢？那肯定也不满足下一个q，因此可以直接不用管，pop就好。
                if j >= q: heapq.heappush(h, [j - i + 1, j])
            
            while h and h[0][1] < q:
                heapq.heappop(h)
            res[q] = h[0][0] if h else -1
        return [res[q] for q in queries]


# 94. Binary Tree Inorder Traversal
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        stack = []
        curr = root
        while curr or stack:
            # 针对每一次的外部while-loop，我们就要把左边的都进入到stack中。
            while curr:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop() # 从后pop，也就是说把最下面的左侧节点pop出来。
            res.append(curr.val) 
            curr = curr.right
        return res

    def preOrderTraversal(self, root: TreeNode) -> list[int]:
        if not root:
            return []
        stack, output = [root], []
        while stack:
            node = stack.pop()
            if node:
                output.append(node.val)
                stack.append(node.right)  # 先右后左，确保左子树先被访问
                stack.append(node.left)
        return output
    
    def postOrderTraversal(self, root: TreeNode) -> list[int]:
        if not root:
            return []
        stack, output = [root], []
        while stack:
            node = stack.pop()
            if node:
                output.append(node.val)  # 添加到输出列表的前端 # 在index=0的位置，添加一个value
                stack.append(node.left)
                stack.append(node.right)
        return output[::-1]

# 2062. Count Vowel Substrings of a String - 这题不错
# 滑动窗口
class Solution:
    def countVowelSubstrings(self, word):
        C = {'a':0,'e':0,'i':0,'o':0,'u':0}
        ans, num, beg, D = 0, 0, 0, C.copy()
        for i,w in enumerate(word):
            # 如果不是元音，那么重置所有
            if w not in "aeiou":
                num, beg, D = 0, i+1, C.copy()
            else:
                D[w] += 1
                # 如果有元音，并且满足所有字母都满足了，我们开始缩小window
                while min(D.values()):   # key
                    D[word[beg]] -= 1
                    beg, num = beg+1, num+1
            ans += num
        return ans

# 1639. Number of Ways to Form a Target String Given a Dictionary
class Solution:
    def numWays(self, words: List[str], target: str) -> int:
        ALPHABET_SIZE = 26
        MOD = 10**9+7
        target_length = len(target)
        word_length = len(words[0])
        char_occurrences = [[0] * word_length for _ in range(ALPHABET_SIZE)] # 出现过多少次

        # char_occurrences记录的是: [[word_length_1],....,[_26]]
        # 哪个char，在哪一col出现过 -> [char][col]
        for col in range(word_length):
            for word in words:
                char_occurrences[ord(word[col]) - ord('a')][col] += 1
        # dp = number of ways to build the prefix of target of length using only col leftmost cols.
        dp = [[0] * (word_length + 1) for _ in range(target_length + 1)]
        dp[0][0] = 1 # dp[0][j] 如果是组成空字符串，那就都不选，无论j是什么都是1
        

        # 了解dp后，我们需要明白如何进行遍历的。
        # 我们每次都会去看当前位的target，一般bottom up都是双for循环，针对每一位的单都会看所有可能的column
        for i in range(target_length + 1):
            for j in range(word_length):
                
                # 1.用j-col的字符去匹配target[i]
                if i < target_length:
                    # dp[l+1][c+1]表示用col和左边能组成什么l，这里l+1和c+1的目的就是为了。处理边界/init情况
                    dp[i + 1][j + 1] += (
                        char_occurrences[ord(target[i]) - ord('a')][j] * 
                        dp[i][j]
                    )
                    dp[i + 1][j + 1] %= MOD

                # 2.不用j-col的字符去匹配target[i]
                # 这样操作可以将最新的结果放在上一个i上，然后在下一个loop计算那个时候的i,j就可以用了。
                dp[i][j + 1] += dp[i][j]
                dp[i][j + 1] %= MOD

        return dp[target_length][word_length]


# 这一题很值得扒
# 1. 我们的dp数组实际上有word_length + 1列。这是为了能够处理一个边界情况：不考虑任何单词的列。
# 2. 在处理dp[i][j]时，考虑的是如何使用前i个目标字符和单词的前j列。因此，我们实际上是在看如何利用第j列来更新dp[i][j+1]的值。i和j表是的意思不一样。
# dp[i][j] -> dp[i][j+1] 核心还是更新列在当前行的结果。
# 这道题的第一次loop和最后一次loop都很有趣。第一次loop为了初始化dp[0][j]；最后一次loop是为了将所有dp[target][j]的结果加起来。

# 210. Course Schedule II
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        inDegree = [0] * numCourses
        graph = collections.defaultdict(list)
        taken = set()

        for nex, pre in prerequisites:
            inDegree[nex] += 1
            graph[pre].append(nex)

        queue = []
        for i, v in enumerate(inDegree):
            if v == 0: queue.append(i)

        res = []
        while queue: 
            cur = queue.pop(0)
            taken.add(cur)
            res.append(cur)
            for nc in graph[cur]:
                inDegree[nc] -= 1
                if inDegree[nc] == 0 and nc not in taken:
                    queue.append(nc)

        return res if len(res) == numCourses else []

        

# 261. Graph Valid Tree
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        if n-1 != len(edges): return False
        graph = collections.defaultdict(list)
        for x, y in edges: 
            graph[x].append(y)
            graph[y].append(x)
        
        seen = set()
        def dfs(node=0):
            # if node in seen: return False
            seen.add(node)
            for nn in graph[node]:
                if nn not in seen:
                    dfs(nn)
            return True
        dfs()
        return len(seen) == n

# 2002. Maximum Product of the Length of Two Palindromic Subsequences
class Solution:
    def check(self, s: str, state: int) -> bool:
        left, right = 0, len(s) - 1
        # 检查 state 对应的子序列是不是回文串
        while left < right:
            # 将 left 和 right 对应上 「状态所对应的字符」 位置
            while left < right and (state >> left & 1) == 0:
                left += 1
            while left < right and (state >> right & 1) == 0:
                right -= 1
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        
        return True

    def maxProduct(self, s: str) -> int:
        n = len(s)
        m = 1 << n # n的长度，也就是一共有2^N的状态
        lst = []
        
        # 牛逼，这里是如何遍历所有sub的，首先m代表了所有sub，然后我们去看所有的sub是不是满足palindromic.
        # 记录所有合法状态的信息
        for i in range(1, m): # 如果是8位的话，那么m就是2048，而下面i最大也是2048，哪怕2048，其本质也是在8位的二进制上进行操作。
            if self.check(s, i):#去查看状态i是否是会回文
                lst.append([i, bin(i).count('1')]) # count(1)标识当前状态里有多少1，也就是有多少子集选中的字符，即子集的长度。
        
        arr = lst
        res = 0
        
        # for-each 优化，j 不需要从 0 开始
        # 这里就要开始进行了合并了。
        for i in range(len(arr)):
            x, len_x = arr[i]
            for j in range(i + 1, len(arr)):
                y, len_y = arr[j]
                # 状态之间没有字符相交，满足题意 这个bitmask操作牛逼
                if (x & y) == 0:
                    res = max(res, len_x * len_y)
        
        return res



# 330. Patching Array
# 没必要做。这题用greedy要明白一个很关键的数学点，如果我们能实现[1-n]的覆盖；那么我们添加新元素n+1 > n的时候，我们就可以是实现[1-(2n+1)]的覆盖。
class Solution:
    def minPatches(self, nums, n):
        reach, ans, idx = 0, 0, 0
        
        while reach < n:
            
            if idx < len(nums) and nums[idx] <= reach + 1:
                # 
                reach += nums[idx]
                idx += 1
            else: 
                # 发现num里面的element是没有办法满足目前reach～n之间的空间
                # 这里默认的插入值是reach+1
                # 如果之前的覆盖值到15了，下一个num里的element是50，肯定没有办法覆盖的。
                # 因此我们可以直接插入16，这样reach就可以来到31。
                # 我们每次插入的值是reach那么远的值，从这个角度来讲是贪心算法。
                ans += 1
                reach = 2*reach + 1       
                
        return ans

# 1235. Maximum Profit in Job Scheduling
# 明白这一题的遍历和DP了。
# 首先dp里肯定是按照结束时间顺序排列的。因为我们已经排序了。
# 那么为什么还要去找index呢？是想看我们当前的p基于过去哪一个时间点。
class Solution:
     def jobScheduling(self, startTime, endTime, profit):
        jobs = sorted(zip(startTime, endTime, profit), key=lambda v: v[1])
        dp = [[0, 0]]
        for s, e, p in jobs:
            # 利用bisect寻找index
            i = bisect.bisect(dp, [s + 1]) - 1
            # dp的pair存的是[end_time, max_profit]
            # bisect找到当前的job能够插在哪个index后面，然后比较profit，如果比当前最大的还要大，那么直接append。
            if dp[i][1] + p > dp[-1][1]:
                dp.append([e, dp[i][1] + p])
        return dp[-1][1]
 

# Cross the threshold

# Perfect pairs