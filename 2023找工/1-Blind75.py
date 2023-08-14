import List;
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

# 238. Product of Array Except Self 🌟
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


# 53. Maximum Subarray 🌟
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        cur_summary = max_total = nums[0]

        for num in nums[1:]:
            # 你陷入了if-else条件的来回挣扎中，因为你不知道怎么判断，或者对这种固定模版理解的不够透彻
            cur_summary = max(cur_summary+num, num) # 关键：没有0在其中，只有要不要从当前的num开始记录->错，不是决定从当前开始，而是决定是否抛弃之前的较大累计值
            # 因为cur_summary是已经遍历了之前的所有元素了，因此，cur_summary关注的只有当前num；放弃之前的值意味着目前cur_summary只有num，如果之前的cur_summary+num小于num，意味着是负值，不值得拥有。
            max_total = max(max_total, cur_summary) # 这个你写对了，只存放历史最大的

        return max_total

# 152. Maximum Product Subarray
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if len(nums) == 1: return nums[0]
        min_prod = max_prod = result = nums[0]

        for n in nums[1:]:
            temp = max_prod
            max_prod = max(n, min_prod*n, max_prod*n)
            min_prod = min(n, min_prod*n, temp * n)
            result = max(result, max_prod)


        return result

# 153. Find Minimum in Rotated Sorted Array
class Solution:
    def findMin(self, nums: List[int]) -> int:
        # If the list has just one element then return that element.
        if len(nums) == 1: return nums[0]

        left, right = 0, len(nums) - 1
        if nums[right] > nums[0]: return nums[0]

        while right >= left:
            mid = left + (right - left) // 2
            if nums[mid] > nums[mid + 1]:
                return nums[mid + 1]
            if nums[mid - 1] > nums[mid]:
                return nums[mid]

            if nums[mid] > nums[0]:
                left = mid + 1
            else:
                right = mid - 1 # 这里已经确保了mid不是最小值的index，因此mid-1不会将搜索空间跳过

class Solution:
    def findMin(self, nums: List[int]) -> int:
        if len(nums) == 1: return nums[0]
        l, r = 0, len(nums)-1
        while l < r:
            m = (l+r) // 2
            r_val = nums[r]
            m_val = nums[m] 
            if m_val > r_val: l = m+1
            else: r = m
        return nums[l]
# 这是比较right 和 mid

# 首先左中右三点的排列组合情况，你可以写出来，然后你就大概知道最小值会落在left-m,m-r的具体哪个区间，如果落在l-m，意味着要缩小右侧区间。
# 拿mid与left比较看似很直观，但其实是缩小左边界，而非右侧。

# 另一个角度说明 问题来了，如果我们是mid 比较 right，因为是找最小值，如果mid < right，
# 立即能判断出来mid到right之间都递增，最小值必不在其中（mid仍可能），因此能移动right。 
# 但如果left < mid，左侧递增，你能直接排除left到mid吗，并不能，因为最小数可能就在left上，你无法据此把这部分排除出去。 

# 当选择left<right时，结尾搜索空间为1，left==right，这个时候你需要决定这个值是否是你要的，在153这道题中，这个题目最后指向的是最小值。刚好
# 如果选择left<=right，所有都要搜索，right<left才会跳出，因此需要在内部判断是否找到的值是最小值，否则就会错过，也有可能导致死循环。
# ✨ <= 一般用于找一个数存不存在；查找特定元素的边界；（找多个元素）；


# ✨ 这里你就犯错了，33题你第一遍尝试用while < 而非left,right；但是思路你写对了！很不错！
# while (left < right) 可能会漏掉最后一个元素的搜索。
# 这是因为我们在更新 left 和 right 指针时，采用的是向下取整（例如：mid = (left + right) / 2），
# 这可能导致 left 和 right 指针在某些情况下没有重叠。
# 如果数组的长度是奇数，当 left 和 right 指针相邻时，mid 会更接近 left，
# 并且 left 的值可能会保留在循环条件中，从而导致循环无法结束。
# 如上，如果确定队列中存在该值，就不会因为找不到目标而无限循环。

# 33. Search in Rotated Sorted Array
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        left, right = 0, n - 1
        while left <= right:
            mid = left + (right - left) // 2
            
            # Case 1: find target
            if nums[mid] == target:
                return mid
            
            # Case 2: subarray on mid's left is sorted
            elif nums[mid] >= nums[left]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
                    
            # Case 3: subarray on mid's right is sorted.
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1
        # 我记住了，因为搜索空间最后是1个元素，因此要比较一下是否真的是我们找的值。
        # 但其实因为我们是在循环哪解决的，因此可以直接返回-1就成。
        return l if nums[l] == target else -1
    

# 15
class Solution:
    # 整体的复杂度还是n^2; 先选定i值，然后在i后面的nums里，继续敲定two sum。
    # two sum有点non-intuitive是通过i值与j值的组合，去匹配已经看过的j值。
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort() # sort可以跳过一些选择，此题的目的是为了满足题意。
        for i in range(len(nums)):
            # 如果i第一位数>0永远不可能。
            if nums[i] > 0:
                break
            if i == 0 or nums[i - 1] != nums[i]: #跳过的
                self.twoSum(nums, i, res)
        return res

    def twoSum(self, nums, i, res):
        seen = set()
        j = i + 1
        while j < len(nums):
            complement = -nums[i] - nums[j]
            if complement in seen:
                res.append([nums[i], nums[j], complement])
                # 跳过重复的元素
                while j + 1 < len(nums) and nums[j] == nums[j + 1]:
                    j += 1
            seen.add(nums[j]) 
            j += 1


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res, dups = set(), set()
        seen = {} # 聪明，放的见过的元素，value存放的是i，这样就知道这个key是不是i这种情况下的。比如之前0的时候，就已经见过x，那么之后发现我们缺一个x，那么此时已经skip过0了，如果不用i进行标识，那么我们会错误添加进res中。（这一段话说的有点乱，看不出来就别看了）
        for i, val1 in enumerate(nums):
            if val1 not in dups: # 跳过重复的
                dups.add(val1)
                for j, val2 in enumerate(nums[i+1:]):
                    complement = -val1 - val2
                    # complement就是已经便利过的v2
                    if complement in seen and seen[complement] == i: # 聪明
                        res.add(tuple(sorted((val1, val2, complement))))
                    seen[val2] = i
        return res

# 11. Container With Most Water
# 这一题主要看思考逻辑，宽度最大是len(height)，那么每一次移动左右指针需要考虑一个问题，在当下，我们是否只有一个因素需要考虑？
# 是的话，移动就有意义；这里的变因是我们宽度是一直缩减的，因此移动短的边界，才有可能获得潜在的benefit。
class Solution:
    def maxArea(self, height: List[int]) -> int:
        maxarea = 0
        left = 0
        right = len(height) - 1
        
        while left < right:
            width = right - left
            maxarea = max(maxarea, min(height[left], height[right]) * width)
            if height[left] <= height[right]:
                left += 1
            else:
                right -= 1
                
        return maxarea
    

# Bit manipulation 
# 371
# 191
# 338
# 268
# 190


# DP	
# 70. Climbing Stairs
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1 or n == 2: return n
        x, y = 1, 2
        for _ in range(n-2):
            x, y = y, x+y
        return y
# 322. Coin Change
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount+1) # 这里用-1不行，因为你下面要去比较dp的大小。
        dp[0] = 0

        for a in range(1, amount+1):
            for c in coins:
                if a >= c and dp[a-c] != -1:
                    dp[a] = min(dp[a-c] + 1,dp[a])

        return dp[-1] if dp[-1] != float('inf') else -1

# 300. Longest Increasing Subsequence
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)

        return max(dp)
# 这题目的思路很好想，复杂度为N^2确实有点顶不住...

# 或者使用下面贪心的算法，目的就是为了维护最小的递增subsequence
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        sub = [nums[0]]
        for num in nums[1:]:
            if num > sub[-1]: # 如果新来的大于最后一位，直接添加到末尾；
                sub.append(num)
            else: # 如果新来的小于最后一位，先前寻找位置，替换！而非插入；
                i = 0
                while num > sub[i]:
                    i += 1
                sub[i] = num
        return len(sub)
# 为什么这个算法是正确的？len(sub)一定是答案？
# -> 1 如果碰到x进入else的情况后，再碰到大的数字y，没关系，直接append，因为这个时候sub里面我们实际考虑的是原来没有x的sub+y
# -> 2 如果y也小，但是刚好是处于末尾，也就意味着我们原来的sub是可以更新的，因为有了更小的末尾，因此也是直接替换就好。


# 1143. Longest Common Subsequence
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        dp = [[0] * (len(text2)+1) for _ in range(len(text1)+1)]
        for i in range(len(text1)):
            for j in range(len(text2)):
                if text1[i] == text2[j]:
                    dp[i+1][j+1] = 1+dp[i][j]
                else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
        return dp[-1][-1]
# 这道题首先你需要理解dp存放的内容是什么？是t1[0..i]和t2[0...j]的LCS
# 如果i,j里面的value相同，直接在(i+1,j+1)+1就成，否则选择一个max((i+1,j), (i,j+1))
# 这一题目是可以倒序的，倒序的唯一好处是dp[i][j] = dp[i+1][j+1] +1 这里的[i][j]与原来text的长度的index保持了一致
# 🌟一般来说倒序视为了避免重复计算，这题目不涉及；怎么避免重复计算呢？试想i取决于i-1, i-2那么就只能倒序, 因为正序的话，i-1会先被更新，导致重复计算；
# 不过这好像是会发生在二维强行一维的情况下。

# 139. Word Break
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        l = 0
        dp = [0] * (len(s)+1)
        dp[0] = 1
        for r in range(len(s)+1):
            if s[l:r] in wordDict and dp[l] == 1: 
                dp[r] = 1
                l = r
        return dp[-1]
# 这里别扭的点，在于l,r在dp中的index中，指代完全不一样。
# s[l:r+1]表明你期望l,r分别是s的左右两个边界；dp[l]想表明某个结尾index是可以组成的，但是我们dp的长度是n+1哦
# ❌上面写的错误！你是想通过一次遍历/贪心的方法找到答案；没有用到for循环是你的败笔，为什么？原本index-3/4都可以与index-0组成答案；
# 但是index-3的时候就更改了l，导致最后的dp其实就是众多可能性中的一种，而非所有可能性的集合。


class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        n = len(s)
        dp = [0] * (n + 1)
        dp[0] = 1

        # 这种双for循环，我们就不用担心l在dp中的意义了，因为我们会遍历每一种情况，只用更新r在dp中的情况就可以了。
        for r in range(1, n + 1):
            for l in range(r):
                if s[l:r] in wordDict and dp[l] == 1: 
                    dp[r] = 1
                    break # 不需要多种情况使得当前r多次成立，一次就可以了

        return dp[-1]

# 非常典型的背包问题解法
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dp = [False] * len(s)
        # for的逻辑就是针对每一个index，我们所有的可能性都要去尝试。
        for i in range(len(s)):
            for word in wordDict:
                # Handle out of bounds case
                if i < len(word) - 1: # i一定是大于等于len(word)
                    continue
                
                if i == len(word) - 1 or dp[i - len(word)]:
                    if s[i - len(word) + 1:i + 1] == word: # 这里的index[]与dp里面的值是割裂开的，i就是代表了最后一位的index
                        dp[i] = True 
                        break

        return dp[-1]

# 377. Combination Sum IV
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [0]*(target+1)
        dp[0] = 1 
        for v in range(1, target+1):
            for n in nums:
                if n > v: continue
                dp[v] += dp[v-n]
        return dp[-1]
    
# 198. House Robber
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [[0]*2 for _ in range(n+1)]
        for i in range(1, n+1):
            v = nums[i-1]
            dp[i][0] = max(dp[i-1][1], dp[i-1][0])
            dp[i][1] = v + dp[i-1][0]
        return max(dp[-1])
    

# 213. House Robber II
# 面对dp的环形没有什么好的办法，
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 0 or nums is None:
            return 0
        if len(nums) == 1:
            return nums[0]

        return max(self.rob_simple(nums[:-1]), self.rob_simple(nums[1:]))

    def rob_simple(self, nums: List[int]) -> int:
        t1 = t2 = 0
        for n in nums:
            t1, t2 = max(n+t2, t1), t1 
            # max有点non-sense这里，但主要就是为了保存最大值，这是一个技巧。
        return t1
# 分四种情况，都选，都不选，a选，b选；分成两次，就排除了都选的情况。剩下几种情况无所谓，因为与题目无关，聪明呀！
# 55. Jump Game
# r就是能够跳的最远的位置。
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        l = len(nums)
        r = 0
        for i in range(l):
            if i > r: continue
            n = nums[i]
            r = max(i+n, r)
            if r >= l-1: return True
                
        return False
    
# 62. Unique Paths
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0]* n for _ in range(m)]
        for i in range(m):
            dp[i][0] = 1
        for i in range(n):
            dp[0][i] = 1

        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]

        return dp[-1][-1]

# 91. Decode Ways
class Solution:
    def numDecodings(self, s: str) -> int:
        if s[0] == '0': return 0
        prev_one = prev_two = 1
        for i in range(1, len(s)):
            cur = 0
            if s[i] != '0':
                cur = prev_one # 这就像跳箱子，存在的可能是从prev_one继承而来的。
            if 10 <= int(s[i-1:i+1]) <= 26: cur += prev_two
            # 如果当前s[i]==0会发生什么？
            # 很有意思：当前的回合结束后，prev_one=0因为从cur而来，prev_two等于上一个回合的prev_one
            # 但是在下一个回合，cur=0，然后也会跳过10~26的判断，因为上一个s[i]=0，最终就会造成prev_one/two均为0的情况，这样就最终返回的也是0
            prev_one, prev_two = cur, prev_one

        return prev_one


# Graph	
# 133. Clone Graph
class Solution:
    def __init__(self):
        self.visited = {}  # 因为存在遍历过的node再次遍历，如果你需要用到递归，那么意思是重复操作，也就是说遍历过的不需要再遍历，因此你需要visited
    # 我确实思考了直接把cloneGraph() as the recursive func or another helper()
    # -> You comes to the point where you need to think about what you are going to return. Deep clone -> what structure is going to help us? Map<original node, copy node>
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node: return node 
        if node in self.visited: return self.visited[node]
        clone_node = Node(node.val, [])
        self.visited[node] = clone_node
        if node.neighbors:
            clone_node.neighbors = [self.cloneGraph(n) for n in node.neighbors]
        
        return clone_node
# 这一题去思考递归是极好的。

# 207. Course Schedule
# 这一题不能用union find，而是应该用topological sort!!!
# uf主要是为了undirect graph; topo是为了direct graph，well this question is to determine if the graph is CYCLIC!
from collections import deque
class Solution:
    def canFinish(self, numCourses, prerequisites):
        # Two vars: InDegrees and NextList
        indegree = [0] * numCourses
        adj = [[] for _ in range(numCourses)]

        # Init: parse data in Prerequisites
        for prerequisite in prerequisites:
            adj[prerequisite[1]].append(prerequisite[0])
            indegree[prerequisite[0]] += 1

        # Init: put all 0-degree into queue
        queue = deque()
        for i in range(numCourses):
            if indegree[i] == 0:
                queue.append(i)

        # Ready to count
        nodesVisited = 0
        # Traverse and using NextList to update Indegrees.
        while queue:
            node = queue.popleft()
            nodesVisited += 1

            for neighbor in adj[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        return nodesVisited == numCourses

# 这一题还有dfs的思路：其实也是backtracking ->
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        adj = [[] for _ in range(numCourses)]
        for nex, cur in prerequisites:
            adj[cur].append(nex)
            
        visited = [False] * numCourses # 去查看某一个node是否visit过，不用在乎它在哪个recursion中，why？-> 因为dfs的原因，如果我们访问该node，就会优先访问该node的所有可能，因此在下一次访问该node的时候，我们就不用检查了。
        inStack = [False] * numCourses # 该数组用于跟踪当前DFS路径上的节点。

        for i in range(numCourses):
            if self.dfs(i, adj, visited, inStack):
                return False
        return True
    
    # if the result is expected, we want to return False 
    def dfs(self, node, adj, visited, inStack):
        # 下面两个判断的顺序也很重要，先判断是否inStack了？如果它在了，意味遇到环了，直接return True；不能先去看是否visited过，好理解吧。
        if inStack[node]: return True
        if visited[node]: return False 
        
        visited[node] = True 
        inStack[node] = True
        for nex in adj[node]:
            if self.dfs(nex, adj, visited, inStack):
                return True
        inStack[node] = False
        return False
    

# 417. Pacific Atlantic Water Flow
# 固定的套路：无论BFS或者DFS都可以，都是要从边界出发向高山进发；都需要一个reachable的set来帮助自己
# return list(pacific_reachable.intersection(atlantic_reachable)) 
# 或者 set1 & set2

# 200. Number of Islands 经典题 - 没必要再刷了。

# 128. Longest Consecutive Sequence
# 这一题的难点在于需要将time complexity维持在o(n)
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        ns = set(nums)
        ans = 0
        for n in ns:
            if n-1 not in ns:
                ta = 1
                cn = n
                while cn+1 in ns:
                    ta += 1
                    cn += 1
                ans = max(ans, ta)
        return ans



# 261. Graph Valid Tree
# 这一题有两个Take-away，关于valid Tree
#   1. 不能有cycle，在union find中如何找到cycle -> 在union(), if root_x == root_y which means they already connected each other -> cycle
#   2. 从其中一个点出发, eg. dfs(0) -> then add node into visited -> return visted == nodes or not. -> also remember to check if the same node is visited multiple times.

# 之所以保留下面这段代码，是为了check line 563 VS line 568; 563保证重复的node不会被check，这里没必要，可以删除，因为我们只有一个起点；568确保neighbor如果在seen里，就是一个cycle。
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
    
        if len(edges) != n - 1: 
            return False
        adj_list = [[] for _ in range(n)]
        for A, B in edges:
            adj_list[A].append(B)
            adj_list[B].append(A)
        
        seen = set()
        
        def dfs(node, parent):
            if node in seen: return
            seen.add(node)
            for neighbour in adj_list[node]:
                if neighbour == parent:
                    continue
                if neighbour in seen:
                    return False
                result = dfs(neighbour, node)
                if not result: return False
            return True
        
        # We return true iff no cycles were detected,
        # AND the entire graph has been reached.
        return dfs(0, -1) and len(seen) == n


# 323. Number of Connected Components in an Undirected Graph
# 这一题是经典的union find题目；如果你要利用DFS的方法做，第一步就是要去构建图；


# 269. Alien Dictionary
# 不理解题意 -> 这题的难点在于我不清楚应该words里面的各个letter应该如何transfer到topo里面的inDegree中
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        # 不能用这个defaultdict，因为你需要判断字母是否存在于dic中，否在生成q的时候，会把所有的v=0存入q中，哪怕该ch并未出现过。
        # dic = collections.defaultdict(set)
        dic = {}
        inDegree = {chr(x):0 for x in range(ord('a'), ord('a')+26)}
        
        # 这个现在就存进来是很有必要的，因为如果最后一位有很长的char_list
        for w in words:
            for c in w:
                dic[c] = set()
                
        for i in range(len(words)-1):
            w1, w2 = words[i], words[i+1]
            n = min(len(w1), len(w2))
            for j in range(n):
                k1, k2 = w1[j], w2[j]
                # the following 2 lines avoid adding the same mapping into our vars
                if k1 != k2:
                    if k2 not in dic[k1]:
                        dic[k1].add(k2)
                        inDegree[k2] += 1
                    break # each w1 VS w2 can only be used once.
                elif j == n-1 and len(w1) > len(w2): return "" # beacause this is invalid input.

        q = collections.deque([k for k, v in inDegree.items() if v == 0 and k in dic])
        result = ''
        while q:
            cur = q.popleft()
            result += cur
            for c in dic[cur]:
                inDegree[c] -= 1
                if inDegree[c] == 0: q.append(c)

        return result if len(result) == len(dic) else ""




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