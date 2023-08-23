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
# 57. Insert Interval
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        output = []
        index = 0
        newStart, newEnd = newInterval

        # phase-1: 看new的首位与intervals[1]比较，将之前的片段添加到output的前侧
        while index < len(intervals) and intervals[index][1] < newStart:
            output.append(intervals[index])
            index += 1
   
        print(output)
        # phase-2: 这题的难点在于，你不知道在phase1结束后，newInterval的图像情况
            # /1 - newInterval位于中空的位置，不与任何重叠
            # /2 - newInterval与index当前的Internval重叠 -> 重叠有很多种情况，部分重叠/全部重叠/超长重叠
            # /3 - newInterval位于末端，此情况应该与/1一样直接append进去
        if index == len(intervals) or intervals[index][0] > newEnd:
            output.append(newInterval)
        else:
            output.append([min(newStart, intervals[index][0]), max(newEnd, intervals[index][1])])
            index += 1 # 这里处理了intervals才需要index++; 上面append(newInterval)是不需要处理index的
            
        
        print(output)
        # phase-3: 你需要为Phase2擦屁股
        while index < len(intervals):
            if intervals[index][0] > output[-1][1]:
                output.append(intervals[index])
            else:
                output[-1][1] = max(intervals[index][1], output[-1][1])
            index += 1

        return output

# 56. Merge Intervals
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        output = [intervals[0]]
        for i in range(1, len(intervals)):
            cur_interval = intervals[i]
            cur_start, cur_end = cur_interval
            if output[-1][1] >= cur_start:
                output[-1][1] = max(cur_end, output[-1][1])
            else:
                output.append(cur_interval)
        return output
    

# 435. Non-overlapping Intervals
# 这一题还是贪心，我担忧的是贪心会不知道remove哪个interval是最优的，这样思考是不对的，要找到锚点anchor point
# NOTE: 你的担忧源于一个很自然的直觉：在复杂问题中，简单的方法通常不能涵盖所有情况，可能会遗漏一些边缘情况。
class Solution():
    def eraseOverlapIntervals(self, intervals):
        if not intervals: return 0
        intervals.sort()
        cnt = 0
        min_reach = intervals[0][1]    
    
        # 针对每一个interval，我们只用比较当前min_reach和start
        # 为什么？我们肯定是想让min_reach右侧越小越好，因为是排序过的。
        for s, e in intervals[1:]:
            # 如果s<min_reach，意味着我们已经尽力避免了，但还是没有办法，因此更新min_reach和cnt
            # 左边界排序后，只看右边界，利用min()决定保留哪一个具体的interval, 不用担心删除较大的end的interval会有什么影响。
            # 因为右边选择更小的end，肯定是更没有影响的，因此你担心的是左边的影响。
            # 假设我们有A,B两个interval，如果A完全包含B，min()选择B，排除A完全没问题；
            # 如果A的end更小，Start也更小，也就是说A和B部分重合，你担心说A_start ~ b_start这一部分会overlap别的interval，但是你根据min()仍选择了A -》 这种顾虑不存在
            # why？因为这种情况下，A将会和其他之前interval比如C overlap，但是明显C的end更小，在之前的循环中就不会选择A了，直接把该可能排除了。
            if s < min_reach:
                cnt += 1
                min_reach = min(min_reach, e) #
            else:
                min_reach = e
                
        return cnt
    


# 252 - meeting room - 没啥难的，排序就成，只需要记录end
# 253. Meeting Rooms II
import heapq
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        if not intervals: return 0
        intervals.sort()
        room = 1 # 这里的1是关键点。
        # we may nead a pq to store end_time
        # do we need ans=max(ans, cur)? NO -> CUZ pq store currently used meeting rooms.
        pq = []
        heapq.heappush(pq, intervals[0][1])
        for s, e in intervals[1:]:
            if s >= pq[0]: 
                heapq.heappop(pq)
                room -= 1
            room += 1
            # room = max(room, len(pq)) 这样就不用+1，-1了。
            heapq.heappush(pq,e)

        return room


# LinkedList	
# 206 Reverse LinkedList
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        cur = head
        while cur:
            # next_node = cur.next
            # cur.next = prev
            # prev = cur
            # cur = next_node
            cur.next, prev, cur = prev, cur, cur.next
        return prev
    
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # still worth considering: where we connect two nodes, but in one recursion.
        # the only way we communicate is .next.next right? because recursion only return next node. And in the current recursion, we cannot access to the previous node, but only next nodes.
        if not head or not head.next: return head # "not head not necessary"

        pn = self.reverseList(head.next)
        pn.next = head
        head.next = None
        return pn
###########上面是我错误的写法，很有借鉴意义，说明我掌握的不是很牢固。
# pn在每一层call中意味着什么意味着从最底层返回的node，这是一个技巧，因此在最后的return中也是最后一个node
# 下面这种写法也错误了！如果在最后一行return进入递归会发生什么？会发生进入递归与递归中操作顺序的混乱。
# 先进行操作再递归，会改变递归原有的数据结构！因此你需要额外的一行代码把递归的结果存储起来。
# 比如 p = self.reverseList(head.next) -> return p;
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next: return head 
        head.next.next = head
        head.next = None
        return self.reverseList(head.next)
    
    
# 141. Linked List Cycle 
    # - 快慢指针
    # - hashmap

# 21. Merge Two Sorted Lists
# 这题也很有意思：可以通过递归做，也可以通过while循环做
class Solution:
    def mergeTwoLists(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        if not l1: return l2
        if not l2: return l1
        head = ptr = ListNode(0)
        # merge two 
        while l1 and l2:
            if l1.val <= l2.val:
                ptr.next = l1
                l1 = l1.next
            else:
                ptr.next = l2
                l2 = l2.next
            ptr = ptr.next
        
        # connect the rest
        if l1: ptr.next = l1
        if l2: ptr.next = l2

        return head.next
    
# 思考一下，每一层recursion返回的是什么？是一个node，此node之后的所有node都已经安排好了。
# 然后在if-else中，如果l1，我们将l1 cur node连接好之后的recursion，然后return l1就成了。
class Solution:
    def mergeTwoLists(self, l1, l2): 
        if not l1: return l2
        if not l2: return l1

        if l1.val <= l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
            

            
# 23. Merge k Sorted Lists
# 这是python2的答案，py3用heapq
"""
暴力方法:

把所有链表的节点值放入一个数组。
对数组进行排序。
创建一个新的已排序链表，并将排序后的数组中的值逐一插入。
时间复杂度: O(N log N) (其中 N 是所有链表中的元素总数)
逐一比较:

比较每个链表头部的节点，选择最小的。
将选中的节点移到结果链表。
时间复杂度: O(kN) (k 是链表数量)
使用优先队列:

使用一个最小堆（或优先队列）来比较每个链表的头部节点。
每次从堆中取出最小节点并将其添加到结果链表。
将被选中的链表头部的下一个节点放入堆中。
时间复杂度: O(N log k)
分而治之:

使用分治的思想，两两合并链表，直到合并为一个链表。
具体来说，假设有 k 个链表，首先将它们分成 k/2 对（如果 k 是奇数，则最后一个独自为一对）。
对每一对进行合并，然后再将结果进行合并，直到合并为一个链表。
时间复杂度: O(N log k)
递归合并:

这与分治方法类似，但更倾向于递归方式的实现。
首先合并前两个链表，然后合并结果与第三个链表，以此类推。
时间复杂度: 取决于具体实现，但在最坏情况下可能为 O(k^2N)
"""
from Queue import PriorityQueue
class Solution(object):
    def mergeKLists(self, lists):
        head = ptr = ListNode(0)
        q = PriorityQueue()
        for l in lists:
            if l:
                q.put((l.val, l))
        while not q.empty():
            val, node = q.get()
            ptr.next = node
            ptr = ptr.next
            node = node.next
            if node: q.put((node.val, node))

        return head.next
    

# 19 Remove Nth Node From End of List

class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        fast = slow = head
        # 这一题中n,fast,slow的关系比较不太好把握。
        while n and fast:
            n -= 1
            fast = fast.next
        
        if not fast: return head.next 


        # 如果fast还有的话，就要同时往后走了
        while fast.next: # 之所以.next是因为我们需要fast走到最后none的位置
            fast, slow = fast.next, slow.next

        slow.next = slow.next.next
        return head

# 143. Reorder List
class Solution:
    def reorderList(self, head: ListNode) -> None:
        if not head:
            return 
        
        # find the middle of linked list [Problem 876]
        # in 1->2->3->4->5->6 find 4 
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next 
            
        # reverse the second part of the list [Problem 206]
        # convert 1->2->3->4->5->6 into 1->2->3->4 and 6->5->4
        # reverse the second half in-place
        prev, curr = None, slow
        while curr:
            curr.next, prev, curr = prev, curr, curr.next       

        # merge two sorted linked lists [Problem 21]
        # merge 1->2->3->4 and 6->5->4 into 1->6->2->5->3->4
        first, second = head, prev
        while second.next:
            first.next, first = second, first.next
            second.next, second = first, second.next



# Matrix	
# 73. Set Matrix Zeroes
# 第一个方法很简单，利用两个set分别记录横纵坐标，然后遍历修改值
# The following method can avoid extra espace.
class Solution(object):
    def setZeroes(self, matrix):
        setFirstRow = False
        R, C = len(matrix), len(matrix[0])
        # Phase-1 Record 0 positions in the first row/col
        for i in range(R):
            if matrix[i][0] == 0: setFirstRow = True # we won't change the first row for now, but will do later.
            for j in range(1, C):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
            
        # Phase-2 change cells into 0's as per first row/col record
        for i in range(1, R):
            for j in range(1, C):
                if not matrix[i][0] or not matrix[0][j]: matrix[i][j] = 0

        
        # Phase-3 change first COL
        if matrix[0][0] == 0: # (0,0)==0 有两种可能性；1. 原本为0；2.first col本身有0； -> 无论如何第一列都要变0
            for j in range(1,C):
                matrix[0][j] = 0

        # phase-4 change first ROW:
        if setFirstRow:
            for i in range(R):
                matrix[i][0] = 0


# 79. Word Search
class Solution(object):
    def exist(self, board, word):
        if not board or not word: return False        

        def dfs(i, j, word):
            
            # 这种写法不正确！太复杂了，既然四个方向某个方向满足就满足，那么可以用下面的found= or就可以了，为什么不能直接返回？因为这一题是回溯，要将修改的数据复原
            # for ni, nj in ((i+1, j),(i-1, j),(i, j+1),(i, j-1)):
            #     if 0 <= ni < len(board) and 0 <= nj <len(board[0]):
            #         return dfs(ni, nj, word[1:])
     

            if not word: return True
            if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[0]:
                return False
            cur = board[i][j]
            board[i][j] = "#"
            
            # Check in all 4 directions
            found = (dfs(i+1, j, word[1:]) or 
                     dfs(i-1, j, word[1:]) or 
                     dfs(i, j+1, word[1:]) or 
                     dfs(i, j-1, word[1:]))
            
            board[i][j] = cur
            return found


        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(i, j, word):
                    return True

        return False

            

# 54. Spiral Matrix
class Solution(object):
    def spiralOrder(self, matrix):
        # 1. for-loop len(matrix) // 2? 并没有这种，因为最后一行 may be right or down
        # 2. count(m*n) ✅
        
        up, down, left, right = 0, len(matrix)-1, 0, len(matrix[0])-1
        res = []
        while len(res) < len(matrix)*len(matrix[0]):
        # 首先聊聊后两个为什么需要if? if up!=down -> 此时边界至少还有多个行，因此可以向左走； 如果==了，那么只有一行了，因此在之前向右走的for循环中就已经记录过了
        # 为什么向下走的时候不需要判断？首先向下走时一定经过了向右走；因此目前来到了可以遍历的最右边；不需要考虑是否单行/列的问题。
        # 为什么向右走的可以如此坚决？因为肯定不满足while的循环，因此一定是有可以走的路的。

            # Right
            for i in range(left, right+1): res.append(matrix[up][i])
            # Down
            for i in range(up+1, down+1): res.append(matrix[i][right])

            # Left
            if up != down:
                for i in range(right-1, left-1, -1): res.append(matrix[down][i])
            
            # Up
            if left != right:
                for i in range(down-1, up, -1): res.append(matrix[i][left])
            # change boundaries
            left += 1
            right -= 1
            up += 1
            down -= 1
        return res

# 48. Rotate Image
# 这道题的思路都想出来了，反转/数学对应旋转，但是都没写出来...
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        # 思路出了问题；如果是按照数学对应关系的写法，我们的基准是matrix四象限中的其中一个象限，然后利用数学关系找到所有值。
        # 看到这题不要怕，你要明确的是，你需要以什么为基准。
        n = len(matrix[0])
        
        for i in range(n // 2 + n % 2): # 如果是2X2刚好，刚好四个格子，四个象限；如果是3*3，每个象限负责2个格子(1*2)，最中间的不需要变化，而非对称的格子四个象限刚好可以互补（这是你没想明白的地方），这也就是为什么在for循环中，我们只需要在一个地方有n%2就可以了。
            for j in range(n // 2):
                tmp = matrix[n - 1 - j][i]
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - j - 1]
                matrix[n - 1 - i][n - j - 1] = matrix[j][n - 1 -i]
                matrix[j][n - 1 - i] = matrix[i][j]
                matrix[i][j] = tmp


class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        self.transpose(matrix)
        self.reflect(matrix)
    
    def transpose(self, matrix):
        n = len(matrix)
        # 只用一半就可以了，交换x,y
        for i in range(n):
            for j in range(i + 1, n):
                matrix[j][i], matrix[i][j] = matrix[i][j], matrix[j][i]

    def reflect(self, matrix):
        n = len(matrix)
        # 只用一半，交换一个对称坐标就可以
        for i in range(n):
            for j in range(n // 2):
                matrix[i][j], matrix[i][-j - 1] = matrix[i][-j - 1], matrix[i][j]

# String	
# 3. Longest Substring Without Repeating Characters - 滑动窗口 - 简单

# 424. Longest Repeating Character Replacement
# 方法1: 利用二分找
    # 这里我们二分的是
    # lo是最长能满足的substring，hi是第一个不满足的substring长度
    # lo+1是为了避免lo == hi，它们俩的含义都不一样，而且mid会一直在lo，而无法前进到hi，从而跳出循环。
    # while lo + 1 < hi: # 这里用这个是这个解法的take-away

# 这里涉及到滑动窗口一个有趣的trick/变体：我们不需要缩小窗口，只用增大就可以了。
# 那么什么情况下可以不用缩小窗口：1. 目标是最大/最长 2. 缩小窗口不会帮助我们 但是记住你需要判断能否扩大窗口。
class Solution:    
     def characterReplacement(self, s, k):
        max_frequency = window_length = 0
        count = collections.Counter()
        
        for r in range(len(s)):
            ch = s[r]
            count[ch] += 1
            max_frequency = max(max_frequency, count[ch]) # to update MAX frequency of chars in our window
            
            # if len - fre < k means: we still can do operations / can add current word into window
            if window_length - max_frequency < k: 
                window_length += 1
            else: 
                l_ch = s[r-window_length]
                count[l_ch] -= 1 
                
        return window_length
     

# 76. Minimum Window Substring
# 遇到了一个点点磕绊，我们是需要minimum window，while循环使用来缩小窗口的，因此寻求答案的过程应该在while循环中。
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        target_dict = collections.Counter(t)
        word_needs = len(target_dict)
        s_cnt = collections.defaultdict(int)
        l = 0
        word_have = 0
        temp_max = float('inf')
        ans = ""
        
        for r in range(len(s)):
            cur = s[r]
            s_cnt[cur] += 1
            if s_cnt[cur] == target_dict[cur]: word_have += 1

            while word_have == word_needs and l <= r:
                if r-l+1 < temp_max: 
                    ans = s[l:r+1]
                    temp_max = r-l+1
                l_ch = s[l]
                s_cnt[l_ch] -= 1
                if s_cnt[l_ch] < target_dict[l_ch]: word_have -= 1
                
                l += 1 
                
        return ans
            
        
# 242. Valid Anagram - 简单秒杀
# 49。Group Anagrams
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # 利用了每个string里的元素(无关顺序)当作index进行归类
        # 利用tuple的哈希可以作key这一特性
        ans = collections.defaultdict(list)
        for s in strs:
            ans[tuple(sorted(s))].append(s)
        return ans.values()
# 20. Valid Parentheses
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
    

# 5. Longest Palindromic Substring
# 1-check all substrings(bf) -O(n^3) 遍历是n^2 检查ifPalindrome是n
# 2-dp-O(n^2)/O(n^2)
# 3-expand from center O(n^2)/O(n^1) -> 利用ans存放最优解，
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        ans = [0, 0]
        
        # Case 1 - 奇数
        for i in range(n):
            dp[i][i] = True
        
        # Case 2 - 偶数
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                dp[i][i + 1] = True
                ans = [i, i + 1]

        for diff in range(2, n): # diff就是substring的长度
            for i in range(n - diff): # i是substring可能的start_index
                j = i + diff
                if s[i] == s[j] and dp[i + 1][j - 1]:
                    dp[i][j] = True
                    ans = [i, j]
                
        # 因为遍历的时候从小往大看，因此最后一个满足条件的一定是最长。
        i, j = ans
        return s[i:j + 1]


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



# 125. Valid Palindrome / 两种方法：1-比较相反的， 2-双指针
class Solution:
    def isPalindrome(self, s: str) -> bool:
        i, j = 0, len(s)-1
        while i < j:
            # isalum只会判断数字和char
            while i < j and not s[i].isalnum():
                i += 1
            while i < j and not s[j].isalnum():
                j -= 1
            
            if s[i].lower() != s[j].lower():
                return False 
            i += 1
            j -= 1
        return True

class Solution:
    def isPalindrome(self, s: str) -> bool:
        result = ''.join([char.lower() for char in s if char.isalnum()])
        return result == result[::-1]

# 647. Palindromic Substrings 
# also 类似第五题的解法，expand from the center可以把空间优化到O(1)
class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        res = 0
        dp = [[0]*n for _ in range(n)]
        for r in range(n): # r是右边界
            for l in range(r, -1, -1): # l是左边界，不过一定要从小往大去找值
                if s[l] == s[r] and (r-l<2 or dp[l+1][r-1]): # i-j<2是为了判断substring为1/2的场景。
                    dp[l][r] = 1
                    res += 1                
        return res
    
# 271. Encode and Decode Strings
# What to use as a delimiter? Each string may contain any possible characters out of 256 valid ascii characters.
class Codec:
    def encode(self, strs: List[str]) -> str:
        if len(strs) == 0: return chr(258)
        return chr(257).join(x for x in strs)
    def decode(self, s: str) -> List[str]:
        if s == chr(258): return []
        return s.split(chr(257))


# Tree	
# 104. Maximum Depth of Binary Tree
class Solution:
    def maxDepth(self, root):
        if not root: return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
        
# 100. Same Tree
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q: return True
        if not p or not q: return False
        if p.val != q.val: return False
        return self.isSameTree(p.right, q.right) and self.isSameTree(p.left, q.left)
    
# 226. Invert Binary Tree
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root: return None
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root 
    
# 124. Binary Tree Maximum Path Sum
class Solution:
    def maxPathSum(self, root):
        result = float('-inf')
        def dfs(node):
            nonlocal result
            if not node: return 0
            val = node.val
            left = dfs(node.left)
            right = dfs(node.right)
            # 这里不需要单独比较val+left, val+right的原因是没必要，当前path左右都考虑的情况(val+left+right)已经包含
            # 不需要单独考虑(val+left/right)，你之所以想考虑的原因是因为存在：当前node+left/right为max；
            # 但是在递归中这种情况已经考虑了，how？-> 首先看reulst, node+left/right为max一定意味着其中left/right一方小于0，我们在return的那一行已经把小于0的排除了，因此left+right+val其实就包含了val+left/right.
            result = max(result, left+right+val) 
            return max(0, left+val, right+val)
        dfs(root)
        return result

# 102. binary tree level order traversal
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root: return []
        res = []
        q = collections.deque([root])
        while q:
            cur_list = []
            for _ in range(len(q)):
                cur_node = q.popleft()
                cur_list.append(cur_node.val)
                if cur_node.left: q.append(cur_node.left)
                if cur_node.right: q.append(cur_node.right)

            res.append(cur_list)
        return res
    

# 297. Serialize and Deserialize Binary Tree
class Codec: 
    # 🌟如果你不想用一个全局变量处理string，那么直接将string当成一个参数行走在各个recursion中。
    def serialize(self, root):
        def helper(node, t):
            if not node:
                t += "#,"
            else:
                t += str(node.val)+","
                t = helper(node.left, t) # 🌟这里必须用t= 否则没有办法更新t，因为t不是全局变量！！！
                t = helper(node.right, t)
            return t
        return helper(root, "") 

    def deserialize(self, data):
        tl = data.split(",")
        def helper(tl):
            if tl[0] == "#":
                tl.pop(0)
                return None
            cur_node = TreeNode(tl.pop(0))
            cur_node.left = helper(tl)
            cur_node.right = helper(tl)
            return cur_node

        return helper(tl)

# 572. Subtree of Another Tree
class Solution:
    # 重点是isSubtree的逻辑应该是怎么样，我本来是想用for循环找所有node，然后调用helper一一比较，这样子的话代码比较复杂
    # 如果利用递归，我们在每次recursion只能比较两个cur_node,是没有办法比较当前cur_node的son nodes的，搞清楚recursion。
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if not root: return False
        if self.isSameTree(root, subRoot): return True
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
    
    def isSameTree(self, p, q): # 这个简单。
        if not p and not q: return True
        if not p or not q or p.val != q.val: return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right,q.right)

# 105. Construct Binary Tree from Preorder and Inorder Traversal
# preorder is to generate nodes in order，这样我们可以自上而下地构建树。
# inorder的特点是，左子树的值都在cur的左边，右子树的值都在cur的右边；
# 我们限制index范围的目的就是确定每个子树的范围，以防止node出现在错误的位置上。
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        
        index_map = {v : i for i, v in enumerate(inorder)}
        pre_index = 0 

        # 只需要传递进去index的范围，当前子树的范围就可以了。
        def construct(l, r):
            nonlocal pre_index
            cur_node = TreeNode(preorder[pre_index])
            in_index = index_map[preorder[pre_index]]
            pre_index += 1
            
            if in_index > l: 
                cur_node.left = construct(l, in_index-1)
            if in_index < r:
                cur_node.right = construct(in_index + 1,r)
            return cur_node
        return construct(0, len(preorder)-1)
# 98. Validate Binary Search Tree
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def checkSubtree(root, l=float('-inf'), r=float('inf')):
            if not root: return True
            if root.val >= r or root.val <= l: return False
            return checkSubtree(root.left, l, root.val) and checkSubtree(root.right, root.val, r)
        return checkSubtree(root)

# 230. Kth Smallest Element in a BST
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        self.k = k
        self.target = -1 
        def inorder(node):
            if not node: return -1

            inorder(node.left)
            self.k -= 1
            if self.k == 0: 
                self.target = node.val
                return 
            inorder(node.right)
        inorder(root)
        return self.target
class Solution:

    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        def inorder(r):
            return inorder(r.left) + [r.val] + inorder(r.right) if r else []
        return inorder(root)[k - 1]
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack = []
        while True:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if not k: return root.val
            root = root.right

def preorderTraversal(root: TreeNode):
    if not root:
        return []
    stack, output = [root], []
    while stack:
        node = stack.pop()
        if node:
            output.append(node.val)
            stack.append(node.right)  # 先右后左，这样左子节点会先出栈
            stack.append(node.left)
    return output

def inorderTraversal(root: TreeNode):
    stack, output = [], []
    current = root
    while current or stack:
        while current:  # 一直到最左边
            stack.append(current)
            current = current.left
        current = stack.pop()
        output.append(current.val)
        current = current.right
    return output

def postorderTraversal(root: TreeNode):
    if not root:
        return []
    stack, output = [root], []
    while stack:
        node = stack.pop()
        if node:
            output.append(node.val)
            stack.append(node.left)
            stack.append(node.right)
    return output[::-1]  # 最后反转得到正确的后序遍历


# 235. Lowest Common Ancestor of a Binary Search Tree
# 236题目是关于没有BST这么强力的设定的。那一题返回的就是True/False，因此需要一个全局的self.node去取recursion中满足条件的值
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        rv,pv,qv = root.val, p.val, q.val
        if rv < pv and rv < qv: return self.lowestCommonAncestor(root.right, p, q)
        if rv > pv and rv > qv: return self.lowestCommonAncestor(root.left, p, q)
        return root

# 208. Implement Trie (Prefix Tree) Trie树也是属于固定套路的东西。
class Trie:
    def __init__(self):
        self.trie = dict()
        self.WORD_KEY = "#"

    def insert(self, word: str) -> None:
        cur = self.trie
        for ch in word:
            cur = cur.setdefault(ch, {})
        cur[self.WORD_KEY] = word

    def search(self, word: str) -> bool:
        cur = self.trie
        for i in range(len(word)):
            ch = word[i]
            
            if ch in cur:
                cur = cur[ch]
                if i == len(word) - 1 and self.WORD_KEY in cur: return True
            else:
                break
        return False
        

    def startsWith(self, prefix: str) -> bool:
        cur = self.trie
        for ch in prefix:
            if ch not in cur: return False
            cur = cur[ch]
        return True
        
# 211# 211. Design Add and Search Words Data Structure
# 这一题用迭代的方法不好做，精髓在于遇到"."要去遍历所有子树，因此利用recursion的方法会比较好一点。
class WordDictionary:
    def __init__(self):
        self.trie = {}


    def addWord(self, word: str) -> None:
        node = self.trie
        for ch in word:
            if not ch in node:
                node[ch] = {}
            node = node[ch]
        node['$'] = True

    def search(self, word: str) -> bool:
        def search_in_node(word, node) -> bool:
            for i, ch in enumerate(word):
                if not ch in node:
                    if ch == '.':
                        for x in node:
                            if x != '$' and search_in_node(word[i + 1:], node[x]): # 注意了只有遇到.的时候才会进入分支，否则直接通过外侧if-else进入
                                return True
                    return False
                else:
                    node = node[ch]
            return '$' in node

        return search_in_node(word, self.trie)
    

# 212. Word Search II
# 这一题还是有一些细节没有理清楚。比如树中哪里决定是一个出现过单词的结尾；比如在遍历的时候，应该按照什么来遍历。
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        wordTree = dict()
        self.res = []


        for w in words:
            cur = wordTree # 这样直接操纵cur，wordTree也会变化
            for ch in w:
                cur = cur.setdefault(ch, {})
            cur["#"] = w # "#"表示当前层是某个word的结尾。

    
        def bt(parent, i, j):
            cur_ch = board[i][j]
            if cur_ch not in parent: return 
            cur_level = parent[cur_ch]
            
            # 如果有重复的值进来，我们需要处理么？
            if "#" in cur_level: 
                self.res.append(cur_level["#"])
                cur_level.pop("#") # 如果一个单词找到过一次，那么就可以不用再找第二次

            board[i][j] = "#"
            for ni, nj in ((i+1,j),(i-1,j),(i,j+1),(i,j-1)):
                if 0 <= ni < len(board) and 0 <= nj < len(board[0]) and board[ni][nj] != "#":
                    bt(cur_level, ni, nj)
            board[i][j] = cur_ch


            if not cur_level: parent.pop(cur_ch) # 如果当前cur_level没有东西了，可以直接剪枝丢弃。
            return 
            
        
        # 直接遍历board，然后去找wordTree
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] in wordTree:
                    bt(wordTree, i, j)

        return self.res

        
     
# 347. Top K Frequent Elements
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = collections.Counter(nums).most_common(k)
        res = []
        for x, y in count:
            res.append(x)
        return res


    def topKFrequent2(self, nums: List[int], k: int) -> List[int]: 
        # O(1) time 
        if k == len(nums):
            return nums
        count = collections.Counter(nums)   
        return heapq.nlargest(k, count.keys(), key=count.get) 

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = Counter(nums)
        unique = list(count.keys())
        
        def partition(left, right, pivot_index) -> int:
            pivot_frequency = count[unique[pivot_index]]
            # 1. move pivot to end
            unique[pivot_index], unique[right] = unique[right], unique[pivot_index]  
            
            # 2. move all less frequent elements to the left
            store_index = left
            for i in range(left, right):
                # store_index从左边开始，不一定随着right移动，只有当满足小于pivot的条件时，才会向左移动，因此，store_index是右侧的第一位，在结束后需要与pivot交换。
                if count[unique[i]] < pivot_frequency:
                    unique[store_index], unique[i] = unique[i], unique[store_index]
                    store_index += 1

            # 3. move pivot to its final place
            unique[right], unique[store_index] = unique[store_index], unique[right]  
            
            return store_index
        
        def quickselect(left, right, k_smallest) -> None:
            if left == right: return
            
            pivot_index = random.randint(left, right)     
            pivot_index = partition(left, right, pivot_index)

            if k_smallest == pivot_index:
                 return 

            elif k_smallest < pivot_index:
                quickselect(left, pivot_index - 1, k_smallest)
   
            else:
                quickselect(pivot_index + 1, right, k_smallest)
         
        n = len(unique) 
        quickselect(0, n - 1, n - k)
        return unique[n - k:]

# 295. Find Median from Data Stream 
# 有点复杂而已。
from heapq import *
class MedianFinder:
    def __init__(self):
        self.small = []  # the smaller half of the list, max heap (invert min-heap)
        self.large = []  # the larger half of the list, min heap

    def addNum(self, num):
        if len(self.small) == len(self.large):
            heappush(self.large, -heappushpop(self.small, -num))
        else:
            heappush(self.small, -heappushpop(self.large, num))

    def findMedian(self):
        if len(self.small) == len(self.large):
            return float(self.large[0] - self.small[0]) / 2.0
        else:
            return float(self.large[0])