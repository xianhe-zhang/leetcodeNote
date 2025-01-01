# Array	
# 1 - two sum / O(n) O(n) 宝刀未老 ✅ 用complement
# 121. Best Time to Buy and Sell Stock ✅  用两个变量 max_profit=max; lowest_pric=min
# 217 Contains Duplicate ✅
# 238. Product of Array Except Self 🌟 ✅ 如何利用ans，也需要L/R单独的变量类似与prefix_product
# 53. Maximum Subarray 🌟 ✅
    # cur_sum = max(cur_sum + n, 0)
    # max_sum = max(max_sum, cur_sum)

# 152. Maximum Product Subarray - Neetcode ㊗️ / ㊗️
from typing import List


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
# 300. Longest Increasing Subsequence  - Neetcode ㊗️ ✅ ｜ 两种方法：一种常规dp；一种bisect_left()
# 1143. Longest Common Subsequence - Neetcode 👍 dp理解不深，看下方注释
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        l1, l2 = len(text1), len(text2)
        dp = [[0] * (l2+1) for _ in range(l1+1)]
         
        for i in range(l1):
            for j in range(l2):
                if text1[i] == text2[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1]) 
                    # 因为dp当前存放的是t1[:i+1]和t2[:j+1]的LCS。
                    # 所以当text1[i] != text2[j]，我们会希望从前一种情况获得最大值，而非再重新比较所有的dp中的数字。dp[i+1][j+1]的前面只有两种情况，一种是dp[i+1][j]/一种是dp[i][j+1]
                    # 好好思考一下。
        return dp[-1][-1]        
          
# 139. Word Break - Neetcode ㊗️ / ✅ 但是写的时候还是没bug free
class Solution:
    def wordBreak(self, w: str, wordDict: List[str]) -> bool:
        dp = [False] * (len(w) + 1)
        dp[0] = True

        for e in range(len(w)):
            for s in range(e+1): # start_index是希望取到end index的，因为切片器slicer的原因。
                cur_str = w[s: e+1]
                if cur_str in wordDict and dp[s]: # dp[s] 是看(s-1) index是否可以成立。s是当前的window
                    dp[e+1] = True
                    break # 忘记了
            
        return dp[-1]

# 377. Combination Sum IV ✅
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [0] * (1+target)
        dp[0] = 1 # 忘记初始化了
        for t in range(1, target+1):
            for n in nums:
                if t >= n:
                    dp[t] += dp[t-n]

        return dp[-1]
# 198. House Robber - Neetcode 👍
# 213. House Robber II - Neetcode ㊗️
# 面对dp的环形没有什么好的办法，只有去除首尾分别判断
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 0 or nums is None:
            return 0
        if len(nums) == 1:
            return nums[0]
            
        def robTMD(nums):
            yes = no = 0
            for n in nums:
                yes, no = no + n, max(yes, no)
            return max(yes, no)
        return max(robTMD(nums[:-1]), robTMD(nums[1:]))

# 55. Jump Game ✅
# 62. Unique Paths - Neetcode 👍 ✅
# 91. Decode Ways - Neetcode ㊗️ / ㊗️ 这里使用了三个变量，用于节省空间！但是你要理解清楚，原本的是dp[i] = ?+dp[i-1] ?+ dp[i-2]
class Solution:
    def numDecodings(self, s: str) -> int:
        if s[0] == '0': return 0
        one = two = 1 # one是i-1, two是i-2
        for i in range(1, len(s)):
            cur = 0
            
            if s[i] != '0':
                cur = one

            if 10 <= int(s[i-1:i+1]) <= 26:
                cur += two
            
            one, two = cur, one
        
        return one
            
# Graph	
# 133. Clone Graph ✅
class Solution:
    def __init__(self):
        self.visited = {}
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node: return None
        if node in self.visited: return self.visited[node]
        clone_node = Node(node.val)
        self.visited[node]=clone_node
        clone_node.neighbors = [self.cloneGraph(n) for n in node.neighbors if node.neighbors]
        return self.visited[node]
    

# 207. Course Schedule ㊗️ 当然也可以用toposort写。
# 用dfs的本质就是用找graph中是否有环。adj很有趣，其中key是前置，只要前置不形成环就可以。
# visited是用来记录哪些遍历过的，因此再来遍历没有意义！不过这里只是用来很有必要的优化！
# inStack是看当前前置之后会不会成环
# O(m+n)/O(m+n)
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        adj = [[] for _ in range(numCourses)]
        for nex, cur in prerequisites:
            adj[cur].append(nex)
        visited = [False] * numCourses 
        inStack = [False] * numCourses 
        for i in range(numCourses):
            if self.dfs(i, adj, visited, inStack):
                return False
            
        return True
    
    # if the result is expected, we want to return False 
    def dfs(self, node, adj, visited, inStack):
        if inStack[node]: return True
        if visited[node]: return False 
        
        visited[node] = True 
        inStack[node] = True
        for nex in adj[node]:
            if self.dfs(nex, adj, visited, inStack):
                return True
        inStack[node] = False
        return False

# 417. Pacific Atlantic Water Flow ✅ 用
# 200. Number of Islands 经典题 - 没必要再刷了。 ✅
# 128. Longest Consecutive Sequence 
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        nums = set(nums)
        if not nums: return 0
        ans = 1
        for n in nums:
            if n-1 not in nums: # 从每个subsequence的第一位开始，保证最优。
                temp_ans = 1
                cur = n
                while cur+1 in nums:
                    temp_ans += 1
                    cur += 1
                ans = max(ans, temp_ans)
        return ans
    

# 261. Graph Valid Tree ✅ 判断树没有环就可以了！所有都要连接起来！
# 323. Number of Connected Components in an Undirected Graph - Neetcode ㊗️
class Solution:
    # 下面是Union-Find的写法，当然也可以构造出来图，然后去dfs，用visited=[False]*n记录。 -》 Time/Space: O(E+V)
    # E = numbers of edges, V = numbers of vertices.
    # O(E*a) a is union ; Space: V 
    def countComponents(self, n, edges):
        def find(x):
            # return parent[x] if parent[x] == x else find(parent[x])
            if x != parent[x]:
                return find(parent[x])
            return x # or parent[x] because parent root -> x == parent[x]
        
        def union(x, y):
            xr, yr = find(x), find(y)
            if xr != yr:
                if rank[xr] < rank[yr]:
                    parent[xr] = yr
                else:
                    parent[yr] = xr
                    if rank[xr] == rank[yr]:
                        rank[xr] += 1
                
        parent, rank = list(range(n)), [0] * n # rank越高，意味着越接近parent
        for x,y in edges: 
            union(x,y)
            
        # return len(set(parent))
        return len(set(find(x) for x in parent))
    
# 269. Alien Dictionary ㊗️
# words比较的规则：第一个不一样的letter后面不算；如果匹配完了短的，但是前面还有，那么就是invalid
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        char_next_map = {} # 字母key必须在 字母value[]之前
        char_in_degree = {chr(x):0 for x in range(ord('a'), ord('a')+26)}
        
        # 这个init的操作有必要的，如果input=["z", "z"]，孤立字符将不会经过char_next_map和in_degree的处理，因此如果单纯用defaultdict()的话，在最后整合的话会被略过。
        for w in words:
            for c in w:
                char_next_map[c] = set()
                
        for i in range(len(words)-1):
            w1, w2 = words[i], words[i+1]
            n = min(len(w1), len(w2))
            for j in range(n):
                c1, c2 = w1[j], w2[j]
                if c1 != c2:
                    if c2 not in char_next_map[c1]:
                        char_next_map[c1].add(c2)
                        char_in_degree[c2] += 1
                    break # each w1 VS w2 can only be used once.
                elif j == n-1 and len(w1) > len(w2): return "" # beacause this is invalid input.

        q = collections.deque([k for k, v in char_in_degree.items() if v == 0 and k in char_next_map])
        result = ''
        while q:
            cur = q.popleft()
            result += cur
            for c in char_next_map[cur]:
                char_in_degree[c] -= 1
                if char_in_degree[c] == 0: q.append(c)

        return result if len(result) == len(char_next_map) else ""


# # Interval	
# 57. Insert Interval ㊗️
class Solution: 
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        result = []
        index = 0
        new_start, new_end = newInterval
        # 1. insert if end < new_start
        while index < len(intervals) and intervals[index][1] < new_start:
            result.append(intervals[index])
            index += 1

        # 2. insert new - [min_start, max_end] 
        if index == len(intervals) or new_end < intervals[index][0]: # 两种情况
            result.append(newInteval)
        else: # 只剩下两种情况，但两种一情况一定是overlap的，不overlap的在上面已经排除了。
            result.append([min(new_start, intervals[index][0]), max(new_end, intervals[index][1])])
            index += 1
            

        # 3. finish up - check res[-1][1] and start
        while index != len(intervals):
            if result[-1][1] < intervals[index][0]:
                result.append(intervals[index])
            else:
                result[-1][1] = max(intervals[index][1], result[-1][1]) # 万一 new interval很大，比之后的所有都大，我们想维持更大的值
                # result[-1][1] = intervals[index][1] 错误答案，容易忽略的地方
        
            index += 1

        return result         
        
        
# 56. Merge Intervals ✅
# 435. Non-overlapping Intervals ㊗️
class Solution():
    def eraseOverlapIntervals(self, intervals):
        if not intervals: return 0
        intervals.sort()
        min_reach = intervals[0][1]
        res = 0
        for s, e in intervals[1:]:
            if min_reach <= s:
                min_reach = e
            else:
                min_reach = min(e, min_reach) # 这个逻辑还是很难。1.如果e大，直接不考虑当前的；2.如果min_reach大，我们从已经遍历过（有序且non-overlap）中更新最后一位，其实就是删除stack top。因为有序，所以不用太考虑start会不会影响之前的元素。
                res += 1

        return res
# 252 - meeting room - ✅
# 253. Meeting Rooms II
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        if not intervals: return 0
        
        intervals.sort()
        rooms_in_use = []
        rooms = 0
        for s, e in intervals:
            if rooms_in_use and rooms_in_use[0] <= s:
                heapq.heappop(rooms_in_use)
                rooms -= 1
            heapq.heappush(rooms_in_use, e)
            rooms += 1

        return rooms

# LinkedList	
# 206 Reverse LinkedList ㊗️
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next: return head
        head_ptr = self.reverseList(head.next) # 这里不同iteration，利用recursion的stack然后处理每个ptr，但是处理ptr的时候，先处理其next_ptr这个很巧妙
        head.next.next = head
        head.next = None

class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        while head:
            temp = head.next
            head.next = prev
            prev = head
            head = temp
        return prev

# 141. Linked List Cycle / hashtable ✅
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow, fast = head, head.next
        while slow != fast:
            if not fast or not fast.next: return False
            slow = slow.next
            fast = fast.next.next
        return True

# 21. Merge Two Sorted Lists ✅
class Solution:
    def mergeTwoLists(self, l1, l2):
        # if not l2: return l1
        # if not l1: return l2

        # if l1.val <= l2.val:
        #     l1.next = self.mergeTwoLists(l1.next, l2)
        #     return l1
        # else:
        #     l2.next = self.mergeTwoLists(l1, l2.next)
        #     return l2

        if not l2: return l1
        if not l1: return l2
        dummy = ptr = ListNode(0)
        while l1 and l2:
            if l1.val <= l2.val:
                ptr.next = l1
                l1 = l1.next
                ptr = ptr.next
            else:
                ptr.next = l2
                l2 = l2.next
                ptr = ptr.next
        
        if l1:
            ptr.next = l1
        if l2:
            ptr.next = l2
        
        return dummy.next

# 23. Merge k Sorted Lists 可以用21题的方法，也可以用priority queue。 from Queue import PriorityQueue；q.put()/.get()/.empty()
# 19 Remove Nth Node From End of List ✅
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        fast = slow = head
        while n and fast:
            n -= 1
            fast = fast.next 
        
        # fast None or not: None->跳过第一个 / not->一起移动，直到None，然后slow.next=slow.next.next
        if not fast: return head.next

        while fast.next: # tricky的点，如果使用while fast，那么slow会停在要跳过的那个node上，因此要使用dummy node来帮助
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return head
# 143. Reorder List 三部曲：找中点 -> reverse -> merge two 这道题细节很有难度的。
# 143. Reorder List [Problem 876] + [Problem 206] + [Problem 21]
class Solution:
    def reorderList(self, head: ListNode) -> None:
        if not head: return None
        
        # find the middle point; slow will be in the middle or to the right
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        
        # reverse the second part
        prev, curr = None, slow
        # slow.next = None #  这里不要切断，因为否则下面反转我们需要用到next，这里就断了！前半段连接着这个中点没关系，但是后半段不需要。
        # print(curr)
        while curr:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        # now prev will the head of second half

        l1, l2 = head, prev
        # 为什么用.next? 因为操作原因，在第二部反转过后实际上的数据结构是 前半段和后半段会指向/共享第一次找到的slow那个node，我们用next是为了跳过最后一次合并，如果不跳过会造成循环。
        # 因为最后一个node被l1,l2共享，因此最后一次合并时l1和l2实际上指向一个同一个node（slow），因此尝试合并时会造成循环。
        # 为什么用l2而非l1? 因为第一步的时候，奇数时slow指向中点，被前后共享，前后元素一致；偶数时slow指向右侧第一个元素，这个时候l2比l1少一位，为了避免操作出界，我们使用l2
        while l2.next: 
            l1.next, l1 = l2, l1.next
            l2.next, l2 = l1, l2.next
        
# Matrix	
# 73. Set Matrix Zeroes ❌ 很容易遇到bug，这个遍历顺序，和更改流程要记住，虽然不太考察。
class Solution:
    def setZeroes(self, matrix):
        setFirstCol = False
        R, C = len(matrix), len(matrix[0])
        # Phase-1 Record 0 - skip first col
        for i in range(R):
            if matrix[i][0] == 0: setFirstCol= True # 因为我们使用了additional var to store first col to help us, so we need  to handle first col last -> phase 4. If we handle first row first, then there may be some edge case we miss.
            for j in range(1, C):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
            
        # Phase-2 change cells into 0's by the record
        for i in range(1, R):
            for j in range(1, C):
                if not matrix[i][0] or not matrix[0][j]: matrix[i][j] = 0

        # Phase-3 change first row
        if matrix[0][0] == 0: # (0,0)==0 有两种可能性；1. 原本为0；2.first col本身有0； -> 无论如何第一列都要变0
            for j in range(1,C):
                matrix[0][j] = 0

        # phase-4 change first col:
        if setFirstCol:
            for i in range(R):
                matrix[i][0] = 0

# 79. Word Search - Neetcode ㊗️ - 主要是内部的return怎么写
class Solution(object):
    def exist(self, board, word):
        if not board or not word: return False        

        def backtrack(i, j, word):
            if not word: return True
            if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[0]:
                return False
            cur = board[i][j]
            board[i][j] = "#"
            
            # Check in all 4 directions
            found = (backtrack(i+1, j, word[1:]) or 
                     backtrack(i-1, j, word[1:]) or 
                     backtrack(i, j+1, word[1:]) or 
                     backtrack(i, j-1, word[1:]))
            
            board[i][j] = cur
            return found # 如果我们要返回递归结果，那么边界条件/edge case一定要有t/f，一般来说edge case常用来剪纸/满足条件


        for i in range(len(board)):
            for j in range(len(board[0])):
                if backtrack(i, j, word):
                    return True

        return False


# 54. Spiral Matrix ㊗️
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix: return []
        output = []
        up, down, left, right = 0, len(matrix)-1, 0, len(matrix[0])-1
        
        while len(output) < len(matrix) * len(matrix[0]):
            # right 
            for i in range(left, right+1):
                output.append(matrix[up][i])

            # down 
            for i in range(up+1, down+1):
                output.append(matrix[i][right])

            # left
            if up != down:
                for i in range(right-1, left-1, -1):
                    output.append(matrix[down][i])
            
            # up
            if left != right:            
                for i in range(down-1, up, -1):
                    output.append(matrix[i][left])

            up += 1
            down -= 1
            left += 1
            right -= 1

        return output

    
# 48. Rotate Image - Neetcode ㊗️
# [i][j]
#   -> [j][i]     transpose 主坐标轴对称
#   -> [n-1-i][j] 水平对称
#   -> [i][n-1-j] 垂直对称
#   旋转90度 == 垂直对称 -> 转置 / 转置 -> 水平对称
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix[0])
        for i in range(n // 2 + n % 2): 
            for j in range(n // 2):
                tmp = matrix[n - 1 - j][i]
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - j - 1]
                matrix[n - 1 - i][n - j - 1] = matrix[j][n - 1 -i]
                matrix[j][n - 1 - i] = matrix[i][j]
                matrix[i][j] = tmp
    # 需要处理的4个点:[i,j], [n-1-i][n-1-j], [j][n-i-1], [n-1-j][i]
    # 如何理解这四个点，其实只用注意四个顶点就好了。 第一个是原点，后三个分别为三个顶点。
    # 这里的方法是in-place rotation

# String	
# 3. Longest Substring Without Repeating Characters - 滑动窗口 - 简单 ✅
# 424. Longest Repeating Character Replacement ㊗️
# 那么什么情况下可以不用缩小窗口：1. 目标是最大/最长 2. 缩小窗口不会帮助我们 但是记住你需要判断能否扩大窗口。
class Solution:    
     def characterReplacement(self, s, k):
        counter = Counter()
        max_freq = 0 
        w_width = 0

        for p in range(len(s)):
            right_ch = s[p]
            counter[right_ch] += 1
            # max_freq = max(counter.values()) # 这里不需要比较所有的，因为我们只更新了counter[right_ch]
            max_freq = max(max_freq, counter[right_ch]) 

            # 目前我们的逻辑window中是包含了最右侧的char，看要不要narrow window了。
            if w_width + 1 - max_freq <= k: # 此刻的w_width还是老的，所以+1，考虑右侧的char
                w_width += 1
            else:
                left_ch = s[p-w_width]
                counter[left_ch] -= 1

        return w_width

# 76. Minimum Window Substring  Hard 题目✅
# 用的helpfer var比较多 -> target_counter; target_needs; words_have_in_window; temp_min_window_length
# 主要的逻辑：进入到window，增加counter，比较是否满足freq要求，是否满足needs，如果满足并且更短，那么更新答案
# 能进入到while，肯定都是满足的了。在while里尝试缩小左边界，因为这是求minimum
# 242. Valid Anagram - 简单秒杀 ✅
# 49. Group Anagrams ✅ ans[tuple(sorted(s))].append(s)


# 20. Valid Parentheses ✅
class Solution:
    def isValid(self, s: str) -> bool:
        m = {
            "]": "[",
            "}": "{",
            ")": "(",
        }

        stack = []
        for c in s:
            if c in m and stack and stack[-1] == m[c]:
                stack.pop()
            else:
                stack.append(c)

        return len(stack) == 0
    
# 5. Longest Palindromic Substring
# 1-check all substrings(bf) -O(n^3) 遍历是n^2 检查ifPalindrome是n
# 2-dp-O(n^2)/O(n^2)
# 3-expand from center O(n^2)/O(n^1) -> 利用ans存放最优解，
class Solution:
    def longestPalindrome(self, s: str) -> str:
        def helper(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return s[left+1: right]
        res = ""
        for i in range(len(s)):
            res = max(res, helper(i, i), helper(i, i+1), key=len)
        return res

# 125. Valid Palindrome / 两种方法：1-比较相反的， 2-双指针，isalnum() ✅
# 647. Palindromic Substrings ㊗️
class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        res = 0
        dp = [[0]*n for _ in range(n)]
        for r in range(n):
            for l in range(r, -1 ,-1): # 一定是r，一定是倒序
                if s[l] == s[r] and (r-l<2 or dp[l+1][r-1]): # r-l<2 一定要在or的第一个选项中判断。
                    dp[l][r] = 1
                    res += 1
        return res




# 271. Encode and Decode Strings
# --------
# if len(strs) == 0: return chr(258)
# return chr(257).join(x for x in strs)
# --------
# if s == chr(258): return []
# return s.split(chr(257))



# Tree	
# 104. Maximum Depth of Binary Tree ✅ return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
# 100. Same Tree ✅
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q: return True
        if not p or not q: return False
        if p.val != q.val: return False
        return self.isSameTree(p.right, q.right) and self.isSameTree(p.left, q.left)
# 226. Invert Binary Tree ✅ root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
# 124. Binary Tree Maximum Path Sum 可以小看一下
class Solution:
    def maxPathSum(self, root):
        res = float('-inf')
        def dfs(node):
            nonlocal res
            if not node: return 0
            left = max(0, dfs(node.left))
            right = max(0, dfs(node.right))
            res = max(res, left + right + node.val)
            return max(0, node.val + max(left, right)) # 这里写的有点复杂
        dfs(root)
        return res 
"""
left = dfs(node.left)
right = dfs(node.right)
下面比较了left+right+val，没有比较left/right + val的原因是如果如果一侧小于0，直接不考虑了，因为最后一行。
result = max(result, left+right+val) 
return max(0,left+val, right+val) 
""" 


# # 102. binary tree level order traversal
# class Solution:
#     def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
#         if not root: return []
#         res = []
#         q = collections.deque([root])
#         while q:
#             cur_list = []
#             for _ in range(len(q)):
#                 cur_node = q.popleft()
#                 cur_list.append(cur_node.val)
#                 if cur_node.left: q.append(cur_node.left)
#                 if cur_node.right: q.append(cur_node.right)

#             res.append(cur_list)
#         return res
# 上面的这种解法是我熟悉的。
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        levels = []
        if not root: return levels

        def dfs(node: TreeNode, level: int) -> None:
            if len(levels) == level: # 精彩，精髓点
                levels.append([])
            
            levels[level].append(node.val)
            if node.left: dfs(node.left, level + 1)
            if node.right: dfs(node.right, level + 1)

        dfs(root, 0)
        return levels
    
# 297. Serialize and Deserialize Binary Tree - 这个可以多
class Codec:
    def serialize(self, root):
        if not root: return "[]"
        ans = []
        queue = deque([root])
        
        while queue:
            node = queue.popleft()
            if node:
                ans.append(str(node.val))
                queue.append(node.left) # 这种traverse的方式其实是level-order/bfs
                queue.append(node.right)
            else:
                ans.append("N")

        return '[' + '/'.join(ans) + ']'

    def deserialize(self, data):
        # invalid case
        if data == "[]":
            return None
        
        values = data[1:-1].split('/') # 去掉首尾
        root = TreeNode(int(values[0]))

        index = 1 # 跳过index==0/root开始。
        queue = deque([root])

        # queue里存的是nodes
        # index遍历完，就不会往queue里面存node了。
        while queue:
            node = queue.popleft()
            if values[index] != "N":
                node.left = TreeNode(int(values[index]))
                queue.append(node.left)
            index += 1

            if values[index] != "N":
                node.right = TreeNode(int(values[index]))
                queue.append(node.right)
            index += 1

        return root
    

# 572. Subtree of Another Tree ✅
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if not root: return False
        if self.isSameTree(root, subRoot): return True

        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
    
    def isSameTree(self, p, q):
        if not p and not q: return True
        if not p or not q or p.val != q.val: return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right,q.right)

      
# 105. Construct Binary Tree from Preorder and Inorder Traversal ㊗️ 
# inorder的特点是，左子树的值都在cur的左边，右子树的值都在cur的右边；
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        pre_index = 0    
        in_index_map = {v:i for i, v in enumerate(inorder)}

        # 只需要传递进去index的范围，当前子树的范围就可以了。
        # 我们为什么需要用l, r去规范当前子树的节点范围呢？ -> 因为我们不确定子树的结构！
        # 我们的preorder只负责一个个按顺序去创建，但是没有规范树的结构。
        # 如果没有范围了，意味着当前节点下没有其他sub-node了.
        def construct(l, r):
            nonlocal pre_index
            cur_val = preorder[pre_index]
            cur_node = TreeNode(cur_val)
            in_index = in_index_map[cur_val]
            pre_index += 1

            if l < in_index: # 意味着[l:in_index]中还有这么多元素没有被分配。
                cur_node.left = construct(l, in_index-1)
            if in_index < r:
                cur_node.right = construct(in_index+1, r)
            return cur_node

        return construct(0, len(preorder)-1)


# 98. Validate Binary Search Tree ✅
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def checkSubtree(node, left=float('-inf'), right=float('inf')):
            if not node: return True
            if not left < node.val < right: return False
            return checkSubtree(node.left, left, node.val) and checkSubtree(node.right, node.val, right)
        return checkSubtree(root)

# 230. Kth Smallest Element in a BST ❌
# 可以用DFS和BFS，都要会
# 前序遍历	根 → 左 → 右	构造树、表达式解析
# 中序遍历	左 → 根 → 右	排序、二叉搜索树遍历
# 后序遍历	左 → 右 → 根	子树计算、删除树
class Solution:
    # dfs方法复杂度为O(n)/O(n)
    def kthSmallestdfs(self, root: Optional[TreeNode], k: int) -> int:
        def inorder(node):
            if not node:
                return []
            else:
                return inorder(node.left) + [node.val] + inorder(node.right)
        return inorder(root)[k-1]
    # bfs的方法复杂度为O(H+k)/O(H) H是tree的高度，最好的情况是balanced tree O(logn)
    def kthSmallestbfs(self, root: Optional[TreeNode], k: int) -> int:
        stack = []
        while 1: 
            while root: 
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if not k:
                return root.val
            root = root.right
            
# 235. Lowest Common Ancestor of a Binary Search Tree ㊗️
class Solution:
    # O(n)/O(n)
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        root_val, p_val, q_val = root.val, p.val, q.val
        if root_val < p_val and root_val < q_val: return self.lowestCommonAncestor(root.right, p, q)
        if root_val > p_val and root_val > q_val: return self.lowestCommonAncestor(root.left, p, q)
        return root
    # O(n)/O(1)
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        root_val, p_val, q_val = root.val, p.val, q.val
        node = root
        while node:
            root_val = node.val
            if root_val < p_val and root_val < q_val:
                node = node.right
            if root_val > p_val and root_val > q_val:
                node = node.left
            else:
                return node

# 236题目是关于没有BST这么强力的设定的。那一题返回的就是True/False，因此需要一个全局的self.node去取recursion中满足条件的值 ㊗️
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        res = TreeNode()
        def findLCA(node):
            nonlocal res
            if not node: return 0
            left = findLCA(node.left)
            right = findLCA(node.right)
            mid = node == p or node == q 
            ts = left+mid+right
            if ts == 2: res = node
            return 1 if ts > 0 else 0
        findLCA(root)
        return res
    
# 208. Implement Trie (Prefix Tree) Trie树也是属于固定套路的东西。
class Trie:
    def __init__(self):
        self.trie = dict()
        self.WORD_KEY = "#"

    def insert(self, word: str) -> None:
        cur = self.trie # 精髓
        for ch in word:
            cur = cur.setdefault(ch, {})
        cur[self.WORD_KEY] = word # 精髓

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
        


# 211. Design Add and Search Words Data Structure
class WordDictionary:
    def __init__(self):
        self.trie = {}

    def addWord(self, word: str) -> None:
        cur = self.trie
        for ch in word:
            cur = cur.setdefault(ch, {})
        cur["$"] = True
        

    def search(self, word: str) -> bool:
        def search_in_node(node_dict, word):
            for i, ch in enumerate(word):
                if ch in node_dict: 
                    node_dict = node_dict[ch] # 满足继续匹配下一项
                else:
                    if ch == '.':
                        # 只有通过后续所有检查，才可以返回True
                        for n in node_dict:
                            if n != "$" and search_in_node(node_dict[n], word[i+1:]):
                                return True # 只要有一个path成功了，就回返回True
                    # no match
                    return False
            return "$" in node_dict # 此时已经在最后一层了

        return search_in_node(self.trie, word)

# 212. Word Search II  = Trie/prefix Tree + Backtracking + Graph
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        wordTree = dict()
        self.res = []

        for w in words:
            cur = wordTree # 这样直接操纵cur，wordTree也会变化
            for ch in w:
                cur = cur.setdefault(ch, {})
            cur["#"] = w # "#"表示当前层是某个word的结尾。

        def bt(parent, i, j): # 不应该叫Parent, parent是trie的一部分，表示了当前i,j的可以取什么值。
            cur_ch = board[i][j]
            if cur_ch not in parent: return
            cur_level = parent[cur_ch]

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
    # O(nlogn) most_common是用排序 / O(n + k)
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = Counter(nums).most_common(k)
        res = []
        for x, y in count:
            res.append(x)
        return res

    #  O(nlogk) k为heap的大小 / O(n + k)
    def topKFrequent(self, nums: List[int], k: int) -> List[int]: 
        if k == len(nums):
            return nums
        count = Counter(nums)   
        return heapq.nlargest(k, count.keys(), key=count.get) 
    # o(n ~ n^2) O（n/ O(n)
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
    

# 295. Find Median from Data Stream  ㊗️
class MedianFinder:
    def __init__(self):
        self.small = []  # the smaller half of the list, max heap (invert min-heap)
        self.large = []  # the larger half of the list, min heap

    # small和large heap之间的转化一定是要负号的，但是最开始往里面存数的时候，small存负，large存正。让他们两个pop的时候，pop口都在median附近。
    def addNum(self, num):
        if len(self.small) == len(self.large):
            heappush(self.large, -heappushpop(self.small, -num)) # heapq.heappop的是堆中最小的新元素，small中存的都是负数，因此它pop出来的就是最大的那个数。
        else:
            heappush(self.small, -heappushpop(self.large, num))

    def findMedian(self):
        if len(self.small) == len(self.large):
            return float(self.large[0] - self.small[0]) / 2.0
        else:
            return float(self.large[0])