# 一周刷完吧...我感觉可行
# 👍 以后可以跳过
# ㊗️ 注意了需要
# ❌ 再刷
import List
import defaultdict


# 👍 36. Valid Sudoku
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        seen_row = [set() for _ in range(len(board))]
        seen_col = [set() for _ in range(len(board))]        
        seen_box = [set() for _ in range(len(board))]
        for i in range(len(board)):
            for j in range(len(board[0])):
                # 👍
                box_index = i // 3 * 3 + j // 3
                cur = board[i][j]
                if cur == '.': continue
                if cur in seen_row[i] or cur in seen_col[j] or cur in seen_box[box_index]:
                    return False
                seen_row[i].add(cur)
                seen_col[j].add(cur)
                seen_box[box_index].add(cur)
        return True
    
# ❌ 42. Trapping Rain Water
class Solution:
    def trap(self, height: List[int]) -> int:
        res, stack = 0, []
        for i in range(len(height)):
            # 首先搞明白是递减栈
            while stack and height[stack[-1]] < height[i]:
                base_index = stack.pop()
                if not stack: break # 👍 意味着左边没有墙可以阻挡，因此没有办法存水
                h = min(height[stack[-1]], height[i]) - height[base_index]
                # 👍 这个是有意义的，因为base到左边墙如果中间有space存水，那么已经存入了。
                # 如果是两个相同的base，那么第一个pop出来的，计算值是0，没有意义。只有left>base的那一对才会有意义！
                diff = i - stack[-1] - 1 
                res += h * diff
            stack.append(i)

        return res
# 167 👍 双指针

# 567 ❌ Permutation in String - 煞笔
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        target = Counter(s1)
        t = len(target)
        w = len(s1)
        records = Counter(s2[:w])

        if records == target: return True
        okay = sum(records[k] >= target[k] for k,v in target.items())
        for i in range(w, len(s2)):
            new, old = s2[i], s2[i-w]
            
            if records[old] == target[old]: # 操作不能放在下面，为什么？看下面两行的note
                okay -= 1
            
            records[old] -= 1 # 这里针对records进行了更新，如果old和new是一个char，那么我们就会对这个数据进行两次修改后才进行的if-else 判断，此时就不准确了。
            records[new] += 1 # 我们的代码是针对每一个更新，就有一个判断。
            
            if records[new] == target[new]:
                okay += 1
            
            if okay == t: return True
        return False


# 👍 239. Sliding Window Maximum
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        stack, res = deque(), []
        # stack is CORE: ASC order (value, i), i should be within the range. we actually don't need value, just index
        for i, v in enumerate(nums):
            while stack and v >= nums[stack[-1]]:
                stack.pop()
            stack.append(i)

            # (i-k+1) == left_boundary_index  -> i - (i-k+1) + 1 = num of elements in window
            if i-k+1 > stack[0]: #stack[0] == i-k:
                stack.popleft()
            
            if i-k+1 >=0: # 我写的是i-k>=0
                res.append(nums[stack[0]])

        return res
# 这一题你能写出来，但是你要好好想想 -> Monotonic Decreasing Stack stack[0]是bottom -> 我们要保持的是stack中的元素永远是range里的，这里通过popleft()删除不是range里的。
# 其次，我们要保证stack[0]一定是最大的。


# ㊗️ 155. Min Stack
class MinStack:
    # remember this is stack question, operation will be only done for [-1] element, not stack bottom
    def __init__(self):
        self.stack = [] # to track the regular queue
        self.min_stack = [] # to track min_value of regular queue

    def push(self, val: int) -> None: 
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> None:
        if self.min_stack[-1] == self.stack[-1]:
            self.min_stack.pop()
        self.stack.pop()
    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]
        

#  👍 150. Evaluate Reverse Polish Notation
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        for t in tokens:
            if t in ('+', '-', '*', '/'):
                v = stack.pop()
                if t == '+':
                    stack[-1] += v
                elif t == '-':
                    stack[-1] -= v
                elif t == '*':
                    stack[-1] *= v
                else:
                    prev = stack.pop()
                    stack.append(int(prev / v)) # int就是不要小数点后面的。
            else:
                stack.append(int(t))
        return sum(stack)
    

# 👍 22. Generate Parentheses
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []
        def bt(t, l, r):
            if l == r == n: 
                res.append(''.join(t))
            if l < n:
                bt(t+['('], l+1, r)    
            if l > r:
                bt(t+[')'], l, r+1)
        bt([],0,0)
        return res



# 739 👍 Daily Temperature 还是min stack 然后先把res = [0 * n] init出来，因为最后留在stack中的元素，我们是没有办法处理的。

# 
# ❌ 853. Car Fleet
class Solution:
    def carFleet(self, target: int, pos: List[int], speed: List[int]) -> int:
        
        # 1 - 按照pos排序
        # 2 - 每个位置只有一辆车，我们只需要计算出来每个位置上到达target的时间，转换一下。
        time = [float(target - p) / s for p, s in sorted(zip(pos, speed))] 

        res = cur = 0
        # 3 - 倒着遍历，一般来说距离target越近的位置，越先到。一个fleet的限制是由当前fleet中，最慢的那个车决定的。
        for t in time[::-1]:
            # 4 - cur存放的当前所用最大的时间。
            if t > cur:
                res += 1
                cur = t
        return res 


# 👍 704 - binary search - 练手基本题
# ㊗️ 74. Search a 2D Matrix
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        l, r = 0, m*n-1

        while l < r:  # 这里需要考虑了，if <，意味着最后l==r的时候你没有考虑，因此需要添加验证！
            mid = (l + r) // 2
            x, y = divmod(mid, n)
            v = matrix[x][y]
        
            if v == target:
                return True
            elif v < target:
                l = mid + 1
            else:
                r = mid

        return True if matrix[l//n][l%n] == target else False
from bisect import bisect_left
# ㊗️981. Time Based Key-Value Store - 不过这种题一般不会考...细节多，没必要，着重点不在算法了...
class TimeMap:
    def __init__(self):
        self.key_to_value = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        t = [timestamp, '']
        key_list = self.key_to_value[key]
        idx = bisect_left(key_list, t)
        if idx == len(key_list):
            key_list.append([timestamp, value])
        elif key_list[idx][0] == timestamp: # 如果当前timestamp被占用，那么就upsert
            key_list[idx][1] = value
        else: # 没有被占用，insert
            key_list = key_list[:idx] + [[timestamp, value]] + key_list[idx:]
        

    def get(self, key: str, timestamp: int) -> str:
        key_list = self.key_to_value[key]
        if not key_list: return ""

        t = [timestamp, '']
        idx = bisect_left(key_list, t)

        if idx == 0: # 如果是0有两种情况，因为用的bisect_left: 一种找到了，一种没找到。
            if key_list[0][0] != timestamp: return "" # 没有timestamp
            else: return key_list[0][1]
        # 没有找到timestamp
        elif idx == len(key_list): return key_list[-1][1]
        # 找到的index不是timestamp，返回左边prev
        elif key_list[idx][0] != timestamp: return  key_list[idx-1][1]
        # 找到的index是timestamp，直接返回。
        else: return key_list[idx][1]

        
# ❌ 4. Median of Two Sorted Arrays
# binary search variant -> the core == find a way to present the range + find a way to locate the mid index + find a way to narrow down the range (if condition)
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        n1 = len(nums1)
        n2 = len(nums2)
        
        if n1 > n2: return self.findMedianSortedArrays(nums2, nums1)
        
        k = (n1 + n2 + 1) // 2

        l, r = 0, n1

        while l < r:
            m1 = (l+r)//2
            m2 = k - m1

            # when I was here, I realized that I don't know how to find median...
            # in this solution version, m1/m2 is considered as smallest elements in the right part
            # m2-1 will be largest e in the left part.
            
            
            if nums1[m1] < nums2[m2-1]: # this means we need more element from nums1 to combine array
                l = m1 + 1
            else: 
                r = m1

        # premise: left part 比 right part 多0/1个元素，m1+m2=k，k是一半/一半+1
        # binary search done when l==r
        m1, m2 = l, k - l
        c1 = max(
        nums1[m1-1] if m1 > 0 else float('-inf'), # 0 意味着没有从当前list中选取值
        nums2[m2-1] if m2 > 0 else float('-inf')
        )
        if (n1+n2)%2 == 1:
            return c1
        c2 = min(
        nums1[m1] if m1 < n1 else float('inf'),
        nums2[m2] if m2 < n2 else float('inf')
        )
        return (c1+c2) * 0.5


# ㊗️ 138. Copy List with Random Pointer
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head: return None
        seen = dict()
        def helper(node):
            if not node: return None
            if node in seen: return seen[node] # copy_node
            copy_node = Node(node.val)
            seen[node] = copy_node
            copy_node.next = helper(node.next) # 关键点，不能直接去访问seen[node.next]
            copy_node.random = helper(node.random)
            return copy_node # for others to connect
        helper(head)
        return seen[head]

# 👍 2. Add Two Numbers
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        head = dummy = ListNode(0)
        carry = 0

        while l1 or l2 or carry:
            v1 = l1.val if l1 else 0 
            v2 = l2.val if l2 else 0 
            carry, val = divmod(v1+v2+carry, 10)
            head.next = ListNode(val)
            head = head.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        
        return dummy.next
    

# ㊗️ 287. Find the Duplicate Number
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        l, r = 1, len(nums)-1 # l~r is value range
        while l < r:
            m = (l+r) // 2 # 这里的m只是通过二分找到一个value当作一个锚点
            cnt = sum( n <= m for n in nums) # 找到所有<=m的值，如果如果这个数比cnt大，意味着1~m中有重复的，否则意味着m~有重复的。
            if cnt <= m:
                l = m + 1
            else:
                r = m
        return l

from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.dict = OrderedDict()
    def get(self, key):
        if key not in self.dict:
            return -1
        self.dict.move_to_end(key)
        return self.dict[key]
    def put(self, key, value):
        if key in self.dict:
            self.dict.move_to_end(key)
        self.dict[key] = value
        if len(self.dict) > self.capacity:
            self.dict.popitem(last=False)
    

# ㊗️ 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit
# 这一题的难点在于，在维护sliding window的时候如何知道sliding window的极值。
from collections import deque 
class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        min_deque, max_deque = deque(), deque()
        l = res = 0
        # 利用stack的原理
        for r in range(len(nums)):
            while min_deque and nums[r] <= nums[min_deque[-1]]:
                min_deque.pop()
            while max_deque and nums[r] >= nums[max_deque[-1]]:
                max_deque.pop()

            min_deque.append(r) # 从小到大排列
            max_deque.append(r) # 从大到小排列

            # 更新deque里的index
            while nums[max_deque[0]] - nums[min_deque[0]] > limit:
                l += 1
                if l > max_deque[0]: max_deque.popleft()
                if l > min_deque[0]: min_deque.popleft()
            res = max(res, r-l+1)
        return res
            
# ㊗️ 543. Diameter of Binary Tree 关键是读清题意，可以不通过root
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        res = 0
        def dfs(node=root):
            nonlocal res
            if not node: return 0
            left = dfs(node.left)
            right = dfs(node.right)
            res = max(res, left+right)
            return max(left, right) + 1
        dfs()
        return res


# ㊗️ 25. Reverse Nodes in k-Group
# count k -> reverse -> connect
import ListNode
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        dummy = ListNode(0, head)
        tail = dummy
        ptr = head

        while ptr:
            count = 0

            # move ptr and count to k
            while ptr and count < k:
                count += 1
                ptr = ptr.next 
                # ptr will point to the start of new group

            # determine if ptr or if meets k
            # if k -> reverse
            if count == k:
                # head is to reverse
                group_head = self.reverse_helper(head, k) # reverse & head will always be the head of each original group.
                tail.next = group_head
                tail = head
                head = ptr
        
        tail.next = head # this is to fix the landmine left by reverse process.
        return dummy.next
        
    
    def reverse_helper(self, head, k):
        ptr, group_head = head, None
        while k:
            ptr.next, group_head,  ptr= group_head, ptr, ptr.next
            k -= 1

        return group_head
        
# 👍 110. Balanced Binary Tree
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        flag = True
        def dfs(node):
            nonlocal flag
            if not node: return 0 
            left = dfs(node.left)
            right = dfs(node.right)
            if abs(left-right) > 1: flag = False
            return max(left, right) + 1
        dfs(root)
        return flag 

# 👍 199. Binary Tree Right Side View

# 👍 1448. Count Good Nodes in Binary Tree
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        cnt = 0
        def dfs(node, ceil=float('-inf')):
            nonlocal cnt
            if not node: return
            if node.val >= ceil: 
                ceil = node.val
                cnt += 1

            dfs(node.left, ceil) 
            dfs(node.right, ceil)

        dfs(root)
        return cnt 

import heapq
# 👍 703. Kth Largest Element in a Stream
class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.heap = nums
        heapq.heapify(self.heap) 
    
        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val: int) -> int:        
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, val)
        else:
            heapq.heappushpop(self.heap, val)
        return self.heap[0]


# 👍 1046. Last Stone Weight

# 一半👍 78. Subsets
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        def bt(path=[], start=0):
            res.append(path[:])
            for i in range(start, len(nums)):
                bt(path+[nums[i]], i + 1) # 这里是I+1, 而非是start+1
        bt()
        return res
    

# 👍 39. Combination Sum
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        def bt(bucket=[], rest=target, cs=candidates):
            if rest == 0: res.append(bucket[:])
            for i in range(len(cs)):
                c = cs[i]
                if c <= rest:
                    bt(bucket+[c], rest-c, cs[i:])
        bt()
        return res
# 👍 46 permutations 简单
# 👍 973.K Closest Points to Origin - 就是用heap

# ㊗️ 215. Kth Largest Element in an Array / quickSelect, counting sort
class Solution:
    # def findKthLargest(self, nums: List[int], k: int) -> int:
    #     min_ = min(nums)
    #     max_ = max(nums)
    #     count = [0] * (max_ - min_ + 1)

    #     for n in nums:
    #         count[n-min_] += 1
        
    #     remain = k

    #     for i in range(len(count)-1, -1, -1):
    #         remain -= count[i]
    #         if remain <= 0: return min_+i

    #     return -1

    def findKthLargest(self, nums: List[int], k: int) -> int:
        def quick_select(nums, k):
            pivot = random.choice(nums)
            left, mid, right = [], [], []
            for n in nums:
                if n > pivot:
                    left.append(n)
                elif n < pivot:
                    right.append(n)
                else:
                    mid.append(n)
            
            if k <= len(left):
                return quick_select(left, k)
            if len(left) + len(mid) < k:
                return quick_select(right, k - len(left) - len(mid))

            return pivot

        return quick_select(nums, k)


# ㊗️ 621. Task Scheduler
class Solution:
    def leastInterval(self, tasks, n):
        
        length = len(tasks)
        if length <= 1:
            return length
    
        task_map = Counter(tasks)
        task_sort = sorted(task_map.items(), key=lambda x: x[1], reverse=True)
        
        max_task_count = task_sort[0][1]
        res = (max_task_count - 1) * (n + 1) # 除了最后一次任务之外，完成任务本身+空档需要的时间。
        
        for t, val in task_sort:
            if val == max_task_count: # 如果是自己的话+1，如果有同样多个元素的话，也是+1。
                res += 1
        
        # 精华如果length太大，意味着虽然interval很小，但是我们有很多出现很少的别的类型的任务，按照最小的interval排最大的类型，是不够所有task执行的，这种情况下，一定是可以避免constraint的，因此只用返回length就可以了。
        return res if res >= length else length

   
from collections import defaultdict
import heapq
# ㊗️ 355. Design Twitter
# 每个用户有自己的tweet List, 通过timestamp排序，利用链表连接起来。
# 如果有followee的话，将不同followee的list merge起来！算法与OOD的结合！
class Tweet:
    def __init__(self, tweetId, timestamp):
        self.id = tweetId
        self.timestamp = timestamp
        self.next = None

    def __lt__(self, other):
        return self.timestamp > other.timestamp


class Twitter:

    def __init__(self):
        self.followings = defaultdict(set)
        self.tweets = defaultdict(lambda: None) # 当访问不存在的键时，默认返回None
        self.timestamp = 0

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.timestamp += 1
        tweet = Tweet(tweetId, self.timestamp)
        tweet.next = self.tweets[userId]
        self.tweets[userId] = tweet

    def getNewsFeed(self, userId: int) -> List[int]:
    # 当Tweet对象被添加到堆（heap）中时，它们的指针（即对象引用）和链表连接关系并不会发生改变。
    # 这些操作只是将Tweet对象的引用添加到了一个新的容器（堆）中，以便进行排序和检索，而不会修改这些对象本身或它们之间的连接（即next属性指向的关系）。
        tweets = []
        heap = []

        tweet = self.tweets[userId]
        if tweet:
            heap.append(tweet)

        for user in self.followings[userId]:
            tweet = self.tweets[user]
            if tweet:
                heap.append(tweet)
        heapq.heapify(heap)

        while heap and len(tweets) < 10:
            head = heapq.heappop(heap)
            tweets.append(head.id)

            if head.next:
                heapq.heappush(heap, head.next)



        return tweets

    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId == followeeId:
            return
        self.followings[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followerId == followeeId:
            return
        self.followings[followerId].discard(followeeId)


# 👍  90. Subsets II
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        def bt(remain, path):
            res.append(path[:])
            for i in range(len(remain)):
                if i == 0 or remain[i] != remain[i-1]:
                    bt(remain[i+1:], path+[remain[i]])
        bt(nums, [])
        return res
# 👍 40. Combination Sum II
# 如果想要更高效，那么就不传递cs进去，而是通过index进行控制。
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        res = []
        def bt(cs, remain, path):
            if remain == 0: res.append(path[:])
            for i in range(len(cs)):
                if i != 0 and cs[i] == cs[i-1]:  # 去重
                    continue
                if cs[i] > remain:  # 剪枝
                    break
                bt(cs[i+1:], remain-cs[i], path+[cs[i]])

        bt(candidates, target, [])
        return res

# ㊗️ 79. Word Search
class Solution(object):
    def exist(self, board, word):
        if not board or not word: return False        

        def dfs(i, j, word):
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
            return found # 如果我们要返回递归结果，那么边界条件/edge case一定要有t/f，一般来说edge case常用来剪纸/满足条件


        for i in range(len(board)):
            for j in range(len(board[0])):
                if dfs(i, j, word):
                    return True

        return False


# ㊗️ 131. Palindrome Partitioning
class Solution:
    def partition(self, s):
        res = []
        def helper(s, path):
            if not s: 
                res.append(path)
                return 
            
            for i in range(1, len(s)+1):
                if s[:i] == s[:i][::-1]: # 这个思路是精髓
                    helper(s[i:], path+[s[:i][:]])

        helper(s, [])
        return res 


#  👍  17. Letter Combinations of a Phone Number - hard code就可以
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if len(digits) == 0: return []
        letters = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", 
                   "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        
        def backtrack(index, path):
            if len(path) == len(digits):
                result.append(''.join(path))
                return
                
            for letter in letters[digits[index]]:
                path.append(letter)
                backtrack(index + 1, path)
                path.pop()
            
        
        result = []
        backtrack(0, [])
        return result
    
    
# ㊗️ 51. N-Queens 不难 - 要明白如何判断对角线，和row/col
class Solution:
    def solveNQueens(self, n):
        def create_board(state):
            board = []
            for row in state:
                board.append("".join(row))
            return board
        
        def backtrack(r, diagonals, anti_diagonals, cols, state):
            if r == n: 
                res.append(create_board(state))
                return 

            for c in range(n):
                cur_dia =  r - c
                cur_anti_dia = r + c

                if (cur_dia in diagonals or cur_anti_dia in anti_diagonals or c in cols):
                    continue

                diagonals.add(cur_dia)
                anti_diagonals.add(cur_anti_dia)
                cols.add(c)
                state[r][c]="Q"

                backtrack(r+1, diagonals, anti_diagonals, cols, state)


                diagonals.remove(cur_dia)
                anti_diagonals.remove(cur_anti_dia)
                cols.remove(c)
                state[r][c]="."

        res = []
        state = [["."] * n for _ in range(n)]
        backtrack(0, set(), set(), set(), state)
        return res





# 👍 695. Max Area of Island
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        if not grid: return 0
        max_area = 0

        def get_area(i, j):
            curr_area = 1
            grid[i][j] = 0

            for ni, nj in ((i+1, j),(i-1, j),(i, j+1),(i, j-1)):
                if 0 <= ni < len(grid) and 0 <= nj < len(grid[0]) and grid[ni][nj] == 1:
                    curr_area += get_area(ni, nj)

            return curr_area

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    max_area = max(max_area, get_area(i, j))
    
        return max_area
    

#  👍 130. Surrounded Regions - 额外的数据结构存储就不变的就可以。
# ㊗️  994. Rotting Oranges
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        
        m, n = len(grid), len(grid[0])
        queue = []
        fresh_cnt = 0
        # 2->rotten; 1->fresh; 0->empty
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    queue.append((i, j))
                elif grid[i][j] == 1:
                    fresh_cnt += 1

        time = 0
        while queue:
            next_queue = []
            for _ in range(len(queue)):
                i, j = queue.pop()
                for ni, nj in ((i+1, j),(i-1, j),(i, j+1),(i, j-1)):
                    if 0 <= ni < len(grid) and 0 <= nj < len(grid[0]) and grid[ni][nj] == 1:
                        grid[ni][nj] = 2
                        next_queue.append((ni, nj))
                        fresh_cnt -= 1 # 精髓 >>> 否则答案不准确
            if next_queue: # 精髓 >>> 否则答案不准确 # next_queue有值才意味着下一个rooten才可能发生
                time += 1
            queue = next_queue

        return time if fresh_cnt == 0 else -1
    

# 👍 286 Gates and walls


# 👍 210. Course Schedule II
class Solution:
    def findOrder(self, num_courses: int, prerequisites: List[List[int]]) -> List[int]:
        in_degree = [0] * num_courses
        course_map = defaultdict(list)
        for c, p in prerequisites:
            in_degree[c] += 1
            course_map[p].append(c)

        course_ready_to_take = []
        for i, v in enumerate(in_degree):
            if v == 0: 
                course_ready_to_take.append(i)


        res = []
        while course_ready_to_take:
            course_taken = course_ready_to_take.pop()
            res.append(course_taken)
            for next_possible_course in course_map[course_taken]:
                in_degree[next_possible_course] -= 1
                if in_degree[next_possible_course] == 0:
                    course_ready_to_take.append(next_possible_course)

        return res if len(res) == num_courses else []
        

        
# ㊗️ 684. Redundant Connection
# 利用x==find(x)可以找到集群
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        n = len(edges)
        parent = [i for i in range(n+1)]
        
        def find(x):
            if x != parent[x]:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py


        for s, e in edges:
            if find(s) == find(e):
                return [s, e]
            union(s, e)
      
# ㊗️ 323. Number of Connected Components in an Undirected Graph
class Solution:
    # 下面是Union-Find的写法，当然也可以构造出来图，然后去dfs，用visited=[False]*n记录。 -》 Time/Space: O(E+V)
    # E = numbers of edges, V = numbers of vertices.
    # O(E*a) a is union ; Space: V 
    def countComponents(self, n, edges):
        def find(x):
            return parent[x] if parent[x] == x else find(parent[x])
        
        def union(x, y):
            xr, yr = find(x), find(y)
            if xr != yr:
                if rank[xr] < rank[yr]:
                    parent[xr] = yr
                else:
                    parent[yr] = xr
                    if rank[xr] == rank[yr]:
                        rank[xr] += 1
                
        parent, rank = list(range(n)), [0] * n
        for x,y in edges: 
            union(x,y)
            
        return len(set(find(x) for x in parent))
# ㊗️ 127. Word Ladder
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        # edge case
        if len(beginWord) != len(endWord) or not endWord or endWord not in wordList or not wordList: return 0

        # build records {str:[word...]}
        records = defaultdict(list)
        for w in wordList:
            for i in range(len(w)):
                word_key = w[:i] + "*" + w[i+1:]
                records[word_key].append(w)

        # init(seen/queue)
        seen = set()
        queue = [(beginWord, 0)]
        # BFS 如果不用deque.popleft()的话，只能用额外的循环确保level的顺序。
        while queue:
            next_queue = []
            for _ in range(len(queue)):
                cur_word, num = queue.pop()
                if cur_word == endWord: 
                    return num + 1
                
                for i in range(len(cur_word)):
                    new_word_key = cur_word[:i] + "*" + cur_word[i+1:]
                     
                    if new_word_key in records:
                        for next_possible_word in records[new_word_key]:
                            if next_possible_word in seen: continue
                            seen.add(next_possible_word)
                            next_queue.append((next_possible_word, num+1))
                        records[new_word_key] = []
            queue = next_queue

        # return 
        return 0
        
# 👍 322. Coin Change 本质上就是找到一个组合（完全背包）
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:

        dp = [float('inf')] * (amount+1)
        dp[0] = 0

        for n in range(1, amount+1):
            for c in coins:
                if n-c >= 0:
                    dp[n] = min(dp[n-c]+1, dp[n])

        return dp[-1] if dp[-1] != float('inf') else -1

# 1584 这道题可以不用看，是MST问题，最小生成树 -> 连接所有点的成本最小。
# 有两种solution：Kruskal和Prim算法
# Kruskal: 计算所有遍权重(d,i,j) -> 排序 -> 利用union-find集合，直到连接了所有点。
# Prim: 初始化（从任意一个点出发构造） -> 使用heap加入最小距离点，然后更新所有点到树的距离 -> 重复
# 两种方法的复杂度都是n^2*logn
class Solution_Kruskal:
    def minCostConnectPoints(self, points):
        n, edges = len(points), []
        
        # 1. 计算所有边的权重
        for i in range(n):
            for j in range(i + 1, n):
                dist = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
                edges.append((dist, i, j))
        
        # 2. 排序所有边
        edges.sort()

        # 3. 初始化 Union-Find
        parent = list(range(n))
        rank = [0] * n

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y: # 利用了rank优
                if rank[root_x] > rank[root_y]:
                    parent[root_y] = root_x
                elif rank[root_x] < rank[root_y]:
                    parent[root_x] = root_y
                else:
                    parent[root_y] = root_x
                    rank[root_x] += 1
                return True
            # 如果是一样的，就没必要在连接起来。
            return False

        # 4. 构造最小生成树
        mst_cost = 0
        edges_used = 0

        for cost, u, v in edges:
            if union(u, v):
                mst_cost += cost
                edges_used += 1
                if edges_used == n - 1:
                    break
        
        return mst_cost



# ㊗️ 743. Network Delay Time
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        if len(times) < n - 1: return -1
        node_map = defaultdict(list) # O(E)
        for (x, y, v) in times: # O(E)
            node_map[x].append((y, v))

        seen = set() # O(n)
        queue = [(0, k)] 
        heapq.heapify(queue)

        while queue: # O(n)
            cost, node = heapq.heappop(queue) # O(logn)
            if node in seen: continue
            seen.add(node)
            if len(seen) == n: 
                return cost
            for new_node, new_cost in node_map[node]:
                heapq.heappush(queue, (cost+new_cost, new_node))
        return -1
        
# 👍 70. Climbing Stairs
class Solution:
    def climbStairs(self, n: int) -> int:
        if n in (0, 1, 2): return n
        one, two = 1, 2
        for _ in range(3,n+1):
            one, two = two, one + two
        return two

# 👍 778. Swim in Rising Water
class Solution:
    def swimInWater(self, grid: List[List[int]]) -> int:
        heap = [(grid[0][0],0,0)]
        seen = set()
        seen.add((0,0))

        while heap:
            t, x, y = heapq.heappop(heap)

            if x == len(grid)-1 and y == len(grid[0])-1:
                return t

            for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    new_t = max(t, grid[nx][ny])
                    heapq.heappush(heap, (new_t, nx, ny))

        
# 👍 746. Min Cost Climbing Stairs
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        pre, cur = 0, 0
        for i in range(2, len(cost)+1):
            cur, pre = min(pre+cost[i-2], cur+cost[i-1]), cur
        return cur

# 👍 198. House Robber
class Solution:
    def rob(self, nums: List[int]) -> int:
        dp = [[0] * 2 for _ in range(len(nums))]
        dp[0][1] = nums[0]

        for i in range(1, len(nums)):
            v = nums[i]
            dp[i][1] = dp[i-1][0] + v
            dp[i][0] = max(dp[i-1][0], dp[i-1][1]) 

        return max(dp[-1])

# ㊗️ 213. House Robber II
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
            t1, t2 = max(n+t2, t1), t1  # max有点non-sense这里，但主要就是为了保存最大值，这是一个技巧。精髓吧...
        return t1
    
    
# ㊗️ 91. Decode Ways
class Solution:
    def numDecodings(self, s: str) -> int:
        if s[0] == '0': return 0
        one = two =1 # one是i-1, two是i-2
        for i in range(1, len(s)):
            cur = 0
            
            if s[i] != '0':
                cur = one

            if 10 <= int(s[i-1:i+1]) <= 26:
                cur += two
            
            one, two = cur, one
        
        return one
            

# ㊗️ 152. Maximum Product Subarray
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if len(nums) == 1: return nums[0]
        min_prod = max_prod = res = nums[0]

        for n in nums[1:]:
            temp = max_prod # 是保留之前的max_prod，用于计算当前的min_prod
            max_prod = max(n,  max_prod * n, min_prod * n) # n的存在是为了避免0的情况发生。
            res = max(res, max_prod)
            min_prod = min(n, temp * n, min_prod * n)

        return res


# ㊗️ 139. Word Break
class Solution:
    def wordBreak(self, w: str, wordDict: List[str]) -> bool:
        dp = [False] * (len(w) + 1)
        dp[0] = True

        for e in range(len(w)):
            for s in range(e+1): # start_index是希望取到end index的，因为切片器slicer的原因。
                cur_str = w[s: e+1]
                if cur_str in wordDict and dp[s]: # dp[s] 是看(s-1) index是否可以成立。s是当前的window
                    dp[e+1] = True
                    break
            
        return dp[-1]

    
# ㊗️ 300. Longest Increasing Subsequence
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n

        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j]+1)
        return max(dp) # 这里注意是max，而非dp[-1]
    def lengthOfLIS(self, nums: List[int]) -> int:
        sub = []
        for num in nums:
            i = bisect_left(sub, num)
        
            if i == len(sub):
                sub.append(num)
            else:
                sub[i] = num # replace the first element in sub greater than or equal to num
                # 这里的替换很有意思，如果替换的最后一位元素很好理解--尽可能使得当前的sub更小，之后的值才更有可能组成increasing。
                # 如果替换的是之前的元素，是为了能够更新最后一位元素做准备，而其本身替换是不make sense的，不过没关系，因为对答案没影响。
        return len(sub)
    

# ㊗️㊗️ 416. Partition Equal Subset Sum
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total % 2 == 1: return False 
        target = total // 2
        dp = [False] * (target + 1) # dp[n] == 是否可以组成n的例子。
        dp[0] = True
        for n in nums:
            for t in range(target, n-1, -1): # 倒序很重要！㊗️ 我之前的写法用的是二维dp[][]。因为我们不希望dp[t-n]受到了当前元素的多次影响。比如0-n-2n-3n更新。
                dp[t] = dp[t] or dp[t - n]
            
        return dp[-1]

# 👍 62. Unique Paths
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0] * n for _ in range(m)]

        for i in range(n):
            dp[0][i] = 1
        for i in range(m):
            dp[i][0] = 1
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]
    

# 👍 1143. Longest Common Subsequence
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

        return dp[-1][-1] # 我们想让答案放在最后。
          

# ❌ 309. Best Time to Buy and Sell Stock with Cooldown
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        sell, hold, cool = 0, -prices[0], 0
        # buy和sell是一个动作，这里的核心还是state，那么hold就是有stock的state，cool是没有stock的state。
        for p in prices[1:]:
            pre_sold = sell
            sell = hold + p
            hold = max(hold, cool - p)
            cool = max(pre_sold, cool)
        return max(cool, sell)

# 👍 518. Coin Change II - 完全背包问题 dp[t] += dp[t-n]
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1
        for coin in coins: 
            for i in range(coin, amount+1):
                dp[i] += dp[i - coin]
        return dp[amount]
# ❌ 494. Target Sum
# BF -- O(2^n): backtracking --> if i==end if total == sum: count++
# recursion with memorization,多存一个memo{[index, curr_sum]: count} 多加一个判断，if seen: return memo[(index, curr_sum)]
# 优化DP的方法 -- O(n*sum)/O(sum)
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:        
        total_sum = sum(nums)
        if abs(target) > total_sum:  # 不可能的情况
            return 0
        dp = [0] * (2 * total_sum + 1) # 因可正可负，所以取值范围是2倍的total_sum
        dp[total_sum] = 1  # 初始化，这里的total_sum其实是原点

        for num in nums:
            next_dp = [0] * (2 * total_sum + 1)
            for j in range(2 * total_sum + 1):
                if dp[j] != 0:  # 只有前一状态有值才需要更新
                    next_dp[j + num] += dp[j]
                    next_dp[j - num] += dp[j]
            dp = next_dp 

        return dp[total_sum + target]

# ㊗️ 97. Interleaving String
# 其实也挺简单的。
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        m, n = len(s1), len(s2)
        if m+n != len(s3): return False
        dp = [[0]*(n+1) for _ in range(m+1)]
        dp[0][0] = 1
        
        # 这里有点confusing，但是注意break，因此这里的init只是看prefix最长可以有多少个匹配。
        for i in range(1, m+1):
            if s1[i-1] == s3[i-1]:
                dp[i][0] = 1
            else:
                break
        for i in range(1, n+1): 
            if s2[i-1] == s3[i-1]:
                dp[0][i] = 1
            else:
                break

        # s3[i-1 + j-1] 对应的就是s1[i-1] + s2[j-1] 所以有新的字符的时候就为s[i+j-1]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if (dp[i-1][j] and s1[i-1] == s3[i+j-1]) or (dp[i][j-1] and s2[j-1] == s3[i+j-1]):
                    dp[i][j] = 1
		
        return dp[-1][-1] == 1
    

# 329 👍 Longest Increasing Path in a Matrix 均为O(mn)因为有seen的原因，因此每个cell只会遍历一次。

# ❌ 115. Distinct Subsequences
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        m, n = len(s), len(t)
        dp = [0] * (n + 1)
        dp[0] = 1  # 空串 t 可以被任意 s 匹配 1 次
        
        for i in range(1, m + 1):
            # 这一题主要学习空间优化！我们j列的数据的更新依赖于j-1列，因此我们可以不用二维，而只用之前的一列，也就是prev。
            # 但是这里，我又更加地优化了一下 -》 我们甚至不用prev，因为s当中的数据只能用一次。我们可以采用倒序直接在原dp中进行更新。
            # prev = dp[:]  # 记录上一行的状态
            for j in range(n, 0, -1):  
                if s[i-1] == t[j-1]:
                    dp[j] = dp[j-1] + dp[j]
                    # dp[j] = prev[j-1] + prv[j]
                # else:
                #     dp[j] = prev[j]
        
        return dp[n]
    

# 72 ❌ Edit Distance
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n1 = len(word1)
        n2 = len(word2)
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]

        for j in range(n2): # 如果 word1 是空串（i=0），要把它变成 word2[:j]，只能通过连续插入操作。
            dp[0][j+1] = dp[0][j] + 1
        for i in range(n1): # 如果 word2 是空串（j=0），要把 word1[:i] 变成空串，只能通过连续删除操作。
            dp[i+1][0] = dp[i][0] + 1

        # 每个元素只能用一次，可以去看target中各个元素的可能性，所以元素放在外面的for中， 1/0背包
        # 如果元素用几次都行，那么元素的循环放在里面，完全背包。
        for i in range(n1):
            for j in range(n2):
                if word1[i] == word2[j]:
                    dp[i+1][j+1] = dp[i][j]
                else:
                    # 这里的状态转移是难点。
                    dp[i+1][j+1] = min(     # dp[i+1][j+1] 对应的是w1[:i+1] 匹配到 w2[:j+1]
                        dp[i+1][j],         # Add. 如果我们要使得word1[:i+1]添加一个字符串可以匹配word2[:j+1]的话，那么原本的字符串应该可以满足匹配word2[:j]
                        dp[i][j+1],         # Delete. 意味着上一个word1[:i]就已经可以满足当前的word2[:j+1]了
                        dp[i][j]            # Replace. word1[:i]和word2[:j]本身就可以满足。
                        ) + 1 
                    # 我困惑的root cause是不理解dp[i][j]存放的值到底是什么意思。
                    # 是表明word1[:i+1] 和 word2[:j+1]匹配的话最少需要多少个操作，而非改变dp这个数据结构所依赖的原始数据。dp[i][j]是已经便利过的[:i]所有的操作的可能性的最小值。不一定是str[:i]
        return dp[-1][-1]



# 312. Burst Balloons
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        # 添加虚拟气球 1
        nums = [1] + nums + [1]
        n = len(nums)
        
        # 初始化 dp 数组
        dp = [[0] * n for _ in range(n)]
        
        # 遍历区间长度
        for length in range(2, n):  # 至少三个气球才能戳
            for i in range(0, n - length):  # 左边界
                j = i + length  # 右边界
                # 尝试戳破每个气球 k
                for k in range(i + 1, j):  # k 在 (i, j) 范围内
                    dp[i][j] = max(dp[i][j],
                                   dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j])
        
        # 返回整个区间的最大积分
        return dp[0][n-1]



# 10. Regular Expression Matching
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        # 初始化DP表
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        
        # 空字符串和空模式匹配
        dp[0][0] = True
        
        # *表示可以用来匹配n个前面的字符/干脆不匹配。也就意味着*前一定有字符。
        # 这里的初始化表示着我们可以从*后面开始进行匹配，如果p有前缀*的话。比如a*b*c*我们可以从c开始进行匹配。
        for i in range(1, n):
            if p[i] == '*':
                dp[0][i+1] = dp[0][i-1]  
        
        for i in range(m):
            for j in range(n):
                if p[j] == s[i] or p[j] == '.':
                    dp[i+1][j+1] = dp[i][j]  # 字符匹配
                elif p[j] == '*':
                    # '*'两种情况:
                    # 1. 不使用'*'和前一个字符 (看dp[i+1][j-1]) ： 当前的text和j-1的pattern匹配，相当于不用当前的字符了。
                    # 2. 使用'*'且前一个字符匹配 (看dp[i][j+1])  ： dp[i][j+1] == True意味着 p[:j+1] 与 s[:i]可以match，无论是否使用了p[j]都没关系，为什么呢？因为我们现在尝试match s[i]，如果之前使用了p[j]，可以多次使用呀，所以可以继续match，如果是.，更没问题了。
                    dp[i+1][j+1] = dp[i+1][j - 1] or (dp[i][j+1] and (s[i] == p[j-1] or p[j-1] == '.')) #
        
        return dp[-1][-1]


# 45. Jump Game II
class Solution:
    def jump(self, nums: List[int]) -> int:
        # 我的困惑点在于哪怕知道可以reach to end，but which way to take？
        res, n = 0, len(nums)

        cur_end = cur_far = 0

        for i in range(n-1):
            cur_far = max(cur_far, i + nums[i])

            # if we finish the starting range of current jump
            # Move on to the starting range of next jump
            # 牛逼，好好研读
            if i == cur_end:
                res += 1
                cur_end = cur_far

        return res
# ㊗️ 134. Gas Station
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        n = len(gas)

        if sum(cost) > sum(gas): return -1
        cur_tank = 0
        start_position = 0

        for i in range(n):
            cur_tank += gas[i] - cost[i]
            if cur_tank < 0: # Greedy的精华，只要有negative sum的，我们就选择新的starting position，可以直接跳过negative的position。
                start_position = i + 1
                cur_tank = 0
        return start_position 


# 👍 846. Hand of Straights
class Solution:
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        if len(hand) % groupSize != 0: return False
        counter = Counter(hand)
        record = list(sorted(counter.keys()))

        while record:
            cur = heapq.heappop(record)
            size = counter.pop(cur, 0)

            for add in range(1, groupSize):
                add_cnt_left = counter[cur+add] - size
                if add_cnt_left < 0: return False
                if add_cnt_left == 0: 
                    heapq.heappop(record)
                    counter.pop(cur+add)
                counter[cur+add] = add_cnt_left
                
        return True
        
# 👍 1899. Merge Triplets to Form Target Triplet
class Solution:
    def mergeTriplets(self, triplets: List[List[int]], t: List[int]) -> bool:
        ready = list()
        f1 = f2 = f3 = False
        for x, y, z in triplets:
            if x <= t[0] and y <= t[1] and z <= t[2]:
                if x == t[0]: f1 = True
                if y == t[1]: f2 = True
                if z == t[2]: f3 = True
                ready.append([x,y,z])
        
        if not ready or (len(ready) == 1 and ready[0] != t): return False

    

        return f1 and f2 and f3

        

# 👍 763. Partition Labels
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        if not s: return 0

        last_seen = {}
        for i, c in enumerate(s):
            last_seen[c] = i

        l = 0 # window = s[l:r+1]
        res = []
        right_max = 0
        for r in range(len(s)):
            right_max = max(right_max, last_seen[s[r]])
            # reach to right boundary
            if r == right_max:
                res.append(r-l+1)
                l = r+1
        return res

# ❌ 678. Valid Parenthesis String
# 这一题decompose了问题，从valid parenthsis -> 要想valid，那么转化/抵消后的string一定不能是)开头
# 也就是说在*的帮助下。(出现的最高频率可能性不能为<0
class Solution:
    def checkValidString(self, s: str) -> bool:
        low = high = 0 # 因为*的存在，这里low/high表示(出现的可能性区间。
        for c in s:
            low += 1 if c == '(' else -1
            high += 1 if c in ('*', '(') else -1
            if high < 0: return False
            low = max(0, low)
            # 上述写法常规版其实是if c == '(' / ')' / '*'
        return low==0 # 最低可能性应该为0，为什么？因为你不想s='((((('根本没有办法close掉对吧。
            

# ❌ 1851. Minimum Interval to Include Each Query
class Solution:
    def minInterval(self, intervals, queries):
        intervals = sorted(intervals, reverse=True) #倒序，这样pop顺序才是顺序
        query_cache = {}
        size_heap = []

        # 这一题核心点利用了heap存放满足当前query位置的所有interval，因此需要有序。
        for q in sorted(queries):

            # 1. 更新 size_heap, 把满足的interval放进去
            while intervals and intervals[-1][0] <= q: # 起点比当前q小的interval都要处理完毕，根据q是否在interval内决定是否更新size_heap
                l, r = intervals.pop()
                if q <= r: heapq.heappush(size_heap, [r-l+1, r]) 
                # 精华：放入r是用于比较interval是否满足当前的q
                # 那为什么在下面不去比较left，而是只比较right呢？
                # 因为过往的q的肯定比当前的q小，因此进去到size_queue的我们能肯定left是满足当前q的，就是不知道right是否仍然满足。

            # 2. prep for size_heap, 里面可能存了之前放进去，但是不满足当前q范围的interval
            while size_heap and size_heap[0][1] < q:
                heapq.heappop(size_heap)
            
            # 3. 缓存q的result并更新答案
            query_cache[q] = size_heap[0][0] if size_heap else -1

        return [query_cache[q] for q in queries]

# 48. Rotate Image 下面是技巧
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
# 👍 202. Happy Number
class Solution:
    def isHappy(self, n: int) -> bool:
        seen = set()
        while n != 1:
            if n in seen: return False
            seen.add(n)
            next_n = 0
            while n:
                n, remainder = divmod(n, 10)
                next_n += remainder**2
            n = next_n
        
        return True
# 👍 66 - Plue one - return [int(x) for x in str(int("".join(map(str,digits)))+1)]


# ㊗️ 50. Pow(x, n)
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0: return 1
        if n < 0: return 1.0 / self.myPow(x, -1*n)

        if n%2 == 1:
            return x * self.myPow(x*x, (n-1)//2)
        else:
            return self.myPow(x*x, n//2)

        # another solution below this line
        if n == 0: return 1
        if n < 0: 
            n = -n
            x = 1.0/x

        result = 1
        while n != 0:
            if n % 2 == 1:
                result *= x
                n -= 1
            x *= x
            n //= 2
            return result
            

# ㊗️ 43. Multiply Strings
# 本题的精髓就是模拟乘法并且操作index进行更新
class Solution:
    def multiply(self, num1, num2):
        m, n = len(num1), len(num2)
        product = [0] * (m + n) # 最后一位为m+n-1

        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                mul = int(num1[i]) * int(num2[j])
                # p2是本位，p1是高位，用于存放carry
                p1, p2 = i + j, i + j + 1  
                total = mul + product[p2]
                product[p1] += total // 10
                product[p2] = total % 10

        result = ''.join(map(str, product))
        return result.lstrip('0') or '0'

# ㊗️ 2013. Detect Squares
class DetectSquares:
    def __init__(self):
        self.points=defaultdict(lambda :defaultdict(int)) # [Y][X]坐标上有多少个点，[x][y]也行我觉得
    def add(self, point: List[int]) -> None:
        x,y = point
        self.points[y][x]+=1
    def count(self, point: List[int]) -> int:
        X,Y = point
        count = 0
        for x in self.points[Y]: # 看看当前Y下有没有其他X点的坐标
            d=abs(x-X) # d是边长
            if d==0: continue # 防止与X,Y同一个位置
            # 因为X, Y已经确定了，而我们另一个水平的点出发可以确定一条水平的边，因此只需要检查上方/下方的square就行了。
            count+=(self.points[Y-d][x]*self.points[Y-d][X]*self.points[Y][x]) # 下方的square 这是其他三个点的坐标
            count+=(self.points[Y+d][x]*self.points[Y+d][X]*self.points[Y][x]) # 上方的square
        return count

        
# 👍 136 Single Number - return reduce(lambda x, y: x ^ y, nums)
# 👍 191 Number of 1 Bits - return str(bin(n)).count('1')
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
# 190. Reverse Bits
# 通过对位的调整，一个1一个1的“对齐”
# 还是对位运算的不熟悉
class Solution:
    def reverseBits(self, n: int) -> int:
        result, power = 0, 31
        while n:
            # 因为input给的是32位，所以我们通过& 1 得到最后一位时，直接对result进行移位处理，直接reverse，这里的power很奇妙
            result += (n & 1) << power
            # &1取得最后一位，但是n还是原来的，因此需要做处理
            # n的最后一位在上一行处理完毕，因此需要同步起来。
            n = n >> 1
            # 下一次就需要变了
            power -= 1
        return result
    
# 268 Missing number
# class Solution:
#     def missingNumber(self, nums: List[int]) -> int:
#         n = len(nums)
#         for i, e in enumerate(nums):
#             n ^= i
#             n ^= e
#         return n 

class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        s = set(nums)
        for i in range(len(nums)+1):
            if i not in s:
                return i

# 371 跳过
# 7. Reverse Integer
class Solution:
    def reverse(self, x: int) -> int:
        flag = 1 if x > 0 else -1
        x = list(str(abs(x)))
        tt = 0 
        for n in x[::-1]:
            tt = tt*10 + int(n)
        tt = flag * tt
        return tt if -2**31<= tt <= 2**31 else 0
        
