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


# ㊗️ 84. Largest Rectangle in Histogram - 这一题的思考很好！尤其是边界问题的思考
class Solution:
    
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []
        max_area = 0
        # monotonic increasing stack,
        # 两个理解的核心点：
        # 1. when considering rectangle, we take current_height as base point to construct as big as possible rec
        # 2. current_height is not current_index, it is acutally the element poped.
        for i in range(len(heights)):
            print(stack,"---",max_area)
            # 这里为什么只考虑当前height和其左边的边界，因为我们是基于当前的height！
            # 原本stack[0,1]，如果满足pop的条件，我的右边界相当于current_index，这是一个技巧。
            while stack and heights[stack[-1]] >= heights[i]:
                current_height = heights[stack.pop()]  # here!
                left_boundary = -1 if not stack else stack[-1]
                current_width = i - left_boundary - 1 
                max_area = max(max_area, current_height * current_width)
            stack.append(i)
        # 最后留下的值，是遍历完所有的，因此，因此右边界是最右边！也考虑到了bottom值。
        while stack:
            current_height = heights[stack.pop()]
            left_boundary = -1 if not stack else stack[-1]
            current_width = len(heights) - left_boundary - 1
            max_area = max(max_area, current_height * current_width)
        return max_area


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

   
# 355
# 90
# 40
# 79
# 131
# 17
# 51
# 695
# 130
# 994
# 286
# 210
# 684
# 323
# 127
# 322
# 1584
# 743
# 778
# 70
# 746
# 198
# 213
# 91
# 322
# 152
# 139
# 300
# 416
# 62
# 1143
# 309
# 518
# 494
# 97
# 329
# 115
# 72
# 312
# 10
# 55
# 45
# 134
# 846
# 1899
# 763
# 678
# 1851
# 48
# 202
# 66
# 50
# 43
# 2013
# 136
# 191
# 338
# 190
# 268
# 371
# 7

# 还有64题