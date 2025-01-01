# 4. Median of Two Sorted Arrays
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        n1, n2 = len(nums1), len(nums2)
        if n1 > n2: return self.findMedianSortedArrays(nums2, nums1)# 保证 nums1 是较短的数组，减少二分范围

        # premise: left part 比 right part 多0/1个元素，m1+m2=k，k是一半/一半+1
        k = (n1 + n2 + 1) // 2 
        l, r = 0, n1

        while l < r:
            # 因为k是按照len()来决定的，所以这里m1/m2应该是right part最小的元素。可以自己举个例子理解一下。
            m1 = (l+r)//2 
            m2 = k - m1

            # 其实我们也可以使用m1-1和m2对比，但是我们需要担心边界问题，为什么？因为nums1比nums2短。
            if nums1[m1] < nums2[m2-1]: # 这个比较的实际意义是，m1在当前放入了右侧，m1比左侧元素(m2-1)还要小，应该放在左侧，意味着nums1的分割点小了，所以要调整l.
                l = m1 + 1
            else: 
                r = m1
        
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
    
# 3355. Zero Array Transformation I
# nb. 自己对于prefix_sum的用法已经忘却了。
class Solution:
    def isZeroArray(self, nums, queries):
        n = len(nums)
        freq = [0] * n
        
        # setup for prefix sum operation
        for s, e in queries:
            freq[s] += 1
            if e + 1 < n:
                freq[e+1] -= 1
                
        for i in range(n):
            if i > 0:
                freq[i] += freq[i-1]
            if freq[i] < nums[i]:
                return False
        
        return True
    
# 3362. Zero Array Transformation III
# 与II不同的是，这里的query没有先后顺序，因此需要先sort一下，然后用根据滑动窗口，动态更新heap得值，然后从heap中拿出来范围最大的点，然后更新diff
class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        queries.sort(key=lambda q: q[0]) # 按照left去排列
        h = [] # heap - 最后留下的就是没有用的。
        diff = [0] * (len(nums) + 1) # 
        j = 0 #与diff（类似prefix_sum）配合，从而达到当前num已经有多少个query会来进行操作了。
        # i是nums的index；j是query的index
        for i, x in enumerate(nums):
            if i > 0: 
                diff[i] += diff[i-1]
            
            # heap准备工作，当前遍历到i，把left <= i的所有queries的right端点放进H中，先pop出大的来。
            while j < len(queries) and queries[j][0] <= i: # 把比当前小的都放进去！
                heappush(h, -queries[j][1])  # 取相反数表示最大堆
                j += 1
            
            # 如果当前diff[i]积累的操作不能够满足x的需求，那么我们尝试去拿query
            # 拿query的逻辑是贪心：比如所有能拿的query中，必须拿一个的话，我们想要尽可能大的query对吧。
            while diff[i] < x and h and -h[0] >= i: # -h[0] >= i 保证拿出来的query一定是允许操作当前I的
                diff[i] += 1
                diff[-heappop(h) + 1] -= 1 # pop出来的右端点
            if diff[i] < x: # 处理完了发现没有当前nums没有被available的query handle住，就不能变成zero了，直接返回-1
                return -1
        return len(h)
    
# 3356. Zero Array Transformation II
class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        cnt = [0 for _ in range(n + 1)] # help to optomize; to record query times
        sum_ = k = 0 # s=presum; k=query_index

        # 这里的sum_和cnt，相当于preSum和diff，只不过我的写法一般是直接在diff里叠加了
        # 这里还是使用经典的var帮助计算前缀和，这种算法叫sweep line
        for i in range(n):
            while sum_ + cnt[i] < nums[i]: # 判断当前i的位置的操作是否够。
                k += 1
                if k - 1 >= len(queries): return -1 # 如果遍历完query了就跳过。
                l, r, val = queries[k - 1]
                
                if r < i: continue # 如果当前query的范围太小，肯定不满足，只能跳过。
                cnt[max(l, i)] += val # 这个max也很有灵性，如果当前query太大，我们只更新之后的。
                cnt[r + 1] -= val # 这个r+1表示之后的不要accumulate，就没效果了。
            sum_ += cnt[i]
        return k

# 2337. Move Pieces to Obtain a String 
# 很繁琐的string题，不过思路简单，只需要三步：比较长短/比较字母的相对顺序/比较index的大小是否满足移动方向的关系。

# 56. Merge Intervals
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        output = [intervals[0]]
        for s, e in intervals[1:]:
            if output[-1][1] >= s:
                output[-1][1] = max(output[-1][-1], e)
            else:
                output.append([s, e])
        return output
    
# 1101. The Earliest Moment When Everyone Become Friends
class Solution:
    def earliestAcq(self, logs: List[List[int]], n: int) -> int:
        friends = list(range(n))
        # ✨这个方法可以看是否所有元素都已经被遍历，并且都已经归为一组！
        seen_num = n # 我们每次不会进行多余的连接的，因此针对n个人，我们只需要合并n-1次就可以看作是一个group了
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                friends[rx] = ry
                nonlocal seen_num
                seen_num -= 1
            
        def find(x):
            if friends[x] != x:
                return find(friends[x])
            return friends[x]

        logs.sort()
        for t, x, y in logs:
            union(x, y)
            if seen_num == 1: return t 
        return -1
# 200. Number of Islands
# DFS / BFS-遇到1的就进while queue呗 / UF？ 正常遍历，然后和周围union就行。用的技巧和1101一样，最开始记录所有1，union一次就-1

# 1136. Parallel Courses
# O(Node+Edge)/same space
# DFS集大成者
class Solution:
    def minimumSemesters(self, N: int, relations: List[List[int]]) -> int:
        graph = {i: [] for i in range(1, N + 1)}
        for start_node, end_node in relations:
            graph[start_node].append(end_node)

        visited = {}

        def dfs_check_cycle(node: int) -> bool:
            if node in visited:
                return visited[node]
            else:
                visited[node] = True # True表示是处在当前的recursion路径中；
            for end_node in graph[node]:
                if dfs_check_cycle(end_node): return True
            # mark as visited
            visited[node] = False
            return False

        for node in graph.keys():
            if dfs_check_cycle(node):
                return -1

        # if no cycle, return the longest path
        visited_length = {}

        def dfs_max_path(node: int) -> int:
            # return the longest path (inclusive)
            if node in visited_length:
                return visited_length[node]
            max_length = 1
            for end_node in graph[node]:
                length = dfs_max_path(end_node)
                max_length = max(length+1, max_length)
            # store it
            visited_length[node] = max_length
            return max_length

        return max(dfs_max_path(node)for node in graph.keys())
# 这种写法更好点
class Solution:
    def minimumSemesters(self, N: int, relations: List[List[int]]) -> int:
        graph = {i: [] for i in range(1, N + 1)}
        in_count = {i: 0 for i in range(1, N + 1)}  # or in-degree
        for start_node, end_node in relations:
            graph[start_node].append(end_node)
            in_count[end_node] += 1

        queue = []
        for node in graph:
            if in_count[node] == 0:
                queue.append(node)

        step = 0
        studied_count = 0
        # start learning with BFS
        while queue:
            # start new semester
            step += 1
            next_queue = []
            for node in queue:
                studied_count += 1
                end_nodes = graph[node]
                for end_node in end_nodes:
                    in_count[end_node] -= 1 # 利用了Topological sort的写法
                    # if all prerequisite courses learned
                    if in_count[end_node] == 0:
                        next_queue.append(end_node)
            queue = next_queue
        return step if studied_count == N else -1

# 1494 - Too hard - 思路简单，还是要遍历所有的+Topo，但是要处理只能上k个课atmost，因此要存储状态...，用bitmask存状态...wuw

# 2779. Maximum Beauty of an Array After Applying Operation
# 1-binary search-O(nlogn)/O(n)space主要是sort? - 因为是subsequence, 所以顺序无所谓，所以可以sort
# 主要的算法就是sort之后，遍历，这样根据每个元素我们就有一个范围，往后binary search去找范围就可以了。
# 2-sliding window-一样的复杂度-也是先sort，就比较左右就可以
# 3-sweep line-O(n+maxValue)/O(maxValue)-每一个元素都可以看作一个query区间，我们只用找到哪个区间元素被被覆盖的最多就行了。
class Solution:
    def maximumBeauty(self, nums: list[int], k: int) -> int:
        # If there's only one element, the maximum beauty is 1
        if len(nums) == 1:
            return 1

        max_value = max(nums)  # Find the maximum value in nums
        count = [0] * (max_value + 1)  # Array to track count changes

        # Update the count array for the range [val - k, val + k]
        for num in nums:
            count[max(num - k, 0)] += 1  # Increment at the start of the range
            if num + k + 1 <= max_value:
                count[num + k + 1] -= 1  # Decrement after the range

        max_beauty = 0
        current_sum = 0  # Tracks the running sum of counts

        # Calculate the prefix sum and find the maximum beauty
        for val in count:
            current_sum += val
            max_beauty = max(max_beauty, current_sum)

        return max_beauty
    
# 21. Merge two sorted List - 链表操作基础题
# recursion的话，就直接.next = self.func(node.next)
# iteration的话，ptr.next = l1; l1=l1.next; ptr = ptr.next

# 69 Sqrt(x) 这一题当个笑话看看的了
class Solution:
    def mySqrt(self, x):
        if x < 2:
            return x
        left, right = 0, x # x + 1 也行
        # 题意是：找到第一个值的平方>x，那么这第一个值-1，就是我们实际要找的值。：
        while left < right:
            mid = (left + right) // 2 
            if mid * mid <= x: # 重点：
                left = mid + 1
            else:
                right = mid
        return left - 1

# 2762. Continuous Subarrays
# 常规做法：典型sliding window，然后维护window元素出现的次数，每次依据record去判断window的操作情况
# 优化做法：大体框架是一致的，但是我们会维护cur_max, cur_min，当不满足题意的时候，已当前的结点为right，向左开始遍历找到开始不满足的点，删除
# tip：长度为n的list的所有subarray和为n*(n+1)//2
class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        right = left = 0
        window_len = total = 0

        # Initialize window with first element
        cur_min = cur_max = nums[0]

        for right in range(len(nums)):
            # Update min and max for current window
            cur_min = min(cur_min, nums[right])
            cur_max = max(cur_max, nums[right])

            # If window condition breaks (diff > 2)
            if cur_max - cur_min > 2:
                # Add subarrays from previous valid window
                window_len = right - left # (right-1) - left + 1
                total += window_len * (window_len + 1) // 2 # 求和公式

                # Start new window at current position
                left = right
                cur_min = cur_max = nums[right]

                # Expand left boundary while maintaining condition
                while left > 0 and abs(nums[right] - nums[left - 1]) <= 2:# 这里用的left-1 是为了确保left在有效窗口内且是首位
                    left -= 1
                    cur_min = min(cur_min, nums[left]) # 这里是重新计算有效窗口
                    cur_max = max(cur_max, nums[left]) # same here

                # 为什么需要remove？因为重复计算了，那么哪里重复计算了？比如这次3-5有效，我们减去3-5，但是其实之后遍历3-7是有效的。
                # 那么我们之后会添加3-7，但是注意了，之前我们也添加了1-5；
                # 1-5；3-5；3-7；所以我们要删除，这里没太搞明白数学关系，但是大概是这么个关系
                # Remove overcounted subarrays if left boundary expanded
                if left < right: # 此时的窗口是有效的
                    window_len = right - left
                    total -= window_len * (window_len + 1) // 2

        # Add subarrays from final window
        window_len = right - left + 1
        total += window_len * (window_len + 1) // 2

        return total
    
# 31. Next Permutation - 这一题可以不用看...太抽象了...
# 这一题的核心是理解how permutation works吧...
# 1- 从后往前遍历，找到第一对升序的pair；num[i] > num[i+1]; 此刻num[i+1:]一定是descending的
# 2- 如果不是全程降序，也就是说能找到升序的pair(i>=0), 在num[i+1:]中从后往前尝试找第一个比num[i]大的数字。从而替换。
# 3- 反转num[i+1:] swap掉关键点之后，后面的num[i+1:]我们使其最小，从而满足next permutation
# [4,7,6,5,3,1] -> [5,7,6,4,3,1] -> [5,1,3,4,6,7] 仔细观察4和5
class Solution:
    def nextPermutation(self, nums):
        if len(nums) <= 1:
            return
        
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]: # 想找到相对降序的pair [3,2]
            i -= 1
        
        if i >= 0:  # 这个if表明，存在降序的pair
            j = len(nums) - 1
            while nums[j] <= nums[i]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        
        left, right = i + 1, len(nums) - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left, right = left + 1, right - 1

# 2054. Two Best Non-Overlapping Events # 因为这一道题只有两个events，所以简单很多...
class Solution:
    def maxTwoEvents(self, events: List[List[int]]) -> int:
        times = []
        for s, e, v in events:
            times.append(s, 1, v)
            times.append(e+1, 0, v) # 0 表示 endpoint, 因为s和e是exclusive，所以这里+1
        
        ans, max_value = 0, 0 # woc牛逼
        times.sort()

        for time_value in times:
            # 如果当前是开始时间，我们就去更新ans，怎么更新？
            # max_value是找之前所有event能积累的最大值，max_value只有end time才会更新。
            # 你去看，因为是TWO! non-overlapping events.
            if time_value[1]: 
                ans = max(ans, time_value[2] + max_value)
            else: # 
                max_value = max(max_value, time_value[2])  
        return ans

# 162. Find Peak Element
# 这里的二分不是用来二分取值范围，而是用于优化搜索速度，不用一个个看...
# 首先一般注意, mid是向下取整，因此在与相邻比较的时候，要使用mid VS mid+1

# 13 Roman To Integer
# hard code 需要注意当 i+1<n的时候，要提前比较一下i和i+1

# 560. Subarray Sum Equals K  这个思路很经典
# 这一道题不用sliding window，而是用preSum + twoSum的hashmap思路。

# 146.LRU Cache 
# 不用的OrderDict()的话，要利用doubleLinkedNode，其中要实现add/remove/move_to_head/pop_tail等功能

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

# 207. Course Schedule
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        adj = [[] for _ in range(numCourses)]
        for nex, cur in prerequisites:
            adj[cur].append(nex)
        visited = [0] * numCourses  # 0 - 没有遍历过；1 - 遍历过了； 2 - 当前遍历成环 == instack
        # inStack = [False] * numCourses # 老方法，现在用一个visited store不同的值，代替了。
        for i in range(numCourses):
            if self.dfs(i, adj, visited, inStack):
                return False
            
        return True
    
    # if the result is expected, we want to return False 
    def dfs(self, node, adj, visited, inStack):
        if visited[node] == 2: return True
        if visited[node] == 1: return False # 如果不用优化的话，也可以的。只用判断是不是在instack就可以了。类似backtracking

        visited[node] = 2
        for nex in adj[node]:
            if self.dfs(nex, adj, visited, inStack):
                return True
        visited[node] = 1
        return False
   
# 42. Trapping Rain Water
class Solution:
    def trap(self, height: List[int]) -> int:
        res, stack = 0, []
        for i in range(len(height)):
            # 首先搞明白是递减栈
            while stack and height[stack[-1]] < height[i]:
                # base_index很关键！！
                # i和stack[-1]是左右两个boundary！base
                base_index = stack.pop()
                if not stack: break # 👍 意味着左边没有墙可以阻挡，因此没有办法存水
                h = min(height[stack[-1]], height[i]) - height[base_index]
                # 👍 这个是有意义的，因为base到左边墙如果中间有space存水，那么已经存入了。
                # 如果是两个相同的base，那么第一个pop出来的，计算值是0，没有意义。只有left>base的那一对才会有意义！
                diff = i - stack[-1] - 1 
                res += h * diff

            stack.append(i)

        return res# 42 Trapping Rain Water

# 51 N queen
# 还是backtracking，但是细节复杂。
# 从row开始遍历，然后去找每一个可能的[r][c]坐标，更新set，判断是否可行。
# 如果r == n 证明到底了，可以存在，那么添加入就好了。


# 206. Reverse Linked List
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



# 234. Palindrome Linked List 三种解法都看看
# 额外变量，nums == nums[::-1]
# reverse the second half. O(n)/O(1)
class Solution:

    def isPalindrome(self, head: ListNode) -> bool:
        if head is None:
            return True

        # Find the end of first half and reverse second half.
        first_half_end = self.end_of_first_half(head)
        second_half_start = self.reverse_list(first_half_end.next)

        # Check whether or not there's a palindrome.
        result = True
        first_position = head
        second_position = second_half_start
        while result and second_position is not None:
            if first_position.val != second_position.val:
                result = False
            first_position = first_position.next
            second_position = second_position.next

        # Restore the list and return the result.
        first_half_end.next = self.reverse_list(second_half_start) # reverse_list永远return得是reversed_linked_list_head
        return result    

    def end_of_first_half(self, head: ListNode) -> ListNode:
        fast = head
        slow = head
        while fast.next is not None and fast.next.next is not None:
            fast = fast.next.next
            slow = slow.next
        return slow

    def reverse_list(self, head: ListNode) -> ListNode:
        previous = None
        current = head
        while current is not None:
            next_node = current.next
            current.next = previous
            previous = current
            current = next_node
        return previous
# recursion 比较 O(n)/O(n)
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        self.front_pointer = head

        def recursively_check(current_node=head):
            if current_node is not None:
                if not recursively_check(current_node.next):
                    return False
                if self.front_pointer.val != current_node.val:
                    return False
                self.front_pointer = self.front_pointer.next
            return True

        return recursively_check()


# 周四搞定列表：
# 287. Find the Duplicate Number
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
# O(n)/O(1) 最优做法 Floyd Tortoise and hare algo
class Solution:
    def findDuplicate(self, nums):
        # Find the intersection point of the two runners.
        # 这种方法是取决于当前题目特定的数据结构的。nums 1~n 而我们的index 1~n, 0可以用来init
        # 因为nums多，所以一定有两个index会指向同一个num
        tortoise = hare = nums[0]  # 乌龟/野兔

        # 首先，从开始出发，找到相交的点
        while True:
            tortoise = nums[tortoise]
            hare = nums[nums[hare]]
            if tortoise == hare:
                break
        
        # Find the "entrance" to the cycle.
        # 让慢指针重新出发，下一次相交就是entrance。
        tortoise = nums[0]
        while tortoise != hare:
            tortoise = nums[tortoise]
            hare = nums[hare]
        
        return hare
# 为什么这么找，可以证明是entrance呢？ 证明如下：
# a = 起点～entrance
# b = entrance ～ 1st meet point
# c = cycle length
# 第一次相遇的时候，乌龟走了a+b，兔子走了a+b+k*c k为走了几圈
# 兔子走的比乌龟快一倍，因此a+b+k*c = 2 * （a+b) -> a+b=k*c
# a = k*c - b
# 此时，让一个返回起点，然后用相同的速度，走a步到entrance点；那么另一个在b点，然后走了(k*c-b) -> (k*c-b)+b = k*c 刚好就是在entrance点

# 1760. Minimum Limit of Balls in a Bag O(nlogk)/O(1)
# cannot be solved using heap cuz we don't know how to divide.
# one thing really triggers is why binary search work? 二分解决问题的关键因素：单调性（优化目标-最大/最小化，搜索空间缩小，分配/划分问题）
class Solution:
    def minimumSize(self, nums: List[int], maxOperations: int) -> int:
        def _is_possible(max_balls):
            total_ops = 0
            for n in nums:
                operations = math.ceil(n / max_balls) - 1
                total_ops += operations
                if total_ops > maxOperations:
                    return False
            return True
        
        l, r = 1, max(nums)
        while l < r:
            mid = (l + r) // 2
            if _is_possible(mid):
                r = mid
            else:
                l = mid + 1
        return l
        
    
# 1346. Check If N and Its Double Exist
class Solution:
    def checkIfExist(self, arr: List[int]) -> bool:
        seen = set()
        for num in arr:
            # Check if 2 * num or num / 2 exists in the set
            if 2 * num in seen or (num % 2 == 0 and num // 2 in seen):
                return True
            # Add the current number to the set
            seen.add(num)
        # No valid pair found
        return False
    

# 939. Minimum Area Rectangle
# O(n2)/O(n)
# 思路还是遍历所有的可能性，问题的关键就是如何遍历，如何计算rectangle
# 按照row作main loop，然后i,j - for循环用来找两条col，也就是两个col坐标，所有的col坐标的组合都会被记录，去显示他们有没有x可以对应。
# 然后就是ans = min
# 这类题目之所以记录的目的就是因为不熟悉。
class Solution:
    def minAreaRect(self, points):
        columns = defaultdict(list)
        for x, y in points:
            columns[x].append(y)
        lastx = {}
        ans = float('inf')

        for x in sorted(columns):
            column = columns[x]
            column.sort() 
            for j, y2 in enumerate(column):
                for i in range(j):
                    y1 = column[i]
                    if (y1, y2) in lastx:
                        ans = min(ans, (x - lastx[y1,y2]) * (y2 - y1))
                    lastx[y1, y2] = x
        return ans if ans < float('inf') else 0
    
# 3371  Identify the Largest Outlier in an Array 数学可以不太看
class Solution:
    def getLargestOutlier(self, A: List[int]) -> int:
        total = sum(A) # == outlier + sum(n-2 special elements) + sum(n-2 special elements)
        count = Counter(A)
        res = -inf
        for a in A: # let's take a as sum(n-2 special elements) 
            outlier = total - a - a
            if count[outlier] > (outlier == a): # if outlier == sum, then occurrence of outlier should at least be 2.
                res = max(res, outlier)
        return res
        
    
# 12 Integer to Roman
# 这种方法有点看数学功底呀...😮‍💨 找极限
#     def intToRoman(self, num: int) -> str:
#         digits = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), 
#                   (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), 
#                   (5, "V"), (4, "IV"), (1, "I")]
        
#         roman_digits = []
#         for value, symbol in digits:
#             if num == 0: break
#             count, num = divmod(num, value)
#             roman_digits.append(symbol * count)
#         return "".join(roman_digits)
    
# hard code会比较好！
class Solution:
    def intToRoman(self, num: int) -> str:
        thousands = ["", "M", "MM", "MMM"]
        hundreds = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"]
        tens = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
        ones = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
        return (thousands[num // 1000] + hundreds[num % 1000 // 100] 
               + tens[num % 100 // 10] + ones[num % 10])
    
# 15 - 3SUm
for i, val1 in enumerate(nums):
    if val1 not in dups: # 跳过重复的
        dups.add(val1)
        for j, val2 in enumerate(nums[i+1:]):
            complement = -val1 - val2
            if complement in seen: 
                res.add(tuple(sorted((val1, val2, complement))))
            seen[val2] = i

# 63 也是dp很简单的，有障碍物的unique path

# 85. Maximal Rectangle O(NM)/O(N)
class Solution:
    def leetcode84(self, heights):
        stack = [-1]

        maxarea = 0
        for i in range(len(heights)):

            while stack[-1] != -1 and heights[stack[-1]] >= heights[i]:
                maxarea = max(
                    maxarea, heights[stack.pop()] * (i - stack[-1] - 1)
                )
            stack.append(i)

        while stack[-1] != -1:
            maxarea = max(maxarea, heights[stack.pop()] * (len(heights) - stack[-1] - 1))
        return maxarea

    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix: return 0
        maxarea = 0
        dp = [0] * len(matrix[0])
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                dp[j] = dp[j] + 1 if matrix[i][j] == "1" else 0 # 每一次dp存的都是当前row平地起高楼的height，然后利用LC84去寻找最大值
            # update maxarea with the maximum area from this row's histogram
            maxarea = max(maxarea, self.leetcode84(dp))
        return maxarea


# 3380 # 没意思 暴利解...
# 114. Flatten Binary Tree to Linked List
# Recursion O(n)/O(n)
class Solution:    
    def flatten(self, node: TreeNode) -> None:        
        if not node: return None
        
        if not node.left and not node.right:
            return node
        
        # 这两行代码用于构造recursion stack -- preorder
        leftTail = self.flatten(node.left)
        rightTail = self.flatten(node.right)
        
        #     2
        # 3       4
        # 拿上面这个举例
        if leftTail:
            leftTail.right = node.right     # 3.right = 4
            node.right = node.left          # 2.right = 3
            node.left = None                # 2.left = None
        
        return rightTail if rightTail else leftTail
        
        
# 134 Gas station
# 两个关键判断 1. sum(gas) > sum(cost);  2. curSum < 0的时候直接将starting_position = i+1

# 152 Maximum Product Subarray - 只用维护两个var，一个max_prod; 一个min_prod

# 1143 longest common subsequence 经典DP

# 240. Search a 2D Matrix II O(nlogn)n-边长/O(logn)-tree depth
# 只在横向进行二分搜索。 这是分治的想法
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix: return False
        def search_rectangle(left, right, up,  down):
            if left > right or up > down or target < matrix[up][left] or target > matrix[down][right]:
                return False
            
            mid = (left + right) // 2 
            row = up
            
            while row <= down and matrix[row][mid] <= target:
                if matrix[row][mid] == target: return True
                row += 1
                
            # 答案要么在左下square，要么在右上square
            return search_rectangle(left, mid - 1, row,  down) or search_rectangle(mid + 1,right, up, row - 1)
            
        
        return search_rectangle(0, len(matrix[0])-1, 0, len(matrix)-1)
     
# 1218. Longest Arithmetic Subsequence of Given Difference
# 这一题的写法比longest increasing subsequence更简便
class Solution:
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        dp = {}
        answer = 1
        for a in arr:
            before_a = dp.get(a - difference, 0)
            dp[a] = before_a + 1
            answer = max(answer, dp[a])
            
        return answer

# 875. Koko Eating Bananas 嘿嘿
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        def canEat(n):
            h_needed = 0
            for p in piles:
                h_needed += math.ceil(p/n)
            return h_needed <= h
    
        l, r = 1, max(piles)
        while l < r:
            mid = (l + r) // 2
            if canEat(mid):
                r = mid
            else:
                l = mid + 1
        return l
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


# 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit
# 基础框架是sliding window，但是需要helper vars - 递增queue，递减queue
# 这两个queue分别是从小到大/从大到小排列，其中元素都是当前sliding window里的数字，目的是为了快速得到sliding window里面的极大值/极小值，queue[0]即为极值
# 在更新完这两个deque后，正常while循环判断，从而缩减l窗口

# 23 Merge K sorted List
# 我的印象里有BF/merge 2 sorted list(dummy 和 ptr，dummy留守，ptr穿针引线)
# 但是还有一个利用heap/priority queue -> put((node.val, node)) -> pop出来后
    # val, node = q.get()
    # ptr.next = node
    # ptr = ptr.next
    # node = node.next
    # if node: q.put((node.val, node))


# 540. Single Element in a Sorted Array
# 这题思路不难，难点在于我没有完全理清楚规律！(left-mid)的元素的奇偶性会影响二分范围的选择！
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1
        while l < r:
            m = (l + r) // 2
            halves_are_even = (r - m) % 2 == 0
            if nums[m + 1] == nums[m]:
                if halves_are_even:
                    l = m + 2
                else:
                    r = m - 1
            elif nums[m - 1] == nums[m]:
                if halves_are_even:
                    r = m - 2
                else:
                    l = m + 1
            else:
                return nums[m]
        return nums[l]


# 34. Find First and Last Position of Element in Sorted Array 
# 除了用api，如果想自己写代码用二分来找到target的第一值和最后值的话，写helper func，然后给一个flag parameter决定找first or last
# 我觉得下面的template也可以在找到target之后用while来返回值。
# if nums[mid] == target:
#     if isFirst:
#         if mid == begin or nums[mid - 1] < target:
#             return mid
#         end = mid - 1
#     else:
#         if mid == end or nums[mid + 1] > target:
#             return mid
#         begin = mid + 1

# 1475. Final Prices With a Special Discount in a Shop
# stack里存的是index，但是我们要将input.copy() -> result，这一道stack里存的应该是增序！

# 41. first missing positive
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        # 答案取值范围为[1, n+1] 一共n个数字
        # 1. 先将负数转变为n+1
        # 2. 遍历所有<n+1的数字，将其值作为下标更改下标处的值为负值。
        # 3. 再次遍历，第一个遇到的正数其坐标+1就是没有遇到过的正数。如果所有数都是负数，那么答案是N+1

        # 整体思路总结：原定[0:n]的数字中与index其实是一一对应的的这种关系。所以可以将index看作我们的cheat auxiliary space
        # 对于出现过的值，我们修改其index的值为负数，对我们的操作没有影响，因为我们都是基于0-n范围内的正数进行操作。这样如果我们发现某个index的值为正数，就意味着我们没有在原来的nums里发现这个value，因为遍历index是从小到大，那么第一个不为正数的index对应的值就是first missing postiive
        n = len(nums)
        for i in range(n):
            if nums[i] <= 0:
                nums[i] = n+1
    
        for i in range(n):
            num = abs(nums[i]) # 之所以用abs，是因为在这个循环，会更改之后的数为负数。
            if num <= n: 
                nums[num-1] = -abs(nums[num-1])
        
        for i in range(n):
            if nums[i] > 0:
                return i + 1
        return n + 1


# 45. Jump Game II
# 这一题已经确定了，我们肯定能到达end，因此不用担心这种edge case。
class Solution:
    def jump(self, nums: List[int]) -> int:
        res, n = 0, len(nums)
        cur_end = cur_far = 0
        # cur_far是理论目前能到的最远距离。cur_end是随着遍历动态更新的最远的end point。
        for i in range(n-1):
            cur_far = max(cur_far, i + nums[i])

            # 此处更新逻辑很关键
            # 刚开始cur_end是0，res+1，意味着从开始位置出发，我们至少需要跳一步，假设第一步跳到了index = x
            # 当i达到cur_end(x)的位置的时候，此刻cur_far已经积累了(0~x)能达到最远y的所有可能性。也就是说从0~x任意位置起跳一次，都可以达到<=y的可能性
            if i == cur_end:
                res += 1
                cur_end = cur_far

        return res
    

# 71. Simplify Path 简单，如果decode string会了，这题很easy

# 75. Sort Colors 三指针 = 双指针 + lazy load
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        red, white, blue = 0, 0, len(nums)-1
        
        while white<=blue:
            if nums[white] == 0:
                nums[red], nums[white] = nums[white], nums[red]
                red += 1
                white += 1
            elif nums[white] == 1:
                white += 1
            else:
                nums[blue], nums[white] = nums[white], nums[blue]
                blue -= 1
                
# 76. Minimum Window Substring 其实不难，经典sliding window + auxiliary variable (words_needs, words_cnt, words_have)

# 105. Construct Binary Tree from Preorder and Inorder Traversal 
# nonlocal pre_index,  if l < in_index: node.left = construct(left, mid—1)

 
# 2940. Find Building Where Alice and Bob Can Meet
class Solution:
    def leftmostBuildingQueries(self, heights, queries):
        mono_stack = []
        result = [-1 for _ in range(len(queries))]
        new_queries = [[] for _ in range(len(heights))]
        for i in range(len(queries)):
            a = queries[i][0]
            b = queries[i][1]
            
            # 这里用来确保query的pair一定是[small_index, large_index],方便处理。
            if a > b:
                a, b = b, a
            
            if heights[b] > heights[a] or a == b: # 如果large_index的height大/ small_index和large_index一样，那么答案直接就是在large_index了。
                result[i] = b
            else:  # large_index的height要小于small's的，因此我们要向右找到最小index的height比这两个原始的height都要高。
                new_queries[b].append((heights[a], i)) # 新query的key是large_index，然后存的值是[大height和答案的index]

        for i in range(len(heights) - 1, -1, -1): # 倒序
            mono_stack_size = len(mono_stack)
            for h, query_index in new_queries[i]: # 因为我们构造new_query的逻辑，这里的h肯定是比current_i的height要大的
                position = self.search(h, mono_stack) # 我们想要search的就是，是否存在比h还要大的值，在stack中。
                if position < mono_stack_size and position >= 0: # 表明找到了。
                    result[query_index] = mono_stack[position][1]
            
            # 这个操作步骤是为了搜索优化，而且我们不用担心pop出了右边小值，因为当前current_height更大，如果前面有query，那一定优先选择current_height的。
            while mono_stack and mono_stack[-1][0] <= heights[i]: # stack.top()/pop() 最小值 -> min_stack 从左往右是从大到小。
                mono_stack.pop()
            mono_stack.append((heights[i], i))
        return result

    def search(self, height, mono_stack):
        left = 0
        right = len(mono_stack) - 1
        ans = -1
        while left <= right:
            mid = (left + right) // 2
            if mono_stack[mid][0] > height:
                ans = max(ans, mid)
                left = mid + 1
            else:
                right = mid - 1
        return ans

# 1071. Greatest Common Divisor of Strings
# 也可以用bf，下面是数学推导
class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        if str1+str2 != str2+str1:
            return ""
        max_len = gcd(len(str1), len(str2))
        return str1[:max_len]        

# 120. Triangle - DP 这个不适合二叉树的2n+1寻找每一层的index对应关系。不用改变原有的数据结构，画个图找state就可以了

# 128. Longest Consecutive Sequence - 比较简单，没有顺序，因此直接用set，用于之后的查找。
# 两个关键点 
# if n-1 not in nums -> 确保我们只从一个sequence的开头去遍历
# while cur+1 in nums

# 643. Maximum Average Subarray I 
# 因为subarray的长度固定，因此我们可以使用prefix_sum的技巧来简化计算过程。
# 也可用双指针

# 131. Palindrome Partitioning
# 🤯：虽然也是bt的路子，但是你要注意什么时候进BT
# for i in range(len(s)+1): if s[:i] == s[:i][::-1] # 直接判断是回文的话再进！不用一个个字母判断。

# 167. Two Sum II - Input Array Is Sorted 
# 这题简单，双指针向中间靠拢就可以。 类似container with most water那一道题

# 2981. Find Longest Special Substring That Occurs Thrice I
# 如果用BF：就是用cnt找到所有的这种类型，去比较谁的len长
class Solution:
    def maximumLength(self, s: str) -> int:
        res = -1
        for c in {*s}: # *为结构符号，结构为单个char，{}放入一个集合汇总，类似set()
            # findall(c+'+,s) 会在s中找到所有由c组成的subarry
            # nlargest返回最长的三个长度，如果没有，则返回现有的
            # 最后的两个0,0是为了防止出界
            l = [*nlargest(3,map(len,findall(c+'+',s))),0,0] 
            # l[0]-2 表明example1: 三次的substring都可以从这一个string中拿出来
            # l1-(l[0]==l[1]): 如果l[0]==l[1]，意味着可以l[0]/l[1] - 1就可以拿到最大答案；如果不相等，那么l1一定比l2小, 那么l2一定是被l1涵盖的。
            # l[2]可以直接参与计算，因为一定被l1,l2涵盖。
            res = max(res, l[0]-2, l[1]-(l[0]==l[1]), l[2]) 

        return res or -1
    

# 721. Accounts Merge
# 总体来说用UF，这题的UF我觉得也挺好的。
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        parents = list(range(len(accounts)))
        def find(x):
            return parents[x] if x == parents[x] else find(parents[x])
        def union(x, y):
            xr,yr  = find(x),find(y)
            if xr != yr:
                parents[xr] = yr
            # parents[find(x)] = find(y)
        
        
        
        ownership = {} # {email: index} 用于合并
        for i, (_, *emails) in enumerate(accounts): # i是index ; _是名字; emails怎么取很有趣。
            for e in emails:
                if e in ownership: # 如果email有，合并index
                    union(i, ownership[e])
                ownership[e] = i 

        ans = collections.defaultdict(list) # {index: [emails]}
        for e, o in ownership.items():
            ans[find(o)].append(e)
            
        # 转化成答案[[name, *emails]]
        return [[accounts[i][0]] + sorted(e) for i,e in ans.items()]



# 727. Minimum Window Subsequence
# 这一题的思路很好，比用dp简单多了。
# 首先尝试从当前start_index查找满足subsequence的end_index，有了end_index，我们反过来再找一次，看看能不能优化start_index
class Solution:
    def minWindow(self, S, T):
        
        # Find - Get ending point of subsequence starting after S[s]
        def find_subseq(s):
            t = 0
            while s < len(S):
                if S[s] == T[t]:
                    t += 1
                    if t == len(T):
                        break
                s += 1
            
            return s if t == len(T) else None       # Ensure last character of T was found before loop ended
        
        # Improve - Get best starting point of subsequence ending at S[s]
        def improve_subseq(s):
            t = len(T) - 1
            while t >= 0:
                if S[s] == T[t]:
                    t -= 1
                s -= 1
            
            return s+1
        
        s, min_len, min_window = 0, float('inf'), ''
        
        while s < len(S):
            if S[s] == T[0]:                    # 小优化，只有满足T的时候，才会执行下列操作，尝试找所有的subsequnce
                end = find_subseq(s)            # Find end-point of subsequence
                if end is None:
                    break
                    
                start = improve_subseq(end)     # Improve start-point of subsequence 
                if end-start+1 < min_len:       # Track min length
                    min_len = end-start+1
                    min_window = S[start:end+1]
                
                s = start+1                     # Start next subsequence search
            else: 
                s += 1

        return min_window
# 216. Combination Sum III 经典回溯 -> 只是有些优化的点可能会被考察。比如取值范围min(target+1, 10)

# 226. Invert Binary Tree BFS/DFS都能做，就是让node.left, node.right = func(node.right), func(node.left)
# 739. Daily temperature - Stack的经典应用[index, temperature]入stack

# 1631. Path With Minimum Effort 
# 如果用回溯，太复杂了，3^n，会超时
# 有多种解法，可以用dijkstra -> 用[difference, x, y]入heap，还需要用到visited帮助我们查看是不会已经便利过的。复杂度O(nlogn)/O(n) 这个dijkstra找路径的也可以思考一下。
# 这一题也可以用二分的方法 -> 二分用来取difference，helper function来判断能不能抵达。
class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        row = len(heights)
        col = len(heights[0])

        def canReachDestinaton(x, y, mid):
            if x == row-1 and y == col-1:
                return True
            visited[x][y] = True
            for dx, dy in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
                adjacent_x = x + dx
                adjacent_y = y + dy
                if 0 <= adjacent_x < row and 0 <= adjacent_y < col and not visited[
                        adjacent_x][adjacent_y]:
                    current_difference = abs(
                        heights[adjacent_x][adjacent_y]-heights[x][y])
                    if current_difference <= mid:
                        visited[adjacent_x][adjacent_y] = True
                        if canReachDestinaton(adjacent_x, adjacent_y, mid):
                            return True
            return False
        left = 0
        right = 10000000
        while left < right:
            mid = (left + right)//2
            visited = [[False]*col for _ in range(row)]
            if canReachDestinaton(0, 0, mid):
                right = mid
            else:
                left = mid + 1
        return left



# 231. Power of Two return n > 0 and str(bin(n)).count('1') == 1 # 位运算无聊...


# Stack-LIFO-[-1]是top，如果top放的小值，就是min_stack, 这一点和heap很像。
# 769. Max Chunks To Make Sorted - 好几种solution，都是O(n)/O(n)，除了最后一种space将变为O(1)
# 1. prefix_max, suffix_min auxiliary vars -> if i == 0 or suffix_min[i] > prefix_max[i - 1] # # 关键：a new chunk can be created when 后面的chunk的最小值 < 前面chunk的最大值
# 2.
# 3. stack。 stack里存的是每个chunk的最大值
#     if not stack or arr[i] > stack[-1]:           # Case 1: Current element is larger, starts a new chunk
#         stack.append(arr[i])
#     else:                                         # Case 2: Merge chunks
#         max_element = stack[-1]
#         while stack and arr[i] < stack[-1]:
#             stack.pop()
#         stack.append(max_element)
# 4. 类似stack。这个算法太爆炸了...因为input的值是[0...n-1]
# max_element = max(max_element, arr[i])
# if max_element == i:
#     chunks += 1


# 1752. Check if Array Is Sorted and Rotated
class Solution:
    def check(self, nums: List[int]) -> bool:
        count = 0
        n = len(nums)
        for i in range(n-1):
            if nums[i] > nums[i+1]: # count unsorted pair
                count += 1
        if nums[0] < nums[-1]: # 意味着没有rotate，那就不应该有unsorted pair
            count += 1
        return count <= 1

# 351. Android Unlock Patterns - 主体思路很常规的backtrack，只给了取值范围，那么你for循环，然后每一个元素都要进去backtrack。


# 354. Russian Doll Envelopes
# 这一题思路很奇妙呀！
# 在排序完之后，我们只用找最长的increasing subsequence就好了！
# 这里的二分是优化写法！O(nlogn)/O(n)
class Solution:
    def maxEnvelopes(self, es: List[List[int]]) -> int:
        es.sort(key=lambda x: (x[0], -x[1]))

        def lis(nums):
            dp = []
            for i in range(len(nums)):
                idx = bisect_left(dp, num[i])
                if idx == len(nums):
                    dp.append(nums[i])
                else:
                    dp[idx] = i
            return len(dp)

        return lis([x[1] for x in es])

# ENG表达
# The extremum of a quadratic equation in two variables occurs at 
# derivative
# 360. Sort Transformed Array 
# 如果强调O(n)的复杂度，那么只有用双指针了
class Solution:
    def sortTransformedArray(self, nums: List[int], a: int, b: int, c: int) -> List[int]:
        answer = []
        left, right = 0, len(nums) - 1
        nums = list(map(lambda x: a*x*x+b*x+c, nums))
    
        # a<0，开头向下，每次while我们都是为了在left,right找到最小值，离极点最远的那个
        # 为什么不能找最大值？因为下一个会在left+1, right中寻找，那么Left+1有可能是比left大的，因为数学图像。
        # 如果找最小值，我们是不用担心图像的。
        if a < 0:
            while left <= right:
                if nums[left] < nums[right]:
                    answer.append(nums[left])
                    left += 1
                else:
                    answer.append(nums[right])
                    right -= 1
        else:
            while left <= right:
                if nums[left] < nums[right]:
                    answer.append(nums[left])
                    left += 1
                else:
                    answer.append(nums[right])
                    right -= 1
            
            answer.reverse()
        return answer


# 843. Guess the Word
# 这题就是在words找到特定的word，要求是尽可能快地找到。
# 那么怎么找就是解题的关键，每一次我门去找most_overlap_word，基于信息论，选择尽可能包含多的信息
# match 这个overlap_word 得到match了几次，然后下一次中循环中，把没有match上的word给筛选出去！
class Solution:
    def findSecretWord(self, wordlist, master):
		
        # count the number of matching characters
        def pair_matches(a, b):         
            return sum(c1 == c2 for c1, c2 in zip(a, b))

        def most_overlap_word():
            counts = [[0 for _ in range(26)] for _ in range(6)]     # counts[i][j] is nb of words with char j at index i
            for word in candidates:
                for i, c in enumerate(word):
                    counts[i][ord(c) - ord("a")] += 1

            best_score = 0
            for word in candidates:
                score = 0
                for i, c in enumerate(word):
                    score += counts[i][ord(c) - ord("a")]           # all words with same chars in same positions
                if score > best_score:
                    best_score = score
                    best_word = word

            return best_word

        candidates = wordlist[:]        # all remaining candidates, initially all words
        while candidates:

            s = most_overlap_word()     # guess the word that overlaps with most others
            matches = master.guess(s)

            if matches == 6:
                return

            candidates = [w for w in candidates if pair_matches(s, w) == matches]   # filter words with same matches

# 1792. Maximum Average Pass Ratio # 最精妙的是是入stach的元素，这里的算的是gain
class Solution:
    def maxAverageRatio(self, classes: List[List[int]], extraStudents: int) -> float:
        def gain(x, y): 
            return (x+1)/(y+1) - x/y
        heap = [ [-gain(x,y), x, y] for x, y in classes]
        while extraStudents:
            _, x, y = heapq.heappop(heap)
            heapq.heappush(heap, [-gain(x+1,y+1), x+1, y+1])
            extraStudents -= 1
        return sum([x/y for _, x, y in heap]) / len(heap)

# 410. Split Array Largest Sum
class Solution:
    def splitArray(self, nums: List[int], m: int) -> int:
        def check_required(max_sum_allowed: int) -> int:
            current_sum = 0
            splits_required = 0
            
            for element in nums:
                if current_sum + element <= max_sum_allowed:
                    current_sum += element
                else:
                    current_sum = element
                    splits_required += 1

            return splits_required + 1
        
        l, r = max(nums), sum(nums)
        while l < r:
            mid = (l + r) // 2
            if check_required(mid) <= m:
                r = mid     
            else:
                l = mid + 1
        
        return l
    

# 1825. Finding MK Average - 这一题思路不难，是个锻炼sortedList很好的题目。
from sortedcontainers import SortedList

class MKAverage:
    def __init__(self, m: int, k: int):
        self.m, self.k = m, k
        self.deque = collections.deque()            # 因为题意要求FIFO，所以这个是用来记录pop的时候应该更新哪个num
        self.sl = SortedList()
        self.total = self.first_k = self.last_k = 0 # 变量保存prefix/suffix_k_sum

    def addElement(self, num: int) -> None:
        self.total += num
        self.deque.append(num)
        
        index = self.sl.bisect_left(num)
        if index < self.k: # 如果插入的num在前K个，我们需要：更新first_k(要判断当前有多少个num了)
            self.first_k += num
            if len(self.sl) >= self.k:
                self.first_k -= self.sl[self.k - 1]

        if index >= len(self.sl) + 1 - self.k: # 同理，不过是处理后k个
            self.last_k += num
            if len(self.sl) >= self.k:
                self.last_k -= self.sl[-self.k]

        self.sl.add(num)

        if len(self.deque) > self.m:            # 如果目前已经有了M个num
            num = self.deque.popleft()          # 我们想要pop最先进来的。
            self.total -= num                   # 更新total，然后去找num是否落在前/后k个，只用更新first/last_k就好，不复杂
            index = self.sl.index(num)
            if index < self.k:                  
                self.first_k -= num
                self.first_k += self.sl[self.k]
            elif index >= len(self.sl) - self.k:
                self.last_k -= num
                self.last_k += self.sl[-self.k - 1]
            self.sl.remove(num)

    def calculateMKAverage(self) -> int:
        if len(self.sl) < self.m:
            return -1
        return (self.total - self.first_k - self.last_k) // (self.m - 2 * self.k)


# 900. RLE Iterator
# 这一题的问题在于会超过Memory Limit.
class RLEIterator:

    def __init__(self, A):
        self.A = A
        self.i = 0  # index
        self.q = 0  # quantity of A[i] that is exhausted, for each number diff

    def next(self, n):
        while self.i < len(self.A): 
            if self.q + n > self.A[self.i]:     # if can exhausted current, go to next 
                n -= self.A[self.i] - self.q    # update n
                self.q = 0                      # init
                self.i += 2                     # jump to next num
            else:
                self.q += n
                return self.A[self.i+1]
        return -1


# 901. Online Stock Span # 这一题针对index的处理很巧妙。
# 因为我们要记录当前index和上一个较大值index之间的关系，原始的方法是记录两个值的index，然后求差，如果这一题使用这个方法的话，我们需要维护一个global var
# 但是这里直接存的不是index，而是类似index_prefix_sum，哪怕某个元素被pop出去了，那么append进来的ans一定会记录它的值，详细看代码吧。
class StockSpanner:
    def __init__(self):
        self.stack = []
        
    def next(self, price: int) -> int:
        ans = 1
        while self.stack and self.stack[-1][0] <= price:
            ans += self.stack.pop()[1]
        self.stack.append([price, ans])
        return ans


# 1838. Frequency of the Most Frequent Element
# 两种方法：
# binary search - O(nlogn)/O(n)  
# sliding window - O(nlogn)/O(n) 最好用这个，这题的二分不明确

# 1. Given an index i, if we treat nums[i] as target, we are concerned with how many elements on the left we can take
# 2. 如果我们sum起来，那么target_sum - original_sum就是需要操作的次数。
# 3. 这道题相当于把平常的二分+check再次嵌套而已...
class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        def check(i):
            target = nums[i]
            left = 0
            right = i
            best = i # best是满足k个操作的leftmost的index
            
            while left <= right:
                mid = (left + right) // 2
                count = i - mid + 1
                final_sum = count * target
                original_sum = prefix[i] - prefix[mid] + nums[mid]
                operations_required = final_sum - original_sum

                if operations_required > k:
                    left = mid + 1
                else:
                    best = mid
                    right = mid - 1
                    
            return i - best + 1
        
        nums.sort()
        prefix = [nums[0]]
        
        for i in range(1, len(nums)):
            prefix.append(nums[i] + prefix[-1])
        
        ans = 0
        for i in range(len(nums)):
            ans = max(ans, check(i))
            
        return ans

# sliding window的思路：我们维护一个window，长度就是max freq，大前提一定是sort的
# 如果可以满足at most k ops，那么我们继续便利，相当于扩大window，如果不能满足就lazy narrow left boundary
class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        nums.sort()
        left = 0
        curr = 0 # sum(window)
        
        for right in range(len(nums)):
            target = nums[right]
            curr += target
            
            if (right - left + 1) * target - curr > k:
                curr -= nums[left]
                left += 1

        return len(nums) - left

# 438 就是典型的用两个counter比较的sliding window算法
# 443 String Compression - fast / slow指针 for c in str(cnt) 用于多位数字情况下的slow指针操作，很有意思

# 450. Delete Node in a BST O(logN)/O(H) - 用来找到key-node为logn，然后H是stack的深度即tree——depth
# 难点：要把left_node —> 连接到右侧子树的最左侧node上
# 这一题和114. Flatten Binary Tree to Linked List异曲同工。
class Solution:
    def deleteNode(self, root: Optional[TreeNode], k: int) -> Optional[TreeNode]:
        def dfs(node):
            if not node: return 
            if k > node.val: node.right = dfs(node.right) # 因为我返回的都是node，因此一定要在recursion中把每一层连接起来。
            elif k < node.val: node.left = dfs(node.left)
            else:
            
                if not node.left and not node.right: return None
                if not node.left: return node.right
                if not node.right: return node.left

                left = node.left
                right_as_root = node.right

                # 下面就是把left放在right的leftmost的leaf上！
                ptr = right_as_root
                while ptr.left:
                    ptr = ptr.left
                ptr.left  = left
                return right_as_root
                
            return node

        return dfs(root)

# 930. Binary Subarrays With Sum -- 这个技巧不错。
class Solution:
    def numSubarraysWithSum(self, nums: List[int], target: int) -> int:
        c = collections.Counter({0: 1})
        psum = res = 0 # psum是前缀和

        # 模版：subarray sum可以用prefix_sum解决 # 就像是二分/dp可以用于subsequence一样！
        for n in nums:
            psum += n
            # 这一句的意思就是说，psum - target是之前我们存入counter的某个preSum，是出现过的！
            # == 两个presum之差就是我们target == 这两个presum差出来的subarry就是我们要找的subarray
            res += c[psum - target] 
            c[psum] += 1
        return res

# 950. Reveal Cards In Increasing Order
# simulation - O(nlogn)/O(n)
class Solution:
    def deckRevealedIncreasing(self, deck: List[int]) -> List[int]:
        N = len(deck)
        queue = deque()

        for i in range(N):
            queue.append(i)
        
        deck.sort() #sort过后是我们想要reveal的order
        result = [0] * N

        # queue最开始模拟的是原始/我们想要的order
        # 按照操作模拟，每一次for循环就是用于reveal card的，然后我们就会知道reveal的这张卡在原来的queue中是哪个index，当前的card就应该放在那个index上
        for card in deck:
            
            # Reveal Card
            result[queue.popleft()] = card 

            # Move next card to bottom
            if queue:
                queue.append(queue.popleft())
                
        return result


# 454. 4Sum II - 这也行....
class Solution:
    def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
        count = 0
        m = collections.defaultdict(int)
        for a in A:
            for b in B:
                m[a+b] += 1
        
        for c in C:
            for d in D:
                count += m[-(c+d)]
        return count

# 1. longest increasing subsequence
# dp[i] = max(dp[j] + 1) for all j < i and nums[j] < nums[i]
# 2. longest common subsequence
# dp[i][j] = dp[i-1][j-1] + 1 if s1[i-1] == s2[j-1]
# dp[i][j] = max(dp[i-1][j], dp[i][j-1]) if s1[i-1] != s2[j-1]
