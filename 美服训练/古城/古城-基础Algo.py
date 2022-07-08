"""
目录:
    1- 扫描线
    2- BFS
    3- DFS
    4- Binary Search
    5- Divide and Conquer
    6- Single Stack
    7- Single Queue
    8- Sliding Window
    9- Sort
    10- Prefix Sum
    ----------------- DS ----------------
    11- Trie
    12- Union Find
    13- Heap
    14- Stack/Queue
    15- LinkedList_1
    16- LinkedList_2
"""


########################  扫描线  ########################
# 252. Meeting Rooms
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0]:
                return False
        return True

# 253. Meeting Rooms II
# 这一题的领悟：解法是priority queue/Heap，这些数据结构拥有一些特性和特性的API，所以被这么叫，实质上和基础数据组合类型没差别。
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0
        rooms = []
        intervals.sort(key = lambda x:x[0])
        heapq.heappush(rooms, intervals[0][1])
        
        for i in intervals[1:]:
            if rooms[0] <= i[0]:
                heapq.heappop(rooms)
            heapq.heappush(rooms, i[1])
        return len(rooms)
# 这道题很厉害耶。通过对heap里面数据的操纵很好地显现了priority queue的思想

# 56. Merge Intervals
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        merged = []
        
        for interval in intervals:
            if not merged or merged[-1][1] < interval[0]:       # Key operation
                merged.append(interval)
            else:
                merged[-1][1] = max(merged[-1][1], interval[1]) # Key operation
        
        return merged
# 这道题也好好呀woc。
# 独特的点在于，他不将眼光限制于原来的interval，而是直接focus在new出来的merged上，
# 两个判断非常关键：1.如果start > 已有的任何end，添加 2. 如果start time小的话，那么就看是expand我们的end还是保持原有的end


# 57. Insert Interval
class Solution:
    def insert(self, intervals: 'List[Interval]', newInterval: 'Interval') -> 'List[Interval]':
        # 初始化数据
        new_start, new_end = newInterval
        idx, n = 0, len(intervals)
        output = []
        
        # 第一种情况：先把newInterval前面的管不着的interval放入res
        while idx < n and new_start > intervals[idx][0]:
            output.append(intervals[idx])
            idx += 1
            
        # 第二种情形：我们要插入newInterval了，那么start有可能在/不在已经有的cover里面
        if not output or output[-1][1] < new_start:
            output.append(newInterval)
        else:
            output[-1][1] = max(output[-1][1], new_end)
        
        # 第三种情况：插入newInterval后，把之后的intervals插入到我们的res中。
        while idx < n:
            interval = intervals[idx]
            start, end = interval
            idx += 1
            if output[-1][1] < start:
                output.append(interval)
            else:
                output[-1][1] = max(output[-1][1], end)
        return output
# 题意已经明确说明，interval排序了，而且没有重叠。
# 这里需要注意几个地方
    # 1. index的位置，要看处理逻辑，否则容易导致index out of range
    # 2. index抽出来，也有利于转换逻辑的边界判断，比如index = j在第一种情况判断后可以继续拿到第二种情况进行判断
    # 3. 这题难点在于如果把题目理解成这几种情形。其他不难


# 1288. Remove Covered Intervals
# 自己写的
class Solution:
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key = lambda x: x[1], reverse=True)
        intervals.sort(key = lambda x: x[0])
        # intervals.sort(key = lambda x: (x[0], -x[1]))
        # 🌟 lambda新用法！
        right = intervals[0][1]
        count = 0
        
        for interval in intervals[1:]:
            if interval[1] <= right:  
                count += 1
            else:
                right = interval[1]
            
        return len(intervals) - count

# 1272. Remove Interval
class Solution:
    def removeInterval(self, intervals: List[List[int]], toBeRemoved: List[int]) -> List[List[int]]:
        output = []
        left, right = toBeRemoved
        
        # 首先把start，end, left, right抽出来，其实是有助于直接append的，本题方便些。
        # 其次对情况的判断，很重要。
        for start, end in intervals:
            # 这个if的情况就是 interval不再我们的remove区间。直接添加
            if end <= left or start >= right:
                output.append([start,end])
            # 这个如果细拆的话，能拆出4种情况，但是代码太长了。
            # 最终我们考虑的点不是哪4种情况，而是把4种情况继续抽象，看remove前面是否有需要keep的和remove后面是否有需要keep的，在remove中间overlap的我们就不管了。
            else:
                if start < left:
                    output.append([start, left])
                if end > right:
                    output.append([right, end])
        return output
# 这道题思路自己写出来了，但是条件判断做的不好。具体看题意里面。
# 复杂度On 


# 435. Non-overlapping Intervals
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort()
        count = 0
        right = float('-inf')
        
        for start, end in intervals:
            if start >= right:
                right = end
            else:
                count += 1
                # 这个min的判断就是直接把几种情况聚合了，只看我们的ending point
                right = min(end, right)
        return count
# 困难的还是对于情况的判断

# 1229. Meeting Scheduler
class Solution:
    def minAvailableDuration(self, slots1: List[List[int]], slots2: List[List[int]], duration: int) -> List[int]:
        slots1.sort()
        slots2.sort()
        p1 = p2 = 0
        # 涉及到两个slots同时判断，双指针
        while p1<len(slots1) and p2< len(slots2):
            # 第一步先求出intersect
            intersect_left = max(slots1[p1][0], slots2[p2][0])
            intersect_right = min(slots1[p1][1], slots2[p2][1])
            # 判断是否满足，满足直接return
            if intersect_right - intersect_left >= duration:
                return [intersect_left, intersect_left + duration]
            # 没有intersect/当前intersect不满足，我们就继续下一位，因为涉及到两个list，所以要一步步移兼顾所有情况，不能同时移动两个pointer
            if slots1[p1][1] < slots2[p2][1]:
                p1 += 1
            else: 
                p2 += 1
        return []

# 986. Interval List Intersections
# 跟上一题1229好像...
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        output = []
        p1 = p2 = 0
        
        while p1 < len(firstList) and p2 < len(secondList):
            intersect_left = max(firstList[p1][0], secondList[p2][0])
            intersect_right = min(firstList[p1][1], secondList[p2][1])
            
            if intersect_right >= intersect_left:
                output.append([intersect_left, intersect_right])
                
            if firstList[p1][1] < secondList[p2][1]:
                p1 += 1
            else: 
                p2 += 1
        return output
    

# 759. Employee Free Time
class Solution:
    def employeeFreeTime(self, schedule: '[[Interval]]') -> '[Interval]':

        # 这种情况下就只能用sorted，而不能用sort咯～
        # 先按照每个人的
        ints = sorted([i for s in schedule for i in s], key=lambda x: x.start)
        res, pre = [], ints[0]
        for i in ints[1:]:
            if i.start <= pre.end and i.end > pre.end:
                pre.end = i.end
            elif i.start > pre.end:
                res.append(Interval(pre.end, i.start))
                pre = i
        return res
# 这道题有自己的数据结构和API
# 总体思路不难

# 218. The Skyline Problem
# 这道题只是看懂了，并没有写出来。
class Solution:
    def getSkyline(self, buildings: 'List[List[int]]') -> 'List[List[int]]':
        """
        利用分治的方法，把buildings拆开，然后一一合并，拆开的逻辑在这个方法中，合并的逻辑在merge方法中
        """
        n = len(buildings)
        # The base cases
        if n == 0:
            return []
        if n == 1:
            x_start, x_end, y = buildings[0]
            return [[x_start, y], [x_end, 0]]

        # If there is more than one building,
        # recursively divide the input into two subproblems.
        left_skyline = self.getSkyline(buildings[: n // 2])
        right_skyline = self.getSkyline(buildings[n // 2 :])

        # Merge the results of subproblem together.
        return self.merge_skylines(left_skyline, right_skyline)

    def merge_skylines(self, left, right):
        "首先两个helper，update和append，分别是更新已有的（overlap）和新增"
        # helper function要放在前面，不能放在后面，否则没办法及时读到
        def update_output(x, y):
            # if skyline change is not vertical -
            # add the new point
            if not output or output[-1][0] != x:
                output.append([x, y])
            # if skyline change is vertical -
            # update the last point
            else:
                output[-1][1] = y

        def append_skyline(p, lst, n, y, curr_y):
            while p < n:
                x, y = lst[p]
                p += 1
                if curr_y != y:
                    update_output(x, y)
                    curr_y = y
                    
        # 正片开始
        n_l, n_r = len(left), len(right)
        p_l = p_r = 0
        curr_y  = left_y = right_y = 0
        output = []

        # while we're in the region where both skylines are present
        # 第一次进来的left/right为单个building的[x1,x2,y]
        while p_l < n_l and p_r < n_r:
            point_l, point_r = left[p_l], right[p_r]
            # 第一步操作：找到最小的左侧坐标，赋值给x
            # 第二步操作：更新各自的y
            # 第三步操作：继续往下探索
            if point_l[0] < point_r[0]:
                x, left_y = point_l
                p_l += 1
            else:
                x, right_y = point_r
                p_r += 1
            
            # 找到该坐标x下的最大值/纵坐标
            max_y = max(left_y, right_y)
            
            # if there is a skyline change， 
            # skyline change会有两种情况，一种是碰到新的点了，那么直接添加；一种是重合只在y上，那么更新[-1]的y就好了。
            if curr_y != max_y:
                update_output(x, max_y)
                curr_y = max_y
                
                
        # 前面把重合的部分弄完了，现在只剩下单侧了，进append页面，所以简单更新就成。
        # there is only left skyline 
        append_skyline(p_l, left, n_l, left_y, curr_y)

        # there is only right skyline
        append_skyline(p_r, right, n_r, right_y, curr_y)

        return output

"""
扫描线类型总结：
1. 什么是该类型的题呢？ 涉及到区间的overlap就是。
2. 该类型的技巧？不去关注每个点，而是关注区间的start与end
3. 承接2，此类题的难点在于start与end的情况判断，其他不难，都是些技巧活了，是否用for/是否取值/是否index抽出来之类的
4. 一般来说都是要先排序的。
"""

########################  BFS  ########################
# 102. Binary Tree Level Order Traversal
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        res = []
        queue = [root]
        while queue:
            length = len(queue)
            level = []
            for i in range(length):
                node = queue.pop(0)
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                
            res.append(level)
        return res

# 自己想歪了
"""
需要独立处理BFS当中每一层时, 有两种做法: 
    1. 第一种直接把当前level作为参数传进queue中
    2. 在while中利用for loop把当前level的queue直接清空(本题的做法), 直接处理当前level, 很便捷。
"""
# 111. Minimum Depth of Binary Tree
# while + for好像是BFS的标配呀。
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        queue = [root]
        res = 0
        while queue:
            n = len(queue)
            res += 1
            for i in range(n):
                node = queue.pop(0)
                if not node.left and not node.right:
                    return res
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
        
# 752. Open the Lock
# 这种问题可以抽象成最短路径，思路不错，可以借鉴学习

class Solution(object):
    def openLock(self, deadends, target):
        def neighbors(node):
            for i in xrange(4):
                # x就是node里的四位数的我们要处理的一位
                x = int(node[i])
                # d要么是1，要么是-1，这里不是range
                for d in (-1, 1):
                    y = (x + d) % 10
                    
                    # 每次call neighbor()的时候，返回一个含有8个对象的generator对象，每一层遍历8个对象。
                    # 返回的是生成器对象，每一次
                    # str[y]是原来的node返回
                    yield node[:i] + str(y) + node[i+1:]        # 切片都是前闭后开，前面包含，后面不包含。
                    

        dead = set(deadends)
        queue = collections.deque([('0000', 0)])
        seen = {'0000'}     # 这里的seen是用来作优化用的
        
        while queue:
            node, depth = queue.popleft()
            if node == target: return depth
            if node in dead: continue
            
            # generator是可以迭代的对象
            for nei in neighbors(node):
                if nei not in seen:
                    seen.add(nei)
                    queue.append((nei, depth+1))
        return -1

# 207. Course Schedule
# 这题已经刷过，但是还要再刷。
# 总体思路是维护两张表：一张是某个课程都是哪些课程的先修课；一张是某门课有几节先修课。看似两张表没有联系，但是我们每次处理数据都是处理一门课和它的先修课。
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # 两张course表
        courseSummary = [0] * numCourses
        courseDetail = collections.defaultdict(list)
        # 把表初始化好
        for course, pre in prerequisites:
            courseSummary[course] += 1
            if courseDetail[pre]:
                courseDetail[pre].append(course)
            else:
                courseDetail[pre] = [course]
        # 把BFS的queue初始化好
        queue = []
        for c in range(numCourses):
            if courseSummary[c] == 0:
                queue.append(c)
        count = 0
        
        # 开始进入BFS，学了一门先修课，就横扫一门看看能学什么其他课程。最终能学多少门课，就进多少次while
        # 计数看与我们的total course是否相同。
        while queue:
            curCourse = queue.pop(0)
            count += 1
            nextCourses = courseDetail[curCourse]
            if nextCourses:
                for after in nextCourses:
                    courseSummary[after] -= 1
                    if courseSummary[after] == 0:
                        queue.append(after)
        return count == numCourses

# 210. Course Schedule II
# 思路与207上一题一样，其他的细节我是自己写的哦～
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        courseSummary = [0] * numCourses
        courseDetail = collections.defaultdict(list)
        for c, p in prerequisites:
            courseSummary[c] += 1
            if courseDetail[p]:
                courseDetail[p].append(c)
            else:
                courseDetail[p] = [c]
        
        queue = []
        res = []
        for i in range(numCourses):
            if courseSummary[i] == 0:
                queue.append(i)
        
        while queue:
            cur = queue.pop(0)
            res.append(cur)
            nex = courseDetail[cur]
            if nex:
                for nc in nex:
                    courseSummary[nc] -= 1
                    if courseSummary[nc] == 0:
                        queue.append(nc)
        return res if len(res) == numCourses else []
                     
     

# 490. The Maze
# 这道题跟自己的思路一致！但是因为条件/边界判断的问题，没有做出来。
# 我的思路是direction用helper function封装，返回一个可能的position list
# 然后利用一个seen的列表去优化成个的循环。但是这道题不用这么做！
# pop(0) is O(n) operation while popleft is o(1)
class Solution:
    def hasPath(self, maze, start, destination):
        Q = [start]
        n = len(maze)
        m = len(maze[0])
        # 把所有可能的dirs写在这里之后可以for拿出来用
        dirs = ((0, 1), (0, -1), (1, 0), (-1, 0))
        
        while Q:
            i, j = Q.pop(0)
            maze[i][j] = 2

            if i == destination[0] and j == destination[1]:
                return True
            
            for x, y in dirs:
                # row, col是新坐标
                row = i + x
                col = j + y
                # 走到底，撞到墙。why this inner while?
                # 题意要求要在终点能停下！才可以！仔细读题。
                while 0 <= row < n and 0 <= col < m and maze[row][col] != 1:
                    row += x
                    col += y
                row -= x
                col -= y
                if maze[row][col] == 0:
                    Q.append([row, col])
        
        return False


# 505. The Maze II
from collections import deque
class Solution:
    def shortestDistance(self, maze: List[List[int]], start: List[int], destination: List[int]) -> int:
        if start == destination:
            return 0
        # 初始化
        #   1- 初始化queue： 进queue的是position+distance
        #   2- 维护一个visited
        #   3- 初始化一个res
        queue = deque( [tuple( start + [0] )] ) 
        visited = { tuple(start) : 0 }
        res = []
        
        while queue:
            prev_x, prev_y, prev_distance = queue.popleft()
            # 针对四个方向的四种情况
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                x, y, dist = prev_x, prev_y, prev_distance
                # 如果移动的方向是没有停止，并且满足条件就可以继续满足
                while 0 <= x+dx < len(maze) and 0 <= y+dy < len(maze[0]) and maze[x+dx][y+dy] == 0:
                    dist += 1
                    x += dx
                    y += dy
                # 如果碰到移动过程中碰到就可以继续走，停下的话就跳出while，进行判断
                if [x, y] == destination:
                        res.append(dist)
                        continue
                # 如果x\y，在xy已经碰见过。并且遇见过的要大的话，或者xy没遇见过，都要进visit
                # 进visit证明能到这里，并且下一个循环要从这里出发去看
                if ((x, y) in visited and visited[(x, y)] > dist) or ((x, y) not in visited):
                    visited[(x, y)] = dist
                    queue.append((x, y, dist))

        return min(res) if res else -1
                
"""
这些刷的题基本上都是BFS的应用题——最短路径/图:
    1. 比如模版, while + for
    2. 额外的数据结构支持, list/defaultdict(list)/seen/visted
    3. 如果操作比较复杂, 可以把距离当成一个参数进行传递
"""

########################  DFS  ########################
"""
优缺点:
BFS:对于解决最短或最少问题特别有效，而且寻找深度小，但缺点是内存耗费量大（需要开大量的数组单元用来存储状态）。
DFS：对于解决遍历和求所有问题有效，对于问题搜索深度小的时候处理速度迅速，然而在深度很大的情况下效率不高
DFS的优点
内存开销较小，每次只需维护一个结点
能处理子节点较多或树层次过深的情况（相比BFS）
一般用于解决连通性问题（是否有解）
DFS的缺点
只能寻找有解但无法找到最优解（寻找最优解要遍历所有路径）

"""
# 78. Subsets
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def dfs(first = 0, cur = []):
            # 如果当前答案满足了，剪枝
            # return没有任何东西的话就
            if len(cur) == k:
                output.append(cur[:])
                return 
            # 添加cur
            # pop(cur)的理解十分重要；
                # 首先我们通过for k进入dfs
                # 通过if后，我们利用for i去遍历剩下的数字
                # 那么为什么在调用dfs后，要pop？因为每一次for i，该位置下的cur我们只有一个，一种情况
                # pop后下一次的for i中的cur会有新的值
                # 这是递归逻辑的理解。
            for i in range(first, n): # 这一层的recur开始index为first，本题巧妙的点在于range的n+1与len(n)结合起来
                cur.append(nums[i])
                dfs(i + 1, cur) # 从下一个index开始添加。
                cur.pop()
        output = []
        n = len(nums)
        for k in range(n + 1): # 因为空集也要算，所以要进n+1次
            dfs()
        return output
# 时间N*2^N N-Copy into output，2^N-generate the number of subsets；空间N 用来维护CUR
# DFS在脱离了树之后是适合的问题就是backtrack回溯以及是否有解的情况。
# 这一题可以用字符串解决。
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        output = [[]]
        
        for num in nums:
            output += [cur + [num] for cur in output]   # 这个逻辑好玩
        return output
# 遍历第一个num时，output里是[[],[num]]；当第二次遍历是num2会和output里已经有的元素再次组队。
# 逻辑好玩

"""
理解一下这题的递归顺序很重要! 
    1. for k: 找到k位答案
    2. for i: k位中i位的答案可以是什么
    3. dfs(+1): k位中进一位的答案可以是什么
    4. pop中, 当这种情况结束后更换i位继续进行dfs看下一个可能答案是什么
"""


# 90. Subsets II
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums, path, res):
            res.append(path)
            for i in range(len(nums)):
                if i > 0 and nums[i] == nums[i - 1]:
                    continue
                dfs(nums[i+1:], path+[nums[i]], res)
        res = []
        # sort是没跑的
        nums.sort()
        dfs(nums, [], res)
        return res
# ✨这里的if i > 0 and nums[i] == nums[i - 1]为什么可以针对去重subset? (# 记好了我们是DFS)
# 首先，我们基于index = 1一直入栈只到全部入栈，然后再一个个出栈。
# 想到这里，你想一下10位数中，你如何用DFS找到8位不同的数字：首先前8位，然后拿出最后一位，如果9/10跟最后一位（8）一样，那么肯定就continue，没有了。
# 所以递归的思路是很重要的。好好想想。

# 46. Permutations
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def dfs(first = 0):
            if first == n:
                res.append(nums[:])
            for i in range(first, n):
                nums[first], nums[i] = nums[i], nums[first]
                dfs(first + 1)
                nums[first], nums[i] = nums[i], nums[first]
        n = len(nums)
        res = []
        dfs()
        return res
# 这一题有点扯淡，逻辑简单，但是如果不清楚这样swap，基本无解吧。
"""
Time complexity should be N x N!.
Initially we have N choices, and in each choice we have (N - 1) choices, and so on. 
Notice that at the end when adding the list to the result list, it takes O(N).

Second, the space complexity should also be N x N! since we have N! solutions and each of them requires N space to store elements.
"""

# 47. Permutations II
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        def dfs(com, counter):
            if len(com) == len(nums):
                res.append(list(com))
                return 
            
            for num in counter:
                if counter[num] > 0:
                    com.append(num)
                    counter[num] -= 1
                    
                    dfs(com, counter)
                    com.pop()
                    counter[num] += 1
                    
        dfs([], Counter(nums))
        return res
# 这一题思路也很简单，那么怎么能找到不重复的呢？采用排序计数制，利用Counter完成这个目标。


# 77. Combinations
# 简单自己写出来了
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        
        def dfs(first, cur):    
            if len(cur) == k:
                res.append(cur[:])  # 如果要是 append list的话，这里不能只写cur，好神奇，不知道为什么
                return 
            for num in range(first, n + 1):
                cur.append(num)
                dfs(num + 1, cur)
                cur.pop()
        res = []
        dfs(1, []) # 传参遇到了一些问题，基础不牢固
        return res

        
# 37. Sudoku Solver
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        n = len(board)
        
        #我们维护一个row,col,boxes
        rows, cols, boxes = collections.defaultdict(set), collections.defaultdict(set), collections.defaultdict(set)

        for r in range(n):
            for c in range(n):
                if board[r][c] == '.':
                    continue
                # v就是当前的值
                v = int(board[r][c])
                rows[r].add(v)
                cols[c].add(v)
                boxes[(r // 3) * 3 + c // 3].add(v)

        # 检查能不能这么put value
        def is_valid(r, c, v):
            box_id = (r // 3) * 3 + c // 3
            return v not in rows[r] and v not in cols[c] and v not in boxes[box_id]

        
        def backtrack(r, c):
            # 如果一直可以成功，那么这个就是用来规定遍历的走向
            if r == n - 1 and c == n:
                return True
            elif c == n:
                c = 0
                r += 1

            # current grid has been filled
            if board[r][c] != '.':
                return backtrack(r, c + 1)

            box_id = (r // 3) * 3 + c // 3
            
            
            # 看当前格子填v行不行
            for v in range(1, n + 1):
                if not is_valid(r, c, v):
                    continue

                # 更改期盼和更新维护的数据结构
                board[r][c] = str(v)
                rows[r].add(v)
                cols[c].add(v)
                boxes[box_id].add(v)
                
                # 自动返回机制：如果c+1返回True，那么这个也为True
                if backtrack(r, c + 1):
                    return True

                # backtrack
                board[r][c] = '.'
                rows[r].remove(v)
                cols[c].remove(v)
                boxes[box_id].remove(v)

            return False

        # 我们从(0, 0)开始进入
        backtrack(0, 0)
# 51. N Queens    
class Solution:
    def solveNQueens(self, n):
        # Making use of a helper function to get the
        # solutions in the correct output format
        def create_board(state):
            board = []
            for row in state:
                board.append("".join(row))
            return board
        
        def backtrack(row, diagonals, anti_diagonals, cols, state):
            # Base case - N queens have been placed
            if row == n:
                ans.append(create_board(state))
                return

            for col in range(n):
                curr_diagonal = row - col
                curr_anti_diagonal = row + col
                # If the queen is not placeable
                if (col in cols 
                      or curr_diagonal in diagonals 
                      or curr_anti_diagonal in anti_diagonals):
                    continue

                # "Add" the queen to the board
                cols.add(col)
                diagonals.add(curr_diagonal)
                anti_diagonals.add(curr_anti_diagonal)
                state[row][col] = "Q"

                # Move on to the next row with the updated board state
                backtrack(row + 1, diagonals, anti_diagonals, cols, state)

                # "Remove" the queen from the board since we have already
                # explored all valid paths using the above function call
                cols.remove(col)
                diagonals.remove(curr_diagonal)
                anti_diagonals.remove(curr_anti_diagonal)
                state[row][col] = "."
        
        # 初始化
        ans = []
        # 初始化棋盘
        empty_board = [["."] * n for _ in range(n)]
        backtrack(0, set(), set(), set(), empty_board)
        return ans

# 144. Binary Tree Preorder Traversal
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def dfs(root):
            if not root: return None
            res.append(root.val)
            dfs(root.left)
            dfs(root.right)
        res = []
        dfs(root)
        return res

# 1986. Minimum Number of Work Sessions to Finish the Tasks
# 好题呀～
class Solution:
    def minSessions(self, tasks: List[int], sessionTime: int) -> int:
        n = len(tasks)
        # 从大到小排序
        tasks.sort(reverse=True)
        # 字面意思session，可以
        sessions = []
        # 初始化最大session数量
        result = n
        
        # 注意这里的index很重要，当作参数进行每层的传递
        def dfs(index):
            nonlocal result
            # 这里用来剪枝，如果当前session数量比目前已知的result大，就没必要继续在这个树上吊死了
            # 你看这里只是单纯的return，那么一定是设置判断t/f的条件
            if len(sessions) > result:
                return
            
            # 如果index == n，代表已经traverse所有的elements了，可以返回啦～
            if index == n:
                result = len(sessions)
                return
            
            # 这里是判断现在已有的session
            for i in range(len(sessions)):
                # woc，我懂了。
                # 假设当前session里面已经有了3个，那么判断当前task从头进行尝试，看看哪个session可以放得下。
                # 因为采取backtrack，所以一个task都会去到所有能去的session中
                # 如果可以放得下，继续放下一个dfs
                # 如果都放不下，跳出for循环，用下方的append，新增session
                if sessions[i] + tasks[index] <= sessionTime:
                    sessions[i] += tasks[index]
                    dfs(index + 1)
                    sessions[i] -= tasks[index]
                    
            # 往session新加我们curLevel的task，新增session
            # 为什么这里也需要backtrack pop，因为一个task有可能扮演新增session的作用，也有可能不是。最重要的是穷尽所有可能性
            sessions.append(tasks[index])
            dfs(index + 1)
            sessions.pop()
        
        dfs(0)
        return result
# 与47属于一个类型，backtrack中的smart memorization. 47是将nums的list -> {num:出现次数}
# 这一题是将worktimes塞进一个个session容器中

    
# 1723. Find Minimum Time to Finish All Jobs
class Solution:
    def minimumTimeRequired(self, jobs: List[int], k: int) -> int:
        workers = [0]*k
        
        self.res = sys.maxsize
        # jobs.sort(reverse = True)
        def dfs(curr):
            if curr == len(jobs):
                self.res = min(self.res, max(workers))
                return
            
            seen = set() # record searched workload of workers
            
            # 通过for与dfs(curr+1)的配合，完美达到所有可能性都有！
            for i in range(k):
                # seen用来记录当前cur_task已经记录过答案的worker；
                # 如何理解？如果有两个worker，目前积累的workload一样，那么cur_task分配给谁都一样，那么剪枝一次就可以省略啦～
                if workers[i] in seen: continue # if we have searched the workload of 5, skip it.
                if workers[i] + jobs[curr] >= self.res: continue # another branch cutting
                    
                seen.add(workers[i])
                workers[i] += jobs[curr]
                dfs(curr+1)
                workers[i] -= jobs[curr]
        
        dfs(0)
        return self.res

"""
DFS总结:
    1. DFS的处理逻辑很重要
    2. def(first = 0)是用来当无参数传递时进行初始化, 当first满足一定条件可以return
    3. 处理backtrack时, 在for末尾记得把操作还原, 这样可以对下一位继续进行同等操作
    4. 其实这些题目更像是backtracking 而非简简单单的dfs
    5. Backtrack的模版
        刚开始初始化条件，把需要维护的数据模版都维护出来；如果有需要设计helper；
        进入backtrack，if/while规定遍历顺序，执行判断逻辑，然后记录+进入backtrack+backtrack+（return）

    如果是修改外部作用域的变量，在内部函数声明nonlocal；如果是修改全局作用域的变量，在内部函数声明global

Smart Memorization 难点总结：
- 减枝的题目一般可以使用二分法去做，也相当于是增加了限制条件，这里不是二分部分，不详解
- 一些题目也可以用状态压缩dp来解决，比如人数少的时候，12个人的状态都压缩在一个intege里面，dfs+memo这里非dp也不详解
常见4把刀减枝方法
1- sort倒序，task先做大的这样可以累积时间先达到终止条件
2- global的result, 如果我们是求最小值，当过程中结果已经大于res的时候我们就直接停止
3- 跳过重复的元素，类似permutation里面
4- 改变搜索思路，单向遍历较多的task可以大幅提升速度。一般大的数据部分pointer单向递增，小数据的部分可以增加backtracking的遍历，比如i为task, backtrack每次for loop为session见最后一题。比如1434题帽子比人多，就单向帽子

难点还在于怎么把题意中数据，换种方式遍历。

"""
########################  Binary Search  ########################
"""
二分搜索的一般模版
start, end = 0, len(n) - 1
while start <= end:
    mid = start + (end - start) // 2
    if nums[mid] < target: start = mid + 1
    else: end = mid - 1
"""
# 4. Median of Two Sorted Arrays
# md 这道题好难
def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        def findKthElement(arr1,arr2,k):
            len1,len2 = len(arr1),len(arr2)
            if len1 > len2:
                return findKthElement(arr2,arr1,k)
            if not arr1:
                return arr2[k-1]
            if k == 1:
                return min(arr1[0],arr2[0])
            # 首先K是中位数，每次二分，目的都是为了将整体array去掉二分之一，但因为不清楚两个array的大小关系，因此没办法很好地去除1/2
            # 但每一次的二分，其中一个的前半边肯定构成了最终array的前半边，因此我们是可以排除的
            # 那么什么是跳出条件？只能是两种情况，
            #       1. 一种是其中一个array已经被全部排除完了，直接返回另一个列表中对应的数就可以了
            #       2. 我们的k=1，那么此时不管还有多少，前面的肯定都被排除完了，因此返回两个array中比较小的就可以了。
            
            # k就是，排除剩下的index还有多少；i,j是两个array分别的k//2的index
            i,j = min(k//2,len1)-1,min(k//2,len2)-1
            if arr1[i] > arr2[j]:
                # 这里进入递归的是，k-j-1排除掉j前面还有多少index
                return findKthElement(arr1,arr2[j+1:],k-j-1)
            else:
                return findKthElement(arr1[i+1:],arr2,k-i-1)
        
        # left, right是我们要找到的index     
        l1,l2 = len(nums1),len(nums2)
        left,right = (l1+l2+1)//2,(l1+l2+2)//2
        return (findKthElement(nums1,nums2,left)+findKthElement(nums1,nums2,right))/2

# 278. First Bad Version
class Solution:
    def firstBadVersion(self, n: int) -> int:
        start, end = 0, n
        while start <= end:
            mid = start + (end - start) // 2
            if not isBadVersion(mid):
                start = mid + 1
            else:
                end = mid - 1
        return start
# start和end越来越得心应手了。




########################  Divide & Conquer  ########################

"""
分解原问题为若干子问题，这些子问题是原问题的规模最小的实例
解决这些子问题，递归地求解这些子问题。当子问题的规模足够小，就可以直接求解
合并这些子问题的解成原问题的解
"""

# 169. Majority Element
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        count = Counter(nums)
        res = sorted(count.items(), key = lambda item: item[1], reverse=True)
        return res[0][0]
        # max的骚用法
        # return max(counts.keys(), key=counts.get)
# 如何使用sorted，以及如何给dict排序！
class Solution:
    def majorityElement(self, nums, lo=0, hi=None):
        # lo/hi是indices，传递数组浪费了
        # return的是majority数字
        def majority_element_rec(lo, hi):
            # base case
            if lo == hi:
                return nums[lo]
            # recurse on left and right halves of this slice.
            mid = (hi-lo)//2 + lo
            left = majority_element_rec(lo, mid)
            right = majority_element_rec(mid+1, hi)
            # if the two halves agree on the majority element, return it.
            if left == right:
                return left

            # otherwise, count each element and return the "winner".
            left_count = sum(1 for i in range(lo, hi+1) if nums[i] == left)
            right_count = sum(1 for i in range(lo, hi+1) if nums[i] == right)

            return left if left_count > right_count else right

        return majority_element_rec(0, len(nums)-1)
# 这道题要从bottom-up开始思考。
# 首先我们的base case是选出我们的majority num
# 递归返回后，index*2，看看左右哪个比较多，然后继续返回。

# 215. Kth Largest Element in an Array
# 本题利用了quick sort的思想
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        n = len(nums)
        self.divide(nums, 0, n - 1, k) # 这里的k传递进去是为了在sort过程中起到一个剪枝的作用
        return nums[n - k]
    
    def divide(self, nums, left, right, k):
        if left >= right: return 
        # position是每一次pivot的位置
        position = self.conquer(nums, left, right)
        # 证明至少position这一个位置已经排好了，那么直接返回
        if position == len(nums) - k: return
        elif position < len(nums) - k: self.divide(nums, position + 1, right, k)
        else: self.divide(nums, left, position - 1, k)
        
    def conquer(self, nums, left, right):
        pivot, wall = nums[right], left
        for i in range(left, right):
            if nums[i] < pivot:
                nums[i], nums[wall] = nums[wall], nums[i]
                wall += 1
        nums[wall], nums[right] = nums[right], nums[wall]
        return wall
# 二分要灵活运用呀

########################  Monotone Stack  ########################
"""
单调栈保持递增或者递减，一般是O(n)的时间复杂度
反向模板内部三步走，正向模板需要把放入res这步集成到保持stack单调同时去做
保持stack递增(递减)
将栈顶元素放入final result
把当前iterate元素放入栈(可以是实际元素value，也可以只是index)
private static int[] nextGreaterElement(int[] nums) {
    int[] res = new int[nums.length];
    Stack<Integer> stack = new Stack<>();
    for (int i= nums.length - 1; i >= 0; i--) {
        while(!stack.isEmpty() && nums[i] >= stack.peek()) stack.pop();
        res[i] = stack.isEmpty() ? -1 : stack.peek();
        stack.push(nums[i]);
    }
    return res;
}


单调栈主要解决以下问题： 寻找下一个更大元素 寻找前一个更大元素 寻找下一个更小元素
"""
# 496. Next Greater Element I
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        greater = {x: -1 for x in nums1}
        stack = []
        # 这里的stack是递减的为什么？如果是递减的没关系
        # 一旦发现有大的数字，就持续pop出来，找到当前num应该在的位置，塞进去
        # 这些pop出来的数字，还是有将就的。
        # 画图理解就好了
        for num in nums2:
            while stack and num > stack[-1]:
                prev = stack.pop()
                if prev in greater:
                    greater[prev] = num
            stack.append(num)
        return [greater[x] for x in nums1]

# 503. Next Greater Element II
# 好多坑！
# 🌟if circular, we got 2 solutions here
#   1. nums * 2
#   2. indices = 2 * nums
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        res = [-1 for _ in range(len(nums))]    # 这里一定要先写出来，因为不存在append操作，否则会出现index out of range
        stack = []
        for i in range(len(nums) * 2 - 1, -1, -1):      # 第一个坑！ 1. 这里是倒序，不能正序；因为next greater顺序问题！在接下来谈！
            index = i % len(nums)
            while stack and nums[index] >= nums[stack[-1]]: # 第二个坑！ 1. 这里入stack的是index，而非num； 2. >=而非>
                stack.pop()
            res[index] = -1 if not stack else nums[stack[-1]] # 第三个坑！ if else的顺序，否则会导致index out of range
            stack.append(index)
        return res
# 其实横向比较496和503这两题，其实有很多值得借鉴学习的地方
# 496利用了dict/hashmap的结构存储了答案，因为是两个array，还OK；本题只有一个array，严格一一对应，所以没关系
# 这里入栈的是index，当然可以直接用num，更直接。为什么要用Index？想明白了么？因为我们最终返回的是和nums一一对应的list，没有index的话，如何修改res？对吧



# 坑！********************
# 1. 如果第一个坑采用正序, 我们的stack是递减的；首先我们pop出已经遍历过的所有比我们小的数，然后我们选择stack[-1]之后，再把当前index append进去。
# 但是这样做的后果就是，当前res index选择的是之前遍历过的大数，即是last greater，所以这一题我们要用倒序。
# Plus，为什么496这一题可以正序呢？因为这个顺序，他是保存在map里面，然后as per nums1的顺序得出的，因此没必要太在乎顺序。
# 2. 第二个坑，为什么=的情况下也要出栈，因为我们肯定知道最大值的答案为-1，但如果不出栈，那么当前最大值会是它自己。所以出栈后，我们清空stack，可以保证代码无法更改res中的-1
# 在上一问中不需要，因为它不是环型的，而且他的正序处理依靠的是刚pop出来的值，而非stack里面的值

# 1019. Next Greater Node In Linked List
class Solution:
    """
    我们monotone stack的基本用法就是这样，但是每次判断的那个值，上一个值/下一个值的处理就是在我们code的不同行，不同位置。
    如果能分析到这些，那么这种类型的题就可以了。
    """
    def nextLargerNodes(self, head: Optional[ListNode]) -> List[int]:
        res, stack = [], []
        # 这里的stack，每一个element是one pair，放index和该index对应的值
        while head:
            while stack and stack[-1][1] < head.val:
                res[stack.pop()[0]] = head.val
            stack.append([len(res), head.val])
            res.append(0)
            head = head.next
        return res


# 739. Daily Temperatures 自己写的哦
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        res = [0 for _ in range(len(temperatures))]
        stack = []
        for i in range(len(temperatures)):
            while stack and temperatures[i] > stack[-1][1]:
                index = stack.pop()[0]
                res[index] = i - index
            stack.append([i, temperatures[i]])
        return res
    
# 316. Remove Duplicate Letters
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        stack, seen = [], set()
        # 单独开一张dict，用来维护每个字母最后出现的位置
        last_occurrence = {c: i for i, c in enumerate(s)}
        
        for i, c in enumerate(s):
            # 只去处理没有seen过的
            if c not in seen:
                # 1. char要是小的
                # 2. char出现的位置要比stack[-1]小？
                # 满足条件的话，就把[-1]从seen中删除，我们为什么这么做？
                # 因为我们要确定我们的stack严格满足我们的题意。上述两个条件表明，当我们遇到的c<[-1]时，并且[-1]在之后的位置还会再次出现，那我们就先暂时把它舍弃。
                while stack and c < stack[-1] and i < last_occurrence[stack[-1]]:
                    seen.discard(stack.pop())
                seen.add(c)
                stack.append(c)
        return ''.join(stack)
        # 这一题利用了比较多的space帮助判断，因为我们不仅要照顾顺序，还要照顾重复，而且顺序也不是稳定的，所以利用了比较局部的算法，同时加了一些限制

# 42. Trapping Rain Water
class Solution:
    def trap(self, height: List[int]) -> int:
        res, stack = 0, []
        for i in range(len(height)):
            # 我们的stack还是递减的哈～
            # 如果当前i比stack最低还要高的话，我们进入while，记录top当前最低value，我们进入while循环
            while stack and height[i] > height[stack[-1]]:
                # 记录
                top = stack.pop()
                # 没有stack的话一般就意味着遍历完了，或者当前是最高，之前没有可以组成沟槽的地方了（要组成沟槽后者必须比前者大，而且宽度要超过1）
                if not stack:
                    break
                # 计算宽度，这一题理解stack[-1]很关键. stack[-1]代表的index与我们当前的index不一定紧密相连，拿这种情况是如何产生的？
                # 当我们遇到一个高点，我们会往前找一个点进行匹配。但是前面那个点可能已经匹配和已经出现过的匹配过了，也从stack中删除了，因此我们stack[-1]的index会与我们现在的i有间隙
                # 那么这样res计算答案的图形可以理解两个点之间的部分方块，而非整个丁字形。
                distance = i - stack[-1] - 1
                b_h = min(height[i], height[stack[-1]]) - height[top]
                res += distance * b_h
            stack.append(i)
        return res
# 这一题主要是难以理解。简单代码，逻辑复杂，可以先不看。

########################  Monotone Queue  ########################
"""
时间复杂度依然是 O(N) 线性时间。要这样想，nums 中的每个元素最多被 offer 和 poll 一次，没有任何多余操作，所以整体的复杂度还是 O(N)。
空间复杂度就很简单了，就是窗口的大小 O(k)。
注意判断等号，我一般是先维持k-1的size，然后offerLast
注意offerLast必须在pollLast之后，也就是手动维持单调递增递减队列
一般能用dq的，如果不想手动维护，都可以使用pq来维持window递增或者递减

public int[] MonotonicQueue(int[] nums, int k){
    int N = nums.length;
    Deque<Integer> q = new ArrayDeque<>();
    int[] res = new int[N - k + 1]
    for (int i = 0; i < N; i++) {
        while (!q.isEmpty() && i - q.peekFirst() >= k) q.pollFirst();
        while (!q.isEmpty() && nums[q.peekLast()] <= nums[i]) q.pollLast();
        q.offerLast(i);
        q.peekFirst();
    }
    return res;
}

单调队列，顾名思义其中所有的元素都是单调的(递增或者递减)，承载的基础数据结构是队列，实现是双端队列，队列中存入的元素为数组索引，队头元素为窗口的最大(最小)元素。
队头删除不符合有效窗口的元素，队尾删除不符合最值的候选元素。

"""
# 1696. Jump Game VI
class Solution:
    def maxResult(self, nums: List[int], k: int) -> int:
        # 这道题用dp，先把score init出来
        n = len(nums)
        score = [0]*n
        score[0] = nums[0]
        # init deque
        dq = deque()
        dq.append(0)
        # 这里我们dp里面存的也是index哈
        for i in range(1, n):
            # pop the old index
            # 区间为k，所以最早的index一定为i-k
            while dq and dq[0] < i-k:
                dq.popleft()
            
            # 为什么这里是这么写的？
            # 首先要理解dp里面存的是什么？1. decreasing 2. 针对目前index可以达的所有index
            # 这里的相加就相当于是dp的transition funciton
            # score的index位置，放的是dp里可以到达的最大的数，再加上当前的num。
            score[i] = score[dq[0]] + nums[i]
            
            # pop the smaller value
            # 这个相当于是index入enqueue时候的限制：
                # 如果发现当前score比他们大的话，把dq中已有的比当前i小的全部pop掉
                # 为什么？ -> 当前index是最新的，而且能保证，dq里的都是可达的最大的index。
            while dq and score[i] >= score[dq[-1]]:
                dq.pop()
            dq.append(i)
        return score[-1]

# 1438. Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit
# 这一题涉及到sliding window，其次才是如何存储最大值/最小值。利用monotonic queue
class Solution:
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        min_deque, max_deque = deque(), deque()
        l = r = 0
        ans = 0
        while r < len(nums):
            while min_deque and nums[r] <= nums[min_deque[-1]]:
                min_deque.pop()
            while max_deque and nums[r] >= nums[max_deque[-1]]:
                max_deque.pop()
            min_deque.append(r)
            max_deque.append(r)
            
            # 我们维护一个slidign window，然后判断sliding window里面的max 和 min，如果超过就shrink the window
            while nums[max_deque[0]] - nums[min_deque[0]] > limit:
                l += 1
                if l > min_deque[0]:
                    min_deque.popleft()
                if l > max_deque[0]:
                    max_deque.popleft()
            
            ans = max(ans, r - l + 1)
            r += 1
                
        return ans



# 239. Sliding Window Maximum
from collections import deque
class Solution:
    def maxSlidingWindow(self, nums: 'List[int]', k: 'int') -> 'List[int]':
        if not nums or len(nums) < 2: return nums
        queue = deque()
        # 先把答案的位置弄出来，防止后面操作res时out of range
        res = [0] *(len(nums)-k+1)
        for i in range(len(nums)):
            # 确保queue里是decresing的，而且element都要比当前nums[i]要大
            while queue and nums[queue[-1]] <= nums[i]:
                queue.pop()
            queue.append(i)
            # 如果queue里的最大值已经超过了，那我们pop出去就好了，我们的queue里面的index是in-order的
            if queue[0] <= i-k:
                queue.popleft()
            if i+1 >= k:
                res[i+1-k] = nums[queue[0]]
        return res
"""
大概的思路: 首先把window遍历出来，然后判断是否取消popleft掉old index，pop掉其他值无所谓，因为我们只要区间内的最大值！
pop掉之后，判断我们是否已经遍历完window了， 遍历完就可以添加答案啦～
这种类型的题好难！


而且需要判断是否需要同时维护两个deque，意味着max和min
"""


########################  Sliding Window  ########################
"""
滑动窗口算法可以用以解决数组/字符串的子元素问题，它可以将嵌套的循环问题，转换为单循环问题，降低时间复杂度。
如何识别滑动窗口？
- 连续的元素，比如string, subarray, LinkedList
- min, max, longest, shortest, key word


1- Easy, size fixed 
    窗口长度确定，比如max sum of size = k

2- Median, size可变，单限制条件
    比如找到subarray sum 比目标值大一点点

3- Median, size可变，双限制条件
    比如longest substring with distinct character

4- Hard, size fix, 单限制条件
    比如sliding window maximum，考察单调队列，请参考单调队列的PPT


Sliding window 套路模板时间复杂度一般为O(n)
一般string使用map作为window，如果说明了只有小写字母也可以用int[26]
多重限制条件的压轴题需要考虑是否为单调队列，在另一节PPT有详解
字母类还可以暴力尝试26个字母，比如1个unique，2个unique，然后内部模板
Exact(k) 可以转换为 atMost(k) - atMost(k - 1)


public int lengthofLongestSubstringKDintinct(String s, int k){
    Map<Character, Integer> map = new Hashmap<>();
    int left = 0, res = 0;
    for (int i = 0; i < s.length(); i++) {
        char cur = s.charAt(i);
        map.put(cur, map.getOrDefault(cur, 0) + 1);
        while (map.size() > k) {
            char c = s.charAt(left);
            map.put(c, map.get(c) - 1);
            if (map.get(c) == 0) map.remove(c);
            left++;
        }
        res = Math.max(res, i - left + 1);
    }
    return res;
}
"""
# 3. Longest Substring Without Repeating Characters
# 自己吭哧吭哧写的，想清楚处理逻辑就好很多，一些情况下可能要先处理一段代码后，前面的条件代码才会变得更清晰。
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s: return 0
        if len(s) == 1: return 1
        l, res = 0, 1
        for i in range(1, len(s)):
            while s[i] in s[l : i]:
                l += 1
            res = max((i - l + 1), res)
        return res
            
            
            
# 159. Longest Substring with At Most Two Distinct Characters
# 哈哈哈哈又是自己写的，太爽了呢。
# 利用defaultdict可以帮助自己的代码优化/精简
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        record = dict()
        left, res = 0, 1
        
        for i in range(len(s)):
            ch = s[i]
            if ch not in record:
                record[ch] = 1
            else:
                record[ch] += 1

            while len(record) > 2:
                del_ch = s[left]
                record[del_ch] -= 1
                if record[del_ch] == 0:
                    del record[del_ch]
                left += 1
            res = max((i - left + 1), res)
        return res
"参考答案的优化点: hashmap存的是index, 而非出现的次数, 对哦, 更新最后一次的位置就可以了! 牛 然后移动left pointer的时候可以直接定位到left+1, 而不用一位位地加。"

# 340. Longest Substring with At Most K Distinct Characters
# 跟上一题一样，但是有一点不一样。用了default，但是注意了时间复杂度为N，worst的话为nk；空间为K，开销为hashmap
class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        
        seen = defaultdict(int)
        left, res = 0, 0
        for i in range(len(s)):
            seen[s[i]] += 1
            
            while len(seen) > k:
                del_ele = s[left]
                seen[del_ele] -= 1
                if seen[del_ele] == 0:
                    del seen[del_ele]
                left += 1
            
            res = max(res, (i-left+1))
            
        return res

# 395. Longest Substring with At Least K Repeating Characters

# 本题难点在于代码逻辑
# 首先，我们的for层是为了确保我们的substring中有几个不一样的元素
# 确定好后，每一层都遍历所有，如果满足，就进result，如果不满足就进下一层。
class Solution:
    def longestSubstring(self, s, k):
        result = 0
        # 这里的T是指有多少个unique number，我们遍历每一种情况
        for T in range(1, len(Counter(s))+1): 
            beg, end, Found, freq, MoreEqK = 0, 0, 0, [0]*26, 0
            while end < len(s):
                # MoreEqk是当前已有的Unique
                # 当前已有的Unique小于我们的目标时
                # 把freq里对应的加一
                # 这个if的情况下，我们扩展右边界
                if MoreEqK <= T:
                    s_new = ord(s[end]) - ord('a')
                    freq[s_new] += 1
                    # 如果==1，意味着新增了一个字母，那么当前unique加一
                    if freq[s_new] == 1:
                        MoreEqK += 1
                    # ==k意味着找到一个字母
                    if freq[s_new] == k:
                        Found += 1
                    end += 1
                
                # 如果否则将左边的端口移动。
                else:
                    symb = ord(s[beg]) - ord('a')
                    beg += 1
                    if freq[symb] == k:
                        Found -= 1
                    freq[symb] -= 1
                    if freq[symb] == 0:
                        MoreEqK -= 1
                            
                if MoreEqK == T and Found == T:
                    result = max(result, end - beg)
                    
        return result

# 或者递归的逻辑，好美丽的用法woc，也算死分治。
class Solution(object):
    def longestSubstring(self, s, k):
        if len(s) < k:
            return 0
        for c in set(s):
            if s.count(c) < k:
                return max(self.longestSubstring(t, k) for t in s.split(c))
        return len(s)



# 424. Longest Repeating Character Replacement
class Solution:    
    def characterReplacement(self, s, k):
        count = collections.Counter()
        start = result = 0
        for end in range(len(s)):
            count[s[end]] += 1
            # API返回一个list，其中list[0]存放的是出现最多的数字和其频率
            # 这里返回的是出现最多的元素的频率
            max_count = count.most_common(1)[0][1]
            # 意味着剩下的元素已经大于>k了，怎么办都没有办法转换
            # 而剩下的元素无所谓是几个字母
            # if 不满足，缩小windown
            # 这里填写if其实也是可以的。wtf！
            # 牛逼呀！因为当满足题意的时候window才会expand，而我们找的是largest，所以不满足条件的时候没必要缩小到最小尺寸
            # 当再次满足题意的时候，我们的window会再次expand的！跟res = max(res, xxx)有异曲同工之妙
            while end - start + 1 - max_count > k:
                count[s[start]] -= 1
                start += 1
            result = max(result, end - start + 1)
        return result

# 209. Minimum Size Subarray Sum
# 简单
class Solution:
    def minSubArrayLen(self, target:int, nums) -> int:
        if not nums or not target: return 0
        left, ans, total = 0, float("inf"), 0
        for i in range(len(nums)):
            total += nums[i]
            while total >= target:
                ans = min(ans, i-left+1)
                total -= nums[left]
                left += 1
        return 0 if ans==float("inf") else ans
        
# 992. Subarrays with K Different Integers
class Solution:
    def subarraysWithKDistinct(self, A, K):
        # k - (k-1)意味着只有k个different number的情况
        # 只用相减，就能得到只为K的结果了
        return self.atMostK(A, K) - self.atMostK(A, K - 1)
    
    # 这里是如果如果最多为k个数字，那么有多少种可能
    def atMostK(self, A, K):
        count = collections.Counter()
        res = i = 0
        for j in range(len(A)):
            # 如果遇到新的数字，那么K减去一
            if count[A[j]] == 0: K -= 1
            # 记录遇到过的J数字
            count[A[j]] += 1
            # 已经碰到满足的sliding window，要左移了！之前是不满足的话左移，这里是满足的话左移动
            while K < 0:
                count[A[i]] -= 1
                if count[A[i]] == 0: K += 1
                i += 1
            # 就是从right point开始，到左边pointer，每一个组合都是OK的，那么在i~j这个窗口中，一共有j-i+1个组合，而且不会重复。
            res += j - i + 1
        return res

# 1248. Count Number of Nice Subarrays
# 与上一题非常像，但是这里只用处理odd number就成
class Solution:
    def numberOfSubarrays(self, A, k):
        def atMost(k):
            res = i = 0
            for j in range(len(A)):
                k -= A[j] % 2
                while k < 0:
                    k += A[i] % 2
                    i += 1
                res += j - i + 1
            return res
        
        return atMost(k) - atMost(k - 1)

########################  Sort  ########################
"""
常考的:
merge sort, 
quick sort (quick select)
bucket sort
counting sort
heap sort

少考的:
pancake sort

不考的:
bubble sort, 
selection sort, 
insertion sort,
shell sort
radix sort

sorting还经常和二分相关，比如在一个sorted array里面去找target等等
quicksort 又可以quick select, 作为部分sort即可。
quicksort Ave O(nlogn), 为了防止worst case O(n2)我们可以shuffle 或者 random pivot

O(N2) bubble sort, insertion sort, selection sort, shell sort(half gap实现)
O(nlogn) merge sort, heap sort,quick sort
Avg O(N) 有quick select
严格接近O(N), bucket sort(N + k), counting sort, radix sort(NK) 


"""
# 215. Kth Largest Element in an Array
# 本题利用了quick sort的思想 ✨
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        n = len(nums)
        self.divide(nums, 0, n - 1, k) # 这里的k传递进去是为了在sort过程中起到一个剪枝的作用
        return nums[n - k]
    
    def divide(self, nums, left, right, k):
        if left >= right: return 
        # position是每一次pivot的位置
        position = self.conquer(nums, left, right)
        # 证明至少position这一个位置已经排好了，那么直接返回
        # 这里涉及到一个剪枝，如果position在我们目标的左边，那么我们排右边就行了，不用排左边。
        if position == len(nums) - k: return
        elif position < len(nums) - k: self.divide(nums, position + 1, right, k)
        else: self.divide(nums, left, position - 1, k)
        
    # 即使在quick sort中，理解pi/conquer也很关键
    def conquer(self, nums, left, right):
        # 选取pivot，选取wall
        pivot, wall = nums[right], left
        # 针对每一个元素，如果小于pivot，那么将wall对换。
        # 大概的样子就是碰到小的，我们将其与wall对换，先把wall的值放在后面，而wall的index先不动，然后把index更新到下一位。
        # 这里不用担心对换的值，因为对换的index一定大于等于wall。首先遇到大数的时候，大数与wall不动，碰到小数的时候才会把wall和大数挤到下一位
        for i in range(left, right):
            if nums[i] < pivot:
                nums[i], nums[wall] = nums[wall], nums[i]
                wall += 1
        nums[wall], nums[right] = nums[right], nums[wall]
        return wall


# 148. Sort List
# 🌟Merge sort很流行
class Solution(object):
    def merge(self, h1, h2):
        # 首先我们需要dummy node技术
        # 这里也很明确，tail去充当合并时的index，而dummy充当最后return时的功能性head node
        dummy = tail = ListNode(None)
        while h1 and h2:
            if h1.val < h2.val:
                tail.next, h1 = h1, h1.next
            else:
                tail.next, h2 = h2, h2.next
            tail = tail.next
    
        tail.next = h1 or h2
        return dummy.next
    
    # 当我们把linkedlist拆到不能拆的地步的时候，我们开始merge最小的。
    # 什么时候不能拆呢？slow进入的下一个递归当head时，下一位没有了。
    def sortList(self, head):
        if not head or not head.next:
            return head
    
        # 很明确哈：pre是用来断开连接的；slow是用来铆钉sub-list的head的；fast是用来暂停while的
        pre, slow, fast = None, head, head
        while fast and fast.next:
            pre, slow, fast = slow, slow.next, fast.next.next
        pre.next = None
        
        """
        *的用法
        f(*[1,2,...]) = f(1,2,...)
        self.merge(*map(self.sortList, (head, slow)))
        equals
        self.merge(self.sortList(head),self.sortList(slow))
        """
        return self.merge(*map(self.sortList, (head, slow)))


# Dijkstra Sort的感觉
# 这题不错，挺有意思的。
# 75. Sort Colors
class Solution:
    def sortColors(self, nums):
        red, white, blue = 0, 0, len(nums)
        while white <= blue:
            # 好聪明的写法，荷兰国旗。
            # 如果碰到0，意味着red和white都应该右移
            if nums[white] == 0:
                nums[white], nums[red] = nums[red], nums[white]
                red += 1
                white += 1
            # 碰到1，white自己移动就成了。
            elif nums[white] == 1:
                white += 1
            # 碰到2的话，就是把blue往左边收缩。
            else: 
                nums[white], nums[blue] = nums[blue], nums[white]
                blue -=1 

# 451. Sort Characters By Frequency
class Solution:
    def frequencySort(self, s: str) -> str:
        counts = collections.Counter(s)
        string_builder = []
        for letter, freq in counts.most_common():
            string_builder.append(letter * freq)
        return "".join(string_builder)
# 利用hashmap
# 利用merge sort
# 利用quick sort
# bucket sort也可以

# 用的是bucket sort
# 164. Maximum Gap
# 这里有个有意思的点
        """
        为什么我们只用去找桶与桶之间的diff, 而不用看桶内部元素的diff呢?
        - 首先我们有hi,lo, n-1个bucket
        - 每一个bucket的range为 (hi-lo)/(n-1)
        - 最大的difference为hi-lo
        - 所以平均的difference为(hi-lo)/n-1
        
        如果有diff小于这个平均数, 那么一定有diff大于这个数字, 意味着较大的diff的两个数字肯定不位于一个桶内!而且是一个最大/一个最小。
        """
class Solution:
    def maximumGap(self, nums):
        
        # 找到最大值，最小值，length，以及完成init
        lo, hi, n = min(nums), max(nums), len(nums)
        if n <= 2 or hi == lo: return hi - lo
        
        # 这里的B是bucket
        B = defaultdict(list)
        # 如果num是highest的话，直接入n-2
        # 如果不是的，看看入哪个桶
        # 这里的桶是怎么区分的？比如我们有n个index，那么我们将max～min这个区间分成n个桶
        for num in nums:
            # (num-lo)//(hi-lo）看看num在hi～lo中的range在哪？然后*(n-1)找到桶的位置。
            index = n-2 if num == hi else (num - lo)*(n-1)//(hi-lo) 
            B[index].append(num)
            
        # 找到每一个桶的最小值和最大值
        cands = [[min(B[i]), max(B[i])] for i in range(n-1) if B[i]]
        
        
        # for x,y in zip(cans, cands[1:]) 其实就是把原来的队列和1:对队列拼起来。
        # x=cands, y=cands[1:] => y[0]-x[1]就是最小值
        return max(y[0]-x[1] for x,y in zip(cands, cands[1:]))


    
########################  Prefix Sum  ########################
"""
2sum系列
rangeSum
sliding window
monotonic queue

考察最多的prefix还是two sum(和，差，余数，0), 如果是range Sum的话，大部分情况会提升到2维。
对于sliding window 还是单调队列，取决于是否有负数。滑动窗口的左缩进是不论subarray sum大小直接缩进，对于全正数是ok的，因为缩进一定会让sum减小，有负数的情况就不可以这样，需要根据subarray sum减小来改变左缩进，也就是单调队列保持最小起点，因为差值sum[i] - sum[queue.peekFisrt()] 就是subarray的和，左缩进不一定是一步一步走的，是根据总window sum的减小来走的，会走到下一个最小的起点。


很多题目包括greedy比如gas station 类似的题目，也是用了prefix sum的思路来对一路上的gas求和，这里的prefix sum就比较广义了，题目过多不再详述

"""

## 我勒哥大去！
## Sliding window一般用来处理区间的最值
# 这一题不能用sliding window，因为你不知道下一个element是大是小，而且而且我们要找的值是定值，只能用一个更大的数/一个更小的数合计起来。
# 560. Subarray Sum Equals K
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        ans, presum = 0, 0
        # 这个意味着本身，我们要把它给init出来
        d = {0:1}
        for num in nums:
            # presum就是一列前缀和
            # prefix[i] - prefix[j] = k 那么prefix[i] - k = prefix[j]
            # 这个j就有可能是范围内的任何一个。
            presum = presum + num
            
            # 没有，就意味着还没有该值的组合
            if (presum - k) in d:
                ans = ans+d[presum-k]
                
            # 往hashmap里面update数据
            if presum not in d:
                d[presum] = 1
            else: 
                d[presum] += 1
        return ans

# 974. Subarray Sums Divisible by K
class Solution:
    def subarraysDivByK(self, nums: List[int], k: int) -> int:
        ans, presum = 0, 0
        hm = {0:1}
        
        for num in nums:
            presum += num
            mod = presum%k
            # 这个get()用法，就是返回mod的值，如果没有就返回默认值，这里我们设置为了0
            target = hm.get(mod, 0)
            ans += target
            # 把mod放到hashmap中，为什么这里只要是mod相同就可以添加到ans里面呢？ 想想，如果%k相同，那么这两个presum相减一定能被k整除，因为两个sum都刚好多了一点
            # 当然你通过推导公式也可以得出
            hm[mod] = target + 1
        return ans


"""
写到这里你可以发现, 变种无非是
    - 计算与推导公式
    - 是否需要维护
    - 是否需要判断处理逻辑
    - 更新total的方式。
"""
# 523. Continuous Subarray Sum
class Solution():
    def checkSubarraySum(self, nums, k):
        hm = {0:-1}
        total = 0
        # 本体不需要计数，因此不需要维护什么东西
        # i=index, n=num, we need index to decide we have at least two element, required by the problem
        for i, n in enumerate(nums):
            # 我们只需要得到sum为0就好了
            if k == 0:
                total += n
            # 如果k不为0，那么我们的total只用获得modulo就可以了。
            else:
                total = (total + n) % k
            
            if total in hm:
                # 这是prefix sum的index，所以至少要为2，相减剩下的element才至少为2.
                if i - hm[total] >= 2:
                    return True
            else: 
                hm[total] = i
        return False

# 525. Contiguous Array
class Solution:
    def findMaxLength(self, nums: List[int]) -> int:
        count = 0
        max_length = 0
        table ={0:0}
        # 从1的index枚举, 为什么呢？错了，这里指的是index从1开始，而不是从index=1的地方开始tranverse
        # 这样的话，在后面计算length的时候不用再➕1了。
        for index, num in enumerate(nums, 1):
            # count是用来表示0与1的关系
            if num == 0:
                count -= 1
            else:
                count += 1
            
            # 如果count已经出现过，表明前面的某个index也出现过0与1的关系，相减刚好可以得到count=0的状态，意味着什么？意味着1和0平衡了！
            # 而且你发现没有？如果出现过，我们只update result，并不修改table里面的数据，为什么？因为我们想要的是这种状态第一次出现的时间，用来求的最大值！
            if count in table:
                max_length = max(max_length, index - table[count])
            else:
                table[count] = index
        return max_length
    
# hashmap的存储逻辑我的拿捏还不是很精准 烦死了！


# 370. Range Addition
# 这一题的方法好聪明，我凑
class Solution:
    def getModifiedArray(self, length: int, updates: List[List[int]]) -> List[int]:
        result = [0] * length
        for start, end, value in updates:
            result[start] += value
            end += 1
            # end还在index range内
            if end < len(result):
                result[end] -= value
        
        for i in range(1, len(result)):
            result[i] += result[i-1]
        return result
# 下面解释为什么这种方法行得通？
# [0,0,0,0,0] 我们想把index 1～3 的数字➕2
# [0,2,0,0,-2] 第一步变化，然后我们求前缀和
# [0,2,2,2,0] 成功。
# 简单来讲，就是先把start add value，然后把end index的后一位进行事先删减。
        
        
        

"""
int[][] sums;

public NumMatrix(int[][] matrix) {
    int row = matrix.length, col = matrix[0].length;
    sums = new int[row+1][col+1];
    for (int i=0; i<row; i++)
        for (int j=0; j<col; j++) 
            sums[i+1][j+1] = sums[i+1][j] + sums[i][j+1] + martrix[i][j] - sums[i+j]
}
# 这个function是用来求square的。
public int sumRegion(int row1, int col1, int row2, int col2) {
    return sums[row2+1][col2+1] - sums[row1][col2+1] - sums[row2+1][col1] + sums[row1][col1]
}
"""
# 304. Range Sum Query 2D - Immutable
# 这道题的难点在于题意的理解，以及如何计算prefix sum
class NumMatrix:
    def __init__(self, matrix: List[List[int]]):
        m, n = len(matrix), len(matrix[0])
        # 生成(m+1)X(n+1)的矩阵，那么为什么要生成m+1和n+1呢？
        # 为了边界条件，为了少一次开头的判断。
        # Basic logic is like首先生成matrix，和我们的辅助matrix sum
        # 我们的辅助matrix——sums是干什么呢？是储存了某一个点左上方的所有点值。
        # 我们的辅助matrix——sums与题意中的matrix给的恰巧就错一个值
        self.sums = [[0] * (n+1) for _ in range(m + 1)]
        for row in range(1, m+1):
            for col in range(1, n+1):
                self.sums[row][col] = self.sums[row-1][col] + self.sums[row][col-1] - self.sums[row-1][col-1] + matrix[row-1][col-1]
                # 这里为什么要减去sums[row-1][col-1] 因为上面相加的两个部分，针对这一部分进行了重复计算，所以减去，你自己画个图看一下就明白了，多余的部分是相交的小矩形

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        row1, row2, col1, col2 = row1+1, row2+1, col1+1, col2+1
        return self.sums[row2][col2] - self.sums[row2][col1-1] - self.sums[row1-1][col2] + self.sums[row1-1][col1-1]
    




+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
至此，古城算法基础Algo部分就结束了。下面进行一下复盘工作呗，接下来该开启DB的工作了。
我们大概刷了十个类型，都挺经典的。
    1- 扫描线
    2- BFS
    3- DFS
    4- Binary Search
    5- Divide and Conquer
    6- Single Stack
    7- Single Queue
    8- Sliding Window
    9- Sort
    10- Prefix Sum