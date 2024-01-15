# Google
# This .py file include GG-taged questions from Sprint 2023/Explore GG/Top questions List



# 3 - Longest Substring Without Repeating Characters
# 3种解法
    # 3.1 正常counter/defaultdict 记录 + 正常更新
    # 3.2 set() + remove()
    # 3.3 map() 记录上一次见到该char的index


# 8 - String to Integer (atoi)
class Solution:
    def myAtoi(self, input: str) -> int:
        sign, result, index, n = 1, 0, 0, len(input)
        INT_MAX, INT_MIN = pow(2,31)-1, -pow(2,31)
        
        while index < n and input[index] == ' ':
            index += 1
        
        if index < n and input[index] == '-':
            index += 1
            sign *= -1
        elif index < n and input[index] == '+':
            index += 1
        
        
        while index < n and input[index].isdigit():
            digit = int(input[index])
            
            if (result > INT_MAX//10) or (result == INT_MAX // 10 and digit > INT_MAX % 10):
                return INT_MAX if sign == 1 else INT_MIN
            
            result = 10*result + digit
            index += 1
            
        return sign * result
        


# 12. Integer to Roman
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

# 13. Roman to Integer
VALUES  = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}

class Solution: 
    def romanToInt(self, s):
        tt = i = 0
        n = len(s)
        while i < n:
            if i+1 < n and VALUES[s[i+1]] > VALUES[s[i]]:
                tt += VALUES[s[i+1]]  - VALUES[s[i]]
                i += 2
            else:
                tt += VALUES[s[i]]
                i += 1
        return tt
    

# 253. Meeting Rooms II
import heapq, collections
from collections import List
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        if not intervals: return 0
        intervals.sort()
        room = 1
        pq = []
        heapq.heappush(pq, intervals[0][1])
        for s, e in intervals[1:]:
            if pq and s >= pq[0]:
                heapq.heappop(pq)
            heapq.heappush(pq,e)
            room = max(room, len(pq))
        return room
# 还是一样的，不用缩小pq！


# 68. Text Justification
class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        # we need 3 temp-vars to record the length, word_number(space needed)
        temp = []
        temp_l = 0
        temp_cnt = 0
        res = []

        # 1. temp -> currentline
        for w in words:
            # 1/当前存不到temp中->处理space/更新
            if temp_l+temp_cnt+len(w) > maxWidth:
                # 之所以用max()是因为avoid当前列表里只有一个word，如果只有一个，我们向其后面添加space
                size = max(1,len(temp)-1)
                for i in range(maxWidth - temp_l):
                    index = i % size # 轮流得到插空的index
                    temp[index] += " "
                res.append("".join(temp))

                temp_l, temp_cnt, temp = 0, 0, []
            
            # 2/当前存的到temp中, continue for-loop
            temp_cnt += 1
            temp_l += len(w)
            temp.append(w)

        # To add the rest in temp.
        if temp:
            res.append(' '.join(temp).ljust(maxWidth)) # 把字符串用space填充到maxWidth，并且左对齐
        
        return res

# 839. Similar String Groups 算是经典的union find题目。
# 如果是dfs的思路：针对每个str，进入dfs去看其他str是否类似，这里灵活运用visited是关键，在main中没有visited过，才会initiate dfs。在dfs中我们会及时更新visited。
# 针对每一个dfs，它的所有情况都会探究，因此不用担心有些不会放问道。
class Solution:
    def numSimilarGroups(self, strs: List[str]) -> int:
        parent = [i for i in range(len(strs))]

        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

            
        def union(i1, i2):
            r1, r2 = find(i1), find(i2)
            if r1 != r2 and self.isSameGroup(strs[i1], strs[i2]):
                parent[r1] = r2                
            

        for i in range(len(strs)):
            for j in range(i+1, len(strs)):
                
                union(i, j)
        print(parent)
        return sum(i == parent[i] for i in range(len(parent)))


    def isSameGroup(self, s1, s2):
        return sum(c1 != c2 for c1,c2 in zip(s1,s2)) <= 2


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
        
# 359 简单题 跳过
class Logger:
    def __init__(self):
        self.record = dict()
        
    def shouldPrintMessage(self, timestamp: int, message: str) -> bool:
        if message not in self.record:
            self.record[message] = timestamp
            return True
        else:
            if timestamp - self.record[message] < 10: return False
            self.record[message] = timestamp
            return True

# 1610. Maximum Number of Visible Points
# 这道题可以跳过，需要你理解数学知识，相当于高考18题，用算法写出来。
class Solution:
    def visiblePoints(self, points: List[List[int]], angle: int, location: List[int]) -> int:
        arr, extra = [], 0
        xx, yy = location
        
        for x, y in points:
            if x == xx and y == yy:
                extra += 1 # point与location重合，一定能观察到。
                continue
            arr.append(math.atan2(y - yy, x - xx)) # 将所有的point的弧度计算出来。
        
        arr.sort()
        arr = arr + [x + 2.0 * math.pi for x in arr] # 这里是为了避免跳过了一些case。比如5度和355度，会被跳过。
        angle = math.pi * angle / 180 # angle要转化为弧度，是因为那些API返回的值的unit是弧度。
        
        # 利用滑动窗口
        l = ans = 0
        for r in range(len(arr)):
            while arr[r] - arr[l] > angle:
                l += 1
            ans = max(ans, r - l + 1)
            
        return ans + extra
    


# 2101
# ❌我犯的错：
    # 1. 我利用参数进行全局传值，不适合当前分支求和的情况，而是可以探究最大深度。
    # 2. 这不是回溯，因此不需要在每一个recursion中add/remove当前node。这样会增加重复计算，因为a-b, b-c，但是返回到a的时候a-c又会计算一遍。
    # 3. 这一题要遍历每一个root，why? -> 因为炸弹彼此引爆也是有方向性的，a->b,但是b不能引爆a
class Solution:
    def maximumDetonation(self, bombs: List[List[int]]) -> int:
        graph = collections.defaultdict(list)
        n = len(bombs)
        
        # Build the graph
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue         
                xi, yi, ri = bombs[i]
                xj, yj, _ = bombs[j]

                # Create a path from node i to node j, if bomb i detonates bomb j.
                if ri ** 2 >= (xi - xj) ** 2 + (yi - yj) ** 2:
                    graph[i].append(j)

        # DFS to get the number of nodes reachable from a given node cur
        def dfs(cur, visited):
            visited.add(cur)
            for neib in graph[cur]:
                if neib not in visited:
                    dfs(neib, visited)
            return len(visited)
        
        answer = 0
        for i in range(n):
            visited = set()
            answer = max(answer, dfs(i, visited))
        
        return answer
    
# 关于二分的小总结 - 一定要明白你找的是什么。
# <    
    # 一般用于寻找第一个满足条件的值
    # 适用的场景更多
# <= 
    # 一般用于寻找某个特定的值
    # 容易遇到无限循环的问题。
    # left, right最后不一定能找到值，需要进行判断

# 528. Random Pick with Weight
# 这一题的精华在于如何能够实现按照weight的权重，随机选取值。
# -> 我们利用prefix，这样n个值，每两个值之间的prefix不一样，就看作total_sum的相对应的权重。
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
        

class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        rows, cols = len(grid), len(grid[0])
        target = (rows-1, cols-1)

        if k >= rows + cols - 2: # -2是因为rows和cols会有一格子重复，因此无论在横向/纵向都是走rows-1, cols-1
            return rows+cols-2
        
        state = (0, 0, k)
        queue = collections.deque([(0, state)]) # (step, state) 注意这里的操作，首先deque是一个[]，然后里面每一项是(step,state)
        seen = set([state])

        while queue:
            step, (row, col, k) = queue.popleft()
            if (row, col) == target: return step

            for nr, nc in [(row+1,col),(row,col+1),(row-1,col),(row,col-1)]:
                if 0 <= nr < rows and 0 <= nc < cols:
                    nk = k - grid[nr][nc]
                    nstate = (nr, nc, nk)
                    if nstate not in seen and nk >= 0: 
                        seen.add(nstate)
                        queue.append((step+1, nstate))
        return -1


# 84. Largest Rectangle in Histogram
class Solution:
    # 精华：如何利用Monotonic Stack找到左右边界是这一道题的精华，i-1是右边界，因为它将是stack中的最大值，找面积也是从右向左找的，而非直觉上的向两端延展。
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = [-1]
        max_area = 0
        # 单调递增栈 
        for i in range(len(heights)):
            while stack[-1] != -1 and heights[stack[-1]] >= heights[i]:
                current_height = heights[stack.pop()] # 如果遇到小的height，就利用之前最大的
                current_width = i - stack[-1] - 1 # 右边界就是i-1 左边界stack中cur_height的左边，没有关系，因为cur_height进入stack为了满足单调递增，会把比它自己本身要大的都会pop出来，因此一定会满足rectangle的要求。
                max_area = max(max_area, current_height * current_width)
            stack.append(i)

        # 如果遍历完了，此时我们的右边界将是len-1
        while stack[-1] != -1:
            current_height = heights[stack.pop()]
            current_width = len(heights) - stack[-1] - 1
            max_area = max(max_area, current_height * current_width)
        return max_area
# 如果这一题要用stack=[]，不利用stack=[-1]帮助解决左边界的话，可以用下面两行代码替代：
# left_boundary = -1 if not stack else stack[-1]
# current_width = len(heights) - left_boundary - 1


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
       
            # 加的本身的；也就意味着每个cell我们只会经历这行代码一次。
            # 如果不是第一次visit呢？会直接return visited[x][y]
            # 那如果这个cell是如何能够拥有2以上的值的呢？在
            visited[x][y] += 1 # 这里有点意思哦～
            return visited[x][y]    

        ans = 0
        for i in range(m):
            for j in range(n):
                ans = max(ans, dfs(i, j))

        return ans



# 715 Range Module
class Node:
    __slots__ = ['left', 'right', 'add', 'v']

    def __init__(self):
        self.left = None
        self.right = None
        self.add = 0
        self.v = False


class SegmentTree:
    __slots__ = ['root']

    def __init__(self):
        self.root = Node()

    def modify(self, left, right, v, l=1, r=int(1e9), node=None):
        if node is None:
            node = self.root
        if l >= left and r <= right:
            if v == 1:
                node.add = 1
                node.v = True
            else:
                node.add = -1
                node.v = False
            return
        self.pushdown(node)
        mid = (l + r) >> 1
        if left <= mid:
            self.modify(left, right, v, l, mid, node.left)
        if right > mid:
            self.modify(left, right, v, mid + 1, r, node.right)
        self.pushup(node)

    def query(self, left, right, l=1, r=int(1e9), node=None):
        if node is None:
            node = self.root
        if l >= left and r <= right:
            return node.v
        self.pushdown(node)
        mid = (l + r) >> 1
        v = True
        if left <= mid:
            v = v and self.query(left, right, l, mid, node.left)
        if right > mid:
            v = v and self.query(left, right, mid + 1, r, node.right)
        return v

    def pushup(self, node):
        node.v = bool(node.left and node.left.v and node.right and node.right.v)

    def pushdown(self, node):
        if node.left is None:
            node.left = Node()
        if node.right is None:
            node.right = Node()
        if node.add:
            node.left.add = node.right.add = node.add
            node.left.v = node.add == 1
            node.right.v = node.add == 1
            node.add = 0


class RangeModule: 
    def __init__(self):
        self.tree = SegmentTree()

    def addRange(self, left: int, right: int) -> None:
        self.tree.modify(left, right - 1, 1)

    def queryRange(self, left: int, right: int) -> bool:
        return self.tree.query(left, right - 1)

    def removeRange(self, left: int, right: int) -> None:
        self.tree.modify(left, right - 1, -1)

# 1146 1146. Snapshot Array
import bisect
class SnapshotArray:
    def __init__(self, length: int):
        self.id = 0
        self.history_records = [[[0, 0]] for _ in range(length)]
        
    def set(self, index: int, val: int) -> None:
        self.history_records[index].append([self.id, val])

    def snap(self) -> int:
        self.id += 1
        return self.id - 1

    def get(self, index: int, snap_id: int) -> int:
        snap_index = bisect.bisect_right(self.history_records[index], [snap_id, 10 ** 9]) # 这种排序技巧在二分很重要！
        return self.history_records[index][snap_index - 1][1]


# 818. Race Car
# 这是medium的解法。情况3是hard的tip
class Solution:
    def racecar(self, target: int) -> int:
        #1. Initialize double ended queue as 0 moves, 0 position, +1 velocity
        queue = collections.deque([(0, 0, 1)])
        while queue:
            # (moves) moves, (pos) position, (vel) velocity)
            moves, pos, vel = queue.popleft()

            if pos == target:
                return moves
            
            #2. Always consider moving the car in the direction it is already going
            queue.append((moves + 1, pos + vel, 2 * vel))
            
            #3. Also consider changing direction only when next move will driving away the target.
            if (pos + vel > target and vel > 0) or (pos + vel < target and vel < 0):
                queue.append((moves + 1, pos, -vel / abs(vel)))



# 729
"""
class MyCalendar {
private:
    set<pair<int, int>> calendar; // set类似python，但有序。
public:
    MyCalendar() {
    }
    
    // 每一次都会比较两个边界。
    bool book(int start, int end) {
        const pair<int, int> event{start, end};
        
        const auto nextEvent = calendar.lower_bound(event); // 第一个不小于event的

        // begin()/end()获得都是迭代器，指向元素的。
        // nextEvent是不小于的
        // 如果是没有的话，那么会跳过；lower_bound返回的也是指向元素的迭代器。
        // 如果event太大，会返回end；不等于end意味着 -> event一定在之前。
            // 在之前分两种情况：一种在范围内，一种小于范围
            // 范围内：指向第一个>=event的，只用两者不重叠就行。因为先根据first排列，因此nextEvent一定>=event，因此，我们只用比较event.second和next的first就行。
        if(nextEvent != calendar.end() && nextEvent->first < end) {
            return false;
        }

        if(nextEvent != calendar.begin()) {
            
            const auto preEvent = prev(nextEvent);
            if(preEvent->second > start) {
                return false;
            }
        }

        calendar.insert(event);
          
        return true;
    }
};


// 729. My Calendar I

class MyCalendar {

    public MyCalendar() {
    }
    
    public boolean book(int start, int end) {
        if (query(root, 0, N, start, end - 1) != 0) return false;
        update(root, 0, N, start, end - 1, 1);
        return true;
    }
    // *************** 下面是模版 ***************
    class Node {
        Node left, right;
        // 当前节点值，以及懒惰标记的值
        int val, add;
    }
    private int N = (int) 1e9;
    private Node root = new Node();
    public void update(Node node, int start, int end, int l_boundary, int r_boundary, int val) {
        if (l_boundary <= start && end <= r_boundary) {
            // val和add的值都很灵活，只要不是0.
            node.val += val;
            node.add += val;
            return ;
        }
        pushDown(node);
        int mid = (start + end) >> 1;
        if (l_boundary <= mid) update(node.left, start, mid, l_boundary, r_boundary, val);
        if (r_boundary > mid) update(node.right, mid + 1, end, l_boundary, r_boundary, val);
        pushUp(node);
    }
    public int query(Node node, int start, int end, int l_boundary, int r_boundary) {
        if (l_boundary <= start && end <= r_boundary) return node.val;
        pushDown(node);
        int mid = (start + end) >> 1, ans = 0;
        if (l_boundary <= mid) ans = query(node.left, start, mid, l_boundary, r_boundary);
        if (r_boundary > mid) ans = Math.max(ans, query(node.right, mid + 1, end, l_boundary, r_boundary));
        return ans;
    }
    private void pushUp(Node node) {
        // push其实也是where存放你的节点逻辑的，可以是区间和，可以是最大值，也可以是是否booked.
        // 每个节点存的是当前区间的最大值 
        node.val = Math.max(node.left.val, node.right.val);
    }
    private void pushDown(Node node) {
        // 无论query还是update/modify，都会pushdown更新。
        // 线段树是只有查询和更新两个操作，如果碰到细分的区间，就会pushDown
        // add的值可以
        if (node.left == null) node.left = new Node();
        if (node.right == null) node.right = new Node();
        if (node.add == 0) return ;
        node.left.val += node.add;
        node.right.val += node.add;
        node.left.add += node.add;
        node.right.add += node.add;
        node.add = 0;
    }
}
"""

# 539 - 这题纯烦
class Solution:
    def findMinDifference(self, timePoints: List[str]) -> int:
        timePoints.sort()
        ans = float('inf')
        for i in range(len(timePoints)-1):
            ans = min(ans, self.cal(timePoints[i], timePoints[i+1]))


        t1 = timePoints[-1]
        t2_first, t2_second = timePoints[0].split(":")
        t2 = str(int(t2_first) +  24) + ":" +t2_second
        
        ans = min(ans, self.cal(t1, t2))
        return ans


    def cal(self, t1, t2) -> int :
        t1_h, t1_m = map(lambda x: int(x),t1.split(':'))
        t2_h, t2_m = map(lambda x: int(x),t2.split(':'))
        diff = (t2_h-t1_h)*60 + t2_m-t1_m
        print(diff)
        return diff


"""
// 419. battleships in a board
// 这道题的难点在于如何判断战舰。
// 1.我们只用搜索战舰的开头。因为它是垂直/水平排列的。但是如何找到开头是很困难的。
// 2.首先判断左边和右边有没有‘x’，有的话就不是开头，可以直接跳过。
// 3.题目中的战舰一定是valid的，因此只会有横纵，因此只用找开头就行了。
// 4.跳过前两个if，意味着x要么是新的，要么就是首行。
class Solution {
    public int countBattleships(char[][] board) {
        int m = board.length, n = board[0].length;
        int ans = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i > 0 && board[i-1][j] == 'X') continue;
                if (j > 0 && board[i][j-1] == 'X') continue;
                if (board[i][j] == 'X') ans++;
            }
        }
        return ans;
    }
}"""

# 489. Robot Room Cleaner

class Solution:
    def cleanRoom(self, robot):
        
        # 退回到上一个格子with same direction
        def go_back(): 
            robot.turnRight()
            robot.turnRight()
            robot.move()
            robot.turnRight()
            robot.turnRight()
        
        def backtrack(cell=(0,0), d=0):
            visited.add(cell)
            robot.clean()

            for i in range(4):
                new_d = (d+i) % 4
                new_cell = (cell[0] + direcs[new_d][0], cell[1] + direcs[new_d][1])
                if not new_cell in visited and robot.move():
                    backtrack(new_cell, new_d)
                    go_back()
                
                robot.turnRight()
        # 涉及到方向的话, turnRight, 方向遍历也要根据clockwise.
        direcs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        visited = set()
        backtrack()


# 778 - Swim in the rising water
class Solution:
    def swimInWater(self, grid: List[List[int]]) -> int:
        # level, x, y
        hp = [[grid[0][0],0,0]]
        ans = grid[0][0]
        visited = set((0,0))

        while hp:
            level, x, y = heapq.heappop(hp)
            ans = max(ans, level)
            if x == len(grid)-1 and y == len(grid[0])-1: break
            for nx, ny in ((x+1,y),(x,y+1),(x-1,y),(x,y-1)):
                if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and (nx,ny) not in visited:
                    heapq.heappush(hp, [grid[nx][ny],nx,ny])
                    visited.add((nx,ny))
        return ans
        


        
    

# 2096. Step-By-Step Directions From a Binary Tree Node to Another
# 明眼一看前序遍历；binaryTree，没有什么特殊的结构；
# 肯定需要有signal表示是否找到。
# 找到nearest parent root；然后左右开找；✅
class Solution:
    def getDirections(self, root: Optional[TreeNode], startValue: int, destValue: int) -> str:
        self.node = root
        start_path, dest_path = [],[]
        self.findIntersectNode(root, startValue, destValue)
        self.getPath(self.node, startValue, start_path) 
        print(start_path)
        print(dest_path)
        self.getPath(self.node, destValue, dest_path)

        print(f'before sp: ${start_path}')
        start_path = "U" * len(start_path)
        print(f'after sp: ${start_path}')
        print(f'before ep: ${dest_path}')
        dest_path = "".join(dest_path[::-1])
        print(f'after ep: ${dest_path}')
        return start_path+dest_path

    def getPath(self, node, val, path):
        if not node: return False 
        if node.val == val: return True

        if self.getPath(node.left, val, path):
            path.append('L')
            return True
        if self.getPath(node.right, val, path):
            path.append('R')
            return True        
        return False


    def findIntersectNode(self, cur, v1, v2):
        if not cur: return 0
        left = self.findIntersectNode(cur.left, v1, v2)
        right = self.findIntersectNode(cur.right, v1, v2)
        mid = cur.val == v1 or cur.val == v2
        tt = left+right+mid
        if tt == 2: self.node = cur
        return 1 if tt == 1 else 0 

""" Take away:
1. when searching lowest common ancestor, you need the signal, and outter var to record the current node. what can be a signal? you need to see 3 factors->left, right, cur(mid). return 1 if tt ==1 else 0 can help us to avoid re-updateing in parent-series roots.
2. when getting path, actually DFS is used here. you also need a signal to determine if outter var should be updated. There if dfs(): return true will be a usual solution.
"""
        

# 1101. The Earliest Moment When Everyone Become Friends

class Solution:
    def earliestAcq(self, logs: List[List[int]], n: int) -> int:
        
        friends = list(range(n))
        # ✨这个方法可以看是否所有元素都已经被遍历，并且都已经归为一组！
        seen_num = n
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

            if seen_num == 1:
                return t

        return -1



# 2158. Amount of New Area Painted Each Day
from sortedcontainers import SortedList
# AVL/Red-Black Tree  -> auto-balanced tree

class Solution:
    def amountPainted(self, paint: List[List[int]]) -> List[int]:
        records = []
        max_pos = 0

        for i, [start,end] in enumerate(paint):
            # use 1/-1 to distinguish type
            records.append((start, i, 1))   
            records.append((end, i, -1))
            max_pos = max(max_pos, end) # max_pos是右侧的最远端。


        # records里放的是什么？-> 起点/终点 
        records.sort()


        # sweep across all position
        ans = [0] * len(paint) 
        indexes = SortedList() # same as set() in C++ 存放的是index，按照index大小自动排序。
        i = 0
        
        # 每个for循环看每一个格子。
        for pos in range(max_pos+1):
            
            # 我们的records有几个特性：首先是有序的，毕竟sort过了，是按照节点的先后顺序。
            # 结合records[i][0] == pos 可以推导出 -> while的逻辑只会适用于当前pos存在于records中的，也就是有节点的，有可能0～n个节点，都会进行处理。。
            while i < len(records) and records[i][0] == pos:
                pos, index, tp = records[i]
                # indexes里面存的是所有在当前pos作用的paint的startPoint，但是indexes[0]是第一个，也就是唯一valid的，也就是当前这个pos最终算到indexes头上。
                if tp == 1:
                    indexes.add(index)
                else:
                    indexes.remove(index)
                i += 1

            # indexes[0]就是valid paint的index
            if indexes:
                ans[indexes[0]] += 1

        return ans

        
# 2158. Amount of New Area Painted Each Day
from sortedcontainers import SortedList
# AVL/Red-Black Tree  -> auto-balanced tree

class Solution:
    def amountPainted(self, paint: List[List[int]]) -> List[int]:
        records = []
        

        for i, [start,end] in enumerate(paint):
            # use 1/-1 to distinguish type
            records.append((start, i, 1))   
            records.append((end, i, -1))

        # records里放的是什么？-> 起点/终点 
        records.sort()


        # sweep across all position
        ans = [0] * len(paint) 
        indexes = SortedList() # same as set() in C++ 存放的是index，按照index大小自动排序。
        last_pos = 0
        
        # for循环看records
        for pos, index, tp in records:
            if indexes:
                ans[indexes[0]] += pos-last_pos
            
            last_pos = pos
            if tp == 1:
                indexes.add(index)
            else:
                indexes.remove(index)

        return ans
    
class SegmentTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (4 * size)
        self.lazy = [0] * (4 * size)

    def update_range(self, v, tl, tr, l, r, addend):
        if self.lazy[v] != 0:
            self.tree[v] += self.lazy[v] * (tr - tl + 1)
            if tl != tr:
                self.lazy[v * 2] += self.lazy[v]
                self.lazy[v * 2 + 1] += self.lazy[v]
            self.lazy[v] = 0

        if l > r:
            return

        if l == tl and r == tr:
            self.tree[v] += addend * (tr - tl + 1)
            if tl != tr:
                self.lazy[v * 2] += addend
                self.lazy[v * 2 + 1] += addend
            return

        tm = (tl + tr) // 2
        self.update_range(v * 2, tl, tm, l, min(r, tm), addend)
        self.update_range(v * 2 + 1, tm + 1, tr, max(l, tm + 1), r, addend)
        self.tree[v] = self.tree[v * 2] + self.tree[v * 2 + 1]

    def query_range(self, v, tl, tr, l, r):
        if l > r:
            return 0

        if self.lazy[v] != 0:
            self.tree[v] += self.lazy[v] * (tr - tl + 1)
            if tl != tr:
                self.lazy[v * 2] += self.lazy[v]
                self.lazy[v * 2 + 1] += self.lazy[v]
            self.lazy[v] = 0

        if l == tl and r == tr:
            return self.tree[v]

        tm = (tl + tr) // 2
        return self.query_range(v * 2, tl, tm, l, min(r, tm)) + self.query_range(v * 2 + 1, tm + 1, tr, max(l, tm + 1), r)

class Solution:
    def amountPainted(self, paint):
        MAX_SIZE = 50005
        seg_tree = SegmentTree(MAX_SIZE)
        ans = []

        for start, end in paint:
            end -= 1  # Adjust to 0-indexed
            painted = end - start + 1 - seg_tree.query_range(1, 0, MAX_SIZE - 1, start, end)
            ans.append(painted)
            seg_tree.update_range(1, 0, MAX_SIZE - 1, start, end, 1)

        return ans


# 2172. Maximum AND Sum of Array
# 第一个循环是用来确定和遍历所有可能的状态，而第二个循环是用来进行状态转移，即考虑如何从当前状态通过放置一个新的数字到达新状态，并计算这种转移所能获得的最大 AND 和。这两个循环共同构成了解决这个动态规划问题的完整框架。
class Solution:
    def maximumANDSum(self, nums: List[int], numSlots: int) -> int:    
        f = [0] * (1 << (numSlots * 2))
        # 这里的i有什么作用？它的取值范围是0~2**(numSlots*2)，因此每个bin(i)都可以代表了一种状态
        for i, fi in enumerate(f):
            c = i.bit_count() # 这里的c是看bin(i)里有多少个slot被占用了，有元素了。
            # 如果上面的c也可以当index，表示已经有多少个元素被放入了。
            if c >= len(nums): continue # 如果占用的格子超过我们最大元素，就没必要继续了。

            # 遍历所有slot
            for j in range(numSlots * 2):
                # 遍历j这个slot在状态i下是否为空；如果为空，就是可以塞进去。
                if (i & (1 << j)) == 0: 
                    s = i | (1 << j) # new一个新的state出来，就是在原来状态i上，将j位的也改为1 -> 表示新状态。
                    f[s] = max(f[s], fi + ((j // 2 + 1) & nums[c])) #
        return max(f)


# 这题的逻辑和难点需要值得讲讲：
# 0. 题意是最多两个放在一组，但是每个元素最后都是和当前组的index进行AND运算。因此其实可以看作是在一个单调坐标轴上Insert
# 1. 首先f，其len == 选0～选所有数字的所有状态的可能性。
# 2. f的index(i)翻译成bin()可以当作当前状态，1为被占用了，0为被占用；
# 3. len(i)是所有槽，c是当前状态i下的1的数量 == 已经放了多少元素，nums[c]就是我们要放的下一个元素，当我们选择哪个number放入我们的考量的时候，参考标准是我们当前放入了几个元素，这些元素就像是stack堆叠在一起的。



# 2115
class Solution:
    def findAllRecipes(self, recipes: List[str], ingredients: List[List[str]], supplies: List[str]) -> List[str]:
        
        records = collections.defaultdict(set)
        inDegree = collections.defaultdict(int)
        supplies = set(supplies)

        for i in range(len(recipes)):
            inputs, output = ingredients[i], recipes[i]
            for single_input in inputs:
                if single_input not in supplies:
                    inDegree[output] += 1
                    records[single_input].add(output)

        queue = []
        res = []
        for r in recipes: 
            if not inDegree[r]:
                queue.append(r)
            
        
        while queue:
            cur = queue.pop(0)
            res.append(cur)
            for nex in records[cur]:
                inDegree[nex] -= 1
                if inDegree[nex] == 0:
                    queue.append(nex)

        return res

# 2034.Stock Price Fluctuation 
# 跟我的大体思路差不多，是需要用到heap的，那如何确保heap中的max/min极值是up to date的？ -> 只需要在pop的时候与最近储存的hashmap检查就可以了。
class StockPrice:
    def __init__(self):
        self.latest_time = 0
        # Store price of each stock at each timestamp.
        self.timestamp_price_map = {}
        
        # Store stock prices in sorted order to get min and max price.
        self.max_heap = []
        self.min_heap = []

    def update(self, timestamp: int, price: int) -> None:
        # Update latestTime to latest timestamp.
        self.timestamp_price_map[timestamp] = price
        self.latest_time = max(self.latest_time, timestamp)

        # Add latest price for timestamp.
        heappush(self.min_heap, (price, timestamp))
        heappush(self.max_heap, (-price, timestamp))

    def current(self) -> int:
        # Return latest price of the stock.
        return self.timestamp_price_map[self.latest_time]

    def maximum(self) -> int:
        price, timestamp = self.max_heap[0]

        # Pop pairs from heap with the price doesn't match with hashmap.
        while -price != self.timestamp_price_map[timestamp]:
            heappop(self.max_heap)
            price, timestamp = self.max_heap[0]
            
        return -price

    def minimum(self) -> int:
        price, timestamp = self.min_heap[0]

        # Pop pairs from heap with the price doesn't match with hashmap.
        while price != self.timestamp_price_map[timestamp]:
            heappop(self.min_heap)
            price, timestamp = self.min_heap[0]
            
        return price




# 833. Find And Replace in String
class Solution:
    # # 1. find the valid sources
    # # 2. exclude invalid targets
    # # 3. transform


    def findReplaceString(self, S, indexes, sources, targets):
        for i, s, t in sorted(zip(indexes, sources, targets), reverse=True):
            S = S[:i] + t + S[i + len(s):] if S[i:i + len(s)] == s else S
            # 用法解读：
            # 1. 如果没有满足if->其实就是if S==S: 就是跳过了。
            # 2. 如果满足if -> 把s[i:i+len(s)]更换掉
            # 3. 倒序Reverse避免了因为替换造成的index影响。
        return S

# 但是这种方法没有处理overlap
    
    

# 792. Number of Matching Subsequences
# 只会暴力解（Not accepted）
# Next pointer:  🌟这种方法我第一次见，有点类似OS的多线程的shared var用法。
#   1. 因为s太大了，所以只要遍历它一次就好
class Solution:
    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        ans = 0
        heads = [[] for _ in range(26)]
        for word in words:
            it = iter(word)
            heads[ord(next(it)) - ord('a')].append(it) # it是迭代器，这一行的目的是将迭代器添加到每一个首字母的位置。

        for letter in s:
            # 当前letter的index
            letter_index = ord(letter) - ord('a')
            old_bucket = heads[letter_index] # 本质上是list，或者里面有没有iterator 
            heads[letter_index] = [] # 并且清空。

            while old_bucket: # 如果当前有可能的字符串的话，我们来一个个看。
                it = old_bucket.pop() 
                nxt = next(it, None)
                # 如果有的话nxt的话，我们抵消了当前的letter把剩下的继续放入heads中
                if nxt: 
                    heads[ord(nxt) - ord('a')].append(it)
                else:
                    # 如果没有nxt意味着该word序列已经全部消除了。可以答案+1了。
                    ans += 1

        return ans
    

# 562. Longest Line of Consecutive One in Matrix
# 这题的dp还是挺简单的，3D-array解题，每个特定的index照顾了一种情况。
# 也不用担心各个情况的互相影响。
class Solution:
    def longestLine(self, mat: List[List[int]]) -> int:
        dp = [[[0,0,0,0] for _ in range(len(mat[0]))] for i in range(len(mat))]
        max_ones = 0
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                if mat[i][j] == 1:
                    dp[i][j][0] = 1 + (dp[i][j-1][0] if j > 0 else 0)  # 水平
                    dp[i][j][1] = 1 + (dp[i-1][j][1] if i > 0 else 0)  # 垂直
                    dp[i][j][2] = 1 + (dp[i-1][j-1][2] if i > 0 and j > 0 else 0)  # 对角线
                    dp[i][j][3] = 1 + (dp[i-1][j+1][3] if i > 0 and j < len(mat[0]) - 1 else 0)  # 反对角线
                    max_ones = max(max_ones, dp[i][j][0], dp[i][j][1], dp[i][j][2], dp[i][j][3])

        return max_ones


# 1606 
# 2162 - 没意思
    
    


# 2421
# 552
# 1105
# 1937

# 1048
# 900
# 1996
# 366
# 1387

# 2242
# 2013
# 1554
# 2135
# 1055
# 418

# 2416
# 2018
# 2128
# 2178
# 843

# 332
# 2345
# 1857
# 2313
# 2104

# 2277
# 581
# 2254
# 33
# 2459
# 946

# 2510
# 1020
# 1254
# 2371
# 13
# 4

# 394
# 875
# 759
# 402
# 1360

# 929
# 975
# 482
# 904
# 3

# 11
# 15
# 31
# 43
# 48
# 55

# 66
# 76
# 158
# 159
# 163
# 681

# 809
# 849
# 42
# 215
# 844
# 857

# 973
# 2
# 138
# 127
# 210

# 222
# 399
# 2829
# 753
# 947

# 951
# 425
# 247
# 351
# 17
# 22

# 34
# 315
# 852
# 5
# 152

# 322
# 518
# 410
# 146
# 155

# 297
# 380
# 642
# 7
# 135
# 205

# 246
# 299
# 308
# 731
# 771
# 939







       

