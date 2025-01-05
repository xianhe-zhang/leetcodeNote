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

# 1606. Find Servers That Handled Most Number of Requests
# k servers -> can only handle one(no concurrent)
from sortedcontainers import SortedList
class Solution:
    def busiestServers(self, k: int, arrival: List[int], load: List[int]) -> List[int]:
        count = [0] * k

        busy, free = [], SortedList(list(range(k))) # 这个写法我写不出来的主要原因是不清楚这个sortedList这个数据结构。

        for i, start in enumerate(arrival):

            # 🌟在去决定选择哪个server的时候，先根据current条件把可以选的再次放进来。
            # busy是sortedList所以可以这么用。
            while busy and busy[0][0] <= start:
                _, server_id = heapq.heappop(busy)
                free.add(server_id)

            if free:
                index = free.bisect_left(i%k) # 应该找i%k这个index，如果有的话
                busy_id = free[index] if index < len(free) else free[0] # 如果<len()意味着当前有发现大于index的server。
                free.remove(busy_id)
                heapq.heappush(busy, ((start + load[i]), busy_id))
                count[busy_id] += 1
        max_job = max(count)
        return [i for i ,n in enumerate(count) if n == max_job]

# biesct_left, bisect_right是处理的插入值的边界。
# 下面是如何使用两个heap的方法。priority queue
class Solution:
    def busiestServers(self, k: int, arrival: List[int], load: List[int]) -> List[int]:
        count = [0] * k
        
        busy, free = [], list(range(k))

        for i, start in enumerate(arrival):
            # 一样的，一个pq中存放所有busy的，一定不满足当亲啊的
            while busy and busy[0][0] <= start:
                _, server_id = heapq.heappop(busy)
                # 用两个pq的难点在于如何通过数学的方法找到next available的server_id
                # 
                heapq.heappush(free, i + (server_id - i) % k)

            if free:
                busy_id = heapq.heappop(free) % k
                heapq.heappush(busy, (start + load[i], busy_id))
                count[busy_id] += 1
        
        max_job = max(count)
        return [i for i, n in enumerate(count) if n == max_job]
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
        

# 2162 - 没意思



# 2421
'''
class UnionFind {
private:
    vector<int> parent, rank;
public:
    UnionFind(int size) {
        parent.resize(size);
        rank.resize(size, 0);
        for(int i = 0; i < size; i++) {
            parent[i] = i;
        }
    }

    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }

    void union_set(int x, int y) {
        int xset = find(x), yset = find(y);
        if (xset == yset) {
            return;
        } else if (rank[xset] < rank[yset]) {
            parent[xset] = yset;
        } else if (rank[xset] > rank[yset]) {
            parent[yset] = xset;
        } else {
            parent[yset] = xset;
            rank[xset]++;
        } 
    }
};

class Solution {
// 解题思路：按照val从小到大的去对node进行unionFind. 这样每次去遍历的时候都不会遇到更大的node，也就是满足了goodpath的要求。
public:
    int numberOfGoodPaths(vector<int>& vals, vector<vector<int>>& edges) {
        int n = vals.size();
        vector<vector<int>> adj(n); // {int: [int]} 存放的是对应的都有哪些node
        for (auto& edge : edges) {
            adj[edge[0]].push_back(edge[1]);
            adj[edge[1]].push_back(edge[0]);
        }


        // 初始化valuesToNodes；每个node有不同的value，这个数据结构是key是value，value存的是node
        map<int, vector<int>> valuesToNodes;
        for (int node = 0; node < n; node++) {
           valuesToNodes[vals[node]].push_back(node);
        }

        // new a UF object
        UnionFind dsu(n);
        int goodPaths = 0;


        for (auto& [value, nodes] : valuesToNodes) { 
            for (int node : nodes) { 
                for (int neighbor : adj[node]) {
                    // 如果当前的node大于其neighbor，就可以合并。可以满足goodPath的requirement
                    if (vals[node] >= vals[neighbor]) {
                        dsu.union_set(node, neighbor);
                    }
                }
            }

            unordered_map<int, int> group;
            // dsu.find(u)是为了找到nodes中每个single node的root；
            // key是root，value是有多少个节点包括自己与这个节点相连。
            for (int u : nodes) {
                group[dsu.find(u)]++;
            }
            for (auto& [_, size] : group) {
                // 求和公式，之所以这么写，是因为group是基于一个唯一的value中的所有nodes的。
                goodPaths += (size * (size + 1) / 2);
            }
        }
        return goodPaths;
    }
};
'''


# 552. Student Attendance Record II
# 这一题的dp有两个点很值得学习：
# 1. 状态机：其实我们的状态只有[2][3]6种情况，然后n+1遍历就行了。我们其实不需要针对A/L/P单独再划分一个纬度；我们可以通过三个并行的if判断当前属于哪种状态，在状态机这种题型中会比较好用。
# 2. for循环和if的statment的配合很赞！要分开想，for循环遍历各个状态，if决定哪个状态下应该进行什么样的状态转换。
class Solution:
    def checkRecord(self, n: int) -> int:
        MOD = 10**9 + 7
        dp = [[[0,0,0] for _ in range(2)] for _ in range(n+1)] # dp[n+1][2][3]
        dp[0][0][0] = 1 
        for i in range(n):
            for j in range(2): # total absent days 1/0
                for k in range(3): # consecutive late days
                    if j == 1 and k == 0: # A - 如果当前为A，那么K肯定为0，不连贯了，J肯定为1；
                        dp[i+1][j][k] += dp[i][j-1][0] % MOD
                        dp[i+1][j][k] += dp[i][j-1][1] % MOD
                        dp[i+1][j][k] += dp[i][j-1][2] % MOD

                    if k != 0: # L - 如果K不为0，当前肯定为L
                        dp[i+1][j][k] += dp[i][j][k-1] % MOD

                    if k == 0: # P - 如果k!=0意味着当前的不为L，而j的取值跟是不是P又没关系，所以可以这么搞。
                        dp[i+1][j][k] += dp[i][j][0]
                        dp[i+1][j][k] += dp[i][j][1]
                        dp[i+1][j][k] += dp[i][j][2]

                     
        ans = 0 
        for j in range(2):
            for k in range(3):
                ans += dp[n][j][k]
                ans %= MOD
        return ans



# 1105. Filling Bookcase Shelves 🌟这道dp也很有意思
    
# 这道题如果是一维DP的很简单，但是有一个trick：什么时候更换新的层？是往下去看？还是往前看？
# 这一题的难点在于如何划分层级？答: 通过向前回溯；没选取一本书的时候，往前累加看看能放最远的是多少？同时记录最高的。 -> 那么dp只需要判断：dp要么不变，之前每个h都去看，要是发现有更小的，就用更小的。
def minHeightShelves(self, books: List[List[int]], shelf_width: int) -> int:
    n = len(books)
    dp = [float('inf')] * (n+1)
    dp[0] = 0
    for i in range(n):
        # 这里省略了取books[i]的操作，直接放进了下面的while-loop中。
        h =0
        j = i
        temp_width = 0
        while j >= 0:
            temp_width += books[j][0]
            if temp_width > shelf_width:
                break
            h = max(h, books[j][1])
            dp[i+1] = min(dp[i+1], dp[j] + h)
            j -= 1

    return dp[-1]


# 1937 Maximum Number of Points with Cost
class Solution:
    def maxPoints(self, ps: List[List[int]]) -> int:
        # ps = points
        m, n = len(ps), len(ps[0])
        if m == 1: return max(ps[0])
        if n == 1: return sum(sum(x) for x in ps)

        def left(arr):
            l = [arr[0]] + [0] * (n-1)
            for i in range(1, n): l[i] = max(l[i-1] - 1, arr[i])
            return l
        def right(arr):
            r = [0] * (n-1) + [arr[-1]] 
            for i in range(n-2, -1, -1): r[i] = max(r[i+1] - 1, arr[i])
            return r
        prev_row = ps[0]
        # 两层for循环需要(M*N)
        for i in range(1, m):
            # 根据每一层的结果，重新计算l,r，然后依据l,r计算下一层的结果。每一次需要2*M 所以复杂度是（M+2M) * N = M*N
            # 如果按照我自己的方法，是O(M*(N^2))，针对每一格还要进行之前的所有计算。
            # 这里优化的方法是通过dp直接将计算简单化，将i位置的左边的最大值和右边的最大值计算出来。利用了额外的空间。
            l, r, cur = left(prev_row), right(prev_row), [0] * n
            for j in range(n):
                cur[j] = ps[i][j] + max(l[j], r[j])    
            prev_row = cur[:]
        return max(prev_row)
# Two levels of dp.


# 1048. Longest String Chain
# 切片器的妙用
class Solution:
    def longestStrChain(self, words):
        dp = collections.defaultdict(int)
        for w in sorted(words, key=len):
            dp[w] = max(dp[w[:i]+w[i+1:]] + 1 for i in range(len(w)))
        return max(dp.values())
        


# 900. RLE Iterator
# 这一题的问题在于会超过Memory Limit.
class RLEIterator:

    def __init__(self, e: List[int]):
        self.records = collections.deque()
        self.cnt = 0
        for i in range(0, len(e), 2):
            t, v = e[i], e[i+1]
            if t == 0: continue
            self.cnt += t

            if self.records and self.records[-2] == v:
                self.records[-1] += t
                
            else:
                self.records += [v, t]
            

    def next(self, n: int) -> int:
        if n > self.cnt:
            self.records = collections.deque()
            return -1
        else:
            self.cnt -= n
            if self.cnt < 0: return -1 # make sure records won't run out.

            # to pick which element to return 
            while n and self.records:
                nex_v, nex_t = self.records[0], self.records[1]    
                if n <= nex_t: # case 1 - 直接不够
                    self.records[1] -= n
                    return nex_v
        
                else: # case 2 - 够的话我们开始下一项
                    self.records.popleft()
                    self.records.popleft()
                    n -= nex_t
                    
            return -1
     
# 1996. The Number of Weak Characters in the Game
# 当有两个属性的关系的时，一定是通过单调性和sort解决的。
class Solution:
    def numberOfWeakCharacters(self, p: List[List[int]]) -> int:
        p.sort(key=lambda x: (-x[0], x[1])) # 精华:将defense按照升序排列，可以避免当attach相同时造成的def_影响
        print(p)
        ans = 0
        maxDef = 0
        # 按照attack从大到小遍历。
        for _, def_ in p:
            if maxDef > def_: #
                ans += 1
            else:
                maxDef = max(maxDef, def_)
        return ans
# 如果用单调栈的话，要保证同attack元素的def是降序的，这样会碰到最大的def，因为针对每一个attack位置，只要前面有比它严格小的，就可以pop出来，然后ans+=1,而同attack下后面的
# def一定比第一个小，因此不会进入判断，也不会更新maxDef
    

# 366 find leaves of binary tree
class Solution:
    def findLeaves(self, root: Optional[TreeNode]) -> List[List[int]]:
        nodes = collections.defaultdict(list)
        def dfs(node):
            if not node: return 0
            left = dfs(node.left)
            right = dfs(node.right)
            level = max(left, right) + 1
            nodes[level].append(node.val)
            return level

        
        dfs(root)
        return list(nodes.values())

# 1387. Sort Integers by The Power Value 想复杂了，不难。
class Solution:
    def getKth(self, lo: int, hi: int, k: int) -> int:
        c=0
        res=[]
        for x in range(lo,hi+1):
            c=0
            temp=x
            while x!=1:
                if x%2==0:
                    x=x//2
                else:
                    x=3*x+1
                c+=1
            res.append([temp,c])
        
        res.sort(key=lambda x: x[1])
        
        ans=res[k-1]
        return ans[0]

class Solution:
    def getKth(self, lo: int, hi: int, k: int) -> int:
        f = {1: 0}

        def getF(x):
            if x in f:
                return f[x]
            f[x] = (getF(x * 3 + 1) if x % 2 == 1 else getF(x // 2)) + 1
            return f[x]
        
        v = list(range(lo, hi + 1))
        v.sort(key=lambda x: (getF(x), x))
        return v[k - 1]


# 2013
class DetectSquares:
    def __init__(self):
        self.points=defaultdict(lambda :defaultdict(int)) # 这是精髓...
    def add(self, point: List[int]) -> None:
        x,y = point
        self.points[y][x]+=1
    def count(self, point: List[int]) -> int:
        X,Y = point
        count = 0
        for x in self.points[Y]:
            d=abs(x-X) # d是边长
            if d==0:continue # 额外情况
            # 因为X,Y已经确定了，而我们另一个水平的点出发可以确定一条水平的边，因此只需要检查上方/下方的square就行了。
            count+=(self.points[Y-d][x]*self.points[Y-d][X]*self.points[Y][x]) # 下方的square
            count+=(self.points[Y+d][x]*self.points[Y+d][X]*self.points[Y][x]) # 上方的square
        return count
            
# 1554. Strings Differ by One Character
# String Hash的用法
def differByOne(self, dict: List[str]) -> bool:
    n, m = len(dict), len(dict[0])
    hashes = [0] * n #存放的是各个位置的hash value；
    MOD = 10**11 + 7
    

    # hashValue <- 2 也是本题算法的核心。有点类似26进制。
    for i in range(n):
        for j in range(m):
            hashes[i] = (26 * hashes[i] + (ord(dict[i][j]) - ord('a'))) % MOD
    

    base = 1
    # for: 按照字符
    for j in range(m - 1, -1, -1):        
        seen = set()
        # for: 去看dict里面每一个string；
        for i in range(n):
            new_h = (hashes[i] - base * (ord(dict[i][j]) - ord('a'))) % MOD
            if new_h in seen:
                return True
            seen.add(new_h)
            # 🌟why works? -> 匹配的逻辑：
            # hashes[i]里永远存的所有字符贡献过后的hash value. sub-for每一次循环做的就是将当前的i的值的贡献从总贡献中减去。然后将这个结果存入seen中。
            # 而且这一题有一很强的前提条件，就是一个字母不同的string，其他都是一样的，因此可以用hashvalue来做。
        base = 26 * base % MOD
    return False        

# 2135. Count Words Obtained After Adding a Letter 这题不难，因为只有一次操作。
def wordCount(self, startWords: List[str], targetWords: List[str]) -> int:
    word_map = {} # 存放有序的key
    for w in startWords:
        key = tuple(sorted(list(w)))
        word_map[key] = word_map.get(key, 0) + 1

    count = 0
    for w in targetWords:
        wl = sorted(list(w))
        for i in range(len(wl)):
            if (tuple(wl[:i]+wl[i+1:])) in word_map:
                count += 1 
                break
    return count


# 1055. Shortest Way to Form String
def shortestWay(self, s: str, t: str) -> int:
    # detect invalid input:
    ss, st = set(list(s)), set(list(t))
    if ss & st != st : return -1

    # to count the minimum number
    n, m = len(s), len(t)
    t_ptr = 0
    def findNext():
        nonlocal t_ptr
        for i in range(n):
            if s[i] == t[t_ptr]:
                t_ptr += 1
                if t_ptr == m: return 
    
    count = 0
    while t_ptr < m:
        findNext()
        count += 1
    return count

# 418. Sentence Screen Fitting
def wordsTyping(self, sentence, rows, cols):

    # Main
    start_ptr = 0 # 用来行进，看来能走多远。这道题怎么manipulate这个ptr是很难的东西。
    # 如何理解这个ptr，在每次开始时，ptr希望指向的是下一行的开始。
    sentence_string = " ".join(sentence) + " " # 这里最后加的空格很重要，因为sentence是要重复地出现在这个grid中，你希望首位中间有空格。如果你要用ptr循环操控指向这个string的话。
    str_len = len(sentence_string)
    # for i in range(rows):
    #     start_ptr += cols 
    #     if sentence_string[start_ptr % str_len] == " ": 
    #         start_ptr += 1
    #     else:
    #         while start_ptr > 0 and sentence_string[(start_ptr - 1) % str_len] != " ": 
    #             start_ptr -= 1
    # return start_ptr // str_len


    # 每指向的是最后一行。
    start_ptr = -1
    for i in range(rows):
        start_ptr += cols 
        if sentence_string[start_ptr % str_len] == " ": 
            continue
    
        elif sentence_string[(start_ptr + 1) % str_len] == " ": 
            start_ptr += 1
        
        else: 
            while start_ptr > 0 and sentence_string[start_ptr % str_len] != " ": 
                start_ptr -= 1
    print(start_ptr)
    print(str_len)
    return (start_ptr+1) // str_len



# 2242. Maximum Score of a Node Sequence
# Fun to think: 因为只有4个node，因此要将中间两个node当作root，其实也就是traverse each edge
class Solution:
    def maximumScore(self, scores: List[int], edges: List[List[int]]) -> int:
        # construct the map
        top_3_nodes = defaultdict(list)

        # construct top_3_nodes {key:current node; values: top 3 nodes with highest score}
        def construct_node_map(x, y, s):
            bisect.insort_left(top_3_nodes[x], [s, y])
            if len(top_3_nodes[x]) > 3:
                top_3_nodes[x].pop(0)


        for x, y in edges:
            construct_node_map(x, y, scores[y])
            construct_node_map(y, x, scores[x])

        ans = -1
        for x, y in edges:
            if len(top_3_nodes[x]) < 2 or len(top_3_nodes[y]) < 2: # 无法满足4个的需求。
                continue
            
            for m in top_3_nodes[x]:
                for n in top_3_nodes[y]:
                    if m[1] not in [x, y] and n[1] not in [x,y]and m[1] != n[1]:
                        ans = max(ans, scores[x]+scores[y]+m[0]+n[0])
        return ans
        

# 2018 Check if Word Can Be Placed In Crossword
class Solution:
    def placeWordInCrossword(self, board: List[List[str]], word: str) -> bool:
        words=[word,word[::-1]] # will contain word and reversed_word
        n=len(word)
        for B in board,zip(*board): # two iterables. The B will take one from each alternately
            for row in B:
                q = ''.join(row).split('#') # KEY: split("#") -> each segment will be considered as a slot for word
                
                # double for-loop is to get every combination
                for w in words:
                    for s in q:
                        if len(s)==n: # if slot len statisfies
                            if all(s[i]==w[i] or s[i]==' ' for i in range(n)): # we need to make sure the pre-placed letter will not have a affect.
                                return True
        return False


# 2416
class TrieNode:
    def __init__(self):
        self.children = [None for _ in range(26)]
        self.is_end = False
        self.count = 0

class Solution:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        current = self.root
        for i in word:
            idx = ord(i) - ord('a')    
            if current.children[idx] == None:       
                current.children[idx] = TrieNode()
            current = current.children[idx] 
            current.count += 1 
        current.is_end = True 

    def search(self, word,ans):   
        current = self.root
        counter = 0
        for i in word:
            idx = ord(i) - ord('a')
            if current.children[idx] == None:   
                return
            current = current.children[idx]
            counter += current.count

        ans.append(counter) 

    def sumPrefixScores(self, words: List[str]) -> List[int]:
        # EASY just refer the 2nd Hint, visulize by incrementing the count of each word on your Trie then again iterate to call search on your trie then add up the count of those letters on the current word
        ans = []
        for word in words:
            self.insert(word)

        for word in words:
            self.search(word,ans)

        return ans
    
# 这个方法我也写出来了，只不过没有那么熟练，我还将Tire Tree的结果转化成了List，结果beyond memory limit.

            

# 2128. Remove All Ones With Row and Column Flips 脑筋急转弯，找规律，一般般。。。
class Solution:
    def removeOnes(self, grid: List[List[int]]) -> bool:
        r1, r1_invert = grid[0], [1-val for val in grid[0]]
        for i in range(1, len(grid)):
            if grid[i] != r1 and grid[i] != r1_invert:
                return False
        return True
# 2178. Maximum Split of Positive Even Integers
# backtrack cannot optimized the process
class Solution:
    # def maximumEvenSplit(self, s: int) -> List[int]:
    #     if s % 2 == 1: return []
    #     res = []

    #     def dfs(residue, start, path):
    #         nonlocal res
    #         if residue == 0 and len(path) > len(res):
    #             res = path[:]

    #         for i in range(start, residue + 1 , 2):
    #             dfs(residue-i, i+2, path+[i])


    #     dfs(s, 2, [])
    #     return res
        
    def maximumEvenSplit(self, f: int) -> List[int]:
        ans, i = [], 2
        if f % 2 == 0:
            while i <= f:
                ans.append(i)
                f -= i
                i += 2
            ans[-1] += f
        return ans
        

        
# 843 Guess the word
# 这一题的难点在于思路，如何narrow down scope.
# 首先，我们从candidate中选出来一个最overlap的单词(most_overlap_word) -> guess会return有几个match的。
# 如果没有找到完全的，返回值为n，假设当前去match的word是x，那么也就意味着x中有n个字母是与最终的secret一致的，也就是说和words中的那个潜在secret中有n个重合
# 因此在下一次的循环中，candidate只用从narrow down后的list中寻找就可以了。探索性的优化算法。
class Solution(object):
    def findSecretWord(self, wordlist, master):
		
        def pair_matches(a, b):         # count the number of matching characters
            return sum(c1 == c2 for c1, c2 in zip(a, b))

        def most_overlap_word():
            # counts[i][j]： i-th index; j是char，value是出现的次数
            counts = [[0 for _ in range(26)] for _ in range(6)]     # counts[i][j] is nb of words with char j at index i
            for word in candidates:
                for i, c in enumerate(word):
                    counts[i][ord(c) - ord("a")] += 1
            # 当前words 某个index上的所有字母的count

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


# 332. Reconstruct Itinerary
# if you can RE-visit a vertice multiple time, then it's not a directed acyclic graph, since there will at least a cycle in the graph
# it's called Eulerian Cycle.
# start / end at the same vertex?
# The main idea consists of two steps: 
# 1.start from any until stuck at certain vertex 
# 2.backtrack and repeat the process until all edges been used.
from collections import defaultdict
class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        flight_map = defaultdict(list)
        for [ori, des] in tickets:
            flight_map[ori].append(des)

        for origin, destinations in flight_map.items():
            # 可能有多张相同的票，倒叙排列有助于我们使用pop，让字母小的先pop出来，然后先进入backtrack的tree，我们是从底部往result中添加的。
            destinations.sort(reverse=True)

        def dfs(origin='JFK'):
            destionations = flight_map[origin]
            while destionations:
                next_dest = destionations.pop()
                dfs(next_dest)
            result.append(origin)

        result = []
        dfs()
        return result[::-1]
        
# 2345  Finding the Number of Visible Mountains
class Solution:
    def visibleMountains(self, peaks: List[List[int]]) -> int:
        c = collections.Counter()    
                          # count frequency for each point
        for (x, y) in peaks:
            c[(x, y)] += 1
        peaks = sorted(c.keys())  
        if not peaks: return 0
       
        def within(pa, pb):                                 # return True if `pb` is within `pa`
            x1, y1 = pa
            x2, y2 = pb 
            b1 = y1 - x1
            b2 = y1 + x1
            return y2 <= x2 + b1 and y2 <= -x2 + b2


        stack = [tuple(peaks[0])]
        for x, y in peaks[1:]:
            # while stack and within([x, y], stack[-1]):
            while stack and within(stack[-1] ,[x, y]):
                stack.pop()
            if not stack or not within(stack[-1], [x, y]):
                stack.append((x, y))
        return len([p for p in stack if c[p] == 1])




# 1857. Largest Color Value in a Directed Graph
# This question has a few highlights:
from collections import deque, defaultdict
class Solution:
    def largestPathValue(self, colors: str, edges: List[List[int]]) -> int:
        n = len(colors)
        adj = defaultdict(list)
        indegree = [0] * n

        for edge in edges:
            adj[edge[0]].append(edge[1])
            indegree[edge[1]] += 1

        count = [[0] * 26 for _ in range(n)]  # H1: each node update, help to store the path color frequency
        q = deque()

        
        for i in range(n):
            if indegree[i] == 0:
                q.append(i)

        answer = 0
        nodes_seen = 0

        while q: # H2: BFS, not dfs to path by path
            node = q.popleft()
            color_index = ord(colors[node]) - ord('a') 
            count[node][color_index] += 1
            answer = max(answer, count[node][color_index]) # H3: update with current change, the change is the only possibility bigger than current answer
            nodes_seen += 1 # H4: each node will appear once. only into q when there is no more dependence(topological sort)

            if node not in adj:
                continue

            for neighbor in adj[node]:
                for i in range(26):
                    # Update the color frequency for the neighbor
                    count[neighbor][i] = max(count[neighbor][i], count[node][i]) # H5: this is how we use `count` data structure

                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    q.append(neighbor)

        return answer if nodes_seen == n else -1



# 2313. Minimum Flips in Binary Tree to Get Result
class Solution:
    def minimumFlips(self, root: Optional[TreeNode], result: bool) -> int:
        def dfs(node=root):
            val = node.val
            if val == 0:
                return 1, 0 
            if val == 1:
                return 0, 1

            if val == 5:
                t, f =  dfs(node.left or node.right)
                return f, t


            lt, lf = dfs(node.left)
            rt, rf = dfs(node.right)
            # OR -> min(lt, rt), lf+rf
            if val == 2:
                return min(lt + rt, lt + rf, lf + rt), lf + rf

            # AND -> lt+rt, min(lf, rf)
            elif val == 3: 
                return lt + rt, min(lt + rf, lf + rt, lf + rf)
            # XOR -> min(lt/rf, lf/rt), min(lt/rt, lf/rf)
            elif val == 4:
                return min(lf + rt, lt + rf), min(lt + rt, lf + rf)
                
            
        t, f = dfs()
        return t if result else f


        
# 2104. Sum of Subarray Ranges
# Approach 1 - two loops - O(n^2)
# Approach 2 - Monotonic stack - O(n)
class Solution:
    def subArrayRanges(self, nums: List[int]) -> int:
        
        n, res = len(nums), 0
        stack = []


        # 单调递增stack
        for right in range(n+1):
            while stack and (right == n or nums[stack[-1]] >= nums[right]): # this condition means the stack top is BOTTOM
                mid = stack.pop()
                left = -1 if not stack else stack[-1]
                res -= nums[mid] * (mid - left) * (right - mid) 
                # mid - left: 以mid结尾的subarray，这里的left是上一个比mid大的index，因此mid-left就可以
                # right - left: 以mid开始的subarray, 这里的right是下一个比mid大的index
                # 两者*，刚好等于当前mid可以覆盖的最小值的所有subarray数量。
            stack.append(right)

        stack.clear()
        for right in range(n+1):
            while stack and (right == n or nums[stack[-1]] <= nums[right]):
                mid = stack.pop()
                left = -1 if not stack else stack[-1]
                res += nums[mid] * (mid - left) * (right - mid) 
            stack.append(right)

        return res


# 2277. Closest Node to Path in Tree
class Solution:
    def closestNode(self, n: int, edges: List[List[int]], query: List[List[int]]) -> List[int]:
        trees = defaultdict(list)
        for x, y in edges:
            trees[x].append(y)
            trees[y].append(x)
        distances = [[float('inf')] * n for _ in range(n)]

        # tree graph === acyclic undirected
        for x in range(n):  # x is the start point
            queue = deque([x])
            distances[x][x] = 0

            while queue:
                cur = queue.popleft()
                for nex in trees[cur]:
                    if distances[x][nex] == float('inf'):
                        distances[x][nex] = distances[x][cur] + 1
                        queue.append(nex)
            
        # Highlight: when trying to find a point from a path closest to a certain point. We are actually looking for a point that has minimal sum of distances from start/ending/target points.
        return [min(range(n), key=lambda x: distances[x][a] + distances[x][b] + distances[x][q]) for a, b, q in query]

# 优化方法，有了approach 1的highlight，那么其实我们在找三个点的lowest common ancestor

# 581. Shortest Unsorted Continuous Subarray
class Solution:
    def findUnsortedSubarray(self, nums: List[int]) -> int:  
        min_val, max_val = float('inf'), float('-inf')
        flag = False
        
        # 不能只找不满足asc顺序的第一个元素。
        # 找到最小的需要排序的元素
        for i in range(1, len(nums)):
            if nums[i] < nums[i - 1]:
                flag = True
            if flag:
                min_val = min(min_val, nums[i])
        
        flag = False
        
        # 找到最大的需要排序的元素
        for i in range(len(nums) - 2, -1, -1):
            if nums[i] > nums[i + 1]:
                flag = True
            if flag:
                max_val = max(max_val, nums[i])
        
        # 找到最左边的和最右边的边界
        l, r = 0, len(nums) - 1
        while l < len(nums) and min_val >= nums[l]:
            l += 1
        while r >= 0 and max_val <= nums[r]:
            r -= 1
        return 0 if r - l < 0 else r - l + 1


# 用sort，用stack都可以。

# 2254. Design Video Sharing Platform
# 不难
class VideoSharingPlatform:
    def __init__(self):
        self.videos = defaultdict(list) # [video_string, views, likes, dislikes]
        self.max_id = 0
        self.free_ids = []

    def upload(self, video: str) -> int:
        if self.free_ids:
            new_id = heapq.heappop(self.free_ids)
        else:
            new_id = self.max_id
            self.max_id += 1
        self.videos[new_id] = [video, 0, 0, 0]
        return new_id

    def remove(self, videoId: int) -> None:
        if videoId in self.videos:
            heapq.heappush(self.free_ids, videoId)
            del self.videos[videoId]
    
    def watch(self, videoId: int, startMinute: int, endMinute: int) -> str:
        if videoId not in self.videos: return '-1'
        cur_video_ptr = self.videos[videoId]
        cur_video_ptr[3] += 1
        return cur_video_ptr[0][startMinute: min(endMinute+1, len(cur_video_ptr[0]))]

    def like(self, videoId: int) -> None:
        if videoId in self.videos:
            self.videos[videoId][1] += 1

    def dislike(self, videoId: int) -> None:
        if videoId in self.videos:
            self.videos[videoId][2] += 1

    def getLikesAndDislikes(self, videoId: int) -> List[int]:
        if videoId not in self.videos: return [-1]
        return [self.videos[videoId][1], self.videos[videoId][2]]

    def getViews(self, videoId: int) -> int:
        if videoId not in self.videos: return -1
        return self.videos[videoId][3]

# 33. Search in Rotated Sorted Array
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
                    r = mid - 1
                else: 
                    l = mid + 1
            else:
                if mid_val < target <= nums[r]:
                    l = mid + 1
                else: 
                    r = mid - 1
       
        # 如果是l<=r, 在最后一遍的循环中，我们其实可以检测当前值，因此如果跳出了，就意味着target不在区间中，因此直接返回-1.
        # return -1
        return r if nums[l] == target else -1


# 2459. Sort Array by Moving Items to Empty Space
class Solution:
    def sortArray(self, nums: List[int]) -> int:
        def process(nums, final_zero_idx):
            res = 0
            idxs = {v:i for i, v in enumerate(nums)}
            process_idx = 1 if final_zero_idx == 0 else 0

            # If zero is at index 3, then we swap zero with the number "3". After swap, the number "3" goes to index "3",
            # which is its final position, and "zero" goes to a new position that could be anywhere
            def swap(i):
                nonlocal res
                num = nums[i]
                zero_idx = idxs[0]
                nums[zero_idx], nums[i] = nums[i], nums[zero_idx]
                idxs[num], idxs[0] = zero_idx, i
                res += 1
            
            offset = 0 if final_zero_idx == 0 else 1 # this is used to add to num 
            while True:
                num = idxs[0]+offset # the number to swap with zero

                if idxs[0] != final_zero_idx:
                    swap(idxs[num])
                
                else: 
                # cannot swap zero if zero is already at final_zero_index
                # swap the first number that isn't on its final position
                    while process_idx < len(nums) and nums[process_idx]==process_idx + offset: # 意味着已经在final position
                        process_idx += 1
                    if process_idx == len(nums)-offset: # 此时已经进行到最后一位需要process的，如果最后一位是0的话，offset就为1.
                        return res
                    swap(idxs[nums[process_idx]])



        return min(process(nums[:], 0), process(nums[:], len(nums)-1))
    
# 946 Validate Stack Sequences
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        stack = list()
        pop_index = 0
        for e in pushed:
            stack.append(e)
            while stack and stack[-1] == popped[pop_index]:
                stack.pop()
                pop_index += 1

        return len(stack) == 0

            
# 2510. Check if There is a Path With Equal Number of 0's And 1's
# the question's core part is to realize: from start to end, there must exist a path, whose sum between min-path-sum and max-path-sum, given that the path sum only change by one at a time.
class Solution:
    def isThereAPath(self, grid: List[List[int]]) -> bool:
        rows, cols = len(grid), len(grid[0])
        if (rows + cols) % 2 == 0:
            return False
        
        min_ = [[0] * cols for _ in range(rows)]
        max_ = [[0] * cols for _ in range(rows)]
        
        min_[0][0] = max_[0][0] = grid[0][0]

        for row in range(1, rows):
            min_[row][0] = min_[row-1][0] + grid[row][0]
            max_[row][0] = max_[row-1][0] + grid[row][0]

        for col in range(1, cols):
            min_[0][col] = min_[0][col-1] + grid[0][col]
            max_[0][col] = max_[0][col-1] + grid[0][col]

        for row in range(1, rows):
            for col in range(1, cols):
                min_[row][col] = min(min_[row-1][col], min_[row][col-1]) + grid[row][col]
                max_[row][col] = max(max_[row-1][col], max_[row][col-1]) + grid[row][col]
            
        target = (rows+cols-1)//2
        return min_[-1][-1] <= target <= max_[-1][-1]

# 1020. Number of Enclaves
# BFS/DFS
class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        def dfs(x, y):
            grid[x][y] = 0
            
            for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 1:
                    dfs(nx, ny)

        m, n = len(grid), len(grid[0])
        for i in range(len(grid)):
            if grid[i][0] == 1: dfs(i, 0)
            if grid[i][n-1] == 1: dfs(i, n-1)
        
        for i in range(1, len(grid[0])-1):
            if grid[0][i] == 1: dfs(0, i)
            if grid[m-1][i] == 1: dfs(m-1, i)
        
        return sum(sum(x) for x in grid)



# 1254. Number of Closed Islands
# 0 -> land;
class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        visit = [[False] * n for _ in range(m)]
        count = 0
        
        for i in range(m):
            for j in range(n):
                # 1. is land
                # 2. not visited or we can modify it to water
                # 3. not boundary & modify connecting land (== sub BFS)
                if grid[i][j] == 0 and not visit[i][j] and self.bfs(i, j, m, n, grid, visit):
                    count += 1
                    
        return count

    def bfs(self, x, y, m, n, grid, visit):
        q = deque([(x, y)])
        visit[x][y] = True
        isClosed = True

        dirx = [0, 1, 0, -1]
        diry = [-1, 0, 1, 0]

        while q:
            x, y = q.popleft()
            
            for i in range(4):
                r, c = x + dirx[i], y + diry[i]
                
                if r < 0 or r >= m or c < 0 or c >= n:
                    # (x, y) is a boundary cell
                    isClosed = False
                elif grid[r][c] == 0 and not visit[r][c]:
                    q.append((r, c))
                    visit[r][c] = True
        
        return isClosed


# 1632. Rank Transform of a Matrix
class Solution:
    def matrixRankTransform(self, matrix: List[List[int]]) -> List[List[int]]:
        m, n = len(matrix), len(matrix[0])

        ##### finding connected parts
        graphs = dict() 
        for i in range(m):
            for j in range(n):
                v = matrix[i][j]
                if v not in graphs:
                    graphs[v] = {}
                
                if i not in graphs[v]:
                    graphs[v][i] = []

                if ~j not in graphs[v]: # graph是2-dimension，第一个存值，第二个存row和col，～表示当前的值是col值。  
                    graphs[v][~j] = []

                graphs[v][i].append(~j)
                graphs[v][~j].append(i)
        
        value2index = {}
        seen = set()
        for i in range(m):
            for j in range(n):
                if (i, j) in seen: 
                    continue
                seen.add((i,j))
                v = matrix[i][j]
                graph = graphs[v]
                # start bfs
                q = [i, ~j]
                rowcols = {i, ~j} # store visited row and col
                while q:
                    node = q.pop(0)
                    for rowcol in graph[node]:
                        if rowcol not in rowcols:
                            q.append(rowcol)
                            rowcols.add(rowcol)

                points = set()
                for rowcol in rowcols:
                    for k in graph[rowcol]:
                        if k >= 0: # k是横坐标
                            points.add((k, ~rowcol))
                            seen.add((k, ~rowcol))
                        else:
                            points.add((rowcol, ~k))
                            seen.add((rowcol, ~k)) 

                if v not in value2index:
                    value2index[v] = []
                value2index[v].append(points) # 将points存到这个值中去。
                # value2index[v]存放的是值为v的connected parts，要明白，我们一个garph中，可能有多个值相同的connected parts

        answer = [[0] * n for _ in range(m)]
        rowmax = [0] * m # the max rank in i row
        colmax = [0] * n # the max rank in j col
        for v in sorted(value2index.keys()):
        
            for points in value2index[v]:
                rank = 1 # rank是针对每个connected parts的，每个connected part应该share同一个rank，这个根据题意可以conclude
                
                # 拿到rank
                for i, j in points:
                    rank = max(rank, max(rowmax[i], colmax[j]) + 1) # 因为我们从小到大值去update rank matrix，因此可以保证rank的计算不会出错。
                # 更新answer和row/col-max
                for i, j in points:
                    answer[i][j] = rank
                    rowmax[i] = max(rowmax[i], rank) 
                    colmax[j] = max(colmax[j], rank)

        return answer

# 13. Roman to Integer 
# O(1)/O(1)
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
        ans = 0
        i = 0
        n = len(s)
        while i < n:
            if i+1 < n and VALUES[s[i+1]] > VALUES[s[i]]:
                ans += VALUES[s[i+1]] - VALUES[s[i]]
                i += 2
            else:
                ans += VALUES[s[i]]
                i += 1
        return ans
    
# 394. Decode String
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        cur_num = 0
        cur_str = ''
        # 利用cur_str有3个highlights需要想明白。
        for c in s:
            if c.isdigit():
                cur_num *= 10
                cur_num += int(c)

            if c.isalpha():
                cur_str += c
            
            if c == '[':
                # stack存放数字和string，我因为正括号和反括号一定是一一对应的，那么我们只需要在[的时候同时添加num和str，就可以保证pop的时候，顺序和数据是我们expect的。
                # 因为确实存在abc34[edf]的情况，而且这里cur_str是build answer的。
                stack.append(cur_num)
                stack.append(cur_str)
                cur_num, cur_str = 0, ''


            # cur是用来build answer的，主体是cur_str，只不过被括号分割开了。如果遇到了[,那么就先把当前的答案存进去，并且cur_num也一定有数
            if c == ']':
                prev_str = stack.pop()
                prev_num = stack.pop()
                cur_str = prev_str + prev_num * cur_str 
                
        return cur_str

        # # 需要了解的前提：stack里面存放的[格外重要！因为其前面一定是数字，其后面一定是letter！
        # stack = []
        # for c in s:
        #     if c == ']':

        #         # 处理str的逻辑，digit和str一定是由[分割开的。因此我们只需要找到
        #         cur_str = ''
        #         while stack and stack[-1] != '[':
        #             cur_str = stack.pop() + cur_str
                
        #         stack.pop() # to pop '['

        #         # 处理digit的逻辑
        #         temp_num = ''
        #         while stack and stack[-1].isdigit():
        #             temp_num = stack.pop() + temp_num
                
        #         stack.append(int(temp_num)*cur_str) # 我们只找到一对数字和str


        #     else:
        #         stack.append(c)

        # return ''.join(stack)


# 759 759. Employee Free Time 很简单...就是处理对象变为object了
class Solution:
    def employeeFreeTime(self, schedule: '[[Interval]]') -> '[Interval]':
        ints = sorted([i for s in schedule for i in s], key=lambda x: (x.start, x.end))

        res, pre = [], ints[0]
        for i in ints[1:]:
            if i.start <= pre.end and pre.end < i.end: #  pre和当前i有overlap 
                pre.end = i.end
            elif i.start > pre.end:
                res.append(Interval(pre.end, i.start))
                pre = i
        return res
        
# 402. Remove K Digits
# 这一题的难点在于：
# 1. 想到用monotonic stack去build increasing subsequence
# 2. 如果首位是0怎么办？
# 3. 如果末尾没清空怎么办？
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        stack = list()
        for n in num:
            
            # 如果K还可以操作，我们想要的是max_stack，递增，保证stack里面顺序组成的是尽可能小的
            while stack and k and stack[-1] > n: 
                stack.pop()
                k -= 1
            
            if stack or n != '0': # to escape the first digit is 0 
                stack.append(n)
        
        if k: # 把最后几位清掉
            stack = stack[0:-k] 
        
        return ''.join(stack) or '0'
        
        
# 929. Unique Email Addresses
class Solution:
    def numUniqueEmails(self, emails: List[str]) -> int:
        def clean(e):
            # 搞清楚清理顺序就可以了。
            [local, domain] = e.split('@')
            local = local.split('+')[0]
            local = ''.join(local.split('.'))
            return local+"@"+domain

        seen = set()
        for e in emails:
            seen.add(clean(e))
        return len(seen)
    
# 482. License Key Formatting
# this question can be medium instead of easy given its details.
# poor test case for understanding.
# 1. 倒序处理，是因为other groups contain exactly k but first group
# 2. 如果不是 ‘-’，那么就count起来，count一旦满足k，就init count，并且添加-
class Solution:
    def licenseKeyFormatting(self, s: str, k: int) -> str:
        n = len(s)
        count = 0
        ans = ''
        
        for i in reversed(range(n)):
            if (s[i] != '-'): # may be many dash
                ans += s[i].upper()
                count += 1
                if (count == k):
                    count = 0
                    ans += '-'
     
        # Make sure the output doesn't start with a dash -> another trap
        if (len(ans) > 0 and ans[len(ans)-1] == '-'):
            ans = ans[:-1]
        
        ans = ans[::-1]
        return ans

# 904. Fruit Into Baskets
# 经典的题，lazy update，只用找到最大值就行了...
# 只要window超过3个了，就要narrow window了。
class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        basket = defaultdict(int)
        left = 0
        
        # Add fruit from the right index (right) of the window.
        for right, fruit in enumerate(fruits):
            basket[fruit] += 1

            if len(basket) > 2:
                basket[fruits[left]] -= 1
                if basket[fruits[left]] == 0:
                    del basket[fruits[left]]
                left += 1
        
        return right - left + 1
    
# 975. Odd Even Jump - O(nlogn)/O(n)
# Odd 可以跳到bigger(smallest index)
# even 可以跳到smaller(smallest index)
class Solution:
    def oddEvenJumps(self, arr: List[int]) -> int:
        n = len(arr)
        next_higher, next_lower = [0] * n, [0] * n # next_higher[i]记录的是当前index下一个可以跳的更高的数字的最小index是多少
        
        # 构建的方法很有趣
        stack = []
        for [a, i] in sorted([[a,i] for i, a in enumerate(arr)]):
            while stack and stack[-1] < i:
                next_higher[stack.pop()] = i
            stack.append(i)

        # 同理构建
        stack = []
        for [a, i] in sorted([[-a,i] for i, a in enumerate(arr)]):
            while stack and stack[-1] < i:
                next_lower[stack.pop()] = i
            stack.append(i)

        # 交替跳跃，相当于两个dp
        higher, lower = [0] * n, [0] * n
        higher[-1] = lower[-1] = 1
        for i in range(n-1)[::-1]: # 牛逼，如果next_higher[i]有值，next_higher[i] -> next index with higher value，就是意味着可以从当前跳到next_index, 我们这里init结尾为1，那么就是意味着可以跳到结尾。
            higher[i] = lower[next_higher[i]]
            lower[i] = higher[next_lower[i]]
        return sum(higher) # 我们是从奇数开始跳的


# 31. Next Permutation
# 这一题的核心是理解how permutation works吧...
# 1- 从后往前遍历，找到第一对升序的pair；num[i] < num[i+1]; 此刻num[i+1:]一定是descending的
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

# 43
# 48. Rotate Image
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



# 158. Read N Characters Given read4 II - Call Multiple Times
# 这一题是用来模拟的。
# 主要的思路是：传递进来n个数，用while循环读取每个数，
# 1. 看看ptr == size与否，相等意味着缓冲区耗尽，因此需要reset并且读取
# 2. sub-while用于读取缓冲区->buf
# return res
class Solution:
    def __init__(self):
        # 用来存储多读的字符
        self.buffer = [""] * 4
        self.buffer_size = 0
        self.buffer_ptr = 0

    def read(self, buf: List[str], n: int) -> int:
        # 这里的n时要读取多少个字符
        total_chars = 0  # 已读取的字符数量
        
        while total_chars < n:
            if self.buffer_ptr == self.buffer_size:  # 如果缓冲区已耗尽，调用 read4
                self.buffer_size = read4(self.buffer)  # 从文件中读取最多 4 个字符到 self.buffer
                self.buffer_ptr = 0
                if self.buffer_size == 0:  # 如果文件已经读取完毕，退出循环
                    break
            
            # 从缓冲区读取字符到 buf
            while total_chars < n and self.buffer_ptr < self.buffer_size:
                buf[total_chars] = self.buffer[self.buffer_ptr]
                total_chars += 1
                self.buffer_ptr += 1

        return total_chars

# 163. Missing Ranges
class Solution:
    def findMissingRanges(self, nums: List[int], lower: int, upper: int) -> List[List[int]]:
        res=[]
        # lower可以看作cur取值的下限，如果大于这个下限，意味着就有range，否则就没有。
        for cur in nums+[upper+1]:
            if cur>lower: # 如果我们写成lower=cur，这里改为cur > lower+1，我们会错过cur==lower+1的情况，这样有可能会导致我们错过第一个元素的范围，因为第一次的lower并不是上一个元素的范围，而是边界lower本身。
                res.append([lower,cur-1])
            lower=cur+1 
        return res
# 681. Next Closest Time
class Solution:
    def nextClosestTime(self, time: str) -> str:
        hh, mm = time.split(':')
        # print(set(hh+mm))
        # digits = set([hh[0],hh[1],mm[0],mm[1]])
        nums = sorted(set(hh + mm)) # 因为hh+mm是一个字符串
        two_digits_values = [a+b for a in nums for b in nums] # 所有两个数的可能

        # check if any minute valid
        i = two_digits_values.index(mm)
        if i + 1 < len(two_digits_values) and two_digits_values[i + 1] < "60":
            return hh + ":" + two_digits_values[i+1] 
        
        # check if any hour valid
        i = two_digits_values.index(hh)
        if i + 1 < len(two_digits_values) and two_digits_values[i + 1] < "24":
            return two_digits_values[i+1] + ":" + two_digits_values[0]
        

        return two_digits_values[0] + ":" + two_digits_values[0] # earliest time of next day.


# 809. Expressive Words
# 这一题还是有点困难的，尤其是双指针的转化
class Solution:
    def expressiveWords(self, S, words):
        return sum(self.check(S, W) for W in words)

    def check(self, S, W):
        j, n = 0, len(S)
        for i in range(n):
            if j < len(W) and S[i] == W[j]:  # 如果当前字符相同
                j += 1
            # 如果当前字符不同，看看上一个字符能不能stretch
            # 这里是两个不等式的and expression
            # [i-1, i, i+1] != [i,i,i] and [i, i, i] != [i-2, i-1, i]
            # [i-1, i, i+1] == [i,i,i] or [i, i, i] or [i-2, i-1, i]  分别对应中间 或者 分别对应最后
            elif S[i - 1:i + 2] !=  S[i] * 3 != S[i - 2:i + 1]:  
                return False
        return j == len(W)
        
# 849. Maximize Distance to Closest Person
# 这里用的是max，也可以使用双指针，每次遇到1，向周围探索。
class Solution:
    def maxDistToClosest(self, seats: List[int]) -> int:
        seats = "".join(list(map(str, seats)))
        zeros = seats.split('1')
        l = len(max(zeros, key=len))
        res = max(len(zeros[0]), len(zeros[-1]), l//2 + l%2) # max(首， 尾， 中间0)
        return res

# 215. Kth Largest Element in an Array
# 1. Sort - O(nlogn)
# 2. Heap - O(nlogk)
# 3. Quick select - O(n)/O(n)
# 4. counting sort - O(n+m)/O(n)
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


# 210 Course schedule II - return res if len(res) == num_courses else []
# O(V+E)/O(V+E)


# 399. Evaluate Division 
# - 常规BFS
# - UF - 太复杂了，可以不用看了。
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:   
        gid_weight = {}  # gid==group_id; gid_weight装了 字母 + 与其对应的字母 + 两者之间的关系

        def find(node_id):
            if node_id not in gid_weight: # 没有遇见过，把node和weight添加进来。
                gid_weight[node_id] = (node_id, 1) 
            group_id, node_weight = gid_weight[node_id]
            # 
            if group_id != node_id: 
                new_group_id, group_weight = find(group_id)
                gid_weight[node_id] = (new_group_id, node_weight * group_weight)
            return gid_weight[node_id]

        def union(dividend, divisor, value):
            dividend_gid, dividend_weight = find(dividend)
            divisor_gid, divisor_weight = find(divisor)
            if dividend_gid != divisor_gid: # 表明两者还没有连接起来
                gid_weight[dividend_gid] = (divisor_gid, divisor_weight * value / dividend_weight)

        
        for (dividend, divisor), value in zip(equations, values):
            union(dividend, divisor, value)

        results = []
        
        for (dividend, divisor) in queries:
            if dividend not in gid_weight or divisor not in gid_weight:
                # case 1). at least one variable did not appear before
                results.append(-1.0)
            else:
                dividend_gid, dividend_weight = find(dividend)
                divisor_gid, divisor_weight = find(divisor)
                if dividend_gid != divisor_gid:
                    # case 2). the variables do not belong to the same chain/group
                    results.append(-1.0)
                else:
                    # case 3). there is a chain/path between the variables
                    results.append(dividend_weight / divisor_weight)
        return results
# 下面是常规BFS的写法
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        import collections
        graph = collections.defaultdict(dict)
        
        # 构建图 graph的结构是第一个知识点！
        for (dividend, divisor), value in zip(equations, values):
            graph[dividend][divisor] = value
            graph[divisor][dividend] = 1 / value
        
        # BFS - 最主要的是queue里面存放的是[node, weight]有一个权重
        def bfs(src, dst):
            if not (src in graph and dst in graph):
                return -1.0
            queue = collections.deque([(src, 1.0)])
            visited = set([src]) # visited可以避免套圈！
            while queue:
                current, currentProduct = queue.popleft()
                if current == dst:
                    return currentProduct
                for neighbor, value in graph[current].items():
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, currentProduct * value))
            return -1.0
        
        # 对每个查询执行BFS
        return [bfs(query[0], query[1]) for query in queries]

# 947. Most Stones Removed with Same Row or Column
# 一旦有坐标连接起来，想一想connected parts，那么每一个parts中的所有stone都可以被Removed but one，那么这一题就变成count parts
# 那么也可以用disjoint set union来做了。
# 难点1:如何构造图 -> 我们只用关注石头的index就可以了！坐标帮助我们判断他们是不是邻居！
# 如果是用DFS的方法做->O(n2)/O(n2)
class Solution:
    def removeStones(self, stones: List[List[int]]) -> int:
        n = len(stones)

        # 构造图
        adjencency = [[] for _ in range(n)]
        for i in range(n):
            x, y = stones[i][0], stones[i][1]
            for j in range(i+1, n):
                if stones[j][0] == x or stones[j][1] == y:
                    adjencency[i].append(j)
                    adjencency[j].append(i)

        visited = set() # 也可以用[False] * n
        num_of_parts = 0
        def dfs(i):
            visited.add(i)
            for ni in adjencency[i]:
                if ni not in visited:
                    dfs(ni)

        for i in range(n):
            if i not in visited:
                dfs(i)
                num_of_parts += 1
        return n-num_of_parts
                
# UF - O(n)/O(n) nb啊
# 这一题的思路，我们的坐标是可以通过x/y连接的，那么这个x和y就是属于一组，我们把所有的x，y坐标放在一起，比如[x,y1],[x,y2]，
# 在union之后，y1和y2肯定也是一个group。我们最后只用找有多少组就可以了。
class Solution:
    def removeStones(self, stones):
        UF = {}
        def find(x):
            if x != UF[x]:
                UF[x] = find(UF[x])
            return UF[x]
        def union(x, y):
            if x not in UF:  # key2: 如果没有办法直接init parent/uf，我们可以在union/find里去init第一次遇到值
                UF[x] = x
            if y not in UF:
                UF[y] = y
            rootX = find(x)
            rootY = find(y)
            if rootX != rootY:
                UF[rootX] = rootY
        
        maxX = 10**4+1
        for x,y in stones:
            union(x,y+maxX) # key1: 给y加上偏移量 add an offset to avoid the conflict with x

        return len(stones) - len({find(n) for n in UF})
    
# 138. Copy List with Random Pointer
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head: return None
        seen = {}
        def copy_helper(node):
            if not node: return None
            if node in seen: return seen[node]
            copy_node = Node(node.val)
            seen[node] = copy_node
            copy_node.next = copy_helper(node.next)
            copy_node.random = copy_helper(node.random)
            return copy_node
        copy_helper(head)
        return seen[head]


# 951. Flip Equivalent Binary Trees
class Solution:
    def flipEquiv(self, r1: Optional[TreeNode], r2: Optional[TreeNode]) -> bool:
        if not r1 and not r2: return True
        if not r1 or not r2: return False
        if r1.val != r2.val: return False
        return (self.flipEquiv(r1.right, r2.right) and self.flipEquiv(r1.left, r2.left)) \
                or (self.flipEquiv(r1.left, r2.right) and self.flipEquiv(r1.right, r2.left))
        

# 753. Cracking the Safe # 这道题很难，可以不用看了
# This is the question about Euler Path(a path visiting every edge exactly once)
# Euler Circuit == an Euler Path ending where it starts
# each possible password    -> node
# each possible digit       -> edge
# "01" -> 路径0 -> "10"
# Eular circuit 是本题的答案，因为我们希望每个组合都出现，每个点都出现过；但又是最短，避免重复访问，因此就是eular Circuit
# 而题意确定了circuit一定存在，存在条件：1.每个node的in/out degree一样。2. 每个结点都在一个连通图里。
class Solution:
    def crackSafe(self, n: int, k: int) -> str:
        seen = set() # node + edge = path
        ans = []
        def dfs(node):
            for x in map(str, range(k)):
                nei = node + x
                if nei not in seen:
                    seen.add(nei)
                    dfs(nei[1:])
                    ans.append(x)

        dfs("0"*(n-1)) # 这是初始node
        print(ans)
        return "".join(ans) + "0"*(n-1) # 为了返回
    
# 857. Minimum Cost to Hire K Workers
class Solution:
    def mincostToHireWorkers(self, quality: List[int], wage: List[int], k: int) -> float:
        n = len(quality)
        total_cost = math.inf
        current_total_quality = 0
        wage_to_quality_ratio = [(wage[i] / quality[i], quality[i]) for i in range(n)]
        wage_to_quality_ratio.sort(key=lambda x: x[0])

        
        workers = []
        # 我们的cost取决于: ratio / quality
        # 我们排序过后ratio只能是越来越大的，但是quality不一定，因此可能会出现小quality，大ratio的情况，不过没关系，我们也考虑到了。
        for i in range(n):
            heapq.heappush(workers, -wage_to_quality_ratio[i][1])
            current_total_quality += wage_to_quality_ratio[i][1]

            if len(workers) > k:
                current_total_quality += heapq.heappop(workers) # workers里存的是负数，这里实际是减去一个最大的quality

            if len(workers) == k:
                total_cost = min(total_cost, current_total_quality * wage_to_quality_ratio[i][0])

        return total_cost
    
# 127 word ladder 就是正常的bfs，不过要先构造word_list，注意辅助变量seen和入queue得元素有step



# 425. Word Squares 这题构建square的思路也是挺有趣的。没遇到过。
# 这题目用回溯，但是回溯的细节是难点。
class Solution:
    def wordSquares(self, words: List[str]) -> List[List[str]]:

        self.words = words
        self.N = len(words[0])
        self.buildTrie(self.words)

        results = []
        word_squares = []
        for word in words:
            word_squares = [word]
            self.backtracking(1, word_squares, results)
        return results

    # trie
    # 每一层都会在node[#]留下index，是为了之后找prefix使用。
    def buildTrie(self, words):
        self.trie = {}

        for wordIndex, word in enumerate(words):
            node = self.trie
            for char in word:
                if char in node:
                    node = node[char]
                else:
                    newNode = {}
                    newNode['#'] = []
                    node[char] = newNode
                    node = newNode
                node['#'].append(wordIndex)

    def backtracking(self, step, word_squares, results):
        if step == self.N:
            results.append(word_squares[:])
            return

        prefix = ''.join([word[step] for word in word_squares]) # 理解这一步你需要搞清楚square是如何构建起来的。
        for candidate in self.getWordsWithPrefix(prefix):
            word_squares.append(candidate)
            self.backtracking(step+1, word_squares, results)
            word_squares.pop()

    def getWordsWithPrefix(self, prefix):
        node = self.trie
        for char in prefix:
            if char not in node:
                return []
            node = node[char]
        return [self.words[wordIndex] for wordIndex in node['#']]
    

# 247. Strobogrammatic Number II
# 时间复杂度(5^[n/2])空间(n)
class Solution:
    def findStrobogrammatic(self, n: int) -> List[str]:
        res = []
        mid = n//2 - n//1
        # 0, 1, 2, 3, 4     n=5 
        # 0, 1, 2, 3        n=4 
        def bt(i, path):
            if i == n:
                res.append(path)
            
            elif i < n//2:
                for c in ('1', '0', '8', '6', '9'):
                    if i == 0 and c == '0': continue
                    bt(i+1, path+c)
            
            elif i == n//2 and n%2 == 1:
                for c in ('1', '0', '8'):
                    bt(i+1, path+c)
                
            else: 
                other_index = (n-1)-i
                other_ch = path[other_index]
                if other_ch == '6':
                    bt(i+1, path+'9')
                elif other_ch == '9':
                    bt(i+1, path+'6')
                else:
                    bt(i+1, path+other_ch)


        bt(0, "")
        return res


# 34. Find First and Last Position of Element in Sorted Array 
# 记得判断 left?=len or left?=target; 如果不用api如何做？
# 如果用正常的二分流程，当nums[mid] != target的时候一切正常，相同的话需要添加额外的判断
# if nums[mid]==target：
#   if isLeft: if mid > 0 and nums[mid-1] == target: 
#       right = mid-1 else return mid
# 需要注意了，这里因为我额外调用的binary search，因此有可能会出现L==R，如果用while l<r的话有可能无法进入while循环，导致测试出错。


# 315. Count of Smaller Numbers After Self
# 使用segment Tree = 二叉树 + 每个节点表示一个区间
# 用于区间查询/修改： 比如区间和/区间最大值/区间最小值
# 因为315这一道题其实就是查询不同区间内比num小的值，因此可以利用segment tree的特点来查询，将复杂度从N2->nlogn
class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        def update(index, value, tree, size):
            index += size  # shift the index to the lea
            # update from leaf to root
            tree[index] += value # 更新的值，有没有出现过
            while index > 1:
                index //= 2
                tree[index] = tree[index * 2] + tree[index * 2 + 1] 

        def query(left, right, tree, size):
            # return sum of [left, right)
            result = 0
            left += size  # shift the index to the leaf
            right += size
            while left < right:
                # if left is a right node
                # bring the value and move to parent's right node
                if left % 2 == 1: # 这里是唯一我没有想明白的点：为什么是右子树，就意味着当前节点是查询范围的最左侧。
                    result += tree[left]
                    left += 1
                
                left //= 2
                if right % 2 == 1:
                    right -= 1
                    result += tree[right]
                # else directly move to parent
                right //= 2
            return result

        offset = 10**4  # offset negative to non-negative
        size = 2 * 10**4 + 1  # total possible values in nums # 那么其实size是leaf的数量
        tree = [0] * (2 * size) # segment tree是complete binary tree，2*size是树所有节点的数量，这一题用的list存放节点，并没有新开一个类。
        result = []
        for num in reversed(nums): # 遍 遍历 遍 创建，这样只用遍历nums一次，否则要2次
            smaller_count = query(0, num + offset, tree, size) # left, right, tree, size
            result.append(smaller_count)
            update(num + offset, 1, tree, size) # 用来更新tree
        return reversed(result)
# 这种segment tree的做法不用掌握，还是用merge sort/divide and conquer吧
class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        n = len(nums)
        arr = [[v, i] for i, v in enumerate(nums)]  # record value and index
        result = [0] * n

        def merge_sort(arr, left, right):
            
            if right <= left + 1:
                return    
            mid = (left + right) // 2
            merge_sort(arr, left, mid)
            merge_sort(arr, mid, right)
            merge(arr, left, right, mid)
        
        def merge(arr, left, right, mid):
            # merge [left, mid) and [mid, right)
            i = left
            j = mid
            temp = []
            
            while i < mid and j < right:
                if arr[i][0] <= arr[j][0]:
                    result[arr[i][1]] += j - mid
                    temp.append(arr[i])
                    i += 1
                else:
                    temp.append(arr[j])
                    j += 1
            # when one of the subarrays is empty
            while i < mid:
                # j - mid numbers jump to the left side of arr[i]
                result[arr[i][1]] += j - mid
                temp.append(arr[i])
                i += 1
            while j < right:
                temp.append(arr[j])
                j += 1
            # restore from temp
            for i in range(left, right):
                arr[i] = temp[i - left]

        merge_sort(arr, 0, n)

        return result
    
# 2271. Maximum White Tiles Covered by a Carpet
# 这一道题有个很关键的点：carpet的右端点放在一段tile的中间，是不如放在tile的右端点的，因为将carpet的右端点从中间移动到右边的过程中，右端点一定会覆盖到tile，而左端点有可能uncover，也有可能跳过某些空白的。
class Solution:
    def maximumWhiteTiles(self, tiles: List[List[int]], carpetLen: int) -> int:
        tiles.sort(key=lambda x: x[0]) # 排序
        ans = cover = left = 0

        for tl, tr in tiles:
            cover += tr - tl + 1 # 当前tile cover的上限、

            while tiles[left][1] < tr - carpetLen + 1:  #  如果carpet的左边离开了left指向的tile，那么cover需要减去整段的tile
                cover -= tiles[left][1] - tiles[left][0] + 1
                left += 1
            uncover = max(tr - carpetLen + 1 - tiles[left][0], 0) # uncover主要是负责找左端点在的那段tile中有多少tile没有覆盖，(tr-carpetLen+1)是carpet的左端点。如果左端点在tile上，那么取前面的值，如果在tiles中间的空白部分，前面的值为负，取后面的值0.
            ans = max(ans, cover - uncover)
        return ans

# 3413. Maximum Coins From K Consecutive Bags
# 这一题是2271的variant，加了权重。
"""
对于本题来说，如果出现两个相邻区间，左边区间 c 大，右边区间 c 小的情况，那么和右端点对齐就不是最优的，和左端点对齐反而是最优的。
所以在 2271 题的基础上，额外跑一遍和左端点对齐的滑动窗口即可。
代码实现时，把 coins 反转，每个区间 [l,r] 改为 [−r,−l]，就可以复用和右端点对齐的代码了。
"""
class Solution:
    # 2271. 毯子覆盖的最多白色砖块数
    def maximumWhiteTiles(self, tiles: List[List[int]], carpetLen: int) -> int:
        ans = cover = left = 0
        for tl, tr, c in tiles:
            cover += (tr - tl + 1) * c
            while tiles[left][1] < tr - carpetLen + 1:
                cover -= (tiles[left][1] - tiles[left][0] + 1) * tiles[left][2]
                left += 1
            uncover = max((tr - carpetLen + 1 - tiles[left][0]) * tiles[left][2], 0)
            ans = max(ans, cover - uncover)
        return ans

    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        coins.sort(key=lambda c: c[0])
        ans = self.maximumWhiteTiles(coins, k) # 正常右端点

        # 下面的反转和取负操作，相当于把intervals按照x=0对折映射到负值区域上。从而实现代码的复用
        coins.reverse()
        for t in coins:
            t[0], t[1] = -t[1], -t[0]
        return max(ans, self.maximumWhiteTiles(coins, k))


# 852. Peak Index in a Mountain Array
# 因为确保会有peak element，就意味着是有序的，至少是局部有序。因此我们可以用二分
# 去判断nums[m] VS num[m+1]

# 5. Longest Palindromic Substring
# 除了用expand from center O(n2)/O(1)
# 还可以用dp O(n2)/O(n2)
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

class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        ans = [0, 0]

        for i in range(n): # 初始化，自己
            dp[i][i] = True

        for i in range(n - 1): # 初始化，Pair
            if s[i] == s[i + 1]:
                dp[i][i + 1] = True
                ans = [i, i + 1]

        for diff in range(2, n): # diff + 1是subarray的长度，因为我们已经init长度为1/2的subarry，所以这里我们从diff=2开始。
            for i in range(n - diff):
                j = i + diff
                if s[i] == s[j] and dp[i + 1][j - 1]:
                    dp[i][j] = True
                    ans = [i, j]

        i, j = ans
        return s[i : j + 1]


# 410. Split Array Largest Sum 
# 用二分解决 - 可以使用 二分搜索 的关键原因在于问题具有单调性：随着数组被分割的子数组数量增加，子数组的最大和会逐渐减小。这种单调特性是二分搜索的核心条件。

# 380. Insert Delete GetRandom O(1)
# 要求是每个func都是O(1)，random我们可以使用random.choice轻松实现
# 我们肯定是希望用set实现，但是问题set没有办法O(1)实现getRandom，因为set转化为list，为O(n)
# 因此这里我们用hashmap来记录某个元素在list中的位置/index，当删除时与最后一位swap，再pop list
from random import choice
class RandomizedSet():
    def __init__(self):
        self.dict = {}
        self.list = []
  
    def insert(self, val: int) -> bool:
        if val in self.dict:
            return False
        self.dict[val] = len(self.list) # 这个就是为了给val上index，这里的Len(self.list)其实就是将要添加的val的index
        self.list.append(val)
        return True

    # dict中存放着index和value，我们把最后一位数往list中替换掉，然后pop最后一位
    def remove(self, val: int) -> bool:
        if val in self.dict:
            last_element, idx = self.list[-1], self.dict[val]
            self.list[idx], self.dict[last_element] = last_element, idx
            self.list.pop()  #更新list
            del self.dict[val] #更新dict
            return True
        return False

    def getRandom(self) -> int:
        return choice(self.list)



# 642. Design Search Autocomplete System
# common_prefix 很有可能就是trie tree
class TrieNode:
    def __init__(self):
        self.children = {}
        self.sentences = defaultdict(int) # 每一个node/前缀节点，都会存有完整的sentence和热度（time/count)

class AutocompleteSystem:
    def __init__(self, sentences: List[str], times: List[int]):
        self.root = TrieNode()
        for sentence, count in zip(sentences, times):
            self.add_to_trie(sentence, count)
            
        self.curr_sentence = []
        self.curr_node = self.root
        self.dead = TrieNode()
        
    def input(self, c: str) -> List[str]:
        # 如果输入结束，那么像是init一样，重置。
        if c == "#":
            curr_sentence = "".join(self.curr_sentence)
            self.add_to_trie(curr_sentence, 1)
            self.curr_sentence = []
            self.curr_node = self.root
            return []
        
        # 这道题的逻辑很困难哦...20-25分钟写不完...题倒是不难。
        self.curr_sentence.append(c)
        if c not in self.curr_node.children:
            self.curr_node = self.dead
            return []
        
        self.curr_node = self.curr_node.children[c]
        sentences = self.curr_node.sentences
        sorted_sentences = sorted(sentences.items(), key = lambda x: (-x[1], x[0]))
        
        ans = []
        for i in range(min(3, len(sorted_sentences))):
            ans.append(sorted_sentences[i][0])
        
        return ans

    def add_to_trie(self, sentence, count): # 
        node = self.root
        for c in sentence:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
            node.sentences[sentence] += count

# 135 Candy
# 1. 直觉很简单，因为我们要照顾到higher ranking child的neighbor，因此我们可以用两个array，一个用来照顾左边的，一个用来照顾右边的，具体分多少，取决于哪个更大
# 2. 把方法一简化，只用一个array O(n)/O(n)
class Solution:
    def candy(self, ratings):
        candies = [1] * len(ratings)
        for i in range(1, len(ratings)):
            if ratings[i] > ratings[i - 1]:
                candies[i] = candies[i - 1] + 1
        sum = candies[-1]
        for i in range(len(ratings) - 2, -1, -1):
            if ratings[i] > ratings[i + 1]:
                candies[i] = max(candies[i], candies[i + 1] + 1)
            sum += candies[i]
        return sum
# 还可以用constant space来解决这个问题。


class Solution:
    def getHint(self, secret: str, guess: str) -> str:
        # cnt = Counter(secret)
        a = b = 0
        
        # for i in range(len(secret)):
        #     sc, gc = secret[i], guess[i]
        #     if sc == gc:
        #         a += 1
        #         cnt[sc] -= 1


        # for i in range(len(secret)):
        #     sc, gc = secret[i], guess[i]
        #     if sc != gc and cnt[gc] > 0:
        #         b += 1
        #         cnt[gc] -= 1
        
        # 下面是只用one-pass的方法
        s_cnt = Counter()
        g_cnt = Counter()
        for i in range(len(secret)):
            sc, gc = secret[i], guess[i]
            if sc == gc:
                a += 1
            else:
                s_cnt[sc] += 1
                g_cnt[gc] += 1
        # print(s_cnt & g_cnt)
        b = (s_cnt & g_cnt).total()

        return str(a)+'A'+str(b)+'B'
    
# 308. Range Sum Query 2D - Mutable
# 如果是update调用的多，那么用BF，如果是query调用的多用presum
# 如果调用的一样多...用binary Indexed Tree，但是这个我不会.


# 731. My Calendar II
from sortedcontainers import SortedDict
class MyCalendarTwo:
    def __init__(self):
        self.booking_count = SortedDict()
        self.max_overlapped_booking = 2

    def book(self, start: int, end: int) -> bool:
        self.booking_count[start] = self.booking_count.get(start, 0) + 1
        self.booking_count[end] = self.booking_count.get(end, 0) - 1

        overlapped_booking = 0
        for count in self.booking_count.values():
            overlapped_booking += count # 因为是sorted，所以这个相当于prefix_sum了
            if overlapped_booking > self.max_overlapped_booking:
                # Rollback changes.
                self.booking_count[start] -= 1
                self.booking_count[end] += 1

                # Remove entries if their count becomes zero to clean up the SortedDict.
                if self.booking_count[start] == 0:
                    del self.booking_count[start]

                return False

        return True



# Binary Indexed Tree (BIT) 
    # 通常基于树状数组，用于动态维护前缀和
    # 快速进行区间查询和单点更新
   
# Segment Tree
    # 用树/数组来模拟
    # 支持更复杂的区间查询，最大值/最小值/和
    # 可扩展至多维



# 如何找所有subarray
# for start in range(len(arr)):
#     for end in range(start, len(arr)):
#         subarrays.append(arr[i:j+1])

# 如何找GCD
def gcd_manual(a, b):
    while b:
        a, b = b, a % b
    return a

# 如何找LCM
def lcm_manual(a, b):
    return (a * b) // gcd_manual(a, b)



       

