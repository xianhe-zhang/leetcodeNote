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


# 329
# 715
# 1146
# 818
# 729
# 539
# 777
# 419
# 489
# 778
# 2096
# 1101
# 2158
# 2172
# 2115
# 2034
# 833
# 792
# 562
# 1606
# 2162
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







       

