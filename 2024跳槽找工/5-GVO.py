# 大概50道题左右。
# 394. Decode String
# 我们必须优先解决innermost string
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        cur_num = 0
        cur_str = ''
        for c in s:
            if c == '[': # we need to reset cur_str, becuase str between [] belongs to next level
                stack.append(cur_str)
                stack.append(cur_num)
                cur_num, cur_str = 0, ''
            if c == ']':
                pre_num = stack.pop()
                pre_str = stack.pop()
                cur_str = pre_str + pre_num * cur_str  #***

            if c.isdigit():
                cur_num = 10 * cur_num + int(c)

            if c.isalpha():
                cur_str += c

        return cur_str
            
def decodeStringIIII(s):
    stk = []
    cur = ""
    num = 0
    for ch in s:
        if ch.isdigit():
            num = num*10 + int(ch)
        elif ch.isalpha():
            cur += ch
        elif ch == "[":
            stk.append(cur)
            cur = ""
        elif ch == "}":
            pre = stk.pop()
            cur = pre + cur*num
            num = 0
    
    while stk:
        cur = stk.pop() + cur
    return cur

s = 'a((cd){3}){2}'
print(decodeStringIIII(s))
# BC I/II/III 通用solution - O(n)/O(n)
class Solution:
    def calculate(self, s):
        def update(op, v):
            if op == "+": stack.append(v)
            if op == "-": stack.append(-v)
            if op == "*": stack.append(stack.pop() * v)           #for BC II and BC III
            if op == "/": stack.append(int(stack.pop() / v))      #for BC II and BC III
    
        it, num, stack, sign = 0, 0, [], "+"
        
        while it < len(s):
            if s[it].isdigit():
                num = num * 10 + int(s[it])
            elif s[it] in "+-*/":
                update(sign, num)
                num, sign = 0, s[it]
            elif s[it] == "(":                                        # For BC I and BC III 
                num, j = self.calculate(s[it + 1:])
                it = it + j
            elif s[it] == ")":                                        # For BC I and BC III
                update(sign, num)
                return sum(stack), it + 1
            it += 1
        update(sign, num)
        return sum(stack)
    
    
# 227 Basic Calculator - 这道题的精髓在于operator要先存起来，然后等到下一次遇到的时候，再去处理cur_num
# follow up - 如何不使用stack？ 需要使用额外的变量：result, last_number, curr_number, 遇到+/-，把last_number放入到res中，然后last_number = (-)curr_number, 遇到*/的时候，直接处理
class Solution:
    def calculate(self, s: str) -> int:
        if not s: return 0
        stack = []
        cur_num = 0
        operator = '+'

        for i in range(len(s)):
            c = s[i]

            if c.isdigit(): cur_num = cur_num*10 + int(c)

            if (not c.isspace() and not c.isdigit()) or i == len(s)-1: # 遇到operator/最后一位了该处理了
                if operator == '+': stack.append(cur_num)
                if operator == '-': stack.append(-cur_num)
                if operator == '*': stack.append(stack.pop()*cur_num)
                if operator == '/': stack.append(int(stack.pop()/cur_num))
                # reset
                operator = c
                cur_num = 0

        return sum(stack)
# 224 Basic Calculator = 227+394
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        operand = 0 # cur_number
        res = 0 # last_number res在这里担任的是每一个stack/()内计算的结果
        sign = 1 # 1 means positive, -1 means negative  

        for ch in s:
            if ch.isdigit():
                operand = (operand * 10) + int(ch)

            elif ch in '+-':
                res += sign * operand # 延迟处理嘛和BC2是一样的道理。
                sign = 1 if ch == '+' else -1
                operand = 0

            elif ch == '(':
                stack.append(res)
                stack.append(sign)
                sign = 1
                res = 0

            elif ch == ')':
                res += sign * operand
                res *= stack.pop() # stack pop 1, sign
                res += stack.pop() # stack pop 2, operand
                operand = 0

        return res + sign * operand
    

# 4 median of two sorted arrays - binary search
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
# 这道题是Torture...

# 295 find median from data stream - heap ✅
class MedianFinder:
    def __init__(self):
        self.large = []
        self.large = []
    def addNum(self, num):
        if len(self.large) == len(self.small):
            heapq.heappush(self.large, -heapq.heappoppush(self.small, -num))
        else:
            heapq.heappush(self.small, -heapq.heappoppush(self.large, num))

    def findMedian(self):
        if len(self.large) == len(self.small):
            return float(self.large[0] - self.small[0]) / 2.0
        else: 
            return float(self.large[0])

# 6 zigzag conversion 额外变量模拟，如果使用index的话，有点不容易操作。
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1: return s
        rows = [""] * numRows
        backward = True
        index = 0
        for char in s:
            rows[index] += char
            if index == 0 or index == numRows - 1:
                backward = not backward
            if backward:
                index -= 1
            else:
                index += 1
        return "".join(rows)
# 62 unique path- follow up设置障碍，求path的数量穿过所有的障碍
# 1. 障碍不能通过，那么设置为0就可以
# 2. 障碍必须通过，那么start - obstacle - stop 两个dp结果相乘就可以，多个障碍也是一样，iterate相乘就行。
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
    
# 68 text justification
class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        # we need 3 temp-vars to record the length, word_number(space needed)
        temp = []
        temp_l = 0
        temp_cnt = 0
        res = []

        # 1. temp -> currentline
        for w in words: # 每一次的循环，我们肯定是要处理当前w的，目的就是放入temp，但是在放入操作之前，我们要实现检查，是否需要更新temp
            # 1/当前存不到temp中->处理space/更新
            if temp_l+temp_cnt+len(w) > maxWidth:
                # 之所以用max()是因为avoid当前列表里只有一个word，如果只有一个，我们向其后面添加space
                size = max(1, temp_cnt-1)
                # size = max(1,len(temp)-1)
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
            res.append(' '.join(temp).ljust(maxWidth,' ')) # 把字符串用space填充到maxWidth，并且左对齐
        
        return res

# 212 word search II 
# 212. Word Search II
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
# trie的话最开始记得用cur

# 218 the skyline problem
# O(nlogn)/O(n)
class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        edges = []
        for i, build in enumerate(buildings):
            edges.append([build[0], i])
            edges.append([build[1], i])
        edges.sort()

        live, answer = [], [] # live是PQ，只存放当前坐标下考虑的building
        idx = 0
        
        while idx < len(edges): # edges里存的只是edge的location，并没有存高度。
            curr_x = edges[idx][0] # curr_x 是当前的x坐标，
            while idx < len(edges) and edges[idx][0] == curr_x:
                b = edges[idx][1] # The index 'b' of this building in 'buildings'
                if buildings[b][0] == curr_x:
                    right = buildings[b][1]
                    height = buildings[b][2]
                    heapq.heappush(live, [-height, right])
                    
                while live and live[0][1] <= curr_x: # 后者表示这个height和right已经pass了，在当前坐标的左边，因此要pop出去。
                    heapq.heappop(live)
                idx += 1
            
            # Get the maximum height from 'live'.
            max_height = -live[0][0] if live else 0
            
            if not answer or max_height != answer[-1][1]: # 只有左侧变化，才会添加，仔细阅读题
                answer.append([curr_x, max_height])
        
        # Return 'answer' as the skyline.
        return answer
# 329. Longest Increasing Path in a Matrix
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        seen = [[0] * len(matrix[0]) for _ in range(len(matrix))] 

        def dfs(i, j, prev=-1):
            
            if i < 0 or i >= len(matrix) or j < 0 or j >= len(matrix[0]): return 0

            cur_val = matrix[i][j]
            if cur_val <= prev: return 0
            if seen[i][j] != 0: return seen[i][j] # 剪枝
                        
            prev = cur_val
            cur_max = max(
                dfs(i+1, j, prev),
                dfs(i-1, j, prev),
                dfs(i, j+1, prev),
                dfs(i, j-1, prev)
            ) + 1
            seen[i][j] = cur_max
            return cur_max            

        res = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                cur_max = dfs(i,j) if seen[i][j] == 0 else seen[i][j] # 剪枝
                res = max(cur_max, res)
        return res 


# 329. Longest Increasing Path in a Matrix
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:

        m, n = len(matrix), len(matrix[0])
        visited = [[0]*n for _ in range(m)]

        def dfs(x, y):
            if visited[x][y]: return visited[x][y]
            for (nx, ny) in ((x-1,y),(x+1,y),(x,y+1),(x,y-1)):
                if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] > matrix[x][y]:
                    visited[x][y] = max(visited[x][y], dfs(nx, ny))
            
            visited[x][y] += 1
            return visited[x][y]
        ans = 0
        for i in range(m):
            for j in range(n):
                ans = max(ans, dfs(i,j))
        return ans
# 359 logger rate limiter ✅ 一个dict解决问题
# 452 Minimum Number of Arrows to Burst Balloons
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        # everytime pick the first ballon's right boundary & shoot, clear all possible burst ballon
        points.sort()
        print(points)
        idx = 0
        res = 0
        while idx < len(points):
            target = points[idx][1]
            idx += 1
            res += 1
            while idx < len(points) and points[idx][0] <= target:
                target = min(target, points[idx][1]) # core关键，要动态更新气球right boudary的位置
                idx += 1
        return res
# 这个按照尾巴排序的思路也很好。
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        if not points: return 0
        points.sort(key = lambda x : x[1])
        
        arrows = 1
        first_end = points[0][1]
        for x_start, x_end in points: 
            if first_end < x_start:
                arrows += 1
                first_end = x_end
        
        return arrows
    
# 621. Task Scheduler
# Counter -> 看一下最多项 -> 计算至少需要多少时间 -> 判断能否满足所有task
# 首先不要看最后一个任务的时间，去看有多少的task_count == max，没满足一次就要+1，你自己想想。
# 在最后比较res和length，返回其中的较大值
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
        
        return res if res >= length else length

# 694 number of distinct island - 这道题的处理思路也很简单，像往常一样，只需要增加path就可以了。path的样子就是岛屿的样子，因为我们遍历的顺序是一致的！
# 不过需要注意你需要额外变量的帮助，比如path，比如seen

# 743. Network Delay Time
# O((V+E)logV) LogV是操作堆的。 / O(V+E)
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        if len(times) < n - 1: return -1
        # build the map
        node_map = defaultdict(list)
        for x, y, w in times:
            node_map[x].append([y, w]) # [cost, dest]

        seen = set()
        q = [[0, k]]
        
        while q:
            cost, node = heapq.heappop(q)
            if node in seen: continue
            seen.add(node)
            if len(seen) == n:
                return cost
            for nex_n, nex_c in node_map[node]:
                heapq.heappush(q, [cost + nex_c, nex_n])
        return -1


# 752 open the lock
# 利用BFS可以
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        q = [["0000", 0]]
        seen = set()

        def findAllNext(s):
            res = []
            for i in range(4):
                x = int(s[i])
                for d in (-1, 1):
                    y = (x + d) % 10
                    res.append(s[:i] + str(y) + s[i+1:])
            return res

        while q:
            next_q = []
            for status, step in q:
                if status == target: return step
                if status in deadends: continue
                for next_status in findAllNext(status):
                    if next_status not in seen:
                        seen.add(next_status)
                        next_q.append([next_status, step+1])
            q = next_q
        return -1
        
# 787 cheapest flights with k stops
class Solution:
    def findCheapestPrice(self, n, flights, src, dst, k):
        visited_city_stop = {}
        flight_map = defaultdict(list)

        for x,y,c in flights:
            flight_map[x].append([y, c])

        q = [[0, 0, src]] # total_cost, stops, city

        while q:
            total_cost, stops, city = heapq.heappop(q)
            if city == dst and stops-1 <= k: return total_cost
            
            # 如果没有访问过，肯定要访问呀。
            # 如果访问过了，看是不是stops会更少，如果少的话，也加入考虑范围内，为什么？这个有趣。
            # 上一次访问[price小，stops大] VS 这一次访问[price大，stops小]
            # 从直觉上来currNode到dst一定是有一条距离的path的，因此price小的不是我们找的值么？
            # 那为什么这一次如果stop小也要考虑呢？因为上一次访问，不会影响的可以用的step少，但是所有情况也随后被加入到了pq中进行考虑
            # 而这一次新的loop虽然price大了，但是stops更多了，也许就刚好访问到结束点，多种可能性都考虑到了。
            if city not in visited_city_stop or stops < visited_city_stop[city]:
                visited_city_stop[city] = stops
                for next_city, cost in flight_map[city]:
                    heapq.heappush(q, [total_cost+cost, stops+1, next_city])
        return -1


# 1244 design a leader board 用sort/heap就可以

# 1406. Stone Game III
class Solution:
    def stoneGameIII(self, stoneValue: List[int]) -> str:
        n = len(stoneValue)
        # dp[i]表示，从当前i堆到最后开始游戏的话，player之间的分差，如果dp[i]>0，表示Alice赢
        dp = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            
            # 这里的dp[i] = stoneVals - dp[next_i]比较难理解。
            # 这里的d[i+1/2/3] 都是可以看作对手的状态，你直接抽象地理解会有点困难，可以用一个实际的例子表示，因为每一个人都是理性人，因此每一个选择，大家都想让结果最大。！
            # 取一堆 
            dp[i] = stoneValue[i] - dp[i + 1]
            # 取2堆
            if i + 2 <= n: dp[i] = max(dp[i], stoneValue[i] + stoneValue[i + 1] - dp[i + 2])
            # 取3堆
            if i + 3 <= n: dp[i] = max(dp[i], stoneValue[i] + stoneValue[i + 1] + stoneValue[i + 2] - dp[i + 3])

        
        if dp[0] > 0:
            return "Alice"
        if dp[0] < 0:
            return "Bob"
        return "Tie"
    
# 1140. Stone Game II
# M 是一个控制玩家可以选择的石子范围的参数，表示当前玩家可以从 1 到 2 * M 堆石子中选择。
# 最开始M是1，如果当前玩家选择过多的石头，那么下一个玩家可以选择更多的石头。两倍at most
class Solution:
    def stoneGameII(self, piles: List[int]) -> int:
    
        suffix_sum = piles[:] # This helps quickly calculate the total stones available from any given starting point.
        for i in range(len(suffix_sum)-2, -1, -1):
            suffix_sum[i] += suffix_sum[i+1]
    
        n = len(piles)
        dp = [[0] * (n+1) for _ in range(n)]

        # dp[i][j] = score staring from pile i with limitation of j
        # stone game好像都是只能倒序
        for i in range(n-1, -1, -1):
            for m in range(1, n+1):
                if i + 2*m >= n: # 如果可以全部拿完，直接拿完了。
                    dp[i][m] = suffix_sum[i]
                else:
                    for x in range(1, 2*m+1): # 可能拿多少个
                        dp[i][m] = max(dp[i][m], suffix_sum[i] - dp[i+x][max(m, x)]) # max（m,x) 是题意要求
                        # 下一个玩家能选择的数量，是由当前m,x中的较大值决定的。
        return dp[0][1]

# O(n3)/O(n2)


# 3398. Smallest Substring With Identical Characters I // 周赛三题/四题经典套路
# 这道题的代码实现并不困难，困难的是想到用二分去解决问题。
class Solution:
    def check(self, s: str, numOps: int, mid: int) -> bool:
        t = numOps
        last = '0'
        # 第一种情况：从 '0' 开始交替
        for char in s:
            if last == char:
                numOps -= 1
            last = '1' if last == '0' else '0'
        if numOps >= 0:
            return True

        numOps = t
        last = '1'
        # 第二种情况：从 '1' 开始交替
        for char in s:
            if last == char:
                numOps -= 1
            last = '1' if last == '0' else '0'
        return numOps >= 0

    def isValid(self, s: str, numOps: int, mid: int) -> bool:
        # 暴力遍历看看，确定好长度之后，
        if mid == 1:
            return self.check(s, numOps, mid)

        count = 0
        last = -1
        for char in s:
            if char == last:
                count += 1
            else:
                numOps -= count // (mid + 1) # 如果连续的元素多余mid，会减去多余的，否则不用管。
                last = char
                count = 1
        numOps -= count // (mid + 1) # 最后一段substring
        return numOps >= 0

    def minLength(self, s: str, numOps: int) -> int:
        # 本题的思路是通过二分找到尽可能小的长度的substring
        start, end = 1, len(s)
        ans = len(s)
        while start <= end:
            mid = start + (end - start) // 2
            if self.isValid(s, numOps, mid): # valid就是在规定的操作次数范围内，能否满足mid这个长度的substring要求。
                ans = mid
                end = mid - 1
            else:
                start = mid + 1
        return ans



# 二分总结
# 拓扑排序 - Indegree = [0] * n / 遍历添加入degree / init把0-degree的进queue / 遍历 更新degree / len(nodes) == num判断是否有环 这是用bfs，如果是dfs就是用visited看有没有环了
# UF - parent = range(n) / rank = [1] * n/ 
def find(self, x):
    # 路径压缩优化：递归将当前节点直接连接到根
    if self.parent[x] != x:
        self.parent[x] = self.find(self.parent[x])
    return self.parent[x]

def union(self, x, y):
    # 按秩合并优化：小秩集合指向大秩集合
    rootX, rootY = self.find(x), self.find(y) 
    # 如果不用的话，直接任意连接就可以。
    if rootX != rootY:
        if self.rank[rootX] > self.rank[rootY]:
            self.parent[rootY] = rootX # x rank 更高，所以把rootY指向X
        elif self.rank[rootX] < self.rank[rootY]:
            self.parent[rootX] = rootY
        else:
            self.parent[rootY] = rootX
            self.rank[rootX] += 1


# string操作的API
# len()
# [:]
# split('.')
# strip(' ')
# replace(old, new)
# find(target)
# index(target)
# count()
# startwith()
# endwith()
# upper/lower
# isdigit/isalpha
# ord/chr

# 第一轮：题目是生成一个车牌号。00000 -> 00001 -> 00002 -> ... -> 99999 -> A0000 -> A0001 -> ... -> A9999 -> ... -> 
# Z9999 -> AA000 -> ... -> ZZ999 -> ... -> AAA00 -> ... -> ZZZ99 -> ... -> AAAA0 -> ... -> ZZZZ9 -> ... -> AAAAA -> ... -> ZZZZZ。
# 输入是一个数字，表示要生成第几个车牌号，输出是一个字符串，对应该数字对应的车牌号。
# 我的解法是暴力法，每个范围依次去找对应的车牌号范围。但写到70%的时候卡住了，
# 面试官给了提示，说可以把问题分成两部分——数字部分按10进制，字母部分按26进制来处理。但当时时间不够，没来得及转换成面试官的方法，最后没能做完。
def generate_license_plate(index):
    def to_letters(n):
        # 将数字 n 转换为 26 进制字母序列
        if n == 0:
            return ""
        letters = []
        while n > 0:
            n -= 1  # 转为从 0 开始的索引
            letters.append(chr(n % 26 + ord('A')))
            n //= 26
        return ''.join(reversed(letters))
    
    # 分离字母部分和数字部分
    letters_index = index // 100000  # 字母部分编号
    numbers = index % 100000         # 数字部分编号
    
    # 转换为字母和数字字符串
    letters = to_letters(letters_index)
    numbers_str = str(numbers).zfill(5)
    
    return letters + numbers_str

# 第二轮：有一个data stream找到最新的K个元素的average，但是元素会一直更新，用queue就好。
# follow-up：同样的data stream，但是要在最新的K个元素中去掉X个最大的元素然后求average。
# 楼主一开始说了个priority queue，面试官说不太efficiency，然后思考了一会儿用sortedList，压哨写完。
class SlidingWindowModifiedAverage:
    def __init__(self, k, x):
        self.k = k
        self.x = x
        self.sorted_window = SortedList()  # 维护有序窗口

    def add(self, val):
        # 添加新元素到窗口
        self.sorted_window.add(val)
        # 如果窗口大小超过 K，移除最老的元素
        if len(self.sorted_window) > self.k:
            self.sorted_window.pop(0)  # 删除最左边（最旧）的元素

    def get_modified_average(self):
        # 如果窗口不足 K 个元素
        if len(self.sorted_window) <= self.x:
            return 0
        
        # 去掉最大的 X 个元素后求平均值
        effective_window = self.sorted_window[:len(self.sorted_window) - self.x]
        return sum(effective_window) / len(effective_window)



# 第二轮：给一个array和prefix然后 detect number of word has prefix，然后这个array是sorted。array里面是string word，用binary search解决。
# # Create the lower bound prefix (e.g., "pre") and upper bound (e.g., "prez")
# start_prefix = prefix
# end_prefix = prefix[:-1] + chr(ord(prefix[-1]) + 1) if prefix else chr(0x10FFFF)

# # Find the range using binary search
# start_idx = bisect.bisect_left(sorted_array, start_prefix)
# end_idx = bisect.bisect_left(sorted_array, end_prefix)


# 第二轮: 设计一个waitlist的structure，满足三种功能，join waitlist, leave waitlist, serve customers with certain size. 
# serve customer这个部分我只需要检查有没有party size和table size完全一样的party就行了。 join 和 leave可能要考虑一下有没有重复加waitlist或者重复leave的可能。
class Waitlist:
    def __init__(self):
        self.waitlist = []  # List of tuples (party_size, name)
        self.parties = set()  # Set to track unique parties

    def join_waitlist(self, party_size, name):
        """Add a party to the waitlist."""
        if name in self.parties:
            print(f"Error: Party '{name}' is already on the waitlist.")
            return False
        self.waitlist.append((party_size, name))
        self.parties.add(name)
        print(f"Party '{name}' with size {party_size} added to the waitlist.")
        return True

    def leave_waitlist(self, name):
        """Remove a party from the waitlist."""
        for i, (_, party_name) in enumerate(self.waitlist):
            if party_name == name:
                del self.waitlist[i]
                self.parties.remove(name)
                print(f"Party '{name}' removed from the waitlist.")
                return True
        print(f"Error: Party '{name}' not found on the waitlist.")
        return False

    def serve_customer(self, table_size):
        """Serve the first party with a size matching the table size."""
        for i, (party_size, name) in enumerate(self.waitlist):
            if party_size == table_size:
                del self.waitlist[i]
                self.parties.remove(name)
                print(f"Party '{name}' with size {party_size} served.")
                return name
        print(f"No matching party found for table size {table_size}.")
        return None

# 第三轮：。给一个list of strings, 要求group strings if they are buddies. 
# Buddies 的定义是： 1. they have same length. 2. the distance between each character is same.
# 举例：“aaa” 和 “bbb”, 都是长度3， 并且a-a-a 的间隔是0-0-0，b-b-b也是，所以是buddies。“zab” 也和 “abc” 是buddies。input只有可能是 “”，或者a-z组合，string里不会有空格。
def group_buddy_strings(strings):
    def compute_pattern(s):
        if not s:
            return ()
        # Compute the distance pattern between consecutive characters
        return tuple((ord(s[i + 1]) - ord(s[i])) % 26 for i in range(len(s) - 1))

    groups = defaultdict(list)

    for string in strings:
        pattern = (len(string), compute_pattern(string))
        groups[pattern].append(string)

    return list(groups.values())

# 题目给一个list of subsequence，subsequence only contains integers.
# 这些subsequence 都是从某个 master sequence 删除一些element得来的。
# master sequence是一个没有重复数字的sequence，比如1, 2, 3, 4, 5, 那subsequence就可能是 1，2，5或者 2，3，4。
# 题目要求判断if all subsequence in the list could possibly come from one same master sequence。
# 比如如果出现1， 2， 5 和 2，1 就说明一个master sequence不可能得出这两个subsequence。
# 利用topological sort可以解决 或者DFS


# 大家会不停的使用Google Search。你调用一个function，叫search。
# search会给你一个timestamp，和用户搜索的内容，比如说“今天天气怎么样？”。
# timestamp是单调递增的。比如说你第一次call search的时候timestamp是5，第二次可能是6，可能是8，但是不可能是4。
# 那么如果说，两次call search的timestamp是小于60的，并且内容是一样的。那么就不用把他送给server，反之则是需要的。
# 比如说，用户在1的时候，提问了“今天天气怎么样？”，然后又在36的时候，提问了“今天天气怎么样？”，那么36秒的这个就不需要送给server。
# 但是如果在68秒的时候提问了“今天天气怎么样？”，那么就需要送给server。
# 小弟的思路是用一个queue来保存这些timestamp，然后用一个hashmap来保存这些提问的string。
from collections import defaultdict, deque

def group_buddy_strings(strings):
    """Group strings that are buddies."""
    def compute_pattern(s):
        if not s:
            return ()
        # Compute the distance pattern between consecutive characters
        return tuple((ord(s[i + 1]) - ord(s[i])) % 26 for i in range(len(s) - 1))

    groups = defaultdict(list)

    for string in strings:
        pattern = (len(string), compute_pattern(string))
        groups[pattern].append(string)

    return list(groups.values())

def validate_subsequences(subsequences):
    """Check if all subsequences could come from one master sequence."""
    # Create a mapping of element -> set of elements that must follow it
    dependency_graph = defaultdict(set)

    for subsequence in subsequences:
        for i in range(len(subsequence) - 1):
            dependency_graph[subsequence[i]].add(subsequence[i + 1])

    # Check for inconsistencies in the dependencies
    visited = set()

    def has_cycle(node, visiting):
        if node in visiting:
            return True
        if node in visited:
            return False

        visiting.add(node)
        for follower in dependency_graph[node]:
            if has_cycle(follower, visiting):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    for node in dependency_graph:
        if has_cycle(node, set()):
            return False

    return True

class SearchHandler:
    def __init__(self):
        self.timestamps = defaultdict(deque)  # Maps search content -> deque of timestamps

    def search(self, timestamp, query): 
        if query in self.timestamps:
            while self.timestamps[query] and timestamp - self.timestamps[query][0] >= 60:
                self.timestamps[query].popleft()

            # If there are still timestamps within 60 seconds, skip sending to server
            if self.timestamps[query]:
                print(f"Query '{query}' at {timestamp} is within 60 seconds, not sent to server.")
                return False

        # Add the current timestamp to the deque and send to server
        self.timestamps[query].append(timestamp)
        print(f"Query '{query}' at {timestamp} is sent to server.")
        return True


# bingo是5 * 5的棋盘。然后第一排是1-15的数，但是这五个数不能是重复的。第二排是16-30的数，第二排的五个数也不能是重复的。以此类推。我现在给你一个random(x, y)，他可以随机生成一个range从x到y的数。
# 然后你给我构造一个valid的bingo棋盘出来。
# 原来第一排就是从1-15里面sample5个数，这5个数不能是重复的。然后第二排就是从16-30里面sample五个数，这五个数也不能重复的。
# 然后小弟就开始写了。对于第一排，我构造一个hashset，重复的使用random(1，15)。如果说这个数在hashset里面，我就重复call，如果不在，那我就把他填到棋盘里面去。
# 大胡子老铁说，你这个不对。你这个可能会重复的使用random(1,15)，每个cell只能用一次。我想了一下，我说，那第一个cell我用random(1,3)，第二个cell我用random(4,6)，第三个cell我用random(7,9)
# 大胡子老铁：你这个也不对。因为你没有办法构造出every possible bingo case。random.sample(numbers, 5)
# 可以用index来sample。我把15个数字排成一个list，然后从1到15里面sample一个数，这个数是这个list的index。
# 选出这个数之后，把这个数从这个list里面pop出去，这个list里面就只有14个数字了。然后再从1到14里面sample一个数。
# 然后大胡子老铁又说，你这个还是不对！因为你用了pop操作，这个操作非常之费时。
# 小弟想了一下，把选出来的数，和list的最后一位swap了一下。这回大胡子老铁总算是满意了。
def generate_bingo_board():
    """Generate a valid 5x5 Bingo board."""
    board = []
    ranges = [(1, 15), (16, 30), (31, 45), (46, 60), (61, 75)]

    for col_range in ranges:
        column_numbers = random.sample(range(col_range[0], col_range[1] + 1), 5)
        board.append(column_numbers)

    # Transpose the board to match Bingo's column-row format
    bingo_board = list(map(list, zip(*board))) # * is used to unpack an iterator
    # Set the center cell to 'FREE'
    bingo_board[2][2] = "FREE"
    return bingo_board



# Serialize object to json string. Input could be number, string, array or dictionary. 
# If there is a cycle in an array or dictionary, serialize it to “recursion”. Example:
# a = {“k”, b} b = [a, 2]
# then serialize(a) = {“k”: [“recursion”, 2]}
def serialize(obj, visited=None):
    if visited is None:
        visited = set()  # Track visited objects to detect cycles

    # Handle primitive types
    if isinstance(obj, (int, float, str, type(None))):
        return obj

    # Handle lists
    if isinstance(obj, list):
        if id(obj) in visited: # 某个obj的唯一标识符
            return "recursion"  # Detected cycle
        visited.add(id(obj))
        return [serialize(item, visited) for item in obj]

    # Handle dictionaries
    if isinstance(obj, dict):
        if id(obj) in visited:
            return "recursion"  # Detected cycle
        visited.add(id(obj))
        return {key: serialize(value, visited) for key, value in obj.items()}

    # Handle unsupported types
    raise TypeError(f"Type {type(obj)} is not supported")


# 1. 给三个integers, 判断这三个integers组成的（a,b,c）是否构成一个valid date. any constraint on these input?
# yyyy-MM-dd也算valid， MM-dd-yyyy也算valid， 要考虑所以可能的合法的date格式。
# yyyy-MM-dd/-dd-mm, mm-yyyy-dd/-dd-yyyy, dd-mm-yyyy/-yyyy-mm
# year 1~2024; m 1~12; d，28/29/30 common year/ leap year
# 2. Follow up: 判断上一题的date是否是ambiguous
#     比如（2018,5,6）可以是2018年5月6号，也可以是2018年6月5号，这就算ambiguous，产生了歧义。

def is_valid_date(a, b, c):
    # 可以把format先init出来，然后把梳子放进去看是否成立！这样就简单多了！
    valid_dates = []
    formats = [
        (a, b, c),  # yyyy-MM-dd
        (a, c, b),  # yyyy-dd-MM
        (b, a, c),  # MM-yyyy-dd
        (b, c, a),  # MM-dd-yyyy
        (c, a, b),  # dd-yyyy-MM
        (c, b, a)   # dd-MM-yyyy
    ]
    
    for year, month, day in formats:
        if 1 <= year <= 2024 and 1 <= month <= 12:
            # Validate day against max days in the month
            try:
                datetime(year=year, month=month, day=day)
                valid_dates.append((year, month, day))
            except ValueError:
                pass
    return valid_dates
def is_ambiguous(a, b, c): # nb
    valid_dates = is_valid_date(a, b, c)
    return len(valid_dates) > 1


# coding 1: design a class that supports insertRange and queryPoint, which returns true if the given point is covered by the given ranges
# 一些followup包括optimize queryPoint to be O(1), points can be negatives, ranges can be continuous instead of discrete
# 两种方法，1. 只用set O(k),2. merge interval O(n), O(logn) 
class RangeManager:
    def __init__(self):
        # 用于存储非重叠、连续合并的区间
        self.ranges = []

    def insertRange(self, intervals, new_start, new_end):
        result = []
        index = 0
        # 1. insert if end < new_start 新intervals之前的全部塞进去。
        while index < len(intervals) and intervals[index][1] < new_start:
            result.append(intervals[index])
            index += 1

        # 2. insert new - [min_start, max_end] 
        if index == len(intervals) or new_end < intervals[index][0]: # 两种情况：最后一位没有overlap / 后面的没有overlap我们的new_interval
            result.append([new_start, new_end])
        else: # 一定有overlap，我们不管怎么overlap的，就把他俩合并放进来。
            result.append([min(new_start, intervals[index][0]), max(new_end, intervals[index][1])])
            index += 1
            

        # 3. finish up - check res[-1][1] and start
        while index != len(intervals):
            if result[-1][1] < intervals[index][0]: # 没有overlap
                result.append(intervals[index])
            else: # 有overlap，合并！取较大值。
                result[-1][1] = max(intervals[index][1], result[-1][1]) # 万一 new interval很大，比之后的所有都大，我们想维持更大的值
        self.ranges = result

    def queryPoint(self, point):
        """
        查询一个点是否在任何范围内。
        """
        # 使用二分查找
        low, high = 0, len(self.ranges) - 1
        while low <= high:
            mid = (low + high) // 2
            if self.ranges[mid][0] <= point <= self.ranges[mid][1]:
                return True
            elif point < self.ranges[mid][0]:
                high = mid - 1
            else:
                low = mid + 1
        return False
    
class RangeManagerOptimized:
    def __init__(self):
        self.covered_points = set()

    def insertRange(self, start, end):
        for i in range(start, end + 1):
            self.covered_points.add(i)

    def queryPoint(self, point):
        return point in self.covered_points




# coding 2 - 68: text justifications but only returns the lines required to accommodate the given text
# follow-up 1: one word might not fit into the width of the page所以需要wrap around到新的lines
# follow-up 2: text现在有newline operators，需要保证return的lines也考虑到newline operator
def justify_text(text, width):
    lines = []
    current_line = []
    current_length = 0

    words = text.split()
    for word in words:

        # followup-1
        while len(word) > width:
            # 如果当前行不为空，先将当前行加入结果
            if current_line:
                lines.append(" ".join(current_line))
                current_line = []
                current_length = 0
            # 将单词截取成一行
            lines.append(word[:width])
            word = word[width:]  # 剩余部分继续处理
    
        # 如果当前单词可以放入当前行
        if current_length + len(word) + len(current_line) <= width:
            current_line.append(word)
            current_length += len(word)
        else:
            # 当前行已满，添加到结果并开始新行
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)

    # 添加最后一行
    if current_line:
        lines.append(" ".join(current_line))

    return lines


# coding 3: given a swipeword string and a list of words, return the potential matches of the words in the swipeword
# 其实就是一个找subsequence的题，刚开始想成了trie但最后用的是map和binary search
# 这题的想法还是挺有趣的。如何找subsequence，比如有一个abc，先看a是不是满足，找到最小index的a，然后去找b[a_index, b_max_index]中找到最小的b_index, 如果能找完，那么就存在。
def find_matches(swipeword, words):
    # Step 1: Build the map of character positions
    char_positions = {}
    for index, char in enumerate(swipeword):
        if char not in char_positions:
            char_positions[char] = []
        char_positions[char].append(index)
    
    def is_subsequence(word):
        current_position = -1
        for char in word:
            if char not in char_positions:
                return False
            # Use binary search to find the next position
            pos_list = char_positions[char]
            next_pos_index = bisect_right(pos_list, current_position) # 精髓，直接用之前的index，在这个pos_list中去寻找，返回右侧的那个index
            if next_pos_index == len(pos_list):
                return False
            current_position = pos_list[next_pos_index]
        return True
    
    # Step 2: Check each word in the list
    result = [word for word in words if is_subsequence(word)]
    return result

# Example usage
swipeword = "abppplee"
words = ["able", "ale", "apple", "bale", "kangaroo"]

# 第二轮：Binary Tree求岛屿数量 双重DFS
def count_islands(root):
    if not root:
        return 0

    # Helper function for DFS traversal
    def dfs(node):
        if not node or node.val == 0:
            return
        node.val = 0
        dfs(node.left)
        dfs(node.right)
        # BFS helper function
    def bfs(start_node):
        queue = deque([start_node])
        while queue:
            node = queue.popleft()
            if node:
                # Mark the node as visited
                node.val = 0
                # Add left and right children to the queue if they are not visited
                if node.left and node.left.val == 1:
                    queue.append(node.left)
                if node.right and node.right.val == 1:
                    queue.append(node.right)

    island_count = 0
    stack = [root]  # Stack for DFS
    while stack:
        node = stack.pop()
        if node and node.val == 1:  # Found a new island
            island_count += 1
            dfs(node)  # Explore the entire island
        # Push children to stack for further exploration
        if node:
            stack.append(node.left)
            stack.append(node.right)

    return island_count

# 第三轮：给一个array，其中两两成对，第一个是数字，第二个是前面数字重复个数，写一个iterator, 包括有下一个和下一个方法。比如[1,2,4,3]，意思就是1重复2次，4重复3次。tricky之处在于数字可以重复零次。
class RepeatIterator:
    def __init__(self, arr):
        self.arr = arr
        self.index = 0  # Pointer to the current position in the array
        self.repeat_count = 0  # Remaining count for the current number
        self.current_value = None

    def next(self):
        # Ensure there is a next element
        if not self.hasNext():
            raise StopIteration("No more elements")
        
        # If no remaining repeats, move to the next pair
        if self.repeat_count == 0:
            self.current_value = self.arr[self.index]  # Get the current value
            self.repeat_count = self.arr[self.index + 1]  # Get its repeat count
            self.index += 2  # Move to the next pair

        # Return the current value and decrease the remaining count
        self.repeat_count -= 1
        return self.current_value

    def hasNext(self):
        # Check if we have remaining repeats or unprocessed pairs
        return self.repeat_count > 0 or self.index < len(self.arr)


# Round 2 美国大选，每个州对应不同数量的选举人票，求算number of the combination of states that to win the elections. - backtracking
# zheyit
def count_combinations(votes):
    target = sum(votes) // 2 + 1
    n = len(votes)
    result = [0]  # Use a list to store the count as a reference

    def backtrack(index, current_sum):
        # If current sum already exceeds the target, count it as a valid combination
        if current_sum >= target:
            result[0] += 1
            return
        # If out of bounds, stop recursion
        if index == n: return
        # Prune: If current sum plus remaining votes cannot reach the target
        if current_sum + sum(votes[index:]) < target: return
        
        # Include the current state
        backtrack(index + 1, current_sum + votes[index])
        # Exclude the current state
        backtrack(index + 1, current_sum)

    backtrack(0, 0)
    return result[0]

# Round 3 类似于群聊的log file，每一行有timestamp， username，text message。
    # Part 1: 求问找到most talkative的user。很简单hashmap。
    # Part 2: 找到top k most talkative users，用的heap， 
    # Part 3: implement helper function parse_log()，主要是string的操作，每一题都问了runtime。


# merge interval变种 (背景是schedule meeting)，给了个interval叫BNS, 在BNS中不能安排meeting
# follow up是BNS有多个，变成了list
def merge_intervals(intervals):
    if not intervals:
        return []
    
    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged

def available_intervals(intervals, bns_list):
    # Step 1: Merge BNS intervals
    merged_BNS = merge_intervals(bns_list)

    # Step 2: Calculate available intervals
    result = []
    for interval in intervals:
        start, end = interval
        current_start = start
        for bns_start, bns_end in merged_BNS:
            if bns_end <= current_start:  # BNS is completely before the current interval
                continue
            if bns_start >= end:  # BNS is completely after the current interval
                break
            if bns_start > current_start:  # Add the part before the BNS
                result.append([current_start, bns_start])
            current_start = max(current_start, bns_end)  # Update current_start after BNS
        if current_start < end:  # Add the remaining part of the interval
            result.append([current_start, end])
    
    return result


# dp，A是长度为N的list，A[i]表示第i天的航行路程。K是energy的最大值，如果选择航行energy-1，否则energy +1但不能超过K
# 一开始用了recursion，面试官提醒可以储存计算过的结果，于是加了一个二维数组储存。
# 问了time complexity和space complexity
def min_energy(A, K):
    N = len(A)

    # DP table: dp[i][j] represents the minimum energy cost at day i with energy j
    dp = [[float('inf')] * (K + 1) for _ in range(N)]

    # Initialize the first day
    dp[0][K] = 0  # Start with full energy

    # Fill the DP table
    for i in range(1, N):
        for j in range(K + 1):
            if j + 1 <= K:  # If we can choose to rest and gain energy
                dp[i][j] = min(dp[i][j], dp[i-1][j+1])
            if j - 1 >= 0:  # If we can choose to sail and consume energy
                dp[i][j] = min(dp[i][j], dp[i-1][j-1] + A[i])

    # Find the minimum cost at the last day
    return min(dp[N-1])

# follow up是如何improve space complexity -> 利用prev和curr = [inf] * (k+1)
# 、和如果加一个列表H，H[i]＝1表示第i天是假期只能休息
#   if H[i] == 1:  # Holiday: can only rest
#     if j + 1 <= K:
#         curr[j] = min(curr[j], prev[j+1])

# 订票系统，卖家输入票子信息，用户买票子，follow-up允许卖家撤回票子，要求实时输出最便宜的票子，额外讨论了OOD设计
class Ticket:
    def __init__(self, price, details):
        self.price = price
        self.details = details
        self.is_active = True  # Mark if ticket is still valid

class QueryEngine:
    def __init__(self):
        self.min_heap = []  # Min-Heap for efficient price queries
        self.ticket_map = {}

    def add_ticket(self, ticket_id, ticket):
        heapq.heappush(self.min_heap, (ticket.price, ticket_id))
        self.ticket_map[ticket_id] = ticket

    def get_cheapest_ticket(self):
        while self.min_heap:
            price, ticket_id = heapq.heappop(self.min_heap)
            if ticket_id in self.ticket_map and self.ticket_map[ticket_id].is_active:
                heapq.heappush(self.min_heap, (price, ticket_id))
                return self.ticket_map[ticket_id]
        return None

    def remove_ticket(self, ticket_id):
        if ticket_id in self.ticket_map:
            self.ticket_map[ticket_id].is_active = False

class TicketSystem:
    def __init__(self):
        self.query_engine = QueryEngine()
        self.ticket_id = 0

    def add_ticket(self, price, details):
        self.ticket_id += 1
        ticket = Ticket(price, details)
        self.query_engine.add_ticket(self.ticket_id, ticket)
        return self.ticket_id

    def get_cheapest_ticket(self):
        return self.query_engine.get_cheapest_ticket()

    def remove_ticket(self, ticket_id):
        self.query_engine.remove_ticket(ticket_id)

# 各大排序/气泡/快排
def bubble_sort(arr): # O(N2)
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):  # 每次减少未排序的部分
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:  # 如果没有发生交换，说明已排序
            break
    return arr

def quick_sort(arr): #O(nlogn) / O(logn) stack depth
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]  # 选择中间元素为基准
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def insertion_sort(arr): # O(n2)
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        # Move elements of arr[0..i-1], that are greater than key, to one position ahead
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def merge_sort(arr): # O(logn)/O(nLogn)
    if len(arr) <= 1:
        return arr  # Base case: an array of 1 element is sorted
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])  # Sort the left half
    right = merge_sort(arr[mid:])  # Sort the right half
    
    return merge(left, right)  # Merge the two halves

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    # Append the remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# https://leetcode.com/discuss/interview-question/5359320/Google-L3-Onsite-Second-Round
# 主体用%分隔开就可以了。
# 困难的点在于后面的dict，可能彼此也有nested relationship.
# resolved_variables
def resolve_nested_variables(variables):
    resolved = {}
    for key, value in variables.items():
        while "%" in value:
            start = value.find("%")
            end = value.find("%", start + 1)
            nested_key = value[start + 1:end]
            value = value[:start] + variables[nested_key] + value[end + 1:]
        resolved[key] = value
    return resolved
variables = {"USER": "%PRONOUN% John", "PRONOUN": "Mr."}
resolved_variables = resolve_nested_variables(variables)
print(resolved_variables)  # Output: {'USER': 'Mr. John', 'PRONOUN': 'Mr.'}

# https://leetcode.com/discuss/interview-question/5359320/Google-L3-Onsite-Second-Round
# backtrack很简单
mp = {
    1  : [2,3],
    2 : [1, 2],
    3 : [1]
}
res =[]
target = 12
def backtrack(cur  , cur_sum , cur_list, start):
    #base case
    if cur  + cur_sum == target and cur in mp[start]:
        res.append(cur_list + [cur])
        return
    
    if cur + cur_sum > target: return         
    vals = mp[cur]
    for v in vals: backtrack(v , cur+cur_sum , cur_list+[cur], start)
        
for i in range(1 , 4): 
    backtrack(i, 0 , [] , i)
    
print(res)

# https://leetcode.com/discuss/interview-question/5369393/Google-Onsite-Final-Coding-Interview 类似UF

# 稍微看一看，最后三天用于复习思路。
https://leetcode.com/discuss/interview-question/1749266/Google-or-Phone-Screen-or-SWE
https://leetcode.com/discuss/interview-question/1673287/Google-or-Virtual-Onsite-or-Maximum-Ancestor-for-Leaves
https://leetcode.com/discuss/interview-question/325845/Google-or-Onsite-or-Decode-string
https://leetcode.com/discuss/interview-question/5092478/Recent-Google-Interview-Questions-L3L4
https://leetcode.com/discuss/interview-question/4342705/Latest-Google-Interview-Questions



