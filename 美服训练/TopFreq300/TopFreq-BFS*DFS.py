# 394. Decode String
# 二刷
from collections import defaultdict
import collections


class Solution(object):
    def decodeString(self, s):
        # curString新理解，这个变量一直focus在当前最内层括号的string；如果没有括号就是全部string
        # stack是用来辅助curString的从某个方面看，之前需要重复几次/括号外的string是什么。
        # curNum就是个打酱油的helper
        stack = []; curNum = 0; curString = ''
        for c in s:    
            print(stack)
            if c == '[':
                # 还是要明白为什么[的时候要into stack
                stack.append(curString)
                stack.append(curNum)
                curString = ''
                curNum = 0
            
            elif c == ']':
                num = stack.pop()
                prevString = stack.pop()
                curString = prevString + num*curString
            
            elif c.isdigit():
                curNum = curNum*10 + int(c)
                
            else:
                curString += c
        return curString


# 721. Accounts Merge
# 这一题建立map是很关键的
class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        visited = [False] * len(accounts)
        emails_map = defaultdict(list)
        res = []
        
        # Build the map. Key是email地址, Value是index表示什么人。
        for i, ac in enumerate(accounts):
            for j in range(1, len(ac)):
                email = ac[j]
                emails_map[email].append(i)
        
        def dfs(i, emails):
            if visited[i]:
                return 
            visited[i] = True
            for j in range(1, len(accounts[i])):
                email = accounts[i][j]
                emails.add(email)
                for neighbor in emails_map[email]:
                    dfs(neighbor, emails)

        for i, ac in enumerate(accounts):
            if visited[i]: continue    
            name, emails = ac[0], set()
            dfs(i, emails)
            res.append([name] + sorted(emails))
            
        return res

# 这一题天生适合玩union find
class UnionFind:
    def __init__(self, N):
        # 长度为N的list，index和val是一一对应的[0,1,2,3,4...]
        self.parents = list(range(N))
    def find(self, x):
        if x != self.parents[x]:
            self.parents[x] = self.find(self.parents[x])
        return self.parents[x]
    def union(self,child, parent):
        self.parents[self.find(child)] = self.find(parent)

class Solution:
    def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
        uf = UnionFind(len(accounts))
        ownership = {}
        # i, (_,*emails)用的好呀
        for i, (_, *emails) in enumerate(accounts):
            for email in emails:
                # 如果email已经遍历过了，那么就union起来。
                if email in ownership:
                    uf.union(i, ownership[email])
                ownership[email] = i
        # owner最后是什么？key是每一个email，value是email对应的某一个index（name）
        
        # 把ownership的dict转化
        ans = collections.defaultdict(list)
        for email, owner in ownership.items():
            ans[uf.find(owner)].append(email)
        
        return [[accounts[i][0]] + sorted(emails) for i, emails in ans.items()]


# 547. Number of Provinces
class Solution:
    def findCircleNum(self, M):
        n = len(M)
        seen = set()
        
        
        def dfs(curIndex):
            for index, neighbor in enumerate(M[curIndex]):
                if neighbor and index not in seen:
                    seen.add(index)
                    dfs(index)
        
        res = 0
        for i in range(n):
            if i not in seen:
                dfs(i)
                res += 1
        return res
    
# 695. Max Area of Island
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        def dfs(grid, row, col):
            if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) or grid[row][col] == 0:
                return 0
            area = 1
            grid[row][col] = 0
            for x, y in ((1,0),(0,1),(-1,0),(0,-1)):
                area += dfs(grid, row+x, col+y)
            return area   
            
        result = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    result = max(result, dfs(grid, i,j))
        
        return result

# 用while循环模拟dfs，我王姐了
class Solution(object):
    def maxAreaOfIsland(self, grid):
        seen = set()
        ans = 0
        for r0, row in enumerate(grid):
            for c0, val in enumerate(row):
                if val and (r0, c0) not in seen:
                    shape = 0
                    stack = [(r0, c0)]
                    seen.add((r0, c0))
                    while stack:
                        r, c = stack.pop()
                        shape += 1
                        for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                            if (0 <= nr < len(grid) and 0 <= nc < len(grid[0])
                                    and grid[nr][nc] and (nr, nc) not in seen):
                                stack.append((nr, nc))
                                seen.add((nr, nc))
                    ans = max(ans, shape)
        return ans



# 130. Surrounded Regions
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        if not board or not board[0]:
            return

        self.ROWS = len(board)
        self.COLS = len(board[0])
        # product()是两个iterables的笛卡尔积
        borders = list(itertools.product(range(self.ROWS), [0, self.COLS-1])) + list(itertools.product([0, self.ROWS-1], range(self.COLS)))
        for row, col in borders:
            self.DFS(board, row, col)
            
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if board[r][c] == 'O':   board[r][c] = 'X'  # captured
                elif board[r][c] == 'E': board[r][c] = 'O'  # escaped
           
    def DFS(self, board, row, col):
        if board[row][col] != 'O':
            return
        board[row][col] = 'E'
        if col < self.COLS-1: self.DFS(board, row, col+1)
        if row < self.ROWS-1: self.DFS(board, row+1, col)
        if col > 0: self.DFS(board, row, col-1)
        if row > 0: self.DFS(board, row-1, col)


# 1631. Path With Minimum Effort

class Solution:
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        # init row/col/diff
        row = len(heights)
        col = len(heights[0])
        difference_matrix = [[math.inf]*col for _ in range(row)]
        
        difference_matrix[0][0] = 0
        visited = [[False]*col for _ in range(row)]
        queue = [(0, 0, 0)]  # difference, x, y
        
        
        while queue:
            # get当前diff最少的
            difference, x, y = heapq.heappop(queue)
            # visited更新
            visited[x][y] = True
            
            for dx, dy in [[0, 1], [1, 0], [0, -1], [-1, 0]]:
                adjacent_x = x + dx
                adjacent_y = y + dy
                # 如果没有visit过，visit过的diff肯定会大于等于了，因为这里我们用的是heapq.heappop()
                if 0 <= adjacent_x < row and 0 <= adjacent_y < col and not visited[adjacent_x][adjacent_y]:
                    # 计算cur_diff
                    current_difference = abs(heights[adjacent_x][adjacent_y]-heights[x][y])
                    # 计算目前碰到过的最大max_diff
                    max_difference = max(current_difference, difference_matrix[x][y])
                    # 这里大于只有一个用处，更新difference_matrix从inf->cur_diff
                    if difference_matrix[adjacent_x][adjacent_y] > max_difference:
                        difference_matrix[adjacent_x][adjacent_y] = max_difference
                        heapq.heappush(
                            queue, (max_difference, adjacent_x, adjacent_y))
        return difference_matrix[-1][-1]        


# 207. Course Schedule
# 很难的一道题
# 这道题有意思的点在于，针对一门课，他的图是怎么样的。
class Solution:
    def canFinish(self, numCourses: int, pre: List[List[int]]) -> bool:
        n = numCourses
        graph = [[] for _ in range(n)]
        visited = [0 for _ in range(n)]        
        # 针对某一门x课，它有多少个先修课ys
        for x, y in pre:
            graph[x].append(y)
        
        # 这里的visited也有意思，有点利用了backtracking的思想。
        # 在同一的track下碰到i先变-1，是因为两门课互相依赖的话，肯定就修不完所有的了。
        # 理想情况下，肯定有课为0；遍历结束后意味着i这门课是可以修完的，那么之后再碰到i的依赖就不用再dfs了，因此变为1直接返回True
        def dfs(i):
            if visited[i] == -1:
                return False
            if visited[i] == 1:
                return True
            
            visited[i] = -1
            for j in graph[i]:
                if not dfs(j):
                    return False
            visited[i] = 1
            return True
        
        
        for i in range(n):
        
            if not dfs(i):
                return False
        return True
            
        
# 279. Perfect Squares
class Solution:
    def numSquares(self, n):
        # 这个square_num是必须的。而且level可以用一个变量进行存储，不必使用tuple入栈。
        square_nums = [i * i for i in range(1, int(n**0.5)+1)]
        level = 0
        queue = {n}
        # 如果使用list.pop的话会tle，具体原因看下方注释
        while queue:
            level += 1
            #！! Important: use set() instead of list() to eliminate the redundancy,
            # which would even provide a 5-times speedup, 200ms vs. 1000ms.
            next_queue = set()
            for remainder in queue:
                for square_num in square_nums:    
                    if remainder == square_num:
                        return level  # find the node!
                    elif remainder < square_num:
                        break
                    else:
                        next_queue.add(remainder - square_num)
            queue = next_queue
        return level
        
# 这是DP的解法
class Solution:
    def numSquares(self, n):
        square_nums = [i**2 for i in range(0, int(math.sqrt(n))+1)] # int是floor所以要+1
        
        dp = [float('inf')] * (n+1)
        # bottom case
        dp[0] = 0
        
        for i in range(1, n+1):
            for square in square_nums:
                # 如果i小于square没必要看了，直接跳过。
                if i < square:
                    break
                dp[i] = min(dp[i], dp[i-square] + 1)
        
        return dp[-1]

# 1319. Number of Operations to Make Network Connected
# 这一题中关于点与点的matrix向Graph转化值得学习。
# 难点还在于如何解题
class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        if len(connections) < n-1: return -1
        G = [set() for _ in range(n)]
        for i, j in connections:
            G[i].add(j)
            G[j].add(i)
        
        seen = [0] * n
        def dfs(i):
            if seen[i]: return 0
            seen[i] = 1
            for j in G[i]: dfs(j)
            return 1
        
        return sum(dfs(i) for i in range(n)) - 1
# 下面是Union Find的方法，也比较好理解。
def makeConnected(self, n: int, connections: List[List[int]]) -> int:
    parents = list(range(n))
    def findParent(c):
        while c != parents[c]:
            c = parents[c]
        return parents[c]

    extraEdgeCount = 0
    for c1, c2 in connections:
        p1 = findParent(c1)
        p2 = findParent(c2)
        # 看看该edge是不是多余的，如果不是多余的就连接起来，是多余的就记录下来。
        if p1 == p2:
            extraEdgeCount += 1
        else:
            parents[p1] = p2
    # 看看有多少群network，这个trick也很重要！！
    connectedNetworkCount = sum(parents[c] == c for c in range(n))
    if extraEdgeCount < connectedNetworkCount - 1:
        return -1
    return connectedNetworkCount - 1


# 934. Shortest Bridge
# 三步走：1.找到第一个为1的cell 2.dfs找到整个岛屿的cell并且入bfs 3.bfs往外走找到第一个1就是另一个岛屿，返回step
# 需要注意的是：dfs中已经把当前岛屿变为-1，而且bfs的基础是所有岛屿，而非只有岛屿轮廓。
class Solution:
    def shortestBridge(self, A):
        def dfs(i, j):
            A[i][j] = -1
            bfs.append((i, j))
            for x, y in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if 0 <= x < n and 0 <= y < n and A[x][y] == 1:
                    dfs(x, y)
        
        def first():
            for i in range(n):
                for j in range(n):
                    if A[i][j]:
                        return i, j
        n, step, bfs = len(A), 0, []
        # 这个感叹号的用法哈
        dfs(*first())
        while bfs:
            # 为什么总是用一个temperary去存数据，而不用pop和append，因为效率低。
            new = []
            for i, j in bfs:
                for x, y in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                    if 0 <= x < n and 0 <= y < n:
                        if A[x][y] == 1:
                            return step
                        elif not A[x][y]:
                            A[x][y] = -1
                            new.append((x, y))
            step += 1
            bfs = new

# 785. Is Graph Bipartite?
# 这一道题一个非常关键的点就是在于认识到二分图的特性。一个node和它相邻的所有node都不应该是一组！
# 知道这个特性的时候，我们就要利用一个分组变量为所有的node打标签，去判断他的下一个变量与当前变量是否为同一组，同一组的话那么就意味着不满足bipartite.
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        # color不止有分组的作用，还有是否已经遍历过的作用！利用0/1分组。
        color = {}
        def dfs(pos):
            for i in graph[pos]:
                if i in color:
                    if color[i] == color[pos]:
                        return False
                else:
                    color[i] = 1 - color[pos]
                    if not dfs(i):
                        return False
            return True
        for i in range(len(graph)):
            if i not in color:
                color[i] = 0
                # 其实你自己清楚dfs的具体是什么作用会对你理解题目有很大的帮助。
                if not dfs(i):
                    return False
        return True


# 994. Rotting Oranges
# S1找到所有rotten进stack；S2开始BFS，边界/空/rotten跳过；S3return和边界判断; 注意level，这里的增加需要额外判断一波。
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        flag = 0
        level = 0
        stack = []
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1: flag += 1
                elif grid[i][j] == 2: stack.append((i,j))
        
        change = 0
        while stack:
            new_stack = []
            temp_flag = 0
            for row, col in stack:
                for dx,dy in ((1,0),(0,-1),(-1,0),(0,1)):
                    ni, nj = row+dx, col+dy
                    if ni < 0 or ni >= len(grid) or nj < 0 or nj >= len(grid[0]) or grid[ni][nj] in (0,2):
                        continue
                    grid[ni][nj] = 2
                    change += 1
                    new_stack.append((ni,nj))
                    temp_flag = 1
            if temp_flag: level += 1 # 你看这里的level需要这样变化，但是你可以直接用[val,level]入栈的方式，这样就不用这一步操作了。tradeoff
            stack = new_stack
        return level if flag == change else -1
                

# 752. Open the Lock
class Solution(object):
    def openLock(self, deadends, target):
        def neighbors(node):
            for i in range(4):
                x = int(node[i])
                for d in (-1, 1):
                    y = (x + d) % 10
                    yield node[:i] + str(y) + node[i+1:]
 
        dead = set(deadends)
        queue = collections.deque([('0000', 0)])
        seen = {'0000'}
        while queue:
            node, depth = queue.popleft()
            if node == target: return depth
            if node in dead: continue
            for nei in neighbors(node):
                if nei not in seen:
                    seen.add(nei)
                    queue.append((nei, depth+1))
        return -1


# 1162. As Far from Land as Possible
# 这道题不难，但是我理解出错了！我focus在水上，就不容易；这题应该focus在land上。
class Solution:
    def maxDistance(self, grid: List[List[int]]) -> int:
        m,n = len(grid), len(grid[0])
        q = deque([(i,j) for i in range(m) for j in range(n) if grid[i][j] == 1])    
        if len(q) == m * n or len(q) == 0: return -1
        level = 0
        while q:
            size = len(q)
            for _ in range(size):
                i,j = q.popleft()
                for x,y in [(1,0), (-1, 0), (0, 1), (0, -1)]:
                    xi, yj = x+i, y+j
                    if 0 <= xi < m and 0 <= yj < n and grid[xi][yj] == 0:
                        q.append((xi, yj))
                        grid[xi][yj] = 1
            level += 1
        return level-1
                    
                    
        