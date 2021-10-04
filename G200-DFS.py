"""
DFS主要解决 可达性 的问题
考虑两点：
1. 栈：用栈来保存当前节点信息，当遍历新节点返回时能够继续遍历当前节点。可以使用递归栈。
2. 标记：和 BFS 一样同样需要对已经遍历过的节点进行标记。
"""

leetcode-695 
#看答案的思路自己写
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:   
        if not grid:
            return 0 
        ans = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    area = self.helper(grid, i, j)
                    ans = max(area, ans)
        return ans

    def helper(self, grid: List[List[int]],i: int, j: int) -> int:
        if i == len(grid) or i < 0: 
            return 0 
        elif j == len(grid[0]) or j < 0:
            return 0 
            
        if grid[i][j] == 1:
            grid[i][j] = 0
            return 1 + 
        return 0 #这里如果不写，会导致helper返回的值为NoneType，从而引起error
#点评：时间复杂度不太好，这里的helper也可以写成内置方法，或许会好些

leetcode-200
class Solution:
    def numIslands(self, grid: List[List[int]]) -> int:   
        if not grid:
            return 0 

        ans = 0

        for i in range(len(grid)):
            for j in range(len(grid[0])):

                if grid[i][j] == "1" :
                    ans += 1
                    self.helper(grid, i ,j) #消除其他陆地
        return ans
    
    #这里递归就是为了消除1。因为只要进入到helper我们就会记录这个岛屿的数量
    def helper(self, grid: List[List[int]],i: int, j: int) -> int:
        """没必要写这些
        if i == len(grid) or i < 0: 
            return False
        elif j == len(grid[0]) or j < 0:
            return False
        """
        if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == "1":  #=1不能省略，否则无限循环将所有值变成0
            grid[i][j] = "0"
        
            #记住缩进：只有grid是1满足，才会继续递归。
            self.helper(grid, i+1, j)
            self.helper(grid, i, j+1) 
            self.helper(grid, i-1, j)  
            self.helper(grid, i, j-1)

        #return True —— 这里没必要写，因为if的存在限制了无限递归.
        #md缩进问题又一次

leetcode-547
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        def findCluster(i):
            for j in range(m):
                if isConnected[i][j] == 1 and j not in visited:     #🌟 这里的j其实是正在遍历城市的关联城市的代号
                    visited.append(j)
                    findCluster(j)
        num = 0
        m = len(isConnected)
        visited = []

        for i in range(m):  #这一题很神奇的一点就是正方形矩阵，因此索引下标就可以代表一座城市
            if i not in visited: 
                num += 1
                findCluster(i)
        return num
#思路：从头遍历，没有遍历过的话+1；成功遍历后，去找其关联城市（递归），看其关联城市的关联城市，是否已经遍历，如果没有，进入递归，并且将城市不断加入遍历城市中。
#这题的索引十分有趣，可以充当城市的代号


leetcode-130
#自己写的——写不出来，但是边界应用的挺好。
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        def solveHelper(board, i, j):
            if 0 <= i < len(board) and 0 <= j < len(board) and (i,j) not in visited:
                visited.append((i,j))
                if i == 0 or i == len(board) - 1 or j == 0 or j == len(board[0]) - 1:
                    count = 1

         
        visited = []

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == "O" and (i,j) not in visited:
                    count = 0
                    solveHelper(board, i, j)

#题解
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        if not board:           #根据题意不用写其实
            return      
        
        n, m = len(board), len(board[0])

        def dfs(x, y):
            if not 0 <= x < n or not 0 <= y < m or board[x][y] != 'O':
                return
            
            board[x][y] = "A"
            dfs(x + 1, y)
            dfs(x - 1, y)
            dfs(x, y + 1)
            dfs(x, y - 1)
        
        for i in range(n):
            dfs(i, 0)   
            dfs(i, m - 1)
        
        for i in range(m - 1):
            dfs(0, i)
            dfs(n - 1, i)
        
        for i in range(n):
            for j in range(m):
                if board[i][j] == "A":
                    board[i][j] = "O"
                elif board[i][j] == "O":
                    board[i][j] = "X"
#先处理特殊值！
#思路牛：因为题意，遍历四个边框和相连接的O，将O值改为A；最后递归完矩阵中会有X，A，O；然后遍历所有修改就好了
#时间复杂度为nm，最主要的是因为最后一个双for循环。


leetcode-417
class Solution:
    def bfs(self, heights: List[List[int]], src: List[List[int]], cnt: List[List[int]]) -> None:
        direction = [(-1, 0), (1, 0), (0, 1), (0, -1)] #河流的四个方向
        m, n = len(heights), len(heights[0])
        visited = [[False] * n for _ in range(m)]
        from collections import deque
        q = deque(src)
        while q:
            x, y = q.popleft()
            for i, j in direction:
                row, col = x + i, y + j #在起始点沿四个方向出发
                if 0 <= row < m and 0 <= col < n and not visited[row][col] and \
                    (x in (-1, m) or y in (-1, n) or heights[row][col] >= heights[x][y]): #不超过边界/没有遍历过/向地势高的地方走/在边界 #这里的(-1,n)是两个元素而非range
                    visited[row][col] = True
                    cnt[row][col] += 1
                    q.append((row, col))
    
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        m, n = len(heights), len(heights[0])
        cnt = [[0] * n for _ in range(m)]   #最后的cnt长度为m，每一个元素的长度为n
        # 1. 从太平洋逆流而上，能达到的陆地进行标记
        self.bfs(heights, [(-1, col) for col in range(n)] + [(row, -1) for row in range(m)], cnt)      #这里边界为-1，是因为之后有row和col，以免index超出边界
        # 2. 从大西洋逆流而上，能达到的陆地进行标记
        self.bfs(heights, [(m, col) for col in range(n)] + [(row, n) for row in range(m)], cnt)
        # 3. 筛选出 cnt[row][col] = 2 的坐标，即既满足太平洋又满足大西洋
        return [[row, col] for row in range(m) for col in range(n) if cnt[row][col] == 2]

#Key Take-Away: 涉及边界问题的话，我们可以从边界满足（从底）开始遍历；
#设计思想：两个方法只做两件事，第一个方法按条件遍历并改变，第二个方法针对结果进行判断
#这题我们用了cnt，visted两个list，q一个deque，xy与ij表示前后的坐标，还有row/col。
#这一题相当于dfs+bfs
