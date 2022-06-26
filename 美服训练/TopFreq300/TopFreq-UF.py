# 200. Number of Islands
class Solution(object):
    def numIslands(self, grid):    
        if len(grid) == 0: return 0
        row = len(grid), col = len(grid[0])
        self.count = sum(grid[i][j] == '1' for i in range(row) for j in range(col))
        # 把所有的岛屿加起来，后续如果合并的话，就-1，最终就是有多少独立的岛屿。       
        parent = [i for i in range(row*col)]
        def find(x):
            if parent[x] != x:
                return find(parent[x])
            return parent[x]
        
        def union(x,y):
            xroot, yroot = find(x), find(y)
            if xroot == yroot: return
            parent[xroot] = yroot
            self.count -= 1
        
        
        for i in range(row):
            for j in range(col):
                print(parent)
                if grid[i][j] == '0':
                    continue
                index = i*col + j
                if j < col-1 and grid[i][j+1] == '1':
                    union(index, index+1)
                if i < row-1 and grid[i+1][j] == '1':
                    union(index, index+col)
        return self.count


