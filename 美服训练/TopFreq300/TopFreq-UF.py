# 200. Number of Islands
from platform import node


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




# 721. Accounts Merge
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


# 399. Evaluate Division
# 这道题太难了。
# union find思路也很难想到，因为是元素之间的关系，还有权重，因此我们的parent可以从通常的list变成块状结构。
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:

        # 这道题的数据结构有点类似什么呢？ 块状结构，
        """
        class Node():
            def __init__(self, node, weight):
                self.node = node
                self.next = Node
                self.weight = weight
        """    
        gid_weight = {}  # gid_weight装了 字母 + 与其对应的字母 + 两者之间的关系

        def find(node_id):
            if node_id not in gid_weight:
                gid_weight[node_id] = (node_id, 1)
            group_id, node_weight = gid_weight[node_id]
            # Lazy Update：我们在第一次update的时候并没有把所有的节点都同步到一个parent上，而只是单纯的连接起来
            # 而在main function中我们才会进行合并加载并更新，比如我们遇到a，但是a依赖于b，那么我们就会find(b)，发现b依赖于c，接着就回find(c)
            # c的group_id == node_id，会返回到上一层(b)，给到weight和group_id，然后将b这一层的group_id和weight计算后返回给最上层a，4
            # 然后a接着计算，最后返回a与c的关系到main function中 # 但是说实话最下层的话计算有点浪费了觉得
            if group_id != node_id: 
                new_group_id, group_weight = find(group_id)
                gid_weight[node_id] = \
                    (new_group_id, node_weight * group_weight)
            return gid_weight[node_id]

        def union(dividend, divisor, value):
            dividend_gid, dividend_weight = find(dividend)
            divisor_gid, divisor_weight = find(divisor)
            if dividend_gid != divisor_gid:
                gid_weight[dividend_gid] = \
                    (divisor_gid, divisor_weight * value / dividend_weight)

        
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

# 1319. Number of Operations to Make Network Connected
class Solution:
    def makeConnected(self, n: int, connections: List[List[int]]) -> int:
        if len(connections) < n-1: return -1
        parent = [i for i in range(n)]
        def find(x):
            if x != parent[x]:
                parent[x] = find(parent[x])
            return parent[x]
    
        def union(x, y):
            px, py = find(x), find(y)
            if find(x) == find(y):
                return 
            parent[find(x)] = find(y) #当parent中x_parent这一位与y_root链接起来
            
        for s, e in connections:
            union(s,e)
        return len([v for i,v in enumerate(parent) if i == v]) - 1
        
#         G = [set() for _ in range(n)]
#         for i, j in connections:
#             G[i].add(j)
#             G[j].add(i)
        
#         seen = [0] * n
#         def dfs(i):
#             if seen[i]: return 0
#             seen[i] = 1
#             for j in G[i]: dfs(j)
#             return 1
        
#         return sum(dfs(i) for i in range(n)) - 1


# 684. Redundant Connection
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        n = len(edges)
        parent = [i for i in range(n+1)]
        
        def find(x):
            if x != parent[x]:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for s, e in edges:
            if find(s) == find(e):
                result = [s,e]
            union(s,e)
        return result