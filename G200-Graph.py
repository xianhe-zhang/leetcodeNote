"""图好像一般不太会考察

但是这几道题好经典呀！
785 我学会了Union find；
207 要学拓扑排序

"""

# 785 Is graph bipartite?
# BFS/DFS/Union Found

# DFS
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        # 有多少个节点
        n = len(graph)
        UNCOLORED, RED, GREEN = 0, 1, 2
        color = [UNCOLORED] * n
        valid = True

        def dfs(node: int, c: int):
            # 这里将valid从外层引入
            nonlocal valid

            # 将当前node染色
            color[node] = c

            # 根据当前node颜色，判读下一个相邻node的颜色。
            cNei = (GREEN if c == RED else RED)

            # 遍历相邻的node
            for neighbor in graph[node]:
                # 如果没有遍历
                if color[neighbor] == UNCOLORED:
                    # 继续进入
                    dfs(neighbor, cNei)
                    # 遍历完记得判断，因为dfs下一层有可能会return valid是false。
                    if not valid:
                        return
                # 【重点】如果遍历过了，那么其应该满足当前节点的颜色要求，如果不满足，就直接返回False就好。
                elif color[neighbor] != cNei:
                    valid = False
                    return

        # 遍历每一个node
        for i in range(n):
            # uncolored意味着还没有经过遍历，进入dfs
            if color[i] == UNCOLORED:
                dfs(i, RED)
                # 如果valid为False，那么直接跳出for循环，not valid意味着不可能存在了。
                if not valid:
                    break
        
        return valid

# BFS
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        n = len(graph)
        UNCOLORED, RED, GREEN = 0, 1, 2
        color = [UNCOLORED] * n
        valid = True

        for i in range(n):
            if color[i] == UNCOLORED:
                q = collections.deque([i])
                color[i] = RED
                while q:
                    node = q.popleft()
                    cNext = (GREEN if color[node] == RED else RED)
                    for neighbor in graph[node]:
                        if color[neighbor] == UNCOLORED:
                            q.append(neighbor)
                            color[neighbor] = cNext
                        elif color[neighbor] != cNext:
                            return False
        return True
        
# UNION FIND 模版题！
# 常见的使用并查集的情景，是动态地快速判断图中节点的连通性（两个节点是否在一个连通分量中）。

class Solution:
    class UnionFind:
        def __init__(self, n):
            # 初始化并查集
            # 维护一个root，记录每一个index分别对应的是哪个root
            self.root = list(range(n))
            # 因为每一个node都可以当root，所以这里rank纯粹是在判断是为了判断哪个node跟的“小弟node”比较多，把rank低的node跟在rank高的root下面去
            self.rank = [1] * n

        # x,y,u,v都是node

        # find是用来修改root里值的；初始化的时候 x == self.root[x]，但是merge/union后会变化
        # 目的是为了找到当前index最终指向的root是谁
        def find(self, x):
            if x != self.root[x]:
                self.root[x] = self.find(self.root[x])
            return self.root[x]

        # 
        def union(self, x, y):
            root_x = self.find(x)
            root_y = self.find(y)
            # 如果是x和y的root相同，属于同一个集内，因此不用执行别的操作
            if root_x == root_y:
                return

            #如果root不一样，这个时候rank权重介入。
            if self.rank[root_x] > self.rank[root_y]:
                self.root[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.root[root_x] = root_y
            else:
                # 两者权重一样的时候，这里介入，并且手动调整权重等级
                self.root[root_y] = root_x
                self.rank[root_x] += 1

        def is_connected(self, x, y):
            return self.find(x) == self.find(y)

    def isBipartite(self, graph: List[List[int]]) -> bool:
        # 通过调用内部类实现初始化
        uf = Solution.UnionFind(len(graph))
        # 遍历所有节点
        for u, vs in enumerate(graph):
            for v in vs:
                # 如果u、v，当前节点和相邻节点，已经在一个集合中了，就说明不是二分图，返回False
                if uf.is_connected(u, v):
                    return False
                # 将当前顶点的所有邻接点进行合并，union是并查集中的合并操作
                # 题意理解：因为这道题是二分图，意味着一个节点和它的所有相邻节点都不是同一个解集，所有相邻节点属于一个解集。
                # 所以这就能理解为什么一个节点下的，每一个v和它的vs[0]可以合并了，合并完在root里显示的就是所有这些节点的最终root！
                uf.union(vs[0], v) 

        return True


#### 复杂度 ###
# 上面三种方法的时间复杂度为N+M, N为顶点数，M为边数；空间为N，因为要新建root和rank


# 207 Course Schedule
"""
总结：拓扑排序问题 
主要解决：有向无回图/解决节点之间的依赖关系
根据依赖关系，构建邻接表、入度数组。
选取入度为 0 的数据，根据邻接表，减小依赖它的数据的入度。
找出入度变为 0 的数据，重复第 2 步。
直至所有数据的入度为 0，得到排序，如果还有数据的入度不为 0，说明图中存在环
"""
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # 初始化入度
        inDegree = [0] * numCourses
        # adjcentMap里面的key是先修课本身，value是修完这门课可以修的课程
        # 为什么需defaultdict，这里我理解的是defaultdict(list)会直接返回list给它的value，即key-value存储的是一个list对象
        adjcentMap = collections.defaultdict(list)
        for c, prer in prerequisites:
            # 将课程c的入度+1
            inDegree[c] += 1 
            if adjcentMap[prer]:
                adjcentMap[prer].append(c)
            else:
                adjcentMap[prer] = [c]
        
        # 这里首先遍历，将课程的入度和每门课的出度（后续课）的dict做好。
        
        q = collections.deque()
        # 遍历indegree，将入度为0，也就是立马就能上的课添加进来。
        for i in range(numCourses):
            if inDegree[i] == 0:
                q.append(i)
        
        count = 0 
        while q:
            # 确定目前take的course
            selected = q.popleft()
            count += 1
            nextCourses = adjcentMap[selected]
            if nextCourses:
                for nc in nextCourses:
                    inDegree[nc] -= 1
                    # 如果indegree为0，入q。
                    if inDegree[nc] == 0:
                        q.append(nc)
        
        # 如果count不想等的话，意味着有的课程形成了环，无论如何就没有办法让我们选上。
        return count == numCourses

# 复杂度均为n+m；n为课程数量，m为先修课要求数量；
# 空间为n+m的原因是了对图进行广度优先搜索，我们需要存储成邻接表的形式，空间复杂度为 O(n+m)。
# 在广度优先搜索的过程中，我们需要最多 O(n) 的队列空间（迭代）进行广度优先搜索。
# 因此总空间复杂度为 O(n+m)O(n+m)。



# 210 Course Schedule II
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        inDegree = [0] * numCourses
        nextCourses = collections.defaultdict(list)
        for c, p in prerequisites:
            inDegree[c] += 1
            if nextCourses[p]:
                nextCourses[p].append(c)
            else:
                nextCourses[p] = [c]
        
        q = collections.deque()
        for i in range(numCourses):
            if inDegree[i] == 0:
                q.append(i)
            
        count = 0
        res = []
        while q:
            selected = q.popleft()
            res.append(selected)
            count += 1
            ncs = nextCourses[selected]
            if ncs:
                for nc in ncs:
                    inDegree[nc] -= 1
                    if inDegree[nc] == 0:
                        q.append(nc)
        print(count)
        if count == numCourses:
            return res
        else: 
            return []
# 与207一模一样，最后判断的时候才不一样


# 684 Redundant connection
# 题目理解：本题的tree是无环无向图，新增一条边可以组成环，那么就去找这条边。主要的切入点是：边的两个vertices是否在一个集合中，如果在那么可以组成。
# UF(并查集)题目的难点在与如何抽象问题，联想到用并查集解决问题
class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        n = len(edges)
        # 0～n，index也为0～n，但是0我们用不到哦～
        root = list(range(n + 1))
        def find(n):
            if root[n] != n:
                root[n] = find(root[n])
            return root[n]

        def union(n1, n2):
            root[find(n1)] = find(n2)

        # 遍历，如果发现root不在一起，也就是我们期望的结果，root起来就好。
        for n1, n2 in edges:
            if find(n1) != find(n2):
                union(n1, n2)
            # 如果root在一起，表明我们之前的操作其实已经将节点相连了。所以这里再相连就表示已经成环了！
            else:
                return [n1, n2]
        return []
# 时间复杂度nlogn，查找find为n，合并为logn; 空间复杂度n
# 关于root这个list：其实是两个队列，用index代表数字，用储存的value代表index所对应的根root。和785的root异曲同工。
# 785那一题还有一些细节添加，这一题的代码比较简便。


