# 787. Cheapest Flights Within K Stops
# Dijkstra's 
# Space O(V^2) Time: O(V^2 * LogV) V^2是因为针对每个node，我们可能visit all its neighbors
# 针对LogV，因为每一次你都要heappush*/heappop
from collections import defaultdict
import heapq
# directed weighted graph
class Solution:
    
   import heapq
# directed weighted graph
class Solution:
    
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, K: int) -> int:
        
        adj_matrix = [[0] * n for _ in range(n)]
        # 将题目中的数据 -> 2-dimensional list
        for s, d, w in flights:
            adj_matrix[s][d] = w
        
        # node到start的距离
        distances = [float("inf") for _ in range(n)]
        # 到达n点时需要多少steps
        current_stops = [float("inf") for _ in range(n)]
        distances[src], current_stops[src] = 0, 0
        
        # Data is (cost, stops, node)
        minHeap = [(0, 0, src)]     
        
        while minHeap:
            # pop出最小cost的，如果cost相同，pop出最小的stops
            cost, stops, node = heapq.heappop(minHeap)
            
            # 一般来说如果遇到node可以直接pop的，dijkstra中因为利用到了heapq，所以pop是有暗含order的
            if node == dst:
                return cost
            
            # If there are no more steps left, continue即跳过该点 # 哪怕dijkstra的算法再复杂，终究也是要遍历所有可能性的。
            if stops == K + 1:
                continue
             
            # Examine and relax all neighboring edges if possible 
            for nei in range(n):
                # >0意味着能从node -> nei
                if adj_matrix[node][nei] > 0:
                    # 这三变量怎么那么恶心，好难懂
                    # du是当前cost，wUV是weighted cost， dV是当前node与src的距离
                    dU, dV, wUV = cost, distances[nei], adj_matrix[node][nei]
                    
                    
                    # 这里if和elif相当于给了一个优先级比较
                    # Better cost? 首先我们看到达当前node的cost是否时最小？如果是，更新距离并且入库
                    if dU + wUV < dV:
                        distances[nei] = dU + wUV
                        heapq.heappush(minHeap, (dU + wUV, stops + 1, nei))
                        # 这个stops绕死我了，当访问一个neighbors第一次肯定就更新，之后再次访问时，可能cost更大，但是step可能更小，如果两者只要有一个不满足，我们肯定就不继续看了！
                        current_stops[nei] = stops
                        
                    # 当一个节点可以从一个很大cost的线路直接过来，或者绕一个node从一个很小cost线路过来。
                    # 本题有限制k的数量，因此最小路径可能并不满足题意。
                    elif stops < current_stops[nei]:
                        #  Better steps?
                        heapq.heappush(minHeap, (dU + wUV, stops + 1, nei))
                        
        # 如果我们没有更新distances[dst]，意味着从src无法抵达dst
        return -1 if distances[dst] == float("inf") else distances[dst]
    

"""
Dijkstra 算法肯定是需要辅助的
要不挺记录每个node 到 src的距离
并且肯定会利用到heapq 因为每次弹出的都是min值
大概思路就是  -  每一次从minheap中找出当前最小的节点 然后看它的neighbors 并且同步更新/append/push
"""

# 743. Network Delay Time
class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        
        graph = defaultdict(list)
        # 利用hashmap，把目的地和cost绑定存入list
        for src, dst, c in times:
            graph[src].append((dst, c)) 
        # queue初始化，把k/src进来
        queue = [(0, k)] #(cost, node)
        visited = set()
        res = 0
            
        while queue:
            #Always pop the min value
            cost, node = heapq.heappop(queue)
            
            # 如果node存过了，一定是小的，因为我们每次拿都是拿离src最近的，这也就是为什么是(distance, node)这么存入的
            if node in visited:
                continue
            
            # 将node入visited防止重复遍历
            visited.add(node)
            # 为什么要在这里更新？首先我们要遍历完所有的node；其次我们的cost是从小到大pop出的；完成遍历所有的node开始，就没有办法再到这里了，会通过前面的if直接跳出
            # 因此最后一次更新是我们的遍历所有点的最小路径！
            res = cost
            neighbours = graph[node]
            
            for neighbour in neighbours:
                new_node, new_cost = neighbour
                if new_node not in visited:
                    curr_cost =  cost + new_cost
                    heapq.heappush(queue, (curr_cost, new_node))
        
        return res if len(visited) == n else -1

# Time complexity: O(N+ElogN)
# Dijkstra's Algorithm takes O(N+ElogN). Finding the minimum time required in signalReceivedAt takes O(N). 
# Space : O(N+E)


"-----------------------TOPOLOGICAL SORT----------------------------------------------"


# 207. Course Schedule
# 这道题有意思的点在于，针对一门课，他的图是怎么样的。
# Time E+V2 Space E+V
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        n = numCourses
        graph = [[] for _ in range(n)]
        visit = [0 for _ in range(n)]
        # 这里的顺序和拓扑排序不太一样topological sort
        # y是先修课，这里就是每一门课下面：有多少先修课； 拓扑排序是一门先修课：可以再上什么post选修课。
        for x, y in prerequisites:
            graph[x].append(y)
        
        def dfs(i):
            if visit[i] == -1:
                return False
            if visit[i] == 1:
                return True
            # 这里的操作很精细，仔细思考路径。
            # 如果能够成功，那么dfs的任意一条路径都已经能够打通，所以在不可能存在环
            # 遍历过后填上-1，这样在这条路的时候，如果之后再次遍历到，就意味着是环了，因此直接-1，return false
            # 好好想想这道题的dfs长什么样子。
            visit[i] = -1

            # 如果graph[i]里面没有值，就不会进入循环/recursion，那么会直接=1，并且返回true，因为它可以到达。
            for j in graph[i]:
                # 只要j里面有一个是false，那么最终就会false
                if not dfs(j):
                    return False
            visit[i] = 1
            return True
                    
            
        for i in range(n):
            # 确保每一个课程都可以上完
            if not dfs(i):
                return False
        return True
            
        
# 210. Course Schedule II
# Time和Space 都是 O(V+E) 
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

# 从上面这两道可以看出topological的精髓，其实就是利用了点与其权重，每次依赖其权重getMin与其neighbors
# 灵活选择bfs/dfs/backtracking


# 329. Longest Increasing Path in a Matrix

"""
Topological Sorting; 时间空间的复杂度都是O(mn)
简单的思路就是遍历每一个cell的周围 去判读啊它的outDegree
找到为0的Degree开始进行BFS 最后到达的level就是答案
class Solution {
    public int longestIncreasingPath(int[][] matrix) {
        int row = matrix.length, col = matrix[0].length;
        if (row == 1 && col == 1) return 1;

        int[][] dirs = {{-1,0},{1,0},{0,-1},{0,1}};
        int[][] outDegrees = new int[row][col];

        // 开始制作outDegree
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                for (int[] dir : dirs){
                    int nextX = i + dir[0];
                    int nextY = j + dir[1];
                    // 如果越界或者值的大小不符合就跳过
                    if (nextX < 0|| nextX >= row || nextY < 0 || nextY >= col || matrix[nextX][nextY] <= matrix[i][j]) {
                        continue;
                    }
                    outDegrees[i][j]++;
                }
                
            }
        }
        
        // 开始制作用于BFS的queue
        // outDegrees == 0 表明该cell一定是队首 因为不存在neighbors比他小
        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                if (outDegrees[i][j] == 0) {
                    // 把{i, j}入列
                    queue.offer(new int[]{i,j});
                }
            }
        }

        int res = 0;
        while (!queue.isEmpty()) {
            res++;
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int[] curPos = queue.poll();
                int x = curPos[0], y = curPos[1];
                for (int[] dir : dirs) {
                    int prevX = x + dir[0];
                    int prevY = y + dir[1];

                    if (prevX < 0 || prevX >= row || prevY < 0 || prevY >= col) {
                        continue;
                    }
                    if (matrix[prevX][prevY] >= matrix[x][y]) {
                        continue;
                    }

                    if (--outDegrees[prevX][prevY] == 0) {
                        queue.offer(new int[]{prevX, prevY});
                    } 
                }
            }
        }
        return res;
    }
}
"""
# 记忆话搜索dp的方法，时间空间也都是为O(mn)
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        m, n = len(matrix), len(matrix[0])
        # 记忆化dp的区别就在于多了这个dp，那么理解这个是干什么用的就很重要。
        dp = [[-1] * n for _ in range(m)]
        
        def dfs(i, j, prev):
            if i < 0 or j < 0 or i >= m or j >= n or matrix[i][j] <= prev:
                return 0
            # 如果== -1意味着还没有经历过这个cell，其他的基本和最原始的方法一致
            if dp[i][j] != -1:
                return dp[i][j]
            
            left = dfs(i, j - 1, matrix[i][j])
            right = dfs(i, j + 1, matrix[i][j])
            top = dfs(i - 1, j, matrix[i][j])
            bottom = dfs(i + 1, j, matrix[i][j])
            
            
            dp[i][j] = max(left, right, top, bottom) + 1
            return dp[i][j]
        
        ans = -1
        for i in range(m):
            for j in range(n):
                ans = max(ans, dfs(i, j, -1))
        return ans



# 跳过这一题
# 1203. Sort Items by Groups Respecting Dependencies
class Solution:
    def sortItems(self, n, m, group, beforeItems):

        # Helper function: returns topological order of a graph, if it exists.
        def get_top_order(graph, indegree):
            top_order = []
            stack = [node for node in range(len(graph)) if indegree[node] == 0]
            while stack:
                v = stack.pop()
                top_order.append(v)
                for u in graph[v]:
                    indegree[u] -= 1
                    if indegree[u] == 0:
                        stack.append(u)
            return top_order if len(top_order) == len(graph) else []

        # STEP 1: Create a new group for each item that belongs to no group. 
        for u in range(len(group)):
            if group[u] == -1:
                group[u] = m
                m+=1

        # STEP 2: Build directed graphs for items and groups.
        graph_items = [[] for _ in range(n)]
        indegree_items = [0] * n
        graph_groups = [[] for _ in range(m)]
        indegree_groups = [0] * m        
        for u in range(n):
            for v in beforeItems[u]:                
                graph_items[v].append(u)
                indegree_items[u] += 1
                if group[u]!=group[v]:
                    graph_groups[group[v]].append(group[u])
                    indegree_groups[group[u]] += 1                    

        # STEP 3: Find topological orders of items and groups.
        item_order = get_top_order(graph_items, indegree_items)
        group_order = get_top_order(graph_groups, indegree_groups)
        if not item_order or not group_order: return []

        # STEP 4: Find order of items within each group.
        order_within_group = collections.defaultdict(list)
        for v in item_order:
            order_within_group[group[v]].append(v)

        # STEP 5. Combine ordered groups.
        res = []
        for group in group_order:
            res += order_within_group[group]
        return res