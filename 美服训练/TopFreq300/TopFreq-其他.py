# 133. Clone Graph

"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

class Solution:
    def __init__(self):
        self.visited = {}
    
    # 无论是最终的return/edge case/special case都是return node
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return node
        if node in self.visited:
            return self.visited[node]
        # copy一个node出来
        clone_node = Node(node.val, [])
        # 把当前node放在visited里面，value就是它的copyNode
        self.visited[node] = clone_node
        
        # 如果node有neighbors的话，那么我们就将其recurse掉。
        # 这一题的逻辑很厉害呀。
        if node.neighbors:
            clone_node.neighbors = [self.cloneGraph(n) for n in node.neighbors]
        return clone_node

# 138. Copy List with Random Pointer
# 这一题的deepcopy其实是用了dict暂时保存。🌟
# 一样的，首先都是用边界条件/返回条件，然后把当前值处理，保存值/判断之类的，然后进入下一个递归。递归的代码就是优雅哈
"""
iterate的思路：差不多一致。最外层traverse all the nodes。每一次遍历的时候，我们用将我们新创立的nodes连接起来，如果已经联结起来就算，如果没有，就修改连接。
当然了以上的修改都是在我们的helper dict中，然后返回head就成。
"""
class Solution(object):
    def __init__(self):
        self.visited = {}
    def copyRandomList(self, head):
        if not head: return None
        if head in self.visited:
            return self.visited[head]
        node = ListNode(head.val, None)
        # 将key:value都存入node，key是原来的值，value是deep copy的值
        self.visited[head] = node
        node.next = self.copyRandomList(head.next)
        node.random = self.copyRandomList(head.random)
        return node
# copy的题非常适合训练recursion的思路！



# 204. Count Primes
# 埃氏筛思路：首先找到numbers；找到prime，然后其倍数依次update为false；最后sum(true)就可以了
# 平常我们的做法利用helper遍历。比如helper(n)就会判断n这个数字是否可以被其他数整除。
class Solution:
    def countPrimes(self, n: int) -> int:
        if n <= 2:
            return 0
        # numbers其实是从0～n一共n+1个数字
        numbers = [False, False] + [True] * (n - 2)
        for p in range(2, int(sqrt(n)) + 1):
            if numbers[p]:
                # Set all multiples of p to false because they are not prime.
                for multiple in range(p * p, n, p):
                    numbers[multiple] = False
        
        return sum(numbers)

# 628. Maximum Product of Three Numbers
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        nums.sort()
        return max(nums[-1]*nums[-2]*nums[-3],
                   nums[-1]*nums[0]*nums[1])

# 509. Fibonacci Number
class Solution:
    def fib(self, n: int) -> int:
        if n == 0 or n == 1:
            return n
        fib = self.fib(n-1) + self.fib(n-2)
        return fib

# 976. Largest Perimeter Triangle
# 突然迷糊了一下
class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums.sort()
        # 排序完，我们只用看连续的三位数就成了，因为左边紧接着的两位数一定是最大的！
        for i in range(len(nums)-3, -1, -1):
            if nums[i] + nums[i+1] > nums[i+2]:
                return sum(nums[i:i+3])
        return 0
            

# 1232. Check If It Is a Straight Line
class Solution:
    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        (x0, y0), (x1, y1) = coordinates[: 2]
        return all((x1 - x0) * (y - y1) == (x - x1) * (y1 - y0) for x, y in coordinates)
# 这一题有两题处理的很漂亮，每个点有两个标准，就是开始亮的两个点，如果斜率一样的话，就证明所有点都在一条直线上。
# 斜率相除的话有可能是0，所以这里我们更换成乘法！


# 202. Happy Number
class Solution:
    def isHappy(self, n: int) -> bool:
        seen = set()
        while n != 1 and n not in seen: # 如果我们碰到1的话，就会无限循环的。
            seen.add(n)
            n = self.getNext(n)
        return n == 1
    
    def getNext(self, n):
        total = 0
        while n>0:
            n, digit = divmod(n, 10) # 注意divmod的用法就可以了，其他没什么需要额外注意的。
            total += digit ** 2
        return total

        
# 29. Divide Two Integers
# 这题麻了
class Solution:  
    def divide(self, A, B):
        if (A == -2147483648 and B == -1): return 2147483647
        a, b, res = abs(A), abs(B), 0
        for x in range(32)[::-1]:
            if (a >> x) - b >= 0:
                res += 1 << x
                a -= b << x
        return res if (A > 0) == (B > 0) else -res


# 464. Can I Win
# 纯享版
class Solution:
    def canIWin(self, maxChoosableInteger, desiredTotal):
        seen = {}
        def can_win(choices, remainder):
            if choices[-1] >= remainder:
                return True
            seen_key = tuple(choices)
            if seen_key in seen:
                return seen[seen_key] 

            for index in range(len(choices)):
                if not can_win(choices[:index] + choices[index + 1:], remainder - choices[index]):
                    seen[seen_key] = True
                    return True

            seen[seen_key] = False
            return False

        summed_choices = (maxChoosableInteger + 1) * maxChoosableInteger / 2
        if summed_choices < desiredTotal:
            return False
        if summed_choices == desiredTotal:
            return maxChoosableInteger % 2
        
        choices = list(range(1, maxChoosableInteger + 1))
        return can_win(choices, desiredTotal)
# 解析版本
class Solution:
    def canIWin(self, maxChoosableInteger, desiredTotal):
        seen = {}
        def can_win(choices, remainder):
            # if the largest choice exceeds the remainder, then we can win!
            if choices[-1] >= remainder:
                return True
            # if we have seen this exact scenario play out, then we know the outcome
            # tuple是hashable的，因此我们将其放在{}
            seen_key = tuple(choices)
            if seen_key in seen:
                return seen[seen_key] # return之前已经见过的

            # 当前recursion，去for没中可能性，只要有一种可能对手全输，我就选这个可能性
            for index in range(len(choices)):
                if not can_win(choices[:index] + choices[index + 1:], remainder - choices[index]): # 就是当我选了index，这里是对手的视角，对手的所有可能性都不可能赢，那我必赢。
                    seen[seen_key] = True
                    return True

            # 注意这里的True/False不是backtrack，而是用来update全局变量的。
            seen[seen_key] = False
            return False

        # let's do some quick checks before we journey through the tree of permutations
        summed_choices = (maxChoosableInteger + 1) * maxChoosableInteger / 2

        # if all the choices added up are less then the total, no-one can win
        if summed_choices < desiredTotal:
            return False

        # if the sum matches desiredTotal exactly then you win if there's an odd number of turns
        if summed_choices == desiredTotal:
            return maxChoosableInteger % 2
        # slow: time to go through the tree of permutations
        choices = list(range(1, maxChoosableInteger + 1))
        return can_win(choices, desiredTotal)


# 486. Predict the Winner
class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        return self.dfs(nums, 0, len(nums)-1) >= 0
    def dfs(self, nums, i, j):
        if i == j: return nums[i]
        return max((nums[i]-self.dfs(nums,i+1,j)),(nums[j]-self.dfs(nums,i,j-1)))
# min-max的思路都有趣，当前选手选的是nums[i],那么dfs(nums,i+1,j)就是对手能够赚取的分数。
# 对手/下层一定是选择对他们最有利/最大的数，而我这里就选择相对大的选择。一路top-down再bottom-up

# Recursion+记忆化搜索
class Solution:
    def PredictTheWinner(self, nums: List[int]) -> bool:
        # memo存的就是dfs的值，dfs的值是什么？是当前回合自己赢得的分数与对手之后回合赢得的分数之差。
        def dfs(i, j):
            if i == j: return nums[i]
            key = tuple((i,j))
            if key in memo:
                return memo[key]
            memo[key] = max((nums[i]-dfs(i+1,j)),(nums[j]-dfs(i,j-1)))
            return memo[key]
        memo = {}
        return dfs(0, len(nums)-1) >= 0

# DP的做法 - 自上而下地思考，自下而上地解决。
class Solution():
    def PredictTheWinner(self, nums):
        dp = {}

        def find(i, j):
            if (i, j) not in dp:
                if i == j:
                    return nums[i]
                dp[i,j] = max(nums[i]-find(i+1, j), nums[j]-find(i, j-1))
            return dp[i,j]

        return find(0, len(nums)-1) >= 0


    
# 877. Stone Game 
# 这里lru_cache的作用其实就和上面两道题中seen/memo的作用一样
from functools import lru_cache
class Solution:
    def stoneGame(self, piles):
        N = len(piles)

        @lru_cache(None)
        def dp(i, j):
            # The value of the game [piles[i], piles[i+1], ..., piles[j]].
            if i > j: return 0
            parity = (j - i - N) % 2
            if parity == 1:  # first player
                return max(piles[i] + dp(i+1,j), piles[j] + dp(i,j-1))
            else:
                return min(-piles[i] + dp(i+1,j), -piles[j] + dp(i,j-1))

        return dp(0, N - 1) > 0



# 1266. Minimum Time Visiting All Points
class Solution:
    def minTimeToVisitAllPoints(self, points: List[List[int]]) -> int:
        time = 0
        for i in range(1,len(points)):
            time += max(abs(points[i][1]-points[i-1][1]), abs(points[i][0]-points[i-1][0]))
        return time


# 892. Surface Area of 3D Shapes
# 这道题的难点就在于你不能从成品中找规律， 而是每放一个ceil，就去减去相应的部分。
class Solution:
    def surfaceArea(self, grid: List[List[int]]) -> int:
          
        n, res = len(grid), 0
        for i in range(n):
            for j in range(n):
                if grid[i][j]: res += 2 + grid[i][j] * 4
                if i: res -= min(grid[i][j], grid[i - 1][j]) * 2
                if j: res -= min(grid[i][j], grid[i][j - 1]) * 2
        return res



# 1401. Circle and Rectangle Overlapping
class Solution:
    def checkOverlap(self, r, x_c, y_c, x1, y1, x2, y2):
        corners = [(x1,y1), (x2,y1), (x2,y2), (x1, y2)]
        # 所有的顶点在不在
        for (x, y) in corners:
            if (x_c - x)**2 + (y_c - y)**2 <= r**2:
                return True
        # 有一条边在圆内
        for x in [x1, x2]:
            if x_c-r <= x <= x_c+r and y1<=y_c<=y2:
                return True
        for y in [y1, y2]:
            if y_c-r <= y <= y_c+r and x1<=x_c<=x2:
                return True
		# 圆在矩形内部
        if x1<=x_c<=x2 and y1<=y_c<=y2:
            return True
        return False            