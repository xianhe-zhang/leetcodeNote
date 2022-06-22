# 2303. Calculate Amount Paid in Taxes
class Solution:
    def calculateTax(self, b: List[List[int]], c: int) -> float:
        l = 0
        z = 0
        for x, y in b:
            z += (min(x, c) - l) * y / 100
            if x > c:
                break
            l = x
        return z
# 可以用一个变量保存上一个循环的信息，灵活运用break/min



class Solution:
    def minPathCost(self, grid: List[List[int]], moveCost: List[List[int]]) -> int:
        res = []
        def dfs(cnt, i, j):
            ch = grid[i][j]
            cnt += ch
            if i == len(grid)-1:
                temp.append(cnt)
                
                return 
            
            for j in range(len(grid[0])):
                dfs(cnt+moveCost[ch][j], i+1, j)

        for j in range(len(grid[0])):
            temp = []
            dfs(0, 0, j)
            res.append(min(temp))
            
            
        return min(res)
    
    
# 2304. Minimum Path Cost in a Grid
# QTNN的，第二题是3重循环+DP
# 思路牛逼：当我们到第i行时，针对每个节点，我们看所有i-1行过来的最小cost，保留下来。
class Solution:
    def minPathCost(self, a: List[List[int]], c: List[List[int]]) -> int:
        n = len(a)
        m = len(a[0])
        f = [[1e9 for j in range(m)]for i in range(n)]
        for i in range(m):
            f[0][i] = a[0][i]
        for i in range(n - 1):
            for j in range(m):
                for k in range(m):
                    f[i + 1][k] = min(f[i + 1][k], f[i][j] + a[i + 1][k] + c[a[i][j]][k])
        return min(f[-1])


# 2309. Greatest English Letter in Upper and Lower Case
# 大神的写法
class Solution:
    def greatestLetter(self, s: str) -> str:
        z = set()
        for i in s:
            if i.isupper() and i.lower() in s:
                z.add(i)
        if len(z) == 0:
            return ''
        return max(z)
# 我的憨批写法，字母可以直接走max()，一个i一个upper/lower让我无地自容
# 你的思路不清晰
class Solution:
    def greatestLetter(self, s: str) -> str:
        res = ""
        lower = [ch for ch in s if ch.islower()]
        upper = [ch for ch in s if ch.isupper()]
        for u in upper:
            if u.lower() in lower and (not res or ord(u)>ord(res)):
                res = u
        return res
        

# 2310. Sum of Numbers With Units Digit K
# 思路解析：其实我们不用在乎其他的，只用关注最后一位就成了，因为最后一位限定为K
# 那么nk的末位一定呈现一定规律，看看需要几个k能和num的末位对应上，其他的用10多大都成，无所谓。
class Solution:
    def minimumNumbers(self, num, k):
        if num == 0: return 0
        for i in range(1, 11):
            if k * i % 10 == num % 10 and i * k <= num:
                return i
        return -1


# 2311. Longest Binary Subsequence Less Than or Equal to K
class Solution:
    def longestSubsequence(self, s: str, k: int) -> int:
        dp = [0]
        for v in map(int, s):
            if dp[-1] * 2 + v <= k:
                dp.append(dp[-1] * 2 + v)
            for i in range(len(dp) - 1, 0, -1):
                dp[i] = min(dp[i], dp[i - 1] * 2 + v)
        return len(dp) - 1
# 最重要的就是理解dp[i]是什么？dp[i]是 minimum value of subsequence with length i
# 知道这个就容易多了，每次遍历了一个v，然后我们就去更新已有的dp里面所有最短路径。
