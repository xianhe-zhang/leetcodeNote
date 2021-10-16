# 树的右视图 199✅
# 统计单词出现次数 剑指offer56-I -II 43 39✅
# 二叉树反转180 226✅
# 二叉树最小深度 111✅
# 岛屿数量 200✅
# 打家劫舍 337✅
# 上台阶 746✅
# 两数之和 1
# 反转链表 206
# 数组的子序列最大和 53 剑指offer
# 数组的topk大数字  剑指offerII 076

#树的右视图 leetcode-199
#BFS
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
      res = []
      if not root:
        return res

      visited = []
      depth = 0
      visited.append(root)

      while visited:
        size = len(visited)
        for i in range(size):
          cur = visited.pop(0)
          if cur.right:
            visited.append(cur.right)
          if cur.left:
            visited.append(cur.left)
          
          if i == 0:
            res.append(cur.val)

      return res
# 思路总结：采用BFS，每一层遍历，每一层是一个while；通过先遍历左边还是右边 确定 每一次遍历的第一项还是最后一项是最右侧的值
# 这一题的关键点在于对于BFS这种题型套路的熟悉

#DFS
class Solution:
    
    def rightSideView(self, root: TreeNode) -> List[int]:
      res = []  
      def dfs(root, depth: int):
        if not root:
          return res
        
        if depth == len(res):
          res.append(root.val)
        
        dfs(root.right, depth + 1)
        dfs(root.left, depth + 1)

      dfs(root, 0)
      return res
        
        

# 剑指 Offer 56 - I. 数组中数字出现的次数
# 自己想到的方法是利用collection.Counter(), 然后for一遍就成了。或者用sort+双指针
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
      res = []
      temp = collections.Counter(nums)
      for item in temp: #这个item是key
        if temp[item] == 1:
          res.append(item)
      return res

#位运算
"""
首先明白，位运算是二进制的32位运算，即0000 0000 16+16
"""
class Solution:
    def singleNumbers(self, nums: List[int]) -> List[int]:
        x, y, n, m = 0, 0, 0, 1
        for num in nums:         # 1. 遍历异或
            n ^= num
        while n & m == 0:        # 2. 循环左移，计算 m
            m <<= 1       
        for num in nums:         # 3. 遍历 nums 分组
            if num & m: x ^= num # 4. 当 num & m != 0
            else: y ^= num       # 4. 当 num & m == 0
        return x, y              # 5. 返回出现一次的数字
"""
通过这道题理解位运算：
1. 本题的要求是时间复杂度为n，空间复杂度为1；
2. 第一次的for循环是求n，n为最后只出现x ^ y, x和y分别为只出现一次的次数，因为最后其他数都两两抵消了？
3. 如何理解两两抵消？位运算，因此每一位只有0和1，可以抵消，你自己想想
4. 那么如何解释第二个while？ 用m=1然后依次向左位移1，当计算结果不为0的时候跳出；为什么呢？当计算结果不为0意味着，此时m当中1所处的位，就是n的1的最低位，这也就意味着，在这一位上，x和y不一样，一个为0，一个为1
5. 下一步就简单了，再依次遍历，此时可以根据m当中1的这一位把整个数组分成两组，分别是这一位为0和这一位为1的，刚好也把x,y分开，因为我们知道x,y这一位上的数字不同，这是key！
6. 通过位与运算&区分，然后分别两组迭代^运算，最后剩下来的就是我们的x和y, 直接return就成。
"""

#剑指 Offer 56 - II. 数组中数字出现的次数 II
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        temp = collections.Counter(nums)
        for item in temp: #这个item是key
            if temp[item] == 1:
                return item


#leetcode-169  数组中出现次数超过一半的数字
#此题有三种解法：哈希表计数（我自己能想到的，可以试试不调库）/摩尔投票法/数组排序法
# 数组排序后，中位数肯定是众数，也就是我们这题中出现的超过一半数量的数。
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
      nums.sort()
      mid = len(nums) // 2
      return nums[mid]

# 哈希表计数（非调库）
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        counts = collections.Counter(nums)
        return max(counts.keys(), key=counts.get) #.keys() 可以直接返回key值，但是我们还有lambda表达式，这里就是按照key=counts.get去获得value的值排序
# 直接看solution都是用调库，那么java是如何处理的呢？
# java处理起来好长呀...感觉有点后悔用java了，然后去查询了一下，感觉java和python都学起来好像也可以。python练习逻辑，java练习oop。

#摩尔投票法：
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        votes = 0
        for num in nums:
            if votes == 0: x = num
            votes += 1 if num == x else -1  #这骚气的写法，😵‍💫。
        return x    #x为我们找的超过一半元素的数字
#摩尔投票法，将x看作1，其他所有值都看作是-1，那么摩尔vote最后的结果肯定>0,因为我们的x大于一半
#与此同时，处理上要灵活一些：因为x肯定与其他数字相抵消，因此在前期，我们允许x为-1/x为1，只要x其他值的正负相异就行。

#1～n 整数中 1 出现的次数 leetcode-233
class Solution:
    def countDigitOne(self, n: int) -> int:
        digit, res = 1, 0
        high, cur, low = n // 10, n % 10, 0                 #将数组n每一位都分开，已经处理过的放在low，正在看的位叫做cur位，未处理的区域是high
        while high != 0 or cur != 0:                        #只要有一位不等于0，就意味着还没有遍历完到n
            if cur == 0: res += high * digit                #第一种情况，如果cur是0，那么在可以去到的1～n中，cur为1的次数为high * digit
            elif cur == 1: res += high * digit + low + 1    #第二种情况，如果cur是1，那么除了high*digit，我们low + 1就是high固定，low随意就行，对进退位没有影响
            else: res += (high + 1) * digit                 #第三种情况，如果cur是其他。这些规律都是有cur=0推导而来。
            low += cur * digit
            cur = high % 10
            high //= 10
            digit *= 10
        return res
#大神思路
"""
1. 首先明白这一题，我们转化为，每一位1可以出现的次数的总和，就是我们1～n出现1的所有次数，这个很关键
2. 如何理解cur=0，距离2304，那么我们能够cur取1的范围为0010～2219，首先，0010在范围内，而2219当中的22我们退位了，所以19是我们能取到没有进位的最大数字。
3. 因此0010～2219，我们只用看非cur位就行了，000～229，怎么组合都行，即230，即原来的high * digit，数学归纳法。
"""

## 二叉树反转180 leetcode-226
#递归
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
      if not root:
        return 
      
      root.left, root.right = root.right, root.left
      self.invertTree(root.left)
      self.invertTree(root.right)
      
      return root

#迭代
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
      if not root:
        return 

      queue = [root]
      while queue:
        node = queue.pop(0)
        node.left, node.right = node.right, node.left 
        if node.left:
          queue.append(node.left)
        if node.right:
          queue.append(node.right)

      return root



# 二叉树最小深度 111
#dfs 
#这一题的take-away：可以利用if进行剪枝；不用声明新变量，直接将depth作为传参的值进行传递; 先递归把调用栈处理完毕，弹出的时候进行处理也行。
class Solution:
  def minDepth(self, root: TreeNode) -> int:
    if not root:
      return 0

    left = self.minDepth(root.left)
    right = self.minDepth(root.right)

    if not root.left and not root.right:
      return 1
    elif not root.left or not root.right: 
      return left + 1 if root.left else right+1
    else: 
      return min(left+1, right+1) + 1 #实际最终判断只有这里
# 三种情况：1.如果没有子树，则返回depth=1就行；2.如果有一个子节点，那么root肯定是要返回有节点的，因为是判断子树要判断最深的，我们要的是min，这两个不一样的概念
# 3.两个子节点都有的话，返回最小的那个，因此这一题肯定是从最底层处理，哪怕顶点没有节点，放在第一种情况里一起处理就好了。
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        if not root.left and not root.right:
            return 1
        
        min_depth = 10**9 #这里就算很大，但是我们在下文中肯定是会处理的，因此不用担心，处理的序列在遇到上面的return时便返回了，不会再次初始化。
        if root.left:
            min_depth = min(self.minDepth(root.left), min_depth)
        if root.right:
            min_depth = min(self.minDepth(root.right), min_depth)
        
        return min_depth + 1

#bfs  这一题注定用bfs会快
class Solution:
  def minDepth(self, root: TreeNode) -> int:
    if not root:
      return 0
    
    queue = [(root,1)]
    while queue:
      curNode, depth = queue.pop(0) #处理对象取值的用法，我第一次用
      if not curNode.left and not curNode.right:
        return depth
      if curNode.left:
        queue.append((curNode.left, depth+1))
      if curNode.right:
        queue.append((curNode.right, depth+1))
    return 0


# 岛屿数量 200
class Solution:
    def numIslands(self, grid: [[str]]) -> int:
        def dfs(grid, i, j):
            if not 0 <= i < len(grid) or not 0 <= j < len(grid[0]) or grid[i][j] == '0': return
            grid[i][j] = '0'
            dfs(grid, i + 1, j)
            dfs(grid, i, j + 1)
            dfs(grid, i - 1, j)
            dfs(grid, i, j - 1)
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    dfs(grid, i, j) #将相邻的1都变为0，同时实现剪枝的目的。
                    count += 1
        return count



# 岛屿周长463
#这一题的想法牛呀，利用边界穿越，去计算周长，因为图像的特殊性，不能用有几块岛屿去计算周长
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
      for i in range(len(grid)):
        for j in range(len(grid[0])):
          if grid[i][j] == 1:
            return self.dfs(grid, i, j)
      return 0
    
    def dfs(self, grid, i, j):
      if not 0<=i<len(grid) or not 0<=j<len(grid[0]) or grid[i][j] == 0:
        return 1

      if grid[i][j] != 1:
        return 0

      grid[i][j] = 2
      return self.dfs(grid, i+1, j)+self.dfs(grid, i, j+1)+self.dfs(grid, i-1, j)+self.dfs(grid, i, j-1)



# 岛屿的最大面积 695
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
      res = 0


    def dfs(self,grid,x,y):
      if grid[x][y] == 1 or 0<=x<len(grid) or 0<=y<len(grid[0]):
        return 
      grid[x][y] = 1
      self.dfs(grid, x, y+1)
      self.dfs(grid, x+1, y)
      self.dfs(grid, x-1, y)
      self.dfs(grid, x, y-1)
      return res + 1


# 最大人工到 827
# 打家劫舍 337
# 上台阶 746
# 两数之和 1
# 反转链表 206
# 数组的子序列最大和 53 剑指offer
# 数组的topk大数字  剑指offerII 076

今天刷完这些题，明天才能更美妙！
