leetcode-104 二叉树的最大深度
#这一题很明显两种阶梯思路：1.DFS 2.BFS
#BFS
class Solution(object):
    def maxDepth(self, root):
        if not root:
            return 0
        queue = [root] #这个很关键，每一层的递归，
        height = 0     #用于记录层高
        while queue:
            currentSize = len(queue)
            for i in range(currentSize):        #关键看visted中的多少是当前层。
                node = queue.pop(0)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            height += 1
        return height
#层序遍历：因为每一层都遍历，不存在剪枝，所以遍历到最后肯定是最深的

#DFS
class Solution:
    def maxDepth(self, root):
        if root is None: 
            return 0 
        else: 
            left_height = self.maxDepth(root.left) 
            right_height = self.maxDepth(root.right) 
            return max(left_height, right_height) + 1 


# 110  平衡二叉树 AVL
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
      if not root: return True
      return abs(self.height(root.left) - self.height(root.right)) <= 1 and \
        self.isBalanced(root.left) and self.isBalanced(root.right)

    def height(self, root: TreeNode) -> int:
      if not root: return 0
      # left = self.height(root.left)
      # right = self.height(root.right)
      # return max(left, right) + 1
      return max(self.height(root.left), self.height(root.right)) + 1  #可以简化成这种写法

      
# 543 二叉树的直径
#这题有个陷阱就是：不能直接来递归，求出root的直径，而是要遍历每一个二叉树的直径，最终求出所有节点的直径，选出最大值（有个trick就是保存最大值的方法）
class Solution:
    def __init__(self) -> None:
      self.maxDiameter = 0
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
      self.findDiameter(root)
      return self.maxDiameter

    def findDiameter(self, root):
      if not root:
        return 0
      left = self.findDiameter(root.left)
      right = self.findDiameter(root.right)
      self.maxDiameter = max(self.maxDiameter, left + right)    #这里是用来更新全局变量的
      return max(left, right) + 1                               #这里的return不是直接针对最后的结果负责，而是用于继续递归求的depth的。
#首先，需要清楚findDiameter()的return对象不是我们求的直径，而是depth，我们求的直径在递归的函数中利用全局变量求出来了。这个思想很重要。



# 226 反转二叉树
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

#递归
class Solution:
  def invertTree(self, root: TreeNode) -> TreeNode:
    if not root:
      return 
    
    root.left, root.right = root.right, root.left
    self.invertTree(root.left)
    self.invertTree(root.right)
    return root




# 617 合并二叉树
# 递归
class Solution(object):
  def mergeTrees(self, t1, t2):
    def dfs(t1, t2):
      if not (t1 and t2):               #本题take-away：如果有一方不满足，就返回，但这里返回的不是空值
        return t1 if t1 else t2         #直接利用有value的子树填补，如果没有的话，那就是个空树
      t1.val += t2.val
      t1.left = dfs(t1.left, t2.left)   #一起进左子树有趣～
      t1.right = dfs(t1.right, t2.right)
      return t1  
    return dfs(t1, t2)
#时间复杂度为N，空间复杂度为H，因为要进h次递归，h为树的高度。

# 迭代
class Solution(object):
  def mergeTrees(self, t1, t2):
    if not (t1 and t2):              
      return t1 if t1 else t2  
    queue = [(t1, t2)]
    
    while queue:
      r1, r2 = queue.pop(0)
      r1.val += r2.val
      if r1.left and r2.left:
        queue.append((r1.left, r2.left))
      elif not r1.left:
        r1.left = r2.left
      if r1.right and r2.right:
        queue.append((r1.right, r2.right))
      elif not r1.right:
        r1.right = r2.right
    return t1
#单击opt/start添加光标，shift+alt+L全选光标
#时间复杂度为N，空间复杂度为N/2，因为要进h次递归，h为树的高度。
#Thought：很多时候也许递归会比迭代复杂度高，以为会调用recursive stack。但递归的优点就是可以解决很多问题，很有创意。



# 112 路径总和
#🌟这题比较classical

#DFS      难点在：如何判断是最终子节点
#思路：判断子节点是否可以满足target要求，如果可以就return boolean最终到root
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
      if not root:
        return False
      if not root.left and not root.right:
        return sum == root.val              #take-away，可以利用sum的方法到这里
      return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)  #这种传参处理太棒了！极简派代表

#类回溯
#思路：记录path；遍历到子接点进行判断；但是本题没有重复利用path，也没有利用res；此外，也可以跟解法一一样在一个method里面。
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
      if not root: return False
      res = []
      return self.dfs(root, targetSum, res,[root.val])

    def dfs(self, root, targetSum, res, path):
      if not root:
        return False
      if not root.left and not root.right and targetSum == sum(path):
        return True

      left_flag, right_flag = False, False
      if root.left:
        left_flag = self.dfs(root.left, targetSum, res, path+[root.left.val])
      if root.right:
        right_flag = self.dfs(root.right, targetSum, res, path+[root.right.val])
      return left_flag or right_flag
      
#BFS
class Solution:
    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if not root:
            return False
        que = collections.deque() #双向队列
        que.append((root, root.val))
        while que:
            node, path = que.popleft()
            if not node.left and not node.right and path == sum:
                return True
            if node.left:
                que.append((node.left, path + node.left.val))
            if node.right:
                que.append((node.right, path + node.right.val))
        return False


#Stack #其实都是大同小异
class Solution(object):
    def hasPathSum(self, root, sum):
        if not root:
            return False
        stack = []
        stack.append((root, root.val))
        while stack:
            node, path = stack.pop()
            if not node.left and not node.right and path == sum:
                return True
            if node.left:
                stack.append((node.left, path + node.left.val))
            if node.right:
                stack.append((node.right, path + node.right.val))
        return False


# 437 路径总和 III
#遍历+DFS（双递归）
#基本思路：遍历每一个root；dfs找到每一个root下面的路径，return满足提议的路径
class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
      self.target = targetSum
      self.ans = 0
      self.dfs1(root)                     #全局变量的CRUD
      return self.ans
    
    #用于遍历tree的所有节点，并且将当前的root带入到dfs判断中去
    def dfs1(self, root):
      if not root:
        return 
      self.dfs2(root, root.val)
      self.dfs1(root.left)
      self.dfs1(root.right)

    def dfs2(self,root, curValue):
      if curValue == self.target:
        self.ans += 1
      if root.left:
        self.dfs2(root.left, curValue + root.left.val)
      if root.right:
        self.dfs2(root.right, curValue + root.right.val)
#Key Take-away: 往下的每次一层我们都进行逻辑判断，看看目前子节点之间的所有组合是否成立。
#如果满足题意的话，直接往全局变量里添加就成。
#时间复杂度：n2 ；空间复杂度：递归带来的空间消耗可以理解为树的高度，也可以理解为O(1)


#前缀和   ✨✨✨✨
#主要思想：
#   1.每个Node储存的数据为前缀和，利用不同节点之间的差与target相比较。
#   2.我们抽象出来一个一维列表，每一个node都有一个属于它的列表（因为我们是动态更新这个列表的），可以知道前面满足

class Solution:
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
      def dfs(root, sumNum):
        if root:                #每一个root我们都要
          sumNum += root.val
          tagDiff = sumNum - targetSum #sum是前缀和，我们想要的是前缀和之差，因此sum和target是这个顺序
          self.res += ans[tagDiff] #ans是一维列表，里面放着的是之前所有节点与当前节点curNode的前缀和之差，value是有多少种这样的组合。
          ans[sumNum] += 1
          dfs(root.left,sumNum)
          dfs(root.right,sumNum)
          ans[sumNum] -= 1
      
      self.res = 0 #要么这里用res = []列表，然后在函数内部赋值的时候,用global解决不了。
      ans = defaultdict(int)
      #下面这个挺重要的，规避了为空造成的隐患。
      ans[0] = 1
      dfs(root,0)
      return self.res
#Bug：如果直接用res=0 -> 会报错；因为解释器不清楚这个res是全局变量还是局部变量；也许传参可以解决，那就要解决return回来的问题了。

# dict1 = defaultdict(int)    #0
# dict2 = defaultdict(set)    #set()
# dict3 = defaultdict(str)    #
# dict4 = defaultdict(list)   #[]

# 572 另一棵树的子树
# 基本的思路，利用递归去：1.遍历node 2.判断是否为相同的树
#如果是subTree跟我们的main tree有部分树是Same，那么前者肯定为后者的子树
class Solution:
  def isSubtree(self, root: TreeNode, subRoot: TreeNode) -> bool:
    if not root and not subRoot:
      return True
    if not root or not subRoot:
      return False
    return self.isSameTree(root,subRoot) or self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

  def isSameTree(self, root, subRoot):
    if not root and not subRoot:
      return True
    if not root or not subRoot:
      return False
    return root.val == subRoot.val and self.isSameTree(root.left, subRoot.left) and self.isSameTree(root.right, subRoot.right)
#这一题的return非常精妙。因为最后返回的都是boolean，所以可以把条件判断都放在return中去
#第一个return是或的关系，只要有一个满足就可以；第二个return 判断sameTree，所以都要满足是and的关系。
#这里涉及到两个递归，递归的退出条件虽然冗余，但是不可或缺。


# 101 对称二叉树
#递归：左子树=右子树
#主要思路：这一题不可以不用helper，因为最终对比的是两个子树！
class Solution:
  def isSymmetric(self, root: TreeNode) -> bool:
    if not root:
      return True
    return self.dfs(root.left, root.right)
  
  def dfs(self, left, right):
    if not left and not right:
      return True
    elif not left or not right or left.val != right.val:
      return False
    
    return self.dfs(left.left, right.right) and self.dfs(right.left,left.right)
#复杂度都为N，时间容易理解，因为都要遍历一遍，空间的话最坏能够演化成一条link，所以也为n

#迭代
class Solution:
  def isSymmetric(self, root: TreeNode) -> bool:
    if not root or not (root.left or root.right):
      return True
    
    queue = [root.left, root.right]
    while queue:  
      left = queue.pop(0)
      right = queue.pop(0)
      #因为与递归的顺序不同，所以这里的判断条件会有所不同
      if not (left or right):   #如果left/right都没有值，continue没关系
        continue
      if not (left and right):  #如果一方没有值，都是错的，因为我们的left和right如果对称必须全等
        return False
      if left.val != right.val:
        return False
      queue.append(left.left) #这个append的数据只要是一对就行了。
      queue.append(right.right)
      queue.append(right.left)
      queue.append(left.right)
    return True
#复杂度都是N，空间主要是有N
#与java的区别在于，主要是数据结构的选择，与API的CURD差别，其他基本上一致。


# 111 二叉树的最小深度
#BFS
class Solution:
  def minDepth(self, root: TreeNode) -> int:
    if not root:
      return 0

    queue = [(root, 1)]
    while queue:
      node, depth = queue.pop(0)
      if not node.left and not node.right:
        return depth
      if node.left:
        queue.append((node.left,depth + 1))
      if node.right:
        queue.append((node.right,depth + 1))
    return 0

#DFS
#这里要传参进去，所以需要helper
#这一题不用helper！而且难点在于如何简单化写条件判断，决定返回哪个子树的depth
#条件判断：如果都为空可以返回；如果有一方不为空，返回有值的；如果都不为空，返回min
#写在一个method中时，需要depth当参数
class Solution:
  def minDepth(self, root: TreeNode) -> int:
    if not root:
      return 0
    left = self.minDepth(root.left)
    right = self.minDepth(root.right)
    if root.left and root.right:          #注意这里的顺序很重要！
      return min(left, right) + 1
    else:
      return left + right + 1 
    
# 404 左叶子之和  
# 判断逻辑：每个子树最底层的左侧叶子
# 如果我们在递归中利用判断储存数据，那么可以在return的对象中添加判断遍历哪个子树的逻辑。
class Solution:
  def sumOfLeftLeaves(self, root: TreeNode) -> int:
    if root is None:            #碰到头都是0，因为没用
      return 0
    if root.left and not root.left.right and not root.left.left:       #如果发现又满足合适的子叶，返回，并且从这个node开始专注于右子树
      return self.sumOfLeftLeaves(root.right) + root.left.val
    else: 
      return self.sumOfLeftLeaves(root.right) + self.sumOfLeftLeaves(root.left)   #如果没有发现合适的继续遍历

# 687 最长同值路径
# 这一题我最开的是想法是：遍历每一个node，把node都当作root，然后看可以延伸出来多少种情况，最终用全局变量做动态储存。但是这样重复的计算量太大了。至少是n2了
# 难度在于：因为遍历的顺序，如何更新我们的全局变量？
# solution给到的思路是：跟我的类似，每一层递归都是在判断当前节点；
#我犯了错误：递归的时候只用判断当前层的逻辑就行了，不要考虑下一层；如果val相同的话，我们连接起来，可以求得left+right的最大值，那么上一层也是这种逻辑，直接在一层递归中判断就可以了。
# see the code
class Solution:
  def longestUnivaluePath(self, root: TreeNode) -> int:
    def findLen(root): #findLen返回的是当前Node值相同的最长子树的Node数量；不可能一个method有多个作用。但是我们可以联合全局变量实现多重功效
      if not root:
        return 0
      left = findLen(root.left)
      right = findLen(root.right)
      left_value, right_value = 0, 0
      if root.left and root.val == root.left.val:
        left_value = left + 1
      if root.right and root.val == root.right.val:
        right_value = right + 1  
      self.ans = max(self.ans, left_value + right_value) #计算的是路径，所以这里不用再+1，计算出子树的节点就可以了。
      return max(left_value, right_value)
      
    self.ans = 0
    findLen(root)
    return self.ans
#通过递归里面的判断，将两层递归/递归之间联系起来。 比如这一题，如果满足条件，那么这一层递归就比下一层递归+1；
#再通过左右子树的递归将满足题意的值通过global传递出去。



# 337 打家劫舍 III
# 关键点在于是否选择当前节点。
class Solution:
  def rob(self, root: TreeNode) -> int:
    def dfs(root):
      if not root:
        return 0, 0 #严格符合规矩，return两个值
      
      left = dfs(root.left)      #在判断前进入递归，满足从下往上走，最终确定return跟节点，满足此题。
      right = dfs(root.right) 
      select = left[1] + right[1] + root.val
      unselect = max(left) + max(right)     #如果不选的话，那么left,right选不选的情况都要考虑。
      return select, unselect 
    return max(dfs(root))



# 671 二叉树中第二小的节点
#基本思想：因为根节点肯定为最小值，那么把根节点带入到所有递归中去，将与根节点.val不同的所有节点带入判断中去找到最小值，这个最小值就是我们要找的值，否则return -1；
#tip: 因为return的default是-1，因此可以一开始直接将-1赋值给我们的变量
#dfs
class Solution:
  def findSecondMinimumValue(self, root: TreeNode) -> int:
    self.ans = -1
    self.dfs(root, root.val)
    return self.ans
    
  def dfs(self, root: TreeNode, target) -> int:
    if not root:
      return 
    
    if root.val != target:
      if self.ans == -1:
        self.ans = root.val  
      else:
        self.ans = min(root.val, self.ans)    

    self.dfs(root.left, target)
    self.dfs(root.right, target)
#这个ans怎么优化？好像没办法优化，即使是在全局进行生命的，在各个method中调用仍然需要使用self
#时间复杂度为On，空间复杂度常数，一般为递归的层数


#637  二叉树的层平均值
class Solution:
  def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
    if not root:
      return 0 
    
    queue = collection.deque([root])
    while queue:
      total = 0
      size = len(queue)
      for _ in range(size):
        node = queue.popleft()
        total += node.val
        left = node.left
        right = node.right
        if left: 
          queue.append(left)
        if right:
          queue.append(right)
      res = total / size
    return res
    #展开用cmd+k,cmd+L
#这一题的有趣点在想到用for和size去帮助解题。


#513 找树左下角的值
class Solution:
    def findBottomLeftValue(self, root: TreeNode) -> int:
      res = root.val
      queue = [root]
      while queue:
        size = len(queue)
        for _ in range(size):   #要想针对同一层进行什么操作，这里就适合用for循环
          if _ == 0:            #很巧妙地用index去看这一层取到的特殊的值
            res = queue[_].val  
          node = queue.pop(0)
          if node.left:
            queue.append(node.left)        
          if node.right:
            queue.append(node.right)
      return res

#144 二叉树的前序遍历
class Solution:
  def preorderTraversal(self, root: TreeNode) -> List[int]:
    def preOrder(root):
      if not root:
        return
      res.append(root.val)
      preOrder(root.left)
      preOrder(root.right)
    res = list()
    preOrder(root)
    return res 

# #中序遍历
# order(node.left)
# res.append(node.val)
# order(node.right)

#后序遍历
# order(node.left)
# order(node.right)
# res.append(node.val)

#我们append放置的地点决定了，我们的该项操作是在何时执行的，是在递归前进行还是递归后进行。

#迭代
class Solution:
  def preorderTraversal(self, root: TreeNode) -> List[int]:
    if not root:
      return []
    
    res = list()
    stack = list()
    stack.append(root)

    while stack:
      node = stack.pop()
      res.append(node.val)
      if node.right:
        stack.append(node.right)
      if node.left:
        stack.append(node.left)

    return res         
#这里注意的是右子树先入栈，左子树再入栈
#因为这里有LIFO的特征，所以可以利用这种BFS的模式去模仿DFS的出入栈。

#中序遍历
class Solution:
  def preorderTraversal(self, root: TreeNode) -> List[int]:
    if not root:
      return []
    
    res = list()
    stack = list()
    node = root

    while stack or node:                  #java中判断某个数据是否为空用的是null; 判断某个数据集合是否为空用的是isEmpty()
      #将left都入栈！
      while node:
        stack.append(node)
        node = node.left
    
      node = stack.pop()
      res.append(node.val)
      if node.right:
        stack.append(node.right)      
    return res     

#后续遍历
class Solution:
  def preorderTraversal(self, root: TreeNode) -> List[int]:
    if not root:
      return []
    
    res = list()
    stack1 = list(root)
    stack2 = list()
    while stack1:
      node = stack1.pop()
      stack2.append(node)
      if node.left:
        stack1.append(node.left)      
      if node.right:
        stack1.append(node.right)
    while stack2:
      res.append(stack2.pop().val)
    return res
#后序遍历的核心思路就是利用两个栈，将前序遍历入栈顺序，再次弹栈入栈


#颜色标记法 - 中序遍历
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        WHITE, GRAY = 0, 1
        res = []
        stack = [(WHITE, root)]
        while stack:
            color, node = stack.pop()
            if node is None: continue
            if color == WHITE:
                stack.append((WHITE, node.right))
                stack.append((GRAY, node))
                stack.append((WHITE, node.left))
            else:
                res.append(node.val)
        return res
# 使用颜色标记节点的状态，新节点为白色，已访问的节点为灰色。
# 如果遇到的节点为白色，则将其标记为灰色，然后将其右子节点、自身、左子节点依次入栈。
# 如果遇到的节点为灰色，则将节点的值输出。


#莫里斯Morris遍历
# 用递归和迭代的方式都使用了辅助的空间，而莫里斯遍历的优点是没有使用任何辅助空间。
# 缺点是改变了整个树的结构，强行把一棵二叉树改成一段链表结构。
class Solution(object):
	def inorderTraversal(self, root):
		res = []
		pre = None
		while root:
			# 如果左节点不为空，就将当前节点连带右子树全部挂到
			# 左节点的最右子树下面
			if root.left:
				pre = root.left
				while pre.right:
					pre = pre.right
				pre.right = root   #此时pre是左子树的最右侧子树，将right指向我们的根节点root
				# 将root指向root的left
				tmp = root           #将root挂到tmp上
				root = root.left     #root替换成root.left
				tmp.left = None      #删除原来的链条 完成转移。
			# 左子树为空，则打印这个节点，并向右边遍历	
			else: #只要左子树存在更复杂的结构，会重复上述，直到简单变为链式。然后在下面依次append进去，就是我们的中序遍历
				res.append(root.val)
				root = root.right
		return res
#有点难理解。
