# 145. Binary Tree Postorder Traversal
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        def dfs(root):
            if not root: return 
            dfs(root.left)
            dfs(root.right)
            ans.append(root.val)
        dfs(root)
        return ans

# 94. Binary Tree Inorder Traversal
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        ans = []
        def dfs(root):
            if not root: return 
            dfs(root.left)
            ans.append(root.val)
            dfs(root.right)
        dfs(root)
        return ans



# 144. Binary Tree Preorder Traversal
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def dfs(root):
            if not root: return None
            nonlocal res
            res.append(root.val)
            dfs(root.left)
            dfs(root.right)
        res = []
        dfs(root)
        return res


# 589. N-ary Tree Preorder Traversal
# 这一题只能用iteration不能用recursion，因为没有left，right只有一个children指针。
# 把children倒序插入stack中，pop出来的时候才能是正序。
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        if root is None: return []
        
        stack, output = [root], []
        while stack:
            root = stack.pop()
            output.append(root.val)
            stack.extend(root.children[::-1])
        return output

# 590. N-ary Tree Postorder Traversal
# 这种N-tree的总是会很难。首先一定是需要[：：-1]，因为pop的顺序是倒序的，但是究竟什么时候倒序，是要讲究的。
class Solution:
    def postorder(self, root: 'Node') -> List[int]:
        if not root: return []
        stack, output = [root], []
        while stack:
            root = stack.pop()
            if root is not None:
                output.append(root.val)
            for c in root.children:
                stack.append(c)
            
        return output[::-1]

# 102. Binary Tree Level Order Traversal
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if root is None:
            return []
        res = []
        q = [root]
        
        while q:
            temp = []
            for i in range(len(q)):
                cur = q.pop(0)
                temp.append(cur.val)
                if cur.left: q.append(cur.left)
                if cur.right: q.append(cur.right)
            res.append(temp)
        return res


# 103. Binary Tree Zigzag Level Order Traversal
from collections import deque
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root: return []
        res = []
        # 这里的temp不能放在while中，不是每一次循环都会把会把结果记录下来，如果直接放进去，那么会导致temp为空
        temp = deque()
        flag = True
        q = deque([root, None])
        
        # 因为利用到了None的缘故，所以不能直接些while q：；而要用len（）来变通一下。
        # 这里None的位置在哪里？会把每一层间隔开？哦？如果没有遇到None，就不会添加进去，遇到None意味着当前Node已结束。
        # 所以总体的逻辑是：没有碰到None，按照方向添加temp，遇到了换方向，添加res，然后开启下一轮。
        while len(q) > 0: 
            cur = q.popleft()
            
            # 我们的q里面是一个node， 一个None存的，主要是为了缓冲。
            # 如果当前遇到节点的话，
            if cur:
                if flag:
                    temp.append(cur.val)
                else:
                    temp.appendleft(cur.val)

                # 添加进q的时候都是正序
                if cur.left: q.append(cur.left)
                if cur.right: q.append(cur.right)
            else:
                res.append(temp)
                if len(q) > 0:
                    q.append(None)
                temp = deque()
                flag = not flag
        return res

# 107. Binary Tree Level Order Traversal II
# 送分题
class Solution:
    def levelOrderBottom(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root is None:
            return []
        res = []
        q = [root]
        
        while q:
            temp = []
            for i in range(len(q)):
                cur = q.pop(0)
                temp.append(cur.val)
                if cur.left: q.append(cur.left)
                if cur.right: q.append(cur.right)
            res.append(temp)
        return res[::-1]



# 108. Convert Sorted Array to Binary Search Tree
# 虽然自己没有秒杀，但还是高兴自己写出来了。
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums: return None
        
        mid = len(nums) // 2
        node = ListNode(nums[mid])
        node.left = self.sortedArrayToBST(nums[:mid])
        node.right = self.sortedArrayToBST(nums[mid + 1:])
        
        return node
        
# 这题的集体关键是什么？
# preorder的顺序去recursion
# inorder的左右是他的sub-tree
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def toTreeHelper(left, right):
            nonlocal pre_index
            if left > right: return None
            val = preorder[pre_index]
            root = TreeNode(val)
            
            pre_index += 1
            
            root.left = toTreeHelper(left, in_map[val] - 1)
            root.right = toTreeHelper(in_map[val] + 1, right)
            
            return root
        
        pre_index = 0
        in_map = {}
        for i, v in enumerate(inorder):
            in_map[v] = i
        
        return toTreeHelper(0, len(preorder) - 1)
        
"""
preorder [3 9 20 15 7]  -- 刚好是在递归中构建node的顺序
postorder = [9,15,7,20,3] -- 树于y轴对称
然后倒序出 3 , 20, 7, 15, 9
先构建right-sub tree 再构建left- sub tree
两种order的list 构建树的顺序 ⬆️
"""
# 106. Construct Binary Tree from Inorder and Postorder Traversal
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        
        def helper(l, r):
            if l > r:
                return None
            val = postorder.pop()  
            root = TreeNode(val)
            
            index = map[val]
            root.right = helper(index + 1, r) 
            root.left = helper(l, index - 1)
            
            return root
        
        map = {val: i for i, val in enumerate(inorder)}
        return helper(0, len(inorder) - 1)
        

# 114. Flatten Binary Tree to Linked List
# woc牛呀
# 最优方法，介绍一下思路。
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        if not root: return []
        node = root
        while node:
            if node.left:
                right_most = node.left
                # while的存在的前提是：我们当下node，有left-tree，意味着需要变换
                # while的目的，找到left-tree中的tail，可以与当前right-tree连接起来
                while right_most.right:
                    right_most = right_most.right
                # tail与right-tree连接
                right_most.right = node.right
                # 将left-tree转移到right-tree
                node.right = node.left
                # 调整left pointer为None
                node.left = None

            # 每次我们只向right寻找为什么？
            # 因为我们针对每一个node，都把其left-tree在当下直接移动到右边。
            # 因此，下一次while的node就是现在while node的left。模拟pre order
            node = node.right

# 直接用一个helper模拟preorder了，但是注意，这里的l,r指的是leaf node也就是我们说的right tail.
class Solution:
    def helper(self, node):
        if not node: return None
        if not node.left and not node.right:
            return node
        l = self.helper(node.left)
        r = self.helper(node.right)
        if l:
            # left-tree -> node.right
            l.right = node.right
            # node.right 不再指向原来的node，而是指向原来的left
            node.right = node.left
            # left pointer -> None
            node.left = None
        return r if r else l

    def flatten(self, root: TreeNode) -> None:
        
        self.helper(root)

# 889. Construct Binary Tree from Preorder and Postorder Traversal
# postorder的node可以看作root，当遍历到node时候，它的左右子树一定遍历完了。可以利用preorder进行左右遍历。
# 而且postorder一定是先遍历完left，再遍历right
"""
class Solution {
    int preIndex = 0, posIndex = 0;
    
// Create a node TreeNode(pre[preIndex]) as the root.

// Becasue root node will be lastly iterated in post order,
// if root.val == post[posIndex],
// it means we have constructed the whole tree,

// If we haven't completed constructed the whole tree,
// So we recursively constructFromPrePost for left sub tree and right sub tree.

// And finally, we'll reach the posIndex that root.val == post[posIndex].
// We increment posIndex and return our root node.
    public TreeNode constructFromPrePost(int[]pre, int[]post) {
        
        TreeNode root = new TreeNode(pre[preIndex++]);
        // 走完左递归一定能找到
        if (root.val != post[posIndex])
            root.left = constructFromPrePost(pre, post);
        // 再走右递归
        if (root.val != post[posIndex])
            root.right = constructFromPrePost(pre, post);
        // 最终再pos想加进入下一层，不好想。
        posIndex++;
        return root;
    }
}
"""


"""
# 115 

class Solution {
    public int numDistinct(String s, String t) {
        int M = s.length();
        int N = t.length();
        
        int[][] dp = new int[M+1][N+1];
        
        for (int j = 0; j <= N; j++) dp[M][j] = 0;
        for (int i = 0; i <= M; i++) dp[i][N] = 1;
        
        for (int i = M-1; i >= 0; i--) {
            for (int j = N-1; j >= 0; j--) {
                dp[i][j] = dp[i+1][j];
                
//              为什么要从[i+1][j+1]中走，因为不能➕[i+1]/[j+1]的原因是，前面有new的计算了，所以为了保险。
                if (s.charAt(i) == t.charAt(j)) dp[i][j] += dp[i+1][j+1]; 
            }
        }    
        return dp[0][0];
    }
}

"""



# 1008. Construct Binary Search Tree from Preorder Traversal
# 借由这个题把构建BST好好盘一下
# 1. 进入helper的left，right是in-order的，为什么？因为inorder中的index才可以左边是左子树的，右边是右子树的
# 2. 本质上就是利用preorder和inorder啦。每一次preorder出来，是为了构建Node。
# 3. 那inorder的作用呢？就是为了判断是否需要弹出了...em 也不错
class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> TreeNode:
        # index也是
        def helper(in_left = 0, in_right = len(preorder)):
            nonlocal pre_idx
            # if there is no elements to construct subtrees
            if in_left == in_right:
                return None
            root_val = preorder[pre_idx]
            root = TreeNode(root_val)
            index = idx_map[root_val]

            # recursion 
            pre_idx += 1
            # build left subtree # 这里的右边界是index，而非index - 1
            root.left = helper(in_left, index)
            # build right subtree
            root.right = helper(index + 1, in_right)
            return root
        
        
        # 🌟 BST的sorted就是 inorder顺序
        inorder = sorted(preorder)
        pre_idx = 0
        idx_map = {val:idx for idx, val in enumerate(inorder)}
        return helper()


# 周赛第二题
class Solution:
    def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
        ans1, ans2 = [],[]
        lost = defaultdict(int)
        people = set()
        for win, lose in matches:
            people.add(win)
            people.add(lose)
        for i, j in matches:
            lost[j] += 1
        for i in people:
            if lost[i] == 0:
                ans1.append(i)
            elif lost[i] == 1:
                ans2.append(i)
        ans1.sort()
        ans2.sort()
        return [ans1,ans2]
        
# 周赛第三题，大猩猩吃香蕉
# 总体思路是：lo和hi区间是每个人能分多少，每一次判断能不能分，然后缩小区间。
class Solution:
    def maximumCandies(self, candies: List[int], k: int) -> int:
        if sum(candies) < k:
            return 0
        
        # 我们返回的值，最大只能是high了
        low, high = 1, sum(candies)//k
        while low != high:
            # 这里的mid就是除2往上走
            mid = (low+high+1) >> 1
            # 每个candy//mid的意义在于这一组能分几个。如果够分low = mid
            if sum(i//mid for i in candies) >= k:
                low = mid
            # 不够分的话再说。
            else:
                high = mid-1
        return low

# 297. Serialize and Deserialize Binary Tree
class Codec:
    # return str with #hashtag seperating each child-tree
    def serialize(self, root):
        def doit(node):
            if node:
                vals.append(str(node.val))
                doit(node.left)
                doit(node.right)
            else:
                vals.append('#')
        vals = []
        doit(root)
        return ' '.join(vals)

    # iter()用来生成迭代器
    # next()迭代器可以用的API
    # 因为serialize的时候是pre-order的，因此decode时也按照这个order
    def deserialize(self, data):
        def doit():
            val = next(vals)
            if val == '#':
                return None
            node = TreeNode(int(val))
            node.left = doit()
            node.right = doit()
            return node
        vals = iter(data.split())
        return doit()


# 104. Maximum Depth of Binary Tree
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        def dfs(root, depth):
            if not root: return depth
            depth += 1
            left = dfs(root.left, depth)
            right = dfs(root.right, depth)
            return max(left, right)
        depth = 0
        return dfs(root, depth)

# 101. Symmetric Tree
# 做过但还是没想起来。如果要同时处理不同枝怎么办？同时入两个tree，但是入function的时候有两种情况。
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def dfs(left, right):
            if left is None and right is None: return True
            if left is None or right is None: return False
            
            return left.val == right.val and dfs(left.left, right.right) and dfs(left.right, right.left)
        
        return dfs(root.left, root.right)
        
# 226. Invert Binary Tree
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def dfs(root):
            if not root: return 
            root.left, root.right = root.right, root.left
            dfs(root.left)
            dfs(root.right)
            
        dfs(root)
        return root
            
# 617. Merge Two Binary Trees
class Solution:
    def mergeTrees(self, t1: Optional[TreeNode], t2: Optional[TreeNode]) -> Optional[TreeNode]:
        if not t1: return t2
        if not t2: return t1
        t1.val += t2.val
        t1.left = self.mergeTrees(t1.left, t2.left)
        t1.right = self.mergeTrees(t1.right, t2.right)
        return t1
        
# 100. The same tree
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q: return True
        if not p or not q: return False    
        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
            
# 112. has path sum
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root: return False
        targetSum -= root.val
        if not root.left and not root.right:
            return targetSum == 0
        
        return self.hasPathSum(root.left,) or self.hasPathSum(root.right)

# 236. Lowest Common Ancestor of a Binary Tree
"""
class Solution {
    
    private TreeNode ans;
    
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        this.recurseTree(root, p, q);
        return this.ans;
    }
    
    private boolean recurseTree(TreeNode currentNode, TreeNode p, TreeNode q) {
        if (currentNode == null) return false;
        int left = this.recurseTree(currentNode.left, p, q) ? 1 : 0;
        int right = this.recurseTree(currentNode.right, p, q) ? 1: 0;
        int mid = (currentNode == p || currentNode == q) ? 1 : 0;
        
        // 为什么会有mid，因为根节点可能为两个中的一个
        if (mid + left + right >= 2) {
            this.ans = currentNode;
        }
        // 如果等于零意味着没有找到任何的
        return (mid + left + right > 0);
            
    }
}
"""
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        self.ans = TreeNode()
        def dfs(root):
            # nonlocal ans
            if not root: return False
            
            left = dfs(root.left)
            right = dfs(root.right)
            mid =  root == p or root == q
            
            if (mid + left + right) >= 2: self.ans = root
            return 1 if (mid+left+right) > 0 else 0
        
        dfs(root)
        return self.ans

# 222. Count Complete Tree Nodes
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        return 1 + self.countNodes(root.left) + self.countNodes(root.right) if root else 0

    
# 113. Path Sum II
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        def dfs(node, remainingSum, pathNodes, pathsList):
            if not node:
                return 
        
            pathNodes.append(node.val)

            if remainingSum == node.val and not node.left and not node.right:
                pathsList.append(list(pathNodes))
            else:    

                dfs(node.left, remainingSum - node.val, pathNodes, pathsList)
                dfs(node.right, remainingSum - node.val, pathNodes, pathsList)

            pathNodes.pop()    
    
        pathsList = []
        dfs(root, sum, [], pathsList)
        return pathsList

# Path Sum III
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> int:       
        def preorder(root, cur_sum):
            # 把count引入进来
            # 写递归的edge case/end case
            nonlocal count
            if not root: return 
            # 把当前值给update出来
            cur_sum += root.val
            # 判断是否满足我们的值
            if cur_sum == k: 
                count += 1
            # 先update count，再去update我们的hashmap
            count += h[cur_sum - k]
            h[cur_sum] += 1
            preorder(root.left, cur_sum)
            preorder(root.right, cur_sum)
            # 如果离开了这个level，就把当前的sum给resume
            h[cur_sum] -= 1
            
        h = defaultdict(int)
        count, k = 0, sum
        preorder(root, 0)
        return count
        

# 124. Binary Tree Maximum Path Sum
# 这一题跟我的思路一致，为什么我写不出来？
# 首先recursion返回的对象是什么？
# base case是什么？是leaf return 0
# left，right递归的是什么？最大值吧应该，如果是负数，那么就可以不用要
# 我的思路重合点：利用递归，利用左右子树，利用一个全局变量
class Solution:
    def maxPathSum(self, root):
        def helper(node):
            nonlocal target
            if not node: return 0
            left = max(helper(node.left), 0)
            right = max(helper(node.right), 0)
            p_sum = node.val + left + right
            target = max(target, p_sum)
            return node.val + max(left, right)
        target = float('-inf')
        helper(root)
        return target



# 297. Serialize and Deserialize Binary Tree
class Codec:
    # return str with #hashtag seperating each child-tree
    def serialize(self, root):
        def doit(node):
            if node:
                vals.append(str(node.val))
                doit(node.left)
                doit(node.right)
            else:
                vals.append('#')
        vals = []
        doit(root)

        return ' '.join(vals)

    # iter()用来生成迭代器
    # next()迭代器可以用的API
    # 因为serialize的时候是pre-order的，因此decode时也按照这个order
    def deserialize(self, data):
        def doit():
            val = next(vals)
            if val == '#':
                return None
            node = TreeNode(int(val))
            node.left = doit()
            node.right = doit()
            return node
        vals = iter(data.split())
        return doit()

# 96. Unique Binary Search Trees
class Solution:
    def numTrees(self, n):
        G = [0]*(n+1)
        G[0], G[1] = 1, 1

        for i in range(2, n+1):
            for j in range(1, i+1):
                G[i] += G[j-1] * G[i-j]

        return G[n]


# 95. Unique Binary Search Trees II
# 亮点: 3个for；list构建树的方式。
class Solution:
    def generateTrees(self, n):
        
        def generate_trees(start, end):
            if start > end:
                return [None,]
            
            all_trees = []
            # 三层for循环很巧妙。刚好把左右所有情况都可以遍历完*
            #   第一层：for每个node，并且计算出左右所有的可能性的tree
            #   第二层：选择一个left
            #   第三层：选择一个right，进行组合。
            # 难点在于如何找到所有可能性的subtree的？
            for i in range(start, end + 1):  # pick up a root
                # all possible left subtrees if i is choosen to be a root
                left_trees = generate_trees(start, i - 1)
                
                # all possible right subtrees if i is choosen to be a root
                right_trees = generate_trees(i + 1, end)
                
                # connect left and right subtrees to the root i
                for l in left_trees:
                    for r in right_trees:
                        current_tree = TreeNode(i)
                        current_tree.left = l
                        current_tree.right = r
                        all_trees.append(current_tree)
            
            return all_trees
        # 这里1~n其实是val，所以不用0-indexed
        return generate_trees(1, n) if n else []




# 230. Kth Smallest Element in a BST
# BFS的解法是有意思的
# 我的思路是先遍历呗，inorder，dfs解决问题，time = space = O(n)
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        pick_list = []
        def dfs(root):
            if not root:
                return 
            dfs(root.left)
            pick_list.append(root.val)
            dfs(root.right)
        dfs(root)
        return pick_list[k-1]
# BFS的iterate的code挺有趣哦
# 针对BST一个节点永远是它right-subtree的最小值。
    def kthSmallest2(self, root, k):
        stack = []
        
        while True:
            # 先把左右最上层left treeNode -> stack
            while root:
                stack.append(root)
                root = root.left
            # 从第一位pop，然后针对每一个node，我们再重新进入while，为什么？
            # 这样模仿了in-order，刚好就是按照顺序来的。
            root = stack.pop()
            k -= 1
            if not k:
                return root.val
            # 如果想左下不去了，那么就该往右下去了，bfs中的stack就相当于递归中的递归stack
            root = root.right

# 99. Recover Binary Search Tree
# 总体的思路就是：inorder -> 找到x,y -> 处理x,y； 其中第三步可以学习
class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        def inorder(r):
            return inorder(r.left) + [r.val] + inorder(r.right) if r else []
        
        def find_two_swapped(nums):
            n = len(nums)
            x = y = None
            # 针对这个循环，我们会对y进行两次赋值，并且在第二次赋值时，通过x的判断-> break跳出循环
            for i in range(n-1):
                if nums[i+1] < nums[i]:
                    y = nums[i+1]
                    if x is None:
                        x = nums[i]
                    else:
                        break
            return x,y
        
        def recover(r, count):
            if r:
                if r.val == x or r.val == y:
                    r.val = y if r.val == x else x
                    count -= 1
                    if count == 0:
                        return 
                recover(r.left, count)
                recover(r.right, count)
        
        nums = inorder(root)
        x, y = find_two_swapped(nums)
        recover(root, 2)

# BFS的方法模拟BST-inorder，用dfs也可以完成。
# 在遍历的同时用x,y指向错位的两个node，最后更换就成了。
class Solution:
    def recoverTree(self, root: TreeNode):
        stack = []
        x = y = pred = None
        
        while stack or root:
            # 固定的模版 —— 模拟inorder
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            # 固定的模版 —— 找到更换的值
            if pred and root.val < pred.val:
                y = root
                if x is None:
                    x = pred 
                else:
                    break
            pred = root
            root = root.right

        x.val, y.val = y.val, x.val

"""
MORRIS遍历 
空间复杂度为1 因为节省了空闲的指针
记作当前节点为cur。

1 如果cur无左孩子 cur向右移动(cur=cur.right)  | 向右的道路早已经铺垫好
2 如果cur有左孩子 找到cur左子树上最右的节点 记为mostright
    2-1 如果mostright的right指针指向空 让其指向cur cur向左移 (cur=cur.left) | 连接起来
    2-2 如果mostright的right指针指向cur 让其指向空 cur向右移 (cur=cur.right) | 此时已经遍历当前节点了 返回途中 应该继续前往right subtree
"""
# Morris方法，就是那种可以通过inorder把tree->linkedlist的算法
# predecessor 就是 predecessor， 只有当.right移动的时候采用检测pred和当前cur的值，因为这是inorder
class Solution:
    def recoverTree(self, root):
        # predecessor is a Morris predecessor. 
        # In the 'loop' cases it could be equal to the node itself predecessor == root.
        # pred is a 'true' predecessor, 
        # the previous node in the inorder traversal.
        x = y = predecessor = pred = None
        
        while root:
            # If there is a left child
            # then compute the predecessor.
            # If there is no link predecessor.right = root --> set it.
            # If there is a link predecessor.right = root --> break it.
            if root.left:       
                # Predecessor node is one step left 
                # and then right till you can.
                predecessor = root.left
                while predecessor.right and predecessor.right != root:
                    predecessor = predecessor.right
 
                # set link predecessor.right = root
                # and go to explore left subtree
                if predecessor.right is None:
                    predecessor.right = root
                    root = root.left
                # break link predecessor.right = root
                # link is broken : time to change subtree and go right
                else:
                    # check for the swapped nodes
                    if pred and root.val < pred.val:
                        y = root
                        if x is None:
                            x = pred 
                    pred = root
                    
                    predecessor.right = None
                    root = root.right
            # If there is no left child
            # then just go right.
            else:
                # check for the swapped nodes
                if pred and root.val < pred.val:
                    y = root
                    if x is None:
                        x = pred 
                pred = root
                
                root = root.right
        
        x.val, y.val = y.val, x.val


# 173. Binary Search Tree Iterator
# Take-away: 利用index全局变量去判断next；利用helper function直接在__init__排序掉
class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        self.nodes_sorted = []
        self.index = -1
        self._inorder(root)
    
    def _inorder(self, root):
        if not root:
            return 
        self._inorder(root.left)
        self.nodes_sorted.append(root.val)
        self._inorder(root.right)

    def next(self) -> int:
        self.index += 1
        return self.nodes_sorted[self.index]

    def hasNext(self) -> bool:
        return self.index + 1 < len(self.nodes_sorted)
        

"--------------------Trie Tree字典树-----------------------------"
# 208. Implement Trie (Prefix Tree) 经典题
class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    

    def insert(self, word: str) -> None:
        current = self.root
        for letter in word:
            current = current.children[letter]
        current.is_word = True

    def search(self, word: str) -> bool:
        current = self.root
        for letter in word:
            current = current.children.get(letter)
            if current is None:
                return False
        return current.is_word

    def startsWith(self, prefix: str) -> bool:
        current = self.root
        for letter in prefix:
            current = current.children.get(letter)
            if current is None:
                return False
        return True
   

# 720. Longest Word in Dictionary
# 根据题意加一点巧思就成。分别是trieNode里面的isEnd, word
class TrieNode(object):
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.isEnd = False
        self.word = ''

class Trie(object):
    def __init__(self):
        self.root = TrieNode()
    def insert(self, word):
        node = self.root
        for c in word:
            node = node.children[c]
        node.isEnd = True
        node.word = word
    def bfs(self):
        q = collections.deque([self.root])
        res = ''
        while q:
            cur = q.popleft()
            for n in cur.children.values():
                if n.isEnd:
                    q.append(n)
                    # 这里非常有意思，什么时候update我们的res呢？
                    # 1.要么你比我的res长，要么你就是smallest lexicographical order
                    if len(n.word)>len(res) or n.word<res:
                        
                        res = n.word
                        
        return res 
    
class Solution(object):
    def longestWord(self, words):
        trie = Trie()
        for w in words: trie.insert(w)
        return trie.bfs()
        

# Word Search II
# 这里的思路是把board转化为trie tree字典树，因此只用进行一次dfs就成了。
# 如果是intuitive的方法，针对每个格子都要进行dfs，会有些浪费。
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        WORD_KEY = '$'
        trie = {}
        # 看我们都有什么词
        for word in words:
            # trie此时是空dict
            
            # 字典有点特殊：这里node的操作其实本质上会影响trie
            node = trie
            # 看有word里有什么字母
            for letter in word:
                # 初始化，所以都没有
                # 如果node没有，那就用node创造一个{}，并且key = letter
                # 字典有点特殊，这里的Node=node.setdefault()会让dict里面新生成一个{},同时赋值给这个node了，这个是如何进入{}的方法。
                node = node.setdefault(letter, {})
            # 最后一层的时候，把值输入进去，666
            node[WORD_KEY] = word
            
        
        rowNum = len(board)
        colNum = len(board[0])
        
        matchedWords = []
        
        def backtracking(row, col, parent):    
            
            letter = board[row][col]
            currNode = parent[letter]
            
            # 看当前curNode有没有WORD_KEY，即到最后一层没有
            word_match = currNode.pop(WORD_KEY, False)
            if word_match:
                # 如果发现"$"的话，意味着我们找到了哈哈。
                matchedWords.append(word_match)
            
            # 这里开始就是正常的了 
            board[row][col] = '#'
            
            # 4个方向
            for (rowOffset, colOffset) in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                newRow, newCol = row + rowOffset, col + colOffset     
                if newRow < 0 or newRow >= rowNum or newCol < 0 or newCol >= colNum:
                    continue
                # 当前letter存在与当前层的trie tree中
                if not board[newRow][newCol] in currNode:
                    continue
                # 如果能到这里可以继续往下走。
                backtracking(newRow, newCol, currNode)
        
            # 回溯返回正常值
            board[row][col] = letter
        
            # 
            # 这里用来优化，如果发现找过的话，直接remove the matched leaf node.
            # Optimization: incrementally remove the matched leaf node in Trie.
            if not currNode:
                parent.pop(letter)

        for row in range(rowNum):
            for col in range(colNum):
                # starting from each of the cells
                if board[row][col] in trie:
                    backtracking(row, col, trie)
        
        return matchedWords    


# 1353. Maximum Number of Events That Can Be Attended
# 这个priority queue
# 总体思路就是：想一个问题，我们某一天应该参加哪个event？参加结束最早的。
# O(nlogn)
class Solution:
    def maxEvents(self, events: List[List[int]]) -> int:
        min_heap = []  # min heap of events end time
        events.sort(key = lambda e: e[0])  # sort events by start time

        i = count_events_attended = cur_day = 0
        # 当i没遍历完/heap中还有值
        while i < len(events) or min_heap:
            # 如果heap没值
            if not min_heap:
                # cur_day是当前天的start_day
                cur_day = events[i][0]
            
            # add open events for cur_day
            # 如果碰到新的event开始的时间要比当前的event早，我们把结束时间入min_heap
            # 考虑到特殊性，其实是吧当前天和之后与其同一天开始的event的结束日子入min_heap
            while i < len(events) and events[i][0] <= cur_day:
                heappush(min_heap, events[i][1])
                i += 1
            
            # 入完之后，去掉一个最早开始的，及出席最早的
            heappop(min_heap)  # attend the event ends earliest
            # 出席+=1
            count_events_attended += 1

            # 逻辑上的next day
            cur_day += 1
            # remove close events for cur_day
            # cur_day之前的所有的event都要出栈
            while min_heap and min_heap[0] < cur_day:
                heappop(min_heap)

        return count_events_attended


"-------------------------------------- Segment Tree 经典应用题 虽然题不好----------------------------------------------------------"
# 307. Range Sum Query - Mutable 
# segment Tree 线段树，主要就是解决区间内的动态查询问题。
"""
    The idea here is to build a segment tree. Each node stores the left and right
    endpoint of an interval and the sum of that interval. All of the leaves will store
    elements of the array and each internal node will store sum of leaves under it.
    Creating the tree takes O(n) time. Query and updates are both O(log n).
"""

#Segment tree node
class Node(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.total = 0
        self.left = None
        self.right = None
        

class NumArray(object):
    def __init__(self, nums):

        #helper function to create the tree from input array
        def createTree(nums, l, r):
            #base case
            if l > r:
                return None
                
            # leaf node
            if l == r:
                n = Node(l, r)
                n.total = nums[l]
                return n
            
            mid = (l + r) // 2
            
            root = Node(l, r)
            
            #recursively build the Segment tree
            root.left = createTree(nums, l, mid)
            root.right = createTree(nums, mid+1, r)
            
            #Total stores the sum of all leaves under root
            #i.e. those elements lying between (start, end)
            root.total = root.left.total + root.right.total
                
            return root
        
        self.root = createTree(nums, 0, len(nums)-1)
            
    def update(self, i, val):
        #Helper function to update a value
        def updateVal(root, i, val):
            
            #Base case. The actual value will be updated in a leaf.
            #The total is then propogated upwards
            if root.start == root.end:
                root.total = val
                return val
        
            mid = (root.start + root.end) // 2
            
            #If the index is less than the mid, that leaf must be in the left subtree
            if i <= mid:
                updateVal(root.left, i, val)
                
            #Otherwise, the right subtree
            else:
                updateVal(root.right, i, val)
            
            #Propogate the changes after recursive call returns
            root.total = root.left.total + root.right.total
            
            return root.total
        # 相当于多传了个参
        return updateVal(self.root, i, val)

    def sumRange(self, i, j):
        #Helper function to calculate range sum
        def rangeSum(root, i, j):
            
            #If the range exactly matches the root, we already have the sum
            if root.start == i and root.end == j:
                return root.total
            
            mid = (root.start + root.end) // 2
            
            #If end of the range is less than the mid, the entire interval lies
            #in the left subtree
            if j <= mid:
                return rangeSum(root.left, i, j)
            
            #If start of the interval is greater than mid, the entire inteval lies
            #in the right subtree
            elif i >= mid + 1:
                return rangeSum(root.right, i, j)
            
            #Otherwise, the interval is split. So we calculate the sum recursively,
            #by splitting the interval
            else:
                return rangeSum(root.left, i, mid) + rangeSum(root.right, mid+1, j)
        
        return rangeSum(self.root, i, j)

