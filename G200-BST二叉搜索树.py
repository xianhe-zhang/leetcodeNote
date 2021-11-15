
"""
二叉搜索树的特点：
若它的左子树不空，则左子树上所有结点的值均小于它的根结点的值； 若它的右子树不空，则右子树上所有结点的值均大于它的根结点的值； 
它的左、右子树也分别为二叉排序树。二叉搜索树作为一种经典的数据结构，它既有链表的快速插入与删除操作的特点，又有数组快速查找的优势；
所以应用十分广泛，例如在文件系统和数据库系统一般会采用这种数据结构进行高效率的排序与检索操作。 
"""



#669 修剪二叉搜索树
#这一题灵活地运用了BST的特性。
class Solution:
  def trimBST(self, root: TreeNode, low: int, high: int) -> TreeNode:
    def trim(root):   #这里写helper只是为了省略传参。
      if not root:
        return None
      #关键逻辑判断：根据BST的特性，如果当前值超过限制，那么其中一子树肯定不满足条件；那么我们只用返回相反的子树就行了。
      elif root.val > high:             
        return trim(root.left)
      elif root.val < low:
        return trim(root.right)
      else:
        #这里是将current node的link指向修剪过后的子树
        root.left = trim(root.left)
        root.right = trim(root.right)
        return root
    return trim(root)


#230 二叉搜索树中第k小的数字
#不要被BST的特性束缚了思路。想想前序中序后序的三种遍历方式 -> 中序遍历BST就是一个有序数组。
class Solution:
  def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
    res = list()
    self.dfs(root,res)
    return res[k - 1]

  def dfs(self, root, res):
    if not root:
      return
    self.dfs(root.left, res)
    res.append(root.val)
    self.dfs(root.right, res)
"""
java版本：
public class Solution {
  List<Integer> list;
  public int kthSmallest(TreeNode root, int k) {
    list = new ArrayList<Integer>();
    dfs(root);
    return list.get(k - 1);
  }

  public void dfs(TreeNode, int k) {
    if(root == null) {
      return;
    }
    dfs(root.left, list);
    list.add(root.val);
    dfs(root.right, right);
  }
}

"""

#方法二： 只保存第k个节点
class Solution:
  def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
    self.res = 0
    self.count = 0
    self.dfs(root, k)   #这里不能使用参数传递，而采用self变成全局变量。 详细explaination is below;
    return self.res
  
  def dfs(self, root, k):
    if not root: return
    self.dfs(root.left, k)
    self.count += 1     #这里的self.count += 1不能放在其他地方，要放在真正逻辑判断的地方，而非是代码书写遍历的任意位置。
    if self.count == k:
      self.res = root.val
    self.dfs(root.right, k)
#注意这题如果dfs(root, k, res, count)将参数进行传参，可以将上层数据传递到下层，但是无法将下层的传递回上层，所以这里利用全局变量。
"""
Java version:
public class Solution {
  int res;
  int count;

  public int kthSmallest (TreeNode root, int k) {
    res = 0;
    count = k;
    dfs(root);
    return res
  }

  public void dfs(TreeNode root) {
    if(root == null) {
      return;
    }
    dfs(root.left);
    count--;
    if(count == 0) {            //注意这里不能用count：0->k 而是应该用 k->0，否则还需要把k从一个方法传递到另一个方法，从k->0的话我们就可以通过全局变量进行控制了。
      res = root.val;
      return;
    }
    dfs(root.right);
  }
}
"""

#235 二叉搜索树的最近公共祖先
#两次遍历，时间空间都为n
"""利用BST简化搜索路径找到target node，记录路径，然后比对两个节点的路径，得到分岔点，即为最近公共祖先"""
class Solution:
  def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    path_p = self.getPath(root, p)
    path_q = self.getPath(root, q)
    ancestor = None
    for i, j in zip(path_p, path_q):   #BREAK will break the current For-Loop and enter next for loop;
                                       #While it does break the whole WHILE-LOOP.
      if i == j:  
        ancestor = i
    return ancestor 

  def getPath(self, root, target):
    path = []    
    curNode = root
    # no if-loop here is needed given that we gonna use while later
    # BTW, we should NOT use ending boundary judgement here cuz this is not recursion.
    while curNode != target:
      path.append(curNode)
      if target.val < curNode.val:
        curNode = curNode.left
      else:
        curNode = curNode.right
    path.append(curNode) # special case0
    return path

#方法二 - 利用特性
# 如果两个节点值都小于根节点，说明他们都在根节点的左子树上，我们往左子树上找
# 如果两个节点值都大于根节点，说明他们都在根节点的右子树上，我们往右子树上找
# 如果一个节点值大于根节点，一个节点值小于根节点，说明他们他们一个在根节点的左子树上一个在根节点的右子树上，那么根节点就是他们的最近公共祖先节点。
# 复杂度都为时间为N, 空间为1
class Solution:
  def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    while((root.val - p.val) * (root.val - q.val) > 0):
      if root.val > p.val:
        root = root.left
      else:
        root = root.right
      # root = root.left if root.val > p.val else root = root.right
    return root

"""
Java版本

public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    //如果根节点和p,q的差相乘是正数，说明这两个差值要么都是正数要么都是负数，也就是说
    //他们肯定都位于根节点的同一侧，就继续往下找
    while ((root.val - p.val) * (root.val - q.val) > 0)
        root = p.val < root.val ? root.left : root.right;
    //如果相乘的结果是负数，说明p和q位于根节点的两侧，如果等于0，说明至少有一个就是根节点
    return root;
}
"""


#236 二叉的最近公共祖先
"""分析和拆解问题最关键！两个节点有哪些情况：
    1. x和y分别位于root的两端
    2. x和y其中一个属于母节点，即x可能在y的左子树或者是在右子树上
"""
#本题采用前序/先序遍历（根左右）
#中序遍历（左根右）；后序遍历（左右根）
class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        #找到才返回值，其他的窦唯None
        if not root or root == p or root == q:
          return root

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        #一侧无值说明，目前两个root还未找到，这一题不用怕，因为肯定有解答， 所以最差的情况就是原始根节点
        if not left: return right 
        if not right: return left
        return root 
#太6了，复杂度为都是n，空间是因为递归栈最差为N；时间是都要遍历一遍

"""
考虑四种情况的写法
if not left and not right: return # 1.
if not left: return right # 3.
if not right: return left # 4.
return root # 2. if left and right:
===================
Java版本
class Solution {
  public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) { 
    if(root == null || root == p || root == q) {
      return root;
    }
    TreeNode left = lowestCommonAncestor(root.left, p, q);
    TreeNode right = lowestCommonAncestor(root.right, p, q);
    
    if(left == null) return right;
    if(right == null) return left;
    return root;

  }
}


"""
#108

#109

#653
#530

#501

