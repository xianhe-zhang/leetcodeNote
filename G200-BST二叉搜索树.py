
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


#108 将有序数组转换为二叉搜索树
"""因为二叉树的递归是有顺序，因此非常适合采用分治的思想，首先想要构建树，每一层的书是我们要规定的"""
class Solution:
  def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
    return self.dfs(nums, 0, len(nums) - 1)

  def dfs(self, nums:List[int], low: int, high: int) -> TreeNode:
    if low > high:
      return None
    mid = low + (high - low) // 2
    root = TreeNode(nums[mid])
    root.left = self.dfs(nums, low, mid - 1)
    root.right = self.dfs(nums, mid + 1, high)
    return root

#109 有序链表转换二叉搜索树
"""有序链表，我有两个想法：1.同108先转化成有序数组，再构建 2.直接中序遍历构建，但是具体是什么结构的，我没有把握
我没有想到的方法：3. 快慢指针（找中点的特性）
"""
  #转化成有序数组
class Solution:
  def sortedListToBST(self, head: ListNode) -> TreeNode:
    curList = self.linkedNodesToArray(head)
    return self.dfs(curList, 0, len(curList) - 1)

  def linkedNodesToArray(self, head):
    res = []
    while head:
      res.append(head.val)
      head = head.next  #head.next = head.next.next
    return res

  def dfs(self, nums:List[int], low: int, high: int) -> TreeNode:
    if low > high:
      return None
    mid = low + (high - low) // 2
    root = TreeNode(nums[mid])
    root.left = self.dfs(nums, low, mid - 1)
    root.right = self.dfs(nums, mid + 1, high)
    return root
#时间复杂度: n + logn = n 空间复杂度n + logn  = n

#快慢指针
#为什么选择从中点开始构造整个树？因为高度差不能超过一！所以左右两个要比较紧密排列
class Solution:
  def sortedListToBST(self, head: ListNode) -> TreeNode:
    if not head: return None
    if not head.next: return TreeNode(head.val)

    slow, fast, pre = head, head, head
    while not fast and not fast.next:
      pre = slow 
      slow = slow.next
      fast = fast.next.next
    #理想情况下，slow是中点
    right = slow.next
    pre.next = None
    root = TreeNode(slow.val)
    root.left = self.sortedListToBST(head)
    root.right = self.sortedListToBST(right)
    return root

#复杂度 时间复杂度nlogn n是每一次递归的时候都熬while遍历 logn是栈 空间复杂度logn 调用栈

#方法三：中序遍历优化 仍需要用到len()求中点mid
class Solution:
  def sortedListToBST(self, head: ListNode) -> TreeNode:
    if not head: return None
    length = 0
    self.ptr = head#不能放在下面，因为下面的head已经不是最开始的head了
    while head:
      length += 1
      head = head.next
    # self.ptr = head #这里不能放在这里
    return self.buildTree(0, length - 1)

  def buildTree(self, start, end):
    if start > end: return None
    #这里的mid不做切断链条的处理，而是用于标记
    mid = start + (end - start) // 2
    leftTree = self.buildTree(start, mid - 1)
    root = TreeNode(self.ptr.val)   #这里是访问不到self.ptr的，self可以传递给子方法。
    #更新链表标记
    s  elf.ptr = self.ptr.next 
    #连接左子树
    root.left = leftTree
    root.right = self.buildTree(mid + 1, end)
    return root



#653（这题好垃圾，没用到bst的feature呀）两数之和 IV - 输入 BST
#写一种解法吧：新建set，用来存储已经遍历过的值，然后去遍历，如果有值，就返回true否则false
class Solution:
  def findTarget(self, root: TreeNode, k: int) -> bool:
    if not root:
      return False
    visited = [root]
    targetList = set()
    while visited:
      cur = visited.pop(0)
      if cur.val in targetList:
        return True
      targetList.add(k - cur.val)
      if cur.left:
        visited.append(cur.left)
      if cur.right:
        visited.append(cur.right)
    
    return False
#嘿嘿自己写的，97%！
#复杂度时间空间都为n


#530 二叉搜索树的最小绝对差
#这题：中序遍历有序数组 + 找相邻的差值
class Solution:
  def getMinimumDifference(self, root: TreeNode) -> int:
    def dfs(root):
      if not root:
        return
      dfs(root.left)
      queue.append(root.val)
      dfs(root.right)


    queue = []
    target = float("inf")
    dfs(root)
    for i in range(0,len(queue) - 1):
      if queue[i + 1] - queue[i] < target:
        target = queue[i + 1] - queue[i]
    return target
#这一题的range卡住我了。不应该为-2
#ps.求target的方法，可以直接放在中序遍历中, 这样就不用维护一个list存放遍历的值了
class Solution:
  def getMinimumDifference(self, root: TreeNode) -> int:
    def dfs(root):
      if not root:
        return
      dfs(root.left)
      if self.pre == -1:
        self.pre = root.val
      else:
        self.target = min(self.target, root.val - self.pre)
        self.pre = root.val
      dfs(root.right)

    self.pre = -1
    self.target = float("inf")
    dfs(root)
    return self.target
#这种速度快非常多！！时间复杂度 n 空间logn

#501二叉搜索树中的众数
"""Morris 中序遍历
主要工具是cur和一个指针，先遍历左边的，按照cur找上一级pre，然后通过建立的连接，回溯到最开始的root位置用于开始遍历右子树！
比较难理解，现在看代码
"""
class Solution:
  def findMode(self, root: TreeNode) -> List[int]:
    cur = root                                        #初始化cur位置
    self.ans = []
    self.count = 0
    self.maxCount = 0
    self.base = float("-inf")
    while cur:                                        #进入循环，开始遍历
      if cur.left == None:                            #如果左子树没有值，直接进入下一个node，并更新数据。
        self.update(cur.val)
        cur = cur.right
        continue
      pre = cur.left                                  #cur的左子树不为空时，让pre下去。
      #去找到cur的pre的末尾位置，注意！这里是结点对比，而非是值的对比✨！
      while (pre.right and pre.right != cur):         #右子树不为空，并且右子树不为cur时，指针停下（这是结合了BST与中序遍历的特性来的）
        pre = pre.right
      #not pre.right同时意味着cur的前续没有相同的
      if not pre.right:                               #如果遍历到的pre的右子树没有了，意味着此时pre就是cur的前面那个，可以连接起来了！
        pre.right = cur 
        cur = cur.left                                #cur往左下放一环，继续重复整个while操作
      else:
        pre.right = None                              #如果相同找到与cur相同的值，更新一次update重数，cur向右移
        self.update(cur.val)
        cur = cur.right

    return self.ans

  def update(self, x: int):
    if x == self.base:
      self.count += 1
    else:
      self.count = 1
      self.base = x
    if self.count == self.maxCount:
      self.ans.append(self.base)
    if self.count > self.maxCount:
      self.maxCount = self.count
      # self.ans.pop(0)   这里不能用这个，因为试想我们append时候针对每一个数，只append一次，但是如果count一直增加，那么我们会一直pop出去，pop到后面是empty list报错！
      self.ans = []
      self.ans.append(self.base)
      



#208 实现前缀树
# 字典实现
class Trie(object):
  def __init__(self):
    """
    Initialize your data structure here.
    """
    self.root = TrieNode()

  def insert(self, word):
    """
    Inserts a word into the trie.
    :type word: str
    :rtype: None
    """
    node = self.root
    for c in word:
      if c not in node.children:
        node.children[c] = TrieNode()
      node = node.children[c]
    node.is_word = True

  def search_prefix(self, word):
    node = self.root
    for c in word:
      if c not in node.children:
        return None
      node = node.children[c]
    return node

  def search(self, word):
    """
    Returns if the word is in the trie.
    :type word: str
    :rtype: bool
    """
    node = self.search_prefix(word)
    return node is not None and node.is_word

  def startsWith(self, prefix):
    """
    Returns if there is any word in the trie that starts with the given prefix.
    :type prefix: str
    :rtype: bool
    """
    return self.search_prefix(prefix) is not None


class TrieNode(object):
  def __init__(self):
    self.children = {}
    self.is_word = False



#✨java中的charAr() VS python中的 ord


"""
class Trie {
    class TrieNode {
        boolean end;
        TrieNode[] tns = new TrieNode[26];
    }

    TrieNode root;
    public Trie() {
        root = new TrieNode();
    }

    public void insert(String s) {
        TrieNode p = root;
        for(int i = 0; i < s.length(); i++) {
            int u = s.charAt(i) - 'a';
            if (p.tns[u] == null) p.tns[u] = new TrieNode();
            p = p.tns[u]; 
        }
        p.end = true;
    }

    public boolean search(String s) {
        TrieNode p = root;
        for(int i = 0; i < s.length(); i++) {
            int u = s.charAt(i) - 'a';
            if (p.tns[u] == null) return false;
            p = p.tns[u]; 
        }
        return p.end;
    }

    public boolean startsWith(String s) {
        TrieNode p = root;
        for(int i = 0; i < s.length(); i++) {
            int u = s.charAt(i) - 'a';
            if (p.tns[u] == null) return false;
            p = p.tns[u]; 
        }
        return true;
    }
}

"""


#677  优美的排列 II
class Solution:
  def constructArray(self, n: int, k: int) -> List[int]:
    res= [0 for _ in range(n)]

    for i in range(n - k - 1):
      res[i] = i + 1
    
    j = 0
    left = n - k
    right = n
    for i in range(n - k - 1, n):
      if j % 2 == 0:
        res[i] = left
        left += 1
      else:
        res[i] = right
        right -= 1
      j += 1
    return res
#这一题的精髓在于思路：一个序列可以分为两部分，其中一部分等差一致，另一部分等差不一致，然后我们就可以随意的控制等差的数量了       

