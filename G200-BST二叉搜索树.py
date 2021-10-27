
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
