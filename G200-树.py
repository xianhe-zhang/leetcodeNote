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
