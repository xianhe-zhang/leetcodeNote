leetcode-241
@设计运算优先级
class Solution:
    def diffWaysToCompute(self, input: str) -> List[int]:
        # 如果只有数字，直接返回
        if input.isdigit():
            return [int(input)]

        res = []
        for i, char in enumerate(input): #enumerate：index, num     collections.Counter：nums，times       find：index
            if char in ['+', '-', '*']:
                # 1.分解：遇到运算符，计算左右两侧的结果集
                # 2.解决：diffWaysToCompute 递归函数求出子问题的解
                left = self.diffWaysToCompute(input[:i])
                right = self.diffWaysToCompute(input[i+1:])
                # 3.合并：根据运算符合并子问题的解
                for l in left:
                    for r in right:
                        if char == '+':
                            res.append(l + r)
                        elif char == '-':
                            res.append(l - r)
                        else:
                            res.append(l * r)
        return res
"""
本题算法思想最重要：
    1. 题中可以随便添加括号，因此乘法在这里并不具备实际意义上的优先级；事实上任何运算符号都可以第一个运算，就是因为控制运算flow的括号优先级最高。
    2. 那么问题就可以拆解成：针对单一运算符OP，运算结果的所有可能取决于参加运算的所有元素可能，即 X OP Y的结果组合取决于 X和 Y的结果组合数（OP为运算符）
    3. 那么我们就可以根据2中所讲，依据OP去分解问题

本题的难点：
    1. 上述所讲的算法思想
    2. 到底如何遍历所有可能性呢？
        A：提出这个问题的你忘记了一个前提，就是上面说到的有括号的存在，乘法与加减无差别，因此我们根据index依次遍历就好，就足以取到left和right的所有可能。
"""
---------------------------------------------------------------------------------------------------------------------------------------

#常见回溯算法的套路
"""
递归{
    递归出口；
    for(int i = 1;i<=n;i++){
    	add(i);
    	进入递归；
   		remove(i);
	}
}
"""

#二叉搜索树Binary Search Tree要满足子树大小条件

leetcode-95
1-递归/分治
#如果1～n中的2作为根节点，那么左子树只能是1，右子树是3～n多种可能；基于这种情况我们可以做递归
#也存在左右子树都不止一种情况，那么俩俩组合就行。
class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        if n == 0:          #递归结束条件
            return ans
        return self.helper(1, n)    #这里利用到helper纯粹是为了更改输入值，写在一个方法里也是可以的。

    def helper(self, start, end):
        ans = []

        """
        最开始的两个if时边界判断/递归终止
        """

        if start > end:     #此时没有数字，将 null 加入结果中
            return [None]

        if start == end:    #只有一个数字，当前数字作为一棵树加入结果中
            return [TreeNode(start)]  
        
        ans = []    #初始化

        #从左往右遍历，每一个点都可以当Root根节点
        for i in range(start, end+1):
            temp = []
            left = self.helper(begin, i - 1)    
            right = self.helper(i + 1, end)
            
            #每一个root得到left和right的所有可能子树后，用双循环两两组合起来。
            for m in left:
                for n in right:
                    root = TreeNode(i)    #🌟将数字i声明TreeNode之后，才能对其使用特定的方法
                    root.left , root.right = m, n
                    ans.append(root)
        return ans
#分治递归的重点模版：就是从左往右遍历，以及最后一Part的子树组合。
#分治题目的核心思考是：利用分治的算法思路，能否满足题目求解逻辑，如果可以那么就可以用分治，再去考虑边界情况。

2-DP解法
思想：通过观察，n = 100 的答案可以由 n = 98 的答案计算而来，这个思路的话我们就可以用DP啦。

