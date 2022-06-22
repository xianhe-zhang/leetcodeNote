"""
matrix考虑两个思路:
    - i与j的关系
    - spiral 4个for如何一层层循环layer

"""


# 867. Transpose Matrix
class Solution:
    def transpose(self, matrix: List[List[int]]) -> List[List[int]]:
        new_matrix = [[0] * len(matrix) for _ in range(len(matrix[0]))]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                new_matrix[j][i] = matrix[i][j]
        return new_matrix
                
        

# 832. Flipping an Image
class Solution:
    def flipAndInvertImage(self, image: List[List[int]]) -> List[List[int]]:
        for i in range(len(image)):
            for j in range(len(image[0])):
                image[i][j] = 1 if image[i][j] == 0 else 0
            image[i] = image[i][::-1]
        return image



# 54. Spiral Matrix
# 记住这种思路唉！
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        res = []        
        if len(matrix) == 0:
            return res
        row_begin = 0
        col_begin = 0
        row_end = len(matrix)-1 
        col_end = len(matrix[0])-1
        
        while (row_begin <= row_end and col_begin <= col_end):
            for i in range(col_begin,col_end+1):
                res.append(matrix[row_begin][i])
            row_begin += 1
            for i in range(row_begin,row_end+1):
                res.append(matrix[i][col_end])
            col_end -= 1
            # 这两个if都是为了避免处理到最后例外的情况。
            # begin和end都是可以取值的，试想只剩下最后一行的时候，第一个for已经遍历过了，然后更新begin，
            # 这个时候begin>end，当前行不能取值了，否则会在当前行重复取值。
            if (row_begin <= row_end):
                for i in range(col_end,col_begin-1,-1):
                    res.append(matrix[row_end][i])
                row_end -= 1
            if (col_begin <= col_end):
                for i in range(row_end,row_begin-1,-1):
                    res.append(matrix[i][col_begin])
                col_begin += 1
        return res
    
                

# 59. Spiral Matrix II
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        matrix = [[0] * n for _ in range(n)]
        cnt = 1
        
        for layer in range((n+1)//2):
            for ptr in range(layer,n-layer):
                matrix[layer][ptr] = cnt
                cnt +=1
            
            for ptr in range(layer+1, n-layer):
                matrix[ptr][n-layer-1] = cnt
                cnt +=1
            
            for ptr in range(layer+1, n-layer):
                matrix[n-layer-1][n-ptr-1] = cnt
                cnt +=1
            
            for ptr in range(layer+1, n-layer-1):
                matrix[n-ptr-1][layer] = cnt
                cnt += 1
        return matrix
                


# 73. Set Matrix Zeroes
# 第一种非常直接的算法，时间复杂度为MxN，空间复杂度M+N
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        record = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    record.append((i,j))
        # print(record)
        for i, j in record:
            self.balala(matrix, i, j)                    
        return matrix
    
    def balala(self, matrix, i, j):
        # print(i,j)
        m, n = len(matrix), len(matrix[0])
        for x in range(m):
            matrix[x][j] = 0
        for y in range(n):
            matrix[i][y] = 0

# 尝试做一些优化吧。    
class Solution(object):
    def setZeroes(self, matrix):
        """
        思路分三步走:
            - 双for遍历 干两件事(1. 如果cell为0则更新其对应的首行首列; 2. 看首列是否为0, if so, stroe with another varible)
            - 双for 根据首行首列更近剩下的matrix
            - 更新首行首列
        """
        is_col = False
        R = len(matrix)
        C = len(matrix[0])
        # 因为我们需要首行首列去记录我们是否应该更新，所以要利用一些技巧去避免额外空间，因此更新要分开
        for i in range(R):
            # 针对每一个row，我们看看首列是否为0，如果为0，最后更新首列为0，现在不动是因为我们要先更新其他的matrix
            if matrix[i][0] == 0:
                is_col = True
            # 这里注意一下：1-C
            for j in range(1, C):
                if matrix[i][j]  == 0:
                    matrix[0][j] = 0
                    matrix[i][0] = 0

        # 根据首行首列进行更新。
        for i in range(1, R):
            for j in range(1, C):
                if not matrix[i][0] or not matrix[0][j]:
                    matrix[i][j] = 0

        
        # 首列容易理解，那么首行怎么理解？
        # 首先要理解遍历的对象，除了首列外都遍历！因此当我们第一次遍历首行的时候，是不动首行的，但是我们会动matrix[0][0]
        if matrix[0][0] == 0:
            for j in range(C):
                matrix[0][j] = 0

        # See if the first column needs to be set to zero as well        
        if is_col:
            for i in range(R):
                matrix[i][0] = 0


# 48. Rotate Image
# 这道题的思路都想出来了，反转/数学对应旋转，但是都没写出来...
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix[0])
        # 自己对这道题的理解一直都错了...图的遍历顺序。
        # 这个图的遍历顺序不是每一圈一圈一层层遍历的。而是基于四个象限中的一个象限，i，j遍历根据数学规则同时update四个象限中的数据，至于坐标关系，利用象限就可以得到了。
        for i in range(n // 2 + n % 2): # 看看有没有单列的情况，四个格子刚好四个象限，如果五行五列，这个n%%2就是处理这个事情的。
            for j in range(n // 2):
                # 关于坐标有个规律，旋转后的坐标轴与旋转之前的坐标轴关系是比较固定的。
                # 但是本题中我们的所有坐标都是根据一个象限来的，所以更容易理解。
                tmp = matrix[n - 1 - j][i]
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - j - 1]
                matrix[n - 1 - i][n - j - 1] = matrix[j][n - 1 -i]
                matrix[j][n - 1 - i] = matrix[i][j]
                matrix[i][j] = tmp
# 下面这个是反转的方法。
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        self.transpose(matrix)
        self.reflect(matrix)
    # transpose 依据diagonal进行反转
    def transpose(self, matrix):
        n = len(matrix)
        for i in range(n):
            for j in range(i + 1, n):
                matrix[j][i], matrix[i][j] = matrix[i][j], matrix[j][i]
    # reflect 依据y轴进行反转
    def reflect(self, matrix):
        n = len(matrix)
        for i in range(n):
            for j in range(n // 2):
                matrix[i][j], matrix[i][-j - 1] = matrix[i][-j - 1], matrix[i][j]
