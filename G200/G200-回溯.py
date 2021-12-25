"""
行前笔记：
    1. 回溯属于DFS，但不同于普通DFS（处理可达性）。Back-Tracking主要用于求解 排列组合。
    2. 注意对元素进行标记

Q：如何理解内嵌方法的递归逻辑？
A：内嵌方法递归，就是利用现有的数据进行循环套娃。
Q：既然是循环套娃，为什么不用while/for循环？
A：因此此时我们不知道边际具体数据，只知道条件；同时嵌套结构的原因可以共用方法内的数据，同时又避免了对某些数据的初始化。

Q：类内方法VS方法内方法
A：我觉得看用途吧，两种其实都是可以写的。

Q：递归 VS 回溯 VS DFS
    A1：递归是一种算法结构，回溯是一种算法思想；
    A2：递归是通过调用函数本身来解决问题；回溯是通过不同的尝试得到问题的解集，类似于穷举，但是和穷举不同的是回溯不会“剪枝”
    A3：回溯是DFS一种；DFS利用了隐氏栈的结构。
"""

leetcode-17
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return list()
        
        phoneMap = {        #画出Map
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }

        def backtrack(index: int):
            if index == len(digits):    #终止条件（终止可以是跳出/也可以是不再进行任何循环操作。
                combinations.append("".join(combination))   #combinations 没有进行初始化  # "".join(combination) 就是将这条分支的所有字母串联起来
            else:
                digit = digits[index]   #每个按键为维度
                for letter in phoneMap[digit]:  #每个按键针对的不同字母展开递归，最终形成一个树形 #此时phoneMap[digit]是一个字符串
                    combination.append(letter)  #combination写在for里面很聪明。表示：每条分支都会有。
                    backtrack(index + 1)
                    combination.pop()           #每一次弹栈保证少一位combination #执行到这里的时候单纯是为了删除，所以无所谓pop(0)/pop(-1)

        combination = list()
        combinations = list()
        backtrack(0)
        return combinations
#第一个按键 -> 对应的字母 -> 第二个按键 -> 对应的字母 -> ... ->最后一位按键 -> 对应的字母并返回 
#思考题：既然combination是用来记录字母的，且只有一个combination，那么我们如何保证顺序是对的呢？或者如何保证不同答案的字母在这里不会弄串呢？
#思考题答案：这个要看执行与递归的flow。这里的结构就相当于首先完成最左子树的递归，也就是其中一条分支的递归。而不是属于BFS，每一次针对所有情况都进行一层递归。
#进展：如果BFS跟DFS有这样的差别，那么可以考虑记忆模版了。

leetcode-93
    #什么时候决定继续前移？
    #什么时候决定几位数？
    #结构是怎么样的？需要helper么？
    #需要index。需要双指针么？
    "提出上述问题，其实是因为没有掌握核心算法：这里的核心算法是指将问题分解，分解成回溯算法可以解决的问题，而非暴力地去想，否则容易出岔"

class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        seg_count = 4
        ans = []
        segments = [0] * seg_count

        def dfs(segId: int, seg_start: int):
            #第一个IF：已经遍历完了，这就是一种答案
            if segId == seg_count:      #遍历的ID已经=4，意味着0～3已经遍历完
                if seg_start == len(s): #遍历的start_index刚好越过末尾。
                    ipAddr = ".".join(str(seg) for seg in segments) # ipAddr是一个答案，将其添加
                    ans.append(ipAddr)
                return      #这个return和第二个if的return都是用来中断递归的，否则下面index会out of range；而上面的双if就是表明只有两个条件都满足，才会记录答案，否则这就是条废枝

            if seg_start == len(s):
                return

            if s[seg_start] == "0": #如果是0继续递归
                segments[segId] = 0
                dfs(segId + 1, seg_start+ 1)
                
            addr = 0
            for seg_end in range(seg_start, len(s)):
                addr = addr * 10 + (ord(s[seg_end]) - ord("0")) #ord是汇编表吧，我记得。这里就是看seg_end对应的数字与0的差距是多少刚好是数字意义上的差
                if 0 < addr <= 255: #这里不能等于0，因为如果一段IP地址为0，那么它只能有1个0，因为0一段地址的开导
                    segments[segId] = addr  #往后遍历，如果还有满足的，那么再更新segment[segID]
                    dfs(segId + 1, seg_end + 1)    #通过方法的调用可以将本轮的end + 1自动转换为下一轮的start
                else: 
                    break
        dfs(0, 0)
        return ans
#最后一块代码揭示了递归的顺序。非常奇妙
"从s的第一个数字开始遍历——满足的话，跳到下一段从第一个数字开始...直到最后一段第一个数字，满足的话返回，如果不满足的话最后一段2个数字，直到边界所有情况，然后依次返回上层"

"""
@举个例子
5个1:
[1,0,0] X
[1,1,0] X
[1,1,1] X
[1,1,11] X
[1,1,111] ✅
[1,11,1]X
[1,11,11]✅
...
运算顺序：沿着边界（左子树）往下走，走到头判定，然后顺序横扫，扫完搜集满足的数据；然后往上走一层，继续横扫，收集；重复下去就好了。
"""

leetcode-79
#自己写的，只能返回False，不明白
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        directions = [(0,1),(0,-1),(1,0),(-1,0)]

        def dfs(board_x, board_y ,word_index):
            if board[board_x][board_y] != word[word_index]:
                return False
            
            if word_index == len(word) - 1: #可以返回True了，貌似最终到不了这里
                return True

            visited.add((board_x,board_y))
            result = False  #初始化当前层

            for i, j in directions:
                board_x, board_y = board_x + i , board_y + j
                if 0 <= board_x < len(board) and 0 <= board_y < len(board[0]):   #超越边界的不处理了
                    if (board_x,board_y) not in visited:    #拜访过的不要了
                        
                        if dfs(board_x,board_y,word_index + 1):
                            result = True
                            break   #只要有一个可能的情况成立，那么直接break所有递归，传送值就可以了。

            visited.remove((board_x,board_y)) #剪枝
            return result #把当前层的递归结果往上层传递
            
        m, n = len(board), len(board[0])
        visited = set()
#1. 每一个都遍历，如果能对应上第一位，那么就展开recursion
        for x in range(m):
            for y in range(n):
                if dfs(x, y, 0):
                    return True
        return False
#少考虑了一点，visited出栈
#小tip如果遇到判断True/False，大概率是要将递归放在if判断里的；因为递归设计到返回True/False, 害怕如果直接返回会导致错失一些可能性
#为什么要剪枝？因为每个root不一样的话，这题是可以重复遍历一个地址的。比如：A->C失败，B->C成功。每次只用剪枝到当前层级就行。✨
#看下面题接吧，上面答案不对，但思路没问题

@题解
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        def check(i: int, j: int, k: int) -> bool:
            if board[i][j] != word[k]:
                return False
            if k == len(word) - 1:
                return True
            
            visited.add((i, j))             
            result = False
            for di, dj in directions:
                newi, newj = i + di, j + dj
                if 0 <= newi < len(board) and 0 <= newj < len(board[0]):
                    if (newi, newj) not in visited:
                        if check(newi, newj, k + 1):   
                            result = True
                            break
            
            visited.remove((i, j))
            return result

        h, w = len(board), len(board[0])
        visited = set()
        for i in range(h):
            for j in range(w):
                if check(i, j, 0):
                    return True
        
        return False

2-解法
"""
核心思路：
    1. 遍历所有元素
    2. 当遇到首字母相同的元素时，进入递归；递归为的是找到对应的字母
    3. 进入递归上下左右试探后，找到匹配的元素一起进入下一层直到最终边界逐层返回。
    4. 返回第n层的时候，记住删除n+1层使用过的元素（剪枝）
"""
class Solution(object):
    
    # 定义上下左右四个行走方向
    directs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    def exist(self, board, word):
        m ,n = len(board), len(board[0])
        mark = [[0 for _ in range(n)] for _ in range(m)]    
                
        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0]:
                    # 将该元素标记为已使用
                    mark[i][j] = 1
                    if self.backtrack(i, j, mark, board, word[1:]) == True:
                        return True
                    else:
                        # 回溯
                        mark[i][j] = 0
        return False
        
        
    def backtrack(self, i, j, mark, board, word):
        if len(word) == 0:
            return True
        
        for d0, d1 in self.directs:
            cur_i, cur_j = i + d0, j +d1
            
            if cur_i >= 0 and cur_i < len(board) and cur_j >= 0 and cur_j < len(board[0]) and board[cur_i][cur_j] == word[0]:
                # 如果是已经使用过的元素，忽略
                if mark[cur_i][cur_j] == 1:
                    continue
                # 将该元素标记为已使用
                mark[cur_i][cur_j] = 1
                if self.backtrack(cur_i, cur_j, mark, board, word[1:]) == True:
                    return True
                else:
                    # 回溯
                    mark[cur_i][cur_j] = 0
        return False


leetcode-257
class Solution:
    def binaryTreePaths(self, root):
        def construct_paths(root, path):
            if root:                        
                path += str(root.val)       #这个操作非常灵性：已经遍历的直接放入到临时解集中，后面根据遍历的顺序，保证了放入的元素有顺序
                if not root.left and not root.right:  # 当前节点是叶子节点
                    paths.append(path)  # 把路径加入到答案中
                else:
                    path += '->'  # 当前节点不是叶子节点，继续递归遍历
                    construct_paths(root.left, path)
                    construct_paths(root.right, path)

        paths = []
        construct_paths(root, '')
        return paths
#首先边界条件为：完全叶子节点，加入到解集中。
#谨记递归的顺序

leetcode-46
1- 算法一
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        def backtrack(nums, tmp):
            if not nums:
                res.append(tmp)
                return 
            for i in range(len(nums)):      
                tmp += nums[i]
                nums[:] = nums[:i] + nums[i+1:]
                backtrack(nums, tmp)
        backtrack(nums, [])
        return res
#算法一：nums中的每个元素用for遍历，然后挑出来，再针对剩下的进行for遍历
"""
Back-Track 模版
result = []
def backtrack(路径, 选择列表):
    if 满足结束条件:
        result.add(路径)
        return
    
    for 选择 in 选择列表:
        做选择
        backtrack(路径, 选择列表)
        撤销选择
"""
2-算法2
class Solution:
    def permute(self, nums):
        def backtrack(first = 0):
            # 所有数都填完了
            if first == n:  
                res.append(nums[:])
            for i in range(first, n):   #(first,n)是nums中还没有排列的下标
                # 动态维护数组
                nums[first], nums[i] = nums[i], nums[first] #first是我们想要插入的下表 #那么这个交换的意思就是，将我们想用的i，插入到我们想插入的地方first
                # 继续递归填下一个数
                backtrack(first + 1)
                # 撤销操作
                nums[first], nums[i] = nums[i], nums[first]
        
        n = len(nums)
        res = []
        backtrack()
        return res
#本文采用动态维护数组替代了标记数组去判断是否已经利用某一元素
#动态维护的算法比较难理解：通过每次在first地方分割已经排序的元素和尚未排序的原序，递归到最下方；
#然后往上跳，每一次跳的时候同时撤销交换操作。（弹栈）

leetcode-47
"""
剪枝条件的理解是最困难：（通过观察递归树） 
"""
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums, size, depth, path, used, res):
            if depth == size:   #遍历到最后一层
                res.append(path.copy()) #不用copy的话实质为引用，path最终为0，所以打不出来值，所以这里要用copy下来。
                return      #回溯返回上一个节点
            for i in range(size):   
                if not used[i]:                 #表明没有用过

                    if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]: #剪枝：i>0：边界；后者跟前者相同并且前者目前not used，就是指前者已经遍历完了，到了同一步骤，下面就没必要进行了，剪枝就行。
                        continue

                    used[i] = True
                    path.append(nums[i])
                    dfs(nums, size, depth + 1, path, used, res) #path没啥用这道题，但是是为了模版写出来。
                    used[i] = False     #回溯复原
                    path.pop()

        size = len(nums)
        if size == 0:
            return []

        nums.sort()
        used = [False] * len(nums)
        res = []
        dfs(nums, size, 0, [], used, res)   #path与depth目前还没声明，因此可以先不用传送进去变量名
        return res
#bug小能手，变量名写错/ 判断等号写成价值等好。

leetcode-77
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        def backTrack(n, k, start, path, result):
            if len(path) == k:
                result.append(path.copy()) #path[:]
                return
            for i in range(start, n + 1):
                path.append(i)
                backTrack(n, k, i + 1, path, result)
                path.pop()
 
        path, result = [], []
        backTrack(n, k, 1, path, result)
        return result
#bug能手: 递归将i写成start
"""
下面加入剪枝条件：
if k == 0:
for i in range(start, n-k+2):           #这里就固定下来了，最后几位直接剪去。
    backTrack(n, k-1, i+1, track, result)
Q：为什么是 n-k+2 / k-1 呢？
A：因为搜索起点有边界要求，比如搜索k=4个数字，那么起点为倒数第二个数就没有意义了。我们可以归纳出：搜索的边界至少要大于（结尾 - 仍待搜索的个数）
Details: 

"""

leetcode-39
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:

        def dfs(candidates, begin, size, path, res, target):
            if target == 0:
                res.append(path)
                return

            for index in range(begin, size):
                residue = target - candidates[index]
                if residue < 0:         #剪枝条件
                    break   

                dfs(candidates, index, size, path + [candidates[index]], res, residue) # nums[i] 是 int， [nums[i]]是list
                #这里不需要path.pop()，因为结果只有两个，不满足就砍掉了，不需要再回溯。
        size = len(candidates)
        if size == 0:
            return []
        candidates.sort()
        path = []
        res = []
        dfs(candidates, 0, size, path, res, target)
        return res
#我的思路就是遍历，每一层都是从小到大遍历，如果组合下去大的话，那么这个情况就不合适了。


leetcode-40
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        def dfs(candidates, begin, size, path, res, target):
            if target == 0:
                res.append(path[:])
                return 

            for i in range(begin, size):
                residue = target - candidates[i]
                if residue < 0: 
                    break
                if i > begin and candidates[i-1] == candidates[i]: #忘记剪枝了，以及题目要求。
                    continue
                path.append(candidates[i])
                dfs(candidates, i+1, size, path, res, residue)#这里总写错，如果写成begin + 1 那么每次递归都是以当前层为基数，而不是以当前层的当前遍历的数字为基数
                path.pop()


        if not candidates:
            return []
        size = len(candidates)
        path = []
        res = []         
        candidates.sort()
        dfs(candidates, 0, size, path, res, target)
        return res
#bug小能手：path=res=[]
#万能bug皇：将递归i+1 写成 begin+1！ 下次不要再错


leetcode-216
#我的代码
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        def dfs(begin, path, res, residue,k):
            if residue == 0 and len(path) == k:
                res.append(path[:])
                return
            

            for i in range(begin,(9+1)-len(path)+1-k): #剪枝:9+1是原本元素的index，也就是末尾的index；k表示距离，len(path)是已经找到，+1：算上一位数
                if i > residue:     #剪枝 
                    return 
                residue -= i
                path.append(i)
                dfs(i+1, path, res, residue,k)
                path.pop()
                residue += i            #bug王者

        
        path = []
        res = []
        dfs(1, path, res, n, k)
        return res
"""
BUG:
1. 回溯没有写完整！少些了residue回溯
2. 没有进行剪枝处理，这一题限制了个数和总数，因此可以操作的剪枝地方有两个
3. 题目条件的位置 
"""
leetcode-78#自己写的
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums,size,begin,path,ans):
            if begin == size+1:
                return 

            ans.append(path[:])
            
            for index in range(begin, size):
                path.append(nums[index])
                dfs(nums,size,index+1,path,ans)
                path.pop()



        size = len(nums)
        ans = []
        path = []
        nums.sort()
        dfs(nums,size,0,path,ans)
        return ans

leetcode-90
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums,size,begin,path,ans):
            if begin == size+1:
                return 

            ans.append(path[:])
            
            for index in range(begin, size):
                if index > begin  and nums[index-1] == nums[index]: #唯一的关键点，遇到重复元素，这里的index > begin至关重要
                    continue
                path.append(nums[index])
                dfs(nums,size,index+1,path,ans)
                path.pop()



        size = len(nums)
        ans = []
        path = []
        nums.sort()
        dfs(nums,size,0,path,ans)
        return ans

leetcode-131
#自己的思路是对的，需要一个额外的method去帮助自己判断是否是回文（DP、中心展开）
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        size = len(s)
        if size == 0:
            return res
        path = []
        self.dfs(s, size, 0, path, res)
        return res
    

    def dfs(self, s, size, start, path, res):
        if start == size:
            res.append(path[:])
            return
        for i in range(start, size):            #这里的start也是每一层/depth，因为是不重复的计算，所以没下一层就要进一层start
"""
这里的for处理很精妙，如果遍历的path不是回文串，只能往后走依次遍历，不能往前回溯减小。
1. 不能往前减小是因为，我们的path是从小往大的来的，每一层都是；前面小的组合已经判定过了。
2. 往后走如果这种情况不符合回文判断(constraints)，那么这种组合永远进不了下一关的dfs，也就没有办法进入到res
"""
            if not self.check_is_palindrome(s, start, i):   #针对这一层已经遍历的元素，即path级，如果不是回文的话，进行剪枝
                continue
            path.append(s[start:i + 1])     #start～i保证了每一层基于start的所有顺序组合
            self.dfs(s, size, i + 1, path, res)     #已经遍历的是的话，就往后走。i+1是往后走的机制，非常重要。
            path.pop()
        
    
    def check_is_palindrome(self, s, left, right):  #中心收敛
        while left < right:
            if s[left] != s[right]:
                return False
            left = left + 1
            right = right - 1
        return 
"""
深入理解：
    1. 524的continue剪枝作用，只是在这一层跳过i这个选项，不继续做深入处理，继续往下走
    2. 每一层for结合path[start:i+1],可以了解到。每一层针对start一个元素，去看它的所有组合。（🌟值得注意的是，层级结构的理解更偏向于BFS，而非DFS的遍历顺序）
    🌟这题非常经典，对理解层序与遍历顺序很有帮助。
    3. 先单个字母数着遍历，然后看每一层针对元素的所有可能连续结果，如果没有回文，进行剪枝。
"""

leetcode-37 解数独(困难)
#直接看答案，不知道怎么遍历，不知道怎么判断
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        def dfs(pos: int):
            nonlocal valid      #nonlocal：将方法外的变量带到方法内
            if pos == len(spaces):          #pos其实代表着已经处理了多少满足题意的值，如果相同，那就是处理完了
                valid = True                #返回True，下面循环中的valid也会True，然后迅速返回。
                return
            
            i, j = spaces[pos]
            for digit in range(9):
                if line[i][digit] == column[j][digit] == block[i // 3][j // 3][digit] == False:
                    line[i][digit] = column[j][digit] = block[i // 3][j // 3][digit] = True
                    board[i][j] = str(digit + 1)
                    dfs(pos + 1)
                    line[i][digit] = column[j][digit] = block[i // 3][j // 3][digit] = False
                if valid:       #为什么这里也需要返回呢？因为数组拥有唯一的接，这里return跳过就是指，这种组合都不行，要重新来。
                    return
            
        line = [[False] * 9 for _ in range(9)]      #用来判定line
        column = [[False] * 9 for _ in range(9)]    #同理：用来判定column
        block = [[[False] * 9 for _a in range(3)] for _b in range(3)]   #将9*9的map分成九宫格
        "这三个分别就是对应数独的三个判别限制：如果(i,j)出现元素x，那么其所在的行、列、九宫格就不能出现x了；其中行，列都用二维数组，分别表示第几行/列的某个元素是否出现过"
        valid = False
        spaces = list()

        for i in range(9):
            for j in range(9):
                if board[i][j] == ".":
                    spaces.append((i, j))   #将所有需要处理的地址放入到spaces中
                else:
                    digit = int(board[i][j]) - 1    #求出来是索引，比如x为4，那column[i][3]就应该就不应该出现4，因此都改为True。
                    line[i][digit] = column[j][digit] = block[i // 3][j // 3][digit] = True

        dfs(0)
#keytake away：1.数独处理方式 2.两个valid跳出所有可能性。
---------------------------------------------------------------------------------------------------------------------------------------------------------------------
@大神解法
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        
        def get_possible_digits(r, c):          #来压缩存储每一行、每一列、每一个 3x3 宫格中 1-9 是否出现，这样就可以知道每一个格子有哪些数字可以填写
            b = (r // 3) * 3 + c // 3
            if not any(list(zip(rows[r], cols[c], boxes[b]))[i]):
                return [i for i in range(1, DIGITS + 1)]

        def toggle_an_cell(r, c, d, used):      #指定方格的状态
            b = (r // 3) * 3 + c // 3
            rows[r][d] = cols[c][d] = boxes[b][d] = used
            board[r][c] = str(d) if used else "."

        # 选择能填的数字最少的格子，从这样的格子开始填，填错的概率最小，回溯次数也会变少。
        def get_next_cell():               
            r, c, min_count = 0, 0, DIGITS + 1
            for i in range(m):
                for j in range(n):
                    if board[i][j] == ".":
                        possible_digits = get_possible_digits(i, j)
                        if len(possible_digits) < min_count:
                            min_count = len(possible_digits)
                            r, c = i, j
            return r, c

        def backtrack(remaining):
            if remaining == 0:
                return True
            nr, nc = get_next_cell()
            possible_digits = get_possible_digits(nr, nc)
            for pd in possible_digits:
                toggle_an_cell(nr, nc, pd, True)
                if backtrack(remaining - 1):        #一口气跑到楼顶 
                    return True
                toggle_an_cell(nr, nc, pd, False) #回溯复原
            return False

        DIGITS = 9
        m, n = len(board), len(board[0])
        remaining = 0
        # True = used, False = not used
        # rows[1][2] = True: 第2行已经有“2”这个数字了
        rows = [[True] + [False] * DIGITS for _ in range(m)]
        cols = [[True] + [False] * DIGITS for _ in range(n)]
        boxes = [[True] + [False] * DIGITS for _ in range(m)]

        for r in range(m):
            for c in range(n):
                d = board[r][c]
                if d == ".":
                    remaining += 1
                else:
                    b = (r // 3) * 3 + c // 3
                    rows[r][int(d)] = cols[c][int(d)] = boxes[b][int(d)] = True

        backtrack(remaining)
#算法创新：1. 通过getNext从最小的九宫格开始判断 2. 针对某个单元格直接判断还能填写哪个值

leetcode-51 N-Queen N皇后
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        def isVaild(board,row, col):    
            #不用判断同一行是否冲突，因为根据dfs的遍历顺序，基准就是每一行，所以每一行肯定不会重复。
            #只判断左上角和右上角也是同理，因为考虑到遍历顺序
            #判断同一列是否冲突
            for i in range(len(board)):
                if board[i][col] == 'Q':
                    return False
                    
            # 判断左上角是否冲突
            i = row -1
            j = col -1
            while i>=0 and j>=0:
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j -= 1
            # 判断右上角是否冲突
            i = row - 1
            j = col + 1
            while i>=0 and j < len(board):
                if board[i][j] == 'Q':
                    return False
                i -= 1
                j += 1
            return True

        def backtracking(board, row, n):
            # 如果走到最后一行，说明已经找到一个解
            if row == n:
                temp_res = []   #temp_res初始化
                for temp in board:              #board是每行 点的集合
                    temp_str = "".join(temp)    #
                    temp_res.append(temp_str)
                res.append(temp_res)
            
            for col in range(n):                #按每一个row是一个depth
                if not isVaild(board, row, col):
                    continue
                board[row][col] = 'Q'
                backtracking(board, row+1, n)
                board[row][col] = '.'


        if not n: 
            return []
        board = [['.'] * n for _ in range(n)] #返回的值是[[],[],[],[]]; ['.']*n 是将'.' 乘 n放在一个list中，是n个字符串，而不是一个字符串哈～
        #注意这里不能写成'.'而要写成['.']，前者为一个字符串，后者为list
        res = []
        backtracking(board, 0, n)
        return res
#DFS更深层次理解，面对无数条岔路，每条岔路有不同的情况，我们不必担心我们当下的选择会对其他岔路产生什么影响，因为我们先将岔路的路标弄出来，然后一次探索每一条到尽头，如果有答案，我们就保留，如果没有，我们就回溯。
#DFS遍历再加深：为什么探索到尽头我们就能保证这条路是对的？因为每一次分岔路我们的判断条件都证明已经走过的路是对的，那么最后一个节点是对的话，我们这条路就是一个可能的解。
#然后往回回溯，看上一个节点，之后没有遍历的节点是否满足题解。已经遍历过的肯定就不满足了，根据上上层的约束；如果我们要想看已经遍历过的这一层节点是否满足，就要看最上层的节点其他选择是否有可能（遍历所有）
#Bug小能手：又是缩进错误！
#复杂度为N!











    
    