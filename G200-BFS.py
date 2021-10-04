leetcode-1091-最短路径
#注意点：通过BFS模版题可以学习路径题的答题技巧，好巧妙的设计，感叹一下～
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        if grid[0][0] == 1 or grid[-1][-1] ==1:         #特殊情况
            return -1 

        #路径的8个方向
        directions = [[1,0],[1,1],[0,1],[1,-1],[-1,0],[-1,-1],[0,-1],[-1,1]] 
        
        #queue的作用：1.记录所有可以走的路径与步数 2.结果判断
        queue = [(0,0,1)] #走过的节点 + 层数
        n = len(grid)

        #BFS
        while len(queue): #只要queue还有值就可以循环，没有值的情况意味着无论如何也走不到终点
            x0, y0, cnt = queue.pop(0)      #因为路径添加是按照层级添加的，最早满足条件的层级一定最短
            if x0 == n - 1 and y0 == n - 1: #arrive at bottom-right
                return cnt 
            
            for i, j in directions:         #这不操作就很有灵性，相当于所有方向可以走的，都形成一个记录并记入queue中，效果：可以同时记录这一层级所有可能的走法
                x, y = x0 + i, y0 + j
                if 0 <= x < n and 0 <= y < n and not grid[x][y]:    #确保走的路径不超过边界，并且不走原来走过的路子
                    queue.append((x, y, cnt + 1))
                    grid [x][y] = 1 #visited 
            
        return -1 #讲究

#bug写作小能手：锁进+1，逻辑or写成and + 1

 
leetcode-279
#BFS三要素：队列queue + 节点(value, step) / (value, visited) + 已访问集合
class Solution:
    def numSquares(self, n: int) -> int:
        p_square = [i * i for i in range(1, int(n**0.5)+1)][::-1] # 从大到小减去，帮助加速   # = 可能的完全平方数的集合
        ps_set = set(p_square)   
        
        
        queue = [n]         #q是存储每一层可以得到n的数字，然后去判断是否是完全平方数。
        cache = {n : 1}      #key是组成元素，value是

        while queue:
            val = queue.pop(0)      #pop(0)确保了最短路径，最小量
 
            if val in ps_set:       #判断终止条件
                return cache[val] 

            for num in p_square:    #开始遍历所有可能数字
                if val - num > 0 and val - num not in cache: 
                    queue.append(val - num)             #append的是所有可能组合的平方数 
                    cache[val - num] = cache[val] + 1   #第一次遍历就是第二层，即2个平方数字

        return -1 #如果没有找到target，就返回-1

"做题前，下面几个思路先想明白"
#假设一个数=两个大的完整平方数之和，那么它很有可能可以由更多小的完整平方数相加；也就是这样的一个数拥有多个解集，而本题是找最小单位的解集。
#首先cache是没有什么值的，如果val - num 存在于cache里，那么意味着val和num
#p_square是倒序，val也是倒序。顺序倒序无所谓，只要层级顺序没错就可以。其他的顺序无非是运算的速度。
#这一题的核心思想是：层级是按照从小到大减去一个元素，得到数字，即val - num. 然后看这个数字是否为完全平方数，如果不是继续拆解，如果是返回层级。



leetcode-127
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:   
        

        if endWord not in wordList:
            return 0

        queue = [(beginWord,0)] 
        visted = []

        while queue:
            cur, step = queue.pop(0)
            step += 1
            if cur == endWord:
                return step
    
            for nex in wordList:
                if nex not in visted:
                    if self.helper(cur, nex):
                        queue.append((nex, step))
                        visted.append(nex)
                    
        return 0
    
    #用来判断是否可以继续往下走
    def helper(self, begin: str, end: str) -> bool:
        x, y= list(begin), list(end) 
        nx, ny = len(x), len(y)
        if nx != ny: 
            return False
        count = 0
        for i in range(0, nx):
            if x[i] != y[i]:
                count += 1
        return count == 1 

#自己写的——超时
#原因：利用了helper? helper去比较的话太多了...能走的步子也太多了，因此在遇到那么多种情况的时候是无法转换的。
#总结来说，就是自己设计的算法有点复杂...

1-单向BFS  #
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        word_set = set(wordList)
        if len(word_set) == 0 or endWord not in word_set:
            return 0

        if beginWord in word_set:
            word_set.remove(beginWord)

        queue = deque()
        queue.append(beginWord)
        visited = set()
        word_len = len(beginWord)     #所有单词长度都是一样，所以存下俩。 
        step = 1

        while queue:        #队列当中不为空
            current_size = len(queue)   #当前队列有多少元素，然后依次拿出来。PS.不能将这一句嵌入下一句，因为每一层我们的元素是会变的。
            for _ in range(current_size):   #每一次大的while循环就是一层，这里i代表当前层级有多少元素
                word = queue.popleft()      #pop(0)

                word_list = list(word)
                for j in range(word_len):
                    origin_char = word_list[j]

                    for k in range(26):
                        word_list[j] = chr(ord('a') + k)
                        next_word = ''.join(word_list)      #next_word就是通过word变换一个字母的所有可能
                        if next_word in word_set:           #如果next_word在字典中/==endword，那么就可以继续往下一步走
                            if next_word == endWord:
                                return step + 1             #转换过后的单词可以为endWord，那么就是下一步我们就能找到end了，所以+1
                            if next_word not in visited:
                                queue.append(next_word)
                                visited.add(next_word)      #visted()就是将已经走过的记录，不再走了；你会有疑问，不同路径为什么要共享同一套visted？ A,B两条路如果都经过C点，那么我们只用看A就好了，因为A是最短路径。
                    word_list[j] = origin_char  
            step += 1
        return 0
"""
Q: 128为什么要存origin_char？139为什么要还原？
A: 下面的for k循环操作是根据word_list的每一位进行操作，因此当针对一位操作一圈完了之后，我们最开始要还原最开始的字符，并且开始下一位的操作。

Tip-1:用哈希表判断单词是否在集合中，时间复杂度是 O(1)，如果是列表，判断是否在集合中，需要遍历，所以会超时。

Q：115为什么要用deque
A：Deque一般用用于双向队列，python中的基础类型还是tuple/hash，比列表的复杂度低，用list也可以解决问题。

Q：123为什么要加这一层for？是看每层要出多少元素。但是不加好像也行？
A：不加的话，step+1的触发条件就是每次pop一次就+1，而不是每一层遍历完+1，会导致结果偏大。
"""
2-双向BFS 
#就是两头分别往对面遍历，两头遍历都各自算一个一层，然后等到两个遍历能够碰到一起的时候（双指针），意味首尾可以打通，这个时候返回就可以。
#比单向BFS来说，遍历的对象变小了，所以优化。
#下面的解法有一处优化，在🌟处
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        word_set = set(wordList)
        if len(word_set) == 0 or endWord not in word_set:
            return 0

        if beginWord in word_set:
            word_set.remove(beginWord)

        visited = set()
        visited.add(beginWord)
        visited.add(endWord)

        begin_visited = set()
        begin_visited.add(beginWord)

        end_visited = set()
        end_visited.add(endWord)

        word_len = len(beginWord)
        step = 1
        # 简化成 while begin_visited 亦可
        while begin_visited and end_visited:
            # 打开帮助调试
            # print(begin_visited)
            # print(end_visited)

            if len(begin_visited) > len(end_visited):               #🌟因为循环操作的实质只针对begin，这里做个交换，先去遍历小的，对时间有优化。
                begin_visited, end_visited = end_visited, begin_visited

            next_level_visited = set()
            for word in begin_visited:
                word_list = list(word)

                for j in range(word_len):
                    origin_char = word_list[j]
                    for k in range(26):
                        word_list[j] = chr(ord('a') + k)
                        next_word = ''.join(word_list)
                        if next_word in word_set:
                            if next_word in end_visited:
                                return step + 1
                            if next_word not in visited:
                                next_level_visited.add(next_word)
                                visited.add(next_word)
                    word_list[j] = origin_char
            begin_visited = next_level_visited
            step += 1
        return 0
