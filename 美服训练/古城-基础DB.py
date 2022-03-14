########################  Trie  ########################
"""
Trie树，又叫字典树、前缀树（Prefix Tree）、单词查找树 或 键树，是一种多叉树结构。如下图：
Trie is like a N-array Tree.
字典树的性质
根节点（Root）不包含字符，除根节点外的每一个节点都仅包含一个字符；
从根节点到某一节点路径上所经过的字符连接起来，即为该节点对应的字符串；
任意节点的所有子节点所包含的字符都不相同；

应用场景：打字预测、拼写检查、IP路由、自动补全

"""
# 涉及到3个模版
    # 模版一 addWord(string word)
    # 模版二 Search(string word)
    # 模版三 searchPrefix(string prefix)
"""
class Trie {
    public Trie() {
        root = new TrieNode();
    }

    public void insert(String word){
        TrieNode node = root;
        for (char c: word.toCharArray()) {
            if (node.children[c-'a'] == null) node.children[c-'a'] = new TrieNode();
            node = node.children[c-'a'];
        }
        node.isWord=true
    }

    public boolean search(String word) {
        TrieNode node = root;
        for(char c: word.toCharArray()){
            if(node.children[c-'a'] == null) return False;
            node = node.children[c-'a'];
        }
        return node.isWord;
    }
    
    public boolean startsWith(String prefix) {
        TrieNode node = root;
        for (char c: prefix.toCharArray() {
            if (node.children[c-'a'] == null) return false;
            node = node.children[c-'a'];
        }
        return True
    }
}   
class TrieNode{
    TrieNode[] children;
    boolean isWord;
    public TrieNode() {
        children = new TrieNode[26];
    }
}
"""
# 208. Implement Trie (Prefix Tree)
     
class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode) # 套娃
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
   

# 211. Design Add and Search Words Data Structure
class TrieNode():
    
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.isWord = False
        
class WordDictionary:
    
    def __init__(self):
        self.root = TrieNode()
        
    def addWord(self, word: str) -> None:
        node = self.root
        for w in word:
            node = node.children[w]
        node.isWord = True

    def search(self, word: str) -> bool:
        node = self.root
        self.res = False
        self.dfs(node, word)
        return self.res
    
    # 涉及到符号的判断，我们用dfs来看看，dfs是可达性，bfs是最短。
    def dfs(self, node, word):
        # 如果没有word的情况，那么要么就是不存在，要么就是找完了。
        # 这里有个小细节，没有找到我们就不动res，如果找到那么返回的肯定就是true
        if not word:
            if node.isWord:
                self.res = True
            return 
        
        # 如果当前word为.，那么我们去看当前root的子节点。
        # children里的value存的可是TireNode
        # 如果是.，权当通过，继续往下执行
        if word[0] == '.':
            for n in node.children.values():
                self.dfs(n, word[1:])
        else:
            # 当不为.时，接下来只能从word的节点走
            # 如果为.时，接下来走哪一个都成。因为.可以变化。
            node = node.children.get(word[0])
            if not node:
                return 
            self.dfs(node, word[1:])



# 212. Word Search II
# 很综合的一个题
class TrieNode():
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.isWord = False
    
class Trie():
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for w in word:
            node = node.children[w]
        node.isWord = True
    
    def search(self, word):
        node = self.root
        for w in word:
            node = node.children.get(w)
            if not node:
                return False
        return node.isWord
    
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        res, trie = [], Trie()
        node = trie.root
        for w in words:
            trie.insert(w)
        
        for i in range(len(board)):
            for j in range(len(board[0])):
                self.dfs(board, node, i, j, "", res)
        return res
        

    def dfs(self, board, node, i, j, path, res):
        # 找到了，并且过河拆桥，把isWord改为false
        if node.isWord:
            res.append(path)
            node.isWord = False
        # 出界了。
        if i<0 or i>=len(board) or j<0 or j>= len(board[0]):
            return
        
        tmp = board[i][j]
        node = node.children.get(tmp)
        if not node:
            return 
        # 这里有点像backtrack了，如果都没发现或者都遍历完了，把当前board[i][j]再变回来。
        board[i][j] = "#"
        self.dfs(board, node, i+1, j, path+tmp, res)
        self.dfs(board, node, i-1, j, path+tmp, res)
        self.dfs(board, node, i, j-1, path+tmp, res)
        self.dfs(board, node, i, j+1, path+tmp, res)
        board[i][j] = tmp


# 官方题解
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        WORD_KEY = '$'
        
        trie = {}
        for word in words:
            node = trie
            for letter in word:
                # retrieve the next node; If not found, create a empty node.
                node = node.setdefault(letter, {})
            # mark the existence of a word in trie node
            node[WORD_KEY] = word
        
        rowNum = len(board)
        colNum = len(board[0])
        
        matchedWords = []
        
        def backtracking(row, col, parent):    
            
            letter = board[row][col]
            currNode = parent[letter]
            
            # check if we find a match of word
            word_match = currNode.pop(WORD_KEY, False)
            if word_match:
                # also we removed the matched word to avoid duplicates,
                #   as well as avoiding using set() for results.
                matchedWords.append(word_match)
            
            # Before the EXPLORATION, mark the cell as visited 
            board[row][col] = '#'
            
            # Explore the neighbors in 4 directions, i.e. up, right, down, left
            for (rowOffset, colOffset) in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                newRow, newCol = row + rowOffset, col + colOffset     
                if newRow < 0 or newRow >= rowNum or newCol < 0 or newCol >= colNum:
                    continue
                if not board[newRow][newCol] in currNode:
                    continue
                backtracking(newRow, newCol, currNode)
        
            # End of EXPLORATION, we restore the cell
            board[row][col] = letter
        
            # Optimization: incrementally remove the matched leaf node in Trie.
            if not currNode:
                parent.pop(letter)

        for row in range(rowNum):
            for col in range(colNum):
                # starting from each of the cells
                if board[row][col] in trie:
                    backtracking(row, col, trie)
        
        return matchedWords    

# 421. Maximum XOR of Two Numbers in an Array
"""
class Node {
    HashMap<Integer, Node> children;
    Node() {
        this.children = new HashMap<>();
    }
}

class Trie {
    Node root;
    
    Trie() {
        this.root = new Node();
    }
    
    public void insert(int[] A) {
        for(int num : A) {
            Node curr = this.root;
            for(int i=31;i>=0;i--) {
                int currBit = (num >> i) & 1;
                if(!curr.children.containsKey(currBit)) 
                    curr.children.put(currBit, new Node());
                curr = curr.children.get(currBit);
            }
        }
    }
}

class Solution {
    public int findMaximumXOR(int[] nums) {
        Trie trie = new Trie();
        trie.insert(nums);
        
        int max = 0;

        for(int num : nums) {
            Node curr = trie.root;
            int currSum = 0;
            for(int i=31;i>=0;i--) {
                int requiredBit = 1-((num >> i) & 1); // if A[i] is 0, we need 1 and if A[i] is 1, we need 0. Thus, 1 - A[i]
                if(curr.children.containsKey(requiredBit)) {
                    currSum |= (1<<i); // set ith bit of curr result
                    curr = curr.children.get(requiredBit);
                } else {
                    curr = curr.children.get(1-requiredBit);
                }
            }
            max = Math.max(max, currSum); // get max number
        }
        return max;
    }
}
"""

"""
Java常用API
常用method
str.substring();
str.charAt(index);
str1.compareTo(str2);


"""
   
########################  Union-Find  ########################
# 并查集 是一种树形的数据结构，用于处理不交集的合并(union)及查询(find)问题，主要处理的问题就是围绕着点与点的连接问题
# 可以优化的方向：1. Path Compression 2.Union by Size 3.Union by Rank


# 547. Number of Provinces
class Solution:
    def findCircleNum(self, M):
        
        # 找根节点的过程
        def find(node):
            # 这个if表明要查找的node，就是根节点了
            if circles[node] == node: return node
            # 如果不是root，那么继续查找其父节点的父节点，直至root，然后命名
            root = find(circles[node])
            # union操作，令当前node改为root，并且返回。
            circles[node] = root
            return root
        
        n = len(M)
        # circles就是我们合并过的数据
        circles = {x:x for x in range(n)}
        # 这里不用扫描每一个element
        for i in range(n):
            for j in range(i, n):
                # i != j 意味着不是自己本身； M[i][j] == 1 意味着有连接状态； find(i) != find(j) 意味着这两个点目前不属于一个group，因此我们需要进行处理。
                # 处理：find(i) 找到i的根结点，把circle中的根结点变为 find(j)即为j的根结点
                if i != j and M[i][j] == 1 and find(i) != find(j):
                    circles[find(i)] = find(j)   
        
        # 只有当k == v时，才意味着这个节点是root，我们count出来有多少个，就意味着有多少个group
        return sum([1 for k, v in circles.items() if k == v])
# DFS 方法
class Solution:
    def findCircleNum(slef, M):
        n = len(M)
        seen = set()
        res = 0
        
        def dfs(node):
            # 针对当前城市，去看看它的周边城市，如果发现它周边城市连接了，并且我们没有访问过，就dfs继续进去。
            # 如果之前的dfs访问过，直接剪枝。
            for nei, adj in enumerate(M[node]):
                if adj == 1 and nei not in seen:
                    seen.add(nei)
                    dfs(nei)
        # 每进一次dfs，就把对面老底掏空。
        for i in range(n):
            if i not in seen:
                dfs(i)
                res += 1
        return res


# 305. Number of Islands II
class Solution(object):
    def numIslands2(self, m, n, positions):
        ans = []
        islands = Union()
        # 🌟这里的map(tuple)不能省略，因为我们之后会用到p作为index
        # tuple is immutable, so is hashable. List couldn't do that.
        # 我们在之后会用到dict存数据，list开销有点大。
        for p in map(tuple, positions):
            islands.add(p)
            for dp in (0, 1), (0, -1), (1, 0), (-1, 0):
                q = (p[0] + dp[0], p[1] + dp[1])
                # 新的q见过的吧，就把它们unit起来
                if q in islands.id:
                    islands.unite(p, q)
            # 把每一次的答案进行合并，更新。
            ans += [islands.count]
        return ans

class Union(object):
    def __init__(self):
        # 这里的id就像是union-find/dis-jointed union里存的那个结构
        self.id = {}
        self.sz = {}
        self.count = 0

    # 添加岛屿： （p是个坐标） 
    # 先把新来的这个island给初始化出来，再说unit的事情。
    def add(self, p):
        self.id[p] = p
        self.sz[p] = 1
        self.count += 1

    # 找root
    def root(self, i):
        # 如果不是它本身，就继续找。先将自己的id变成其parent node, 然后将i再更新。
        while i != self.id[i]:
            self.id[i] = self.id[self.id[i]]
            i = self.id[i]
        return i
    
    # unit两个点：
    #   1. 把两个点的root坐标找出来。
    #   2. 如果相同，那么就是找到了；
    #   3. 优化！如果看看是谁的sz大，小的合并到大的里面
    #   4. 调整count
    def unite(self, p, q):
        i, j = self.root(p), self.root(q)
        if i == j:
            return
        if self.sz[i] > self.sz[j]:
            i, j = j, i
        self.id[i] = j
        self.sz[j] += self.sz[i]
        self.count -= 1



# 128. Longest Consecutive Sequence
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        def find(i):
            if i != parent[i]:
                parent[i] = find(parent[i])
            return parent[i]
        
        # 利用rank来优化
        def union(i,j):
            pi, pj = find(i), find(j)
            # 表明还没有合并
            if pi != pj:
                if rank[pi] >= pj:
                    parent[pj] = pi
                    rank[pi] += 1
                else:
                    parent[pi] = pj
                    rank[pj] += 1
        
        if not nums:
            return 0 # corner case
        
        # first pass is initialize parent and rank for all num in nums
        # 初始化
        parent, rank, nums = {}, {}, set(nums)
        for num in nums: # init
            parent[num] = num
            rank[num] = 0
            
        # second pass: union nums[i] with nums[i]-1 and nums[i]+1 if ums[i]-1 and nums[i]+1 in nums
        # 看看有没有，这里注意了，因为前后都判断了，所以要在union里添加限制，以免造成重复浪费
        for num in nums:
            if num-1 in nums:
                union(num-1, num)
            if num+1 in nums:
                union(num+1, num)
                    
        # second pass find numbers under the same parent
        d = collections.defaultdict(list)
        for num in nums:
            d[find(num)].append(num)
        return max([len(l) for l in d.values()]) 
# sorting的方法：
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if not nums: return 0
        nums.sort()
        res = 1
        count = 1
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1]+1:
                count += 1
            elif nums[i] == nums[i - 1]: 
                continue
            else:
                count = 1
            res = max(res, count)
        
        return res
        
"""
简单总结一下吧
1- union find的核心 = 初始化 + find + union (模版性强) 以及那个存储root的dict很重要
2- 难点在于如何把题目抽象成union find 以及 一些细节处理的点
"""    
########################  Heap  ########################
# Heapify
def heapify(arr, n, i): 	#n代表有多少个节点，i代表目前针对那个节点做文章  
    largest = i
    l = 2 * i + 1 # left = 2*i + 1  i节点下的左节点的index
    r = 2 * i + 2 # right = 2*i + 2   i节点下的右节点的index 
    #i节点的根节点为 root=(i-1)/2

 #下面3个if是用来交换二叉树中的值的，构造完全二叉树
    if l < n and arr[i] < arr[l]: 
        largest = l 
    if r < n and arr[largest] < arr[r]: 
        largest = r 
 #上述两个if是用来比较，下面这个是把根节点交换成最大值
    if largest != i:
        arr[i],arr[largest] = arr[largest],arr[i] # 交换
        heapify(arr, n, largest) #对下面节点继续heapify
#上面这个对子树进行heapify比较难理解：首先我们是从最子节点开始构建堆的，所以已经确保了子树的堆结构，如果我们进行了交换，就没有办法保证已经确保的子树还满足完全二叉树的结构，也就是大小关系，因此要对交换过的子树进行这种处理。
def heapSort(arr): 
    n = len(arr) 
 
# Build a maxheap. 
# 这里可以优化一下，从n//2进行。就是倒数第二层。
    for i in range(n, -1, -1):
        heapify(arr, n, i) #倒序建堆，意味着从最底层子树开始建堆

    # 一个个交换元素
    # 为什么要这么处理就是，上面已经建好堆了， 也就是最大值就在root处。 root跟尾节点交换，然后切断尾部。交换过后破坏了堆的形式，那么这个时候只用针对根节点进行heapify就好了。
    for i in range(n-1, 0, -1): #i 倒序 —— 这里就是切尾巴的操作
        arr[i], arr[0] = arr[0], arr[i] # 交换
        heapify(arr, i, 0) # size为i，每次针对0根节点进行heapify
 

# HeapSrot
"""
public void sort(int arr[]) {
    int n = arr.length;
    
    // build heap
    for (int i = n / 2 - 1; i>=0; i--) {
        heapify(arr, n, i);
    }
    // extract from heap one by one
    for (int i=n-1; i>0; i--) {
        // swap i 
        int temp = arr[0];
        arr[0] = arr[i];
        arr[i] = temp;

        heapify(arr, i, 0);
    }
}

Heap也可以用来实现prioirty queue - 这里就不做演示了。


peekMax() or peekMin(), depends on which heap you implement, takes O(1)
add()/remove() a new number to heap would do logn times heapify, so time complexity is O(logn)
Delete random would cause O(n), because search O(n) + delete O(logn)
Build heap only takes O(n), we are doing heapify only for half of the element
"""

########################  Stack/Queue  ########################

# 155. Min Stack
# 要么使用一个stack，但是push in的是一个数组，分别是当前val和当前_min
class MinStack:

    def __init__(self):
        self.stack = []
        self.min_stack =[]

    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> None:
        if self.stack[-1] == self.min_stack[-1]:
            self.min_stack.pop()
        self.stack.pop()
        

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]

# 232. Implement Queue using Stacks
class MyQueue:

    def __init__(self):
        self.inStack = []
        self.outStack = []

    def push(self, x: int) -> None:
        self.inStack.append(x)

    def pop(self) -> int:
        self.move()
        return self.outStack.pop()

    def peek(self) -> int:
        self.move()
        return self.outStack[-1]


    def empty(self) -> bool:
        return (not self.inStack) and (not self.outStack) 

    def move(self):
        if not self.outStack:
            while self.inStack:
                self.outStack.append(self.inStack.pop())

# 225. Implement Stack using Queues
class MyStack:

    def __init__(self):
        self.enque = []
        self.deque = []

    def push(self, x: int) -> None:
        self.enque.append(x)

    def pop(self) -> int:
        while len(self.enque) > 1:
            self.deque.append(self.enque.pop(0))
        pop = self.enque.pop(0)
        # 这个交换也比较重要，否则遇到两次pop就完犊子
        self.enque, self.deque = self.deque, self.enque
        return pop

    # 我们也可以用一个变量去存top
    def top(self) -> int:
        while self.deque:
            self.enque.append(self.deque.pop(0))
        return self.enque[-1]

    def empty(self) -> bool:
        # 这里的判断需要注意，要用两个not
        return not self.enque and not self.deque


# 622. Design Circular Queue
# 看了_init_其他都是自己写的。不错不错嘿嘿嘿
class MyCircularQueue:

    def __init__(self, k: int):
        self.queue = []
        self.capacity = k
        self.index = 0
        self.count = 0
        

    def enQueue(self, value: int) -> bool:
        if self.index == self.capacity:
            return False
        self.queue.append(value)
        self.index += 1
        return True

    def deQueue(self) -> bool:
        if self.index == 0:
            return False
        self.queue.pop(0)
        self.index -= 1
        return True
        

    def Front(self) -> int:
        return self.queue[0] if self.queue else -1

    def Rear(self) -> int:
        print(self.queue)
        return self.queue[-1] if self.queue else -1
        

    def isEmpty(self) -> bool:
        return self.index == 0

    def isFull(self) -> bool:
        return self.index == self.capacity
        

# 641. Design Circular Deque
# 看了_init_其他都是自己写的。不错不错嘿嘿嘿
class MyCircularDeque:
    def __init__(self, k: int):
        self.queue = []
        self.index = 0
        self.capacity = k

    def insertFront(self, value: int) -> bool:
        if self.index == self.capacity:
            return False
        self.index += 1
        self.queue = [value] + self.queue
        return True
         

    def insertLast(self, value: int) -> bool:
        if self.index == self.capacity:
            return False
        self.index += 1
        self.queue.append(value)
        return True

    def deleteFront(self) -> bool:
        if self.index == 0:
            return False
        self.index -= 1
        self.queue.pop(0)
        return True

    def deleteLast(self) -> bool:
        if self.index == 0:
            return False
        self.index -= 1
        self.queue.pop()
        return True
        
    def getFront(self) -> int:
        if self.index == 0: return -1
        return self.queue[0]

    def getRear(self) -> int:
        if self.index == 0: return -1
        return self.queue[-1]
        

    def isEmpty(self) -> bool:
        return self.index == 0
    def isFull(self) -> bool:
        return self.index == self.capacity

      
# 1381. Design a Stack With Increment Operation
class CustomStack:
    def __init__(self, maxSize):
        self.stack = []
        self.size = maxSize

    def push(self, x):
        if len(self.stack) < self.size:
            self.stack.append(x)

    def pop(self):
        return self.stack.pop() if self.stack else -1
    
    def increment(self, k, val):
        for i in range(min(k, len(self.stack))):
            self.stack[i] += val
    



########################  LinkedList-1  ########################

# 206. Reverse Linked List
# 经典题再次复习
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
#         if not head or head.next is None:
#             return head
        
#         prev_node = self.reverseList(head.next)
#         head.next.next = head
#         head.next = None
        
          #🌟 这个tail_node一直从队尾return到队首
#         return prev_node
    
        prev = None
        curr = head
        
        while curr != None:
            temp = curr.next
            curr.next = prev
            prev = curr
            curr = temp
        return prev


# 92. Reverse Linked List II
# 还是卡了半个小时，刚开始的边界没想明白，有点绕。
class Solution:
    def reverseBetween(self, head, m, n):
        if not head:
            return None

        right, left = head, None
        while m > 1:
            left = right
            right = right.next
            m, n = m - 1, n - 1

        # Con是处理前链表的第一个node，pre前面未处理链表的结尾
        
        # 注意如果这里没有经过第一个while，这里的left会为None
        pre, con = left, right
        while n:
            tempNode = right.next
            right.next = left
            left = right
            right = tempNode
            n -= 1
        
        # 等到这个while技术后，我们的prev处于处理link的最后一位，也就是reverse之后的头部；
        # 我们的cur在之后链表中的第一位，因此tail.next=cur就可以🔗上。
        # 害怕我们最开始的con有可能是None，也有可能是从任一Node
        if pre:
            pre.next = left
        # 这里为什么要这么处理？因为如果从第一个遍历开始，left在第一个while是none，而且这个时候我们的head已经切换了。第二个while肯定会进。
        # 所以这里用if pre就用来处理这个事情
        else:
            head = left
        con.next = right
        return head
            
            
# 25. Reverse Nodes in k-Group
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        pointer = head
        new_head = None
        ktail = None
        
        while pointer:
            # 每一次遍历后，我们都要重制count/head/pointer
            count = 0
            pointer = head
            
            # 遍历看有没有符合条件的sub-linkedlist
            while pointer and count < k:
                pointer = pointer.next
                count += 1
            # 判断是否符合
            if count == k:
                # revHead是reverse linked-list 的最后一个node
                revHead = self.reverseLinkedList(head, k)
                
                
                # 这两个if not都是为了处理第一段的ll
                if not new_head:
                    new_head = revHead
                    
                # 这里用来连接k groups的
                if ktail:
                    ktail.next = revHead
                
                # 这里的操作就是让ktail指向已经排序过的head，即最后一位
                # 并且把head更新到pointer，下一个片段的开始
                ktail = head
                head = pointer
        
        # 如果最后还有一些没有进行reverse，把他们连接起来
        # 不同k groups怎么连接起来？
        if ktail:
            ktail.next = head
        return new_head if new_head else head
    def reverseLinkedList(self, head, k):
        pointer, new_head = head, None
        while k:
            temp = pointer.next
            pointer.next = new_head
            new_head = pointer
            pointer = temp
            k -= 1
        return new_head
        
# 2. Add Two Numbers
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        head = dummy
        carry = 0
        
        while l1 or l2 or carry:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            carry, out = divmod(val1 + val2 + carry, 10)
            head.next = ListNode(out)
            head = head.next
            
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
            
        return dummy.next

# 445. Add Two Numbers II
class Solution:
    def addTwoNumbers(self, l1, l2):
        x1, x2 = 0, 0
        while l1:
            x1 = x1*10+l1.val
            l1 = l1.next
        while l2:
            x2 = x2*10+l2.val
            l2 = l2.next
        x = x1 + x2

        head = ListNode(0)
        if x == 0: return head
        while x:
            v, x = x%10, x//10
            # 这里的指针操作太巧妙了
            head.next, head.next.next = ListNode(v), head.next

        return head.next

"""
public class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        Stack<Integer> s1 = new Stack<Integer>();
        Stack<Integer> s2 = new Stack<Integer>();
        
        while(l1 != null) {
            s1.push(l1.val);
            l1 = l1.next;
        };
        while(l2 != null) {
            s2.push(l2.val);
            l2 = l2.next;
        }
        
        int sum = 0;
        ListNode list = new ListNode(0);
        while (!s1.empty() || !s2.empty()) {
            if (!s1.empty()) sum += s1.pop();
            if (!s2.empty()) sum += s2.pop();
            list.val = sum % 10;
            ListNode head = new ListNode(sum / 10);
            head.next = list;
            list = head;
            sum /= 10;
        }
        
        return list.val == 0 ? list.next : list;
    }
}
"""

# 21. Merge Two Sorted Lists
# 自己写的代码又臭又长
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        head = dummy 
        l1 = list1
        l2 = list2
        while l1 or l2:
            if not l2:
                head.next = l1
                break
            if not l1:
                head.next = l2
                break
            
            
            if l1.val > l2.val:
                head.next = l2
                head = head.next
                l2 = l2.next
            elif l2.val >= l1.val:
                head.next = l1
                head = head.next
                l1 = l1.next
        
        return dummy.next
# 优化版本
    def xx(self, l1, l2): 
        while l1 and l2:
            if l1.val <= l2.val:
                prev.next = l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next
            # 把这个提取出来               
            prev = prev.next
        # 把最后的接尾工作放在这里。
        prev.next = l1 if l1 is not None else l2

# 23. Merge k Sorted Lists
class Solution(object):
    def mergeKLists(self, lists):
        
        self.nodes = []
        head = point = ListNode(0)
        # 所有的nodes放在一个list里，然后排序，然后generate
        for l in lists:
            while l:
                self.nodes.append(l.val)
                l = l.next
        
        for x in sorted(self.nodes):
            point.next = ListNode(x)
            point = point.next
        
        return head.next

# 141. Linked List Cycle
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        nodes_seen = set()
        while head is not None:
            if head in nodes_seen:
                return True
            nodes_seen.add(head)
            head = head.next
        return False

# 142. Linked List Cycle II
class Solution(object):
    def detectCycle(self, head):
        visited = set()
        node = head
        while node is not None:
            if node in visited:
                return node
            else:
                visited.add(node)
                node = node.next

        return None

# 287. Find the Duplicate Number
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        nums.sort()
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                return nums[i]

# 203. 203. Remove Linked List Elements
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]: 
        if not head: return None
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        while head:
            if head.val == val:
                prev.next = head.next
            else:
                prev = head
            head = head.next
        
        return dummy.next
        
# 82. Remove Duplicates from Sorted List II
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        
        while head:
            if head.next and head.val == head.next.val:
                while head.next and head.val == head.next.val:
                    head = head.next
                prev.next = head.next
            else:
                prev = prev.next
            head = head.next
        return dummy.next
        

# 19. Remove Nth Node From End of List
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
     
        dummy = ListNode(0, head)
        prev = dummy
        
        while n > 0:
            head = head.next
            n -= 1
                
        while head:
            head = head.next
            prev = prev.next
        
        if prev.next:
            prev.next = prev.next.next
            
        return dummy.next

# 1171. Remove Zero Sum Consecutive Nodes from Linked List
# prefix的方法
class Solution:
    def removeZeroSumSublists(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prefix_dict = collections.OrderedDict()
        cur = dummy = ListNode(0)
        dummy.next = head
        # print(dummy.next)
        prefix = 0
        while cur:
            prefix += cur.val
            node = prefix_dict.get(prefix, cur)
            # 理解为什么要popitem，就能理解为什么起那面需要orderdict
            # 首先我们有prefix，然后吧prefix存入key，对应的value为当前对应的node和之后的。
            # 如果遇到相同的值了，那么我们直接用将两个节点相连就可以了。
            # 为什么要pop，因为如果相连之间的node的prefix都不会存在。
            while prefix in prefix_dict:
                prefix_dict.popitem()
            prefix_dict[prefix] = node
            node.next = cur = cur.next
        return dummy.next

########################  LinkedList-2  ########################
########################  Comparator  ########################
