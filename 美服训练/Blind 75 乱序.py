# Prefix Sum - 238. Product of Array Except Self

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        length = len(nums)
        answers = [0] * length
        
        # 第一遍，answer[i] = [0~i-1]所有的乘积
        answers[0] = 1
        for i in range(1, length):
            answers[i] = answers[i - 1] * nums[i - 1]
            
        # 第二遍，我们用一个R来代替，answer[j]就是第一遍[0~j-1]的乘积，R是[j+1~n]的乘积！
        R = 1
        for j in reversed(range(length)):
            
            answers[j] = answers[j] * R
            R *= nums[j]
            
        return answers
# DP - 322 Coin Change
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        if amount == 0: return 0
        dp = [float("inf") for _ in range(amount + 1)]
        # 🌟这个太关键了
        dp[0] = 0
        for i in range(amount+ 1):
            for c in coins:
                if i >= c:
                    dp[i] = min(dp[i], dp[i-c] + 1)
       
        return dp[amount] if dp[amount] < float("inf") else -1
# Substring - 647
class Solution:
    def countSubstrings(self, s: str) -> int:
        if not s: return 0
        
        res = 0
        n = len(s)
        for length in range(1, n+1):
            for l in range(n - length + 1):
                temp = s[l:l+length]
                if temp == temp[::-1]:
                    res += 1
        return res
                    
            
# LinkedList - 143. Reorder List
class Solution:
    def reorderList(self, head: ListNode) -> None:
        if not head:
            return 
        
        # 找到中点，都从head出发的，slow会在中点或者后半段的首
        # reverse lists
        # merge
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            
        # 类似dummy，最后我们用的是prev
        prev, cur = None, slow
        while cur:
            cur.next, prev, cur = prev, cur, cur.next
            
        # merge 这里我们能清楚左侧的list永远比左侧长。因为左侧的指针一直指导中点/后端的首节点。
        # 中点/后端的首节点指向None。
        # 因此前端要不和后端一样长，要么就是多一个指向None的节点。
        # 因此只用while second.next就好，要么就刚好处理完，要么就是最后一个节点不用处理，因为本身就是最后一位。
        first, second = head, prev
        while second.next:
            first.next, first = second, first.next
            second.next, second = first, second.next
            
# Sliding window - 424. Longest Repeating Character Replacement
class Solution:    
    def characterReplacement(self, s, k):
        # 首先确认一点，一定是sub-sequence.
        # 而且一定是相连的，所以sliding window；利用res维护
        
        count = collections.Counter()
        start = result = 0
        for end in range(len(s)):
            count[s[end]] += 1
            # API返回一个list，其中list[0]存放的是出现最多的数字和其频率
            # 这里返回的是出现最多的元素的频率
            # [0]取出来的是list中的第一项；[1]是取出来的tuple中的第二项。
            max_count = count.most_common(1)[0][1]
        
            # 意味着剩下的元素已经大于>k了，怎么办都没有办法转换
            # 而剩下的元素无所谓是几个字母
            # if 不满足，缩小windown
            while end - start + 1 - max_count > k:
                count[s[start]] -= 1
                start += 1
            result = max(result, end - start + 1)
        return result


# 数据结构/API - 347. Top K Frequent Elements
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = collections.Counter(nums).most_common(k)
        res = []
        for x, y in count:
            res.append(x)
        return res
#另一种数据结构的做法       
from collections import Counter
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]: 
        # O(1) time 
        if k == len(nums):
            return nums
        
        # 1. build hash map : character and how often it appears
        # O(N) time
        count = Counter(nums)   
        # 2-3. build heap of top k frequent elements and
        # convert it into an output array
        # O(N log k) time
        return heapq.nlargest(k, count.keys(), key=count.get) 


# 扫描线 - 435. Non-overlapping Intervals
# 这道题最难的点在于，怎么理解清楚解题思路。你知道看首尾端点，但困难的是，为什么这种思路可以解题
# 只保留较小的末尾，因为碰见不满足条件的肯定要删除一方，那就删除影响较大的一方。
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        # 很明显这题是扫描线，扫描线一般都需要sort
        intervals.sort()
        #
        res = 0
        right = float("-inf")
        for s, e in intervals:
            if s >= right:
                right = e
            else: 
                res += 1
                right = min(right, e)
        return res 
        
"""
关于边界问题很重要！🌟
下面这里为什么要用l+i+1?
这里第一个for循环的i 并不==长度！ 而是==长度-1 也就是index的diff 
所以当i=0的时候意味着他想取本身 但这个时候[i:i+0]就取不到i本身 所以要+1
而我们的i+l 是我们想取到右边界的index 因此在切片器中要+1
这里的+几 理解为向右取几位

====================================================
还有另一种操作方法：
1. 在outter for loop 取的是长度！
2. 在inner for loop一样是取index 但是inner for的condition要写成(len(s)-i+1)是因为外侧的i是长度
3. 长度为5 index的差为4 搞清楚这点 就容易写了。
4. 最后的t = s[l:l+i]

总结 要么全程按照index差写 要么就是一开始比较直观按照length来写
但是在inner for取index的时候需要注意一下 需要+1 也可以相当于len(s) - (i-1)

"""
# String - 5. Longest Palindromic Substring
class Solution:
    def longestPalindrome(self, s: str) -> str:
        # 因为找最长的，所以为了减少浪费，从最长的开始遍历
        for i in range(len(s)-1, -1, -1):
            for l in range(len(s) - i):
                # 🌟这里比较难，右边界是为l+i+1
                # 我们想要n位数，那么左边第一位为l，最后一位就为n+l
                t = s[l:l+i+1]
                if t == t[::-1]:
                    return t
# 复杂度N^3
        
# 但是上面这种暴力方法回TLE
# 下面这种方法是遍历n2，比上面简单粗暴的记性比较会好很多。上面的一大问题是没有进行剪枝！
class Solution:
    def longestPalindrome(self, s):
        res = ""
        for i in range(len(s)):
            # API用法
            # res = max(self.helper(s,i,i), self.helper(s,i,i+1), res, key=len)
            
            # odd case, like "aba"
            tmp = self.helper(s, i, i)
            if len(tmp) > len(res):
                res = tmp
            # even case, like "abba"
            tmp = self.helper(s, i, i+1)
            if len(tmp) > len(res):
                res = tmp
        return res

    # get the longest palindrome, l, r are the middle indexes   
    # from inner to outer
    def helper(self, s, l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            # 这个分号的用法，wow
            l -= 1; r += 1
        # 这么写，因为最后一层循环不满足了，因此l+1 ～ r-1
        return s[l+1:r]
# 还有一种方法，DP
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s:
            return ''
        l = len(s)
        dp = [[None for j in range(l)] for i in range(l)]
        lp = s[0]
        for i in range(l - 1, -1, -1):
            for j in range(i, l):
                # 这一串逻辑主要是为了判读substring的情况下s[i:j+1]是否成立
                # 1. s[i]自己
                if i == j:
                    dp[i][j] = True
                # 2. substring长度为2
                elif j == i + 1:
                    dp[i][j] = s[i] == s[j]
                # 3. substring长度大于2，需要依靠之前的子字符串进行判断
                elif j > i + 1:
                    dp[i][j] = dp[i + 1][j - 1] and s[i] == s[j]
                # 一次遍历完成后，保存最大值。
                if dp[i][j] and j - i + 1 > len(lp):
                    lp = s[i:j + 1]

        return lp
                
# prefix sum - 152. Maximum Product Subarray
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0

        # ma是用来存放所有的max值的
        # mi是用来存放所有的min值的，主要用来对付negative值...
        ma = nums[0]
        mi = nums[0]
        result = ma

        # 利用两个list来处理，反而避免掉了那些不连续的值。
        for i in range(1, len(nums)):
            curr = nums[i]
            temp_max = max(curr, ma * curr, mi * curr)
            mi       = min(curr, ma * curr, mi * curr)

            ma = temp_max

            result = max(ma, result)

        return result

# DP - 91. Decode Ways
class Solution:
    def numDecodings(self,s):     
        if not s:
            return 0
        # 这里dp的index是什么意思？长度/实际的数字，那么可以用index么？可以是可以啦，但是要额外判断开头的的情况
        dp = [0 for _ in range(len(s) + 1)]
        # 为什么把dp[0]设置为1？ # 主要针对dp[2]，如果dp[2]满足两位的情况，那么就要从dp[0]当中增加，即为1
        # 
        dp[0] = 1
        dp[1] = 0 if s[0] == "0" else 1
        # 切片器的index是可以取不到的，但是其间覆盖到的值却可以没有影响地被取到。
        # 所以要从2开始
        for i in range(2, len(s) + 1):
            if "0" < s[i-1] <= "9":
                dp[i] += dp[i-1]
            if "10" <= s[i-2:i] <= "26":
                dp[i] += dp[i-2]
            
        return dp[len(s)]

# 191. Number of 1its
class Solution:
    def hammingWeight(self, n: int) -> int:
        bits = 0
        for i in range(32):
            if (n & 1) != 0:
                bits += 1
            n >>= 1
        return bits

class Solution:
    def solve(self, n):
        return bin(n).count("1")

# 207. Course Schedule
# topological sort
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        n = numCourses
        S = [[] for _ in range(n)]
        degree = [0] * n
        
        for i, j in prerequisites:
            S[j].append(i)
            degree[i] += 1
        
        # 放的是可以选修课程的index
        q = [i for i in range(n) if degree[i] == 0]
        
        for i in q:
            for j in S[i]:
                degree[j] -= 1
                if degree[j] == 0:
                    q.append(j)
        return len(q) == n
        

# 这道题有意思的点在于，针对一门课，他的图是怎么样的。
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        n = numCourses
        graph = [[] for _ in range(n)]
        visit = [0 for _ in range(n)]
        # 这里的顺序和拓扑排序不太一样topological sort
        # y是先修课，这里就是每一门课下面：有多少先修课； 拓扑排序是一门先修课：可以再上什么post选修课。
        for x, y in prerequisites:
            graph[x].append(y)
        
        def dfs(i):
            if visit[i] == -1:
                return False
            if visit[i] == 1:
                return True
            # 这里的操作很精细，仔细思考路径。
            # 如果能够成功，那么dfs的任意一条路径都已经能够打通，所以在不可能存在环
            # 遍历过后填上-1，这样在这条路的时候，如果之后再次遍历到，就意味着是环了，因此直接-1，return false
            # 好好想想这道题的dfs长什么样子。
            visit[i] = -1

            # 如果graph[i]里面没有值，就不会进入循环/recursion，那么会直接=1，并且返回true，因为它可以到达。
            for j in graph[i]:
                # 只要j里面有一个是false，那么最终就会false
                if not dfs(j):
                    return False
            visit[i] = 1
            return True
                    
            
        for i in range(n):
            # 确保每一个课程都可以上完
            if not dfs(i):
                return False
        return True
            
        

# 33. Search in Rotated Sorted Array
class Solution:
    
    # 处理逻辑很复杂，为什么要分这么多case？
    # 因为我们只能单调的序列，而不单调的序列有多种情况，但是处理方式一样。
    # 外层的if就是将图形先分个类，然后根据mid与边界的值确定出一部分是否是有序。
    def search(self, nums: List[int], target: int) -> int:
        if not nums: return - 1
        lo, hi = 0, len(nums)-1
        
        # 这里为什么是=？  如果我们还是为了找到一个数，用不用等号都可以
        # 但是本题哪怕范围缩小到最后，我们还是要判断是否是我们要找的数字，如果是可以返回，如果不是就删除。
        # 而下面的子问题中，是+-1都是因为无法取到值，或者能取到的值已经在最开始判断过了。
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            
            
            
            # 1
            if nums[mid] == target:
                return mid
            
            # 2
            # 先判断是哪种图形。
            # 哪怕是这里的=都不能省略，为什么？因为我们在计算mid的时候已经吃亏了，所以这里不能吃亏，否则与原图就不是一回事了～
            elif nums[mid] >= nums[lo]:
                if nums[lo] <= target < nums[mid]:
                    # 如果是这样的话，hi是取不到的
                    hi = mid - 1
                # target 有两种情况，要么是比mid还大，要么是比lo还下，分别对应着图像的不同位置。
                # mid也没办法取到，因为上面的一个if已经排除了。
                else:
                    lo = mid + 1
            # 3
            # mid < lo 意味着小的数字比较多一点。
            else:
                if nums[mid] < target <= nums[hi]:
                    lo = mid + 1
                else:
                    hi = mid - 1
            
        return -1



# 76. Minimum Window Substring

# 这题的难点在哪？
# sliding window不难，两个while/或者一个for 一个while
# 难在左移右移的条件是如何判断的？
# 1. 利用两个dict去判断
# 2. 利用一个变量保存
# 3. ans利用tuple存起来。
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not s or not t: return ""
        dict_t = collections.Counter(t)
        required = len(dict_t)
        l = r = 0
        formed = 0  # window中有几个字母满足要求了。
        window_counts = dict()
        ans = (float("inf"), None, None)
        
        while r < len(s):
            character = s[r]
            window_counts[character] = window_counts.get(character, 0) + 1
            
            if character in dict_t and window_counts[character] == dict_t[character]:
                formed += 1
            
            # 满足condition的情况下，我们开始缩小左边届
            while l <= r and formed == required:
                character = s[l]
                if r - l + 1 < ans[0]:
                    ans = (r - l + 1, l, r)
                
                window_counts[character] -= 1
                if character in dict_t and window_counts[character] < dict_t[character]:
                    formed -= 1
                    
                l += 1
            r += 1
        return "" if ans[0] == float("inf") else s[ans[1] : ans[2] + 1]


# 271. Encode and Decode Strings

# What to use as a delimiter? Each string may contain any possible characters out of 256 valid ascii characters.
class Codec:
    def encode(self, strs: List[str]) -> str:
        """Encodes a list of strings to a single string.
        """
        if len(strs) == 0: return chr(258)
        return chr(276).join(x for x in strs)
    
        

    def decode(self, s: str) -> List[str]:

        if s == chr(258): return []
        return s.split(chr(276))

# 129. Sum Root to Leaf Numbers
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        res = 0
        # 首先保证是pre order
        def dfs(root, temp):
            nonlocal res
            # 两层嵌套
            if root:
                # 这里的temp是局部变量，不同于回溯，因此没必要
                temp = temp*10 + root.val
                # if not(root.left or root.right)
                if not root.left and not root.right:
                    res += temp
                dfs(root.left, temp)
                dfs(root.right, temp)

        dfs(root, 0)
        return res

# 用一下bfs方法做
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        res = 0
        stack = [(root, 0)]
        while stack:
            root, cur = stack.pop()
            # 因为涉及到leaf的判断，所以要多加一层。
            if root:
                cur = cur * 10 + root.val
                if not (root.left or root.right):
                    res += cur
                else:
                    # 注意顺序
                    stack.append((root.right, cur))
                    stack.append((root.left, cur))
        return res
    
# 23. Merge K sorted LinkedLists
class Solution(object):
    def mergeKLists(self, lists):
        nodes = []
        head = ptr = ListNode(0)
        for l in lists:
            while l: 
                nodes.append(l.val)
                l = l.next
        for x in sorted(nodes):
            ptr.next = ListNode(x)
            ptr = ptr.next
        return head.next


# 212. Word Search II
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
            
            # 4个方向 注意操作流程
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
        
            # 这里用来优化，如果发现找过的话，直接remove the matched leaf node.
            # Optimization: incrementally remove the matched leaf node in Trie.
            # 这里的优化非常小，如果curNode是匹配的，而且是leaf node的话是可以删除的。不要也行。
            # 仔细思考一下，条件非常严格。首先trie树每一层一定只有一个答案，因为tire树要考虑前缀，如果有一位不一样，甚至位置不一样都会有岔路，有点类似bfs的感觉。
            # 所以当not currNode意味着当前的wordkey已经pop了，所以可以删除，无所谓。
            if not currNode:
                parent.pop(letter)

        for row in range(rowNum):
            for col in range(colNum):
                # starting from each of the cells
                if board[row][col] in trie:
                    backtracking(row, col, trie)
        
        return matchedWords    

# 2243. Calculate Digit Sum of a String
# 周赛第一题，我的思路是：因为需要去看每k个的集合，直到总len<=k;
# 因此，我用了t当作每一次的input，temp存暂时的集合，temp_list存每次需要处理的k个元素
class Solution:
    def digitSum(self, s: str, k: int) -> str:
        if len(s) <= k: return s
        t = [ch for ch in s]
        while len(t) > k:
            temp = []
            while t:
                i = 0
                temp_list = []
                while i < k and i < len(s) and t:
                    i+=1
                    temp_list.append(t.pop(0))
                temp.append(str(sum(int(x) for x in temp_list)))
            
            t = [x for x in "".join(temp)]
        return "".join(t)

# 大神的写法：第一个for 针对0～len(s)之间每组(k)的开头元素；第二个for针对每组的k个元素；用r来存sum并且进行int和str的转化‘
# 最绝的是通过recursion来做这道题
class Solution:
    def digitSum(self, s: str, k: int) -> str:
        if len(s) <= k:
            return s
        t = ''
        # Holy Crap！
        for i in range(0, len(s), k):
            r = 0
            for j in s[i:i+k]:
                r += int(j)
            t += str(r)
        return self.digitSum(t, k)


# 2244. Minimum Rounds to Complete All Tasks
# 周赛第二题
class Solution:
    def minimumRounds(self, tasks: List[int]) -> int:
        cnt = collections.Counter(tasks)
        res = 0
        for k, v in cnt.items():
            if v == 1:
                return -1
            x,y = divmod(v, 3)
            res += x
            if y: res += 1
        return res
# 下面是大神的写法，我跟大神唯一不一样的地方是条件的处理。因为每组最多为3个，因此(c[i]+2) // 3 相当于直接进一位！秒
class Solution:
    def minimumRounds(self, a: List[int]) -> int:
        c = Counter(a)
        z = 0
        for i in c:
            # 关于1的处理也很妙
            if c[i] == 1:
                return -1
            z += (c[i] + 2) // 3
        return z

# 2245. Maximum Trailing Zeros in a Cornered Path
# 这道题之所以放弃是因为有3个关键点我么有思路：第一是如何查询末尾的0；第二是我的traverse逻辑有没有问题？；第三那个是现在的代码跟我现有的逻辑怎么对不上
# 对比大神的答案之后有了些许思路
#   1. 查询末尾的0不是通过path和count，而是利用了helper function，%2与%5，分别查询倍数的数量；然后利用前缀和记录2/5的数量
#   2. 这道题不是考察遍历md，遍历所有位置，直接查最大值，我就说哪里不对劲，dfs太容易造成浪费了。
#   3. 我的代码问题在哪？你没有把path给还原，son function里面是可以针对list进行修改的，因为list里面是指针，而不是局部变量。

class Solution:
    def maxTrailingZeros(self, grid: List[List[int]]) -> int:
        def backtracking(grid,r,c,path, flag, direction):
            if r < 0 or r == row or c < 0 or c == col:
                print(path)
                # print(str(reduce(lambda x,y:x*y,path)))
                return str(reduce(lambda x,y:x*y,path)).count('0')
        
            path.append(grid[r][c])
            if direction == 1 and flag:
                return max(backtracking(grid,r+1,c,path,flag,1),backtracking(grid,r,c+1,path,False,-1))
            elif direction == 1 and not flag:
                return backtracking(grid,r+1,c,path,flag,1)    
            elif direction == -1 and flag:
                return max(backtracking(grid,r,c+1,path,flag,-1),backtracking(grid,r+1,c,path,False,1))
            elif direction == -1 and not flag:
                return backtracking(grid,r,c+1,path,flag,-1)
                
            
         
        row, col = len(grid), len(grid[0])
        res = max(backtracking(grid, 0, 0, [], True, 1), backtracking(grid,0,0,[],True,-1))
        return res
        
class Solution:
    def maxTrailingZeros(self, grid: List[List[int]]) -> int:
        def get_25(v) :
            to_ret = [0, 0]
            while v % 2 == 0 :
                v = v // 2
                to_ret[0] += 1
            while v % 5 == 0 :
                v = v // 5
                to_ret[1] += 1
            return to_ret
        
        m, n = len(grid), len(grid[0])
        pre_sum1 = [[[0, 0] for a in range(n+1)] for b in range(m+1)] # 向上
        pre_sum2 = [[[0, 0] for a in range(n+1)] for b in range(m+1)] # 向左
        
        for i in range(m) :
            for j in range(n) :
                gs = get_25(grid[i][j])
                pre_sum1[i+1][j+1][0] = pre_sum1[i][j+1][0] + gs[0]
                pre_sum1[i+1][j+1][1] = pre_sum1[i][j+1][1] + gs[1]
                pre_sum2[i+1][j+1][0] = pre_sum2[i+1][j][0] + gs[0]
                pre_sum2[i+1][j+1][1] = pre_sum2[i+1][j][1] + gs[1]
        
        to_ret = 0
        for i in range(m) :
            for j in range(n) :
                
                a, b = pre_sum1[i+1][j+1]
                r1 = min(a+pre_sum2[i+1][j][0], b+pre_sum2[i+1][j][1])
                to_ret = max(to_ret, r1)
                r2 = min(a+pre_sum2[i+1][-1][0]-pre_sum2[i+1][j+1][0], b+pre_sum2[i+1][-1][1]-pre_sum2[i+1][j+1][1])
                to_ret = max(to_ret, r2)
                a = pre_sum1[-1][j+1][0] - pre_sum1[i][j+1][0]
                b = pre_sum1[-1][j+1][1] - pre_sum1[i][j+1][1]
                r3 = min(a+pre_sum2[i+1][j][0], b+pre_sum2[i+1][j][1])
                to_ret = max(to_ret, r3)
                r4 = min(a+pre_sum2[i+1][-1][0]-pre_sum2[i+1][j+1][0], b+pre_sum2[i+1][-1][1]-pre_sum2[i+1][j+1][1])
                to_ret = max(to_ret, r4)
        return to_ret

# 11. Container With Most Water
# brutal force导致tle
class Solution:
    def maxArea(self, height: List[int]) -> int:
        res = 0
        for i in range(len(height)):
            for j in range(i, len(height)):
                res = max(res, (j-i)*min(height[i],height[j]))
        return res
# tle
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r, area = 0, len(height) - 1, 0
        while l < r:
            area = max(area, (r - l) * min(height[l], height[r]))
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
				
        return area

# 49. Group Anagrams
# 这题记录下来是因为
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # 利用了每个string里的元素(无关顺序)当作index进行归类
        ans = collections.defaultdict(list)
        for s in strs:
            # 把拥有相同的元素，不同排列顺序的s放在一个dict的value中
            ans[tuple(sorted(s))].append(s)
        # value也很好。
        return ans.values()


# 235. Lowest Common Ancestor of a Binary Search Tree
# 转变下思路，我原本的思路是true/false传值，如果遇到左true/右true的话，进行返回，但是return的是node与true/false不符合
# 答案的思路：利用特性好粗糙。不！你一点都不粗糙，你没有理解题意！
# 我们求的是ancestor，因此只要发现这个值不在一个node的同侧，那么这个node就是我们要找的值，而且node的左右子树也有讲究
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':      
        if p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        elif p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        else:
            return root


# 1574. Shortest Subarray to be Removed to Make Array Sorted
# 题目要求remianing的是non-decreasing
class Solution:
    def findLengthOfShortestSubarray(self, arr: List[int]) -> int:
        l, r = 0, len(arr) - 1
        # 一旦出现降序就会跳出while
        while l < r and arr[l+1] >= arr[l]:
            l += 1
            
        if l == len(arr) - 1:
            return 0 # whole array is sorted
        # 如果是增序的话，r降低为什么？
        while r > 0 and arr[r-1] <= arr[r]:
            r -= 1
            
        # case1和2分别意味着增序还是降序
        # 目前整个list被l和r分割 ...l...r...左侧/右侧是升序
        # 看看是左还是右侧保留下来
        toRemove = min(len(arr) - l - 1, r) # case (1) and (2)
		
		# case (3): try to merge，接下来我们要去看左右两侧合并了
        # 这里你有个疑问，为什么不去看中间部分？
        # 因为题意笨蛋，我们remove的是什么？是一个substring，连续的！
        # 所以要想最后结果是一定是从两头走的。
        for iL in range(l+1):
            if arr[iL] <= arr[r]:
                toRemove = min(toRemove, r - iL - 1)
            elif r < len(arr) - 1:
                r += 1
            else:
                break
        return toRemove

# 417. Pacific Atlantic Water Flow
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not matrix or not matrix[0]: return []
        num_rows, num_cols = len(matrix), len(matrix[0])
        pacfic_q = deque()
        atlantic_q = deque()
        
        # 把周围接壤的地方围上
        for i in range(num_rows):
            pacific_q.append((i,0))
            atlantic_q.append((i, num_cols-1))
        for i in range(num_cols):
            pacific_queue.append((0, i))
            atlantic_queue.append((num_rows - 1, i))
        
        def bfs(queue):
            reachable = set()
            while queue:
                (row, col) = queue.popleft()
                reachable.add((row, col))
                for (x, y) in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    new_row, new_col = row + x, col + y
                    if new_row < 0 or new_row >= num_rows or new_col < 0 or new_col >= num_cols:
                        continue
                    if (new_row, new_col) in reachable:
                        continue
                    if matrix[new_row][new_col] < matrix[row][col]:
                        continue
                    queue.append((new_row, new_col))
            return reachable
        
        pacific_reachable = bfs(pacific_queue)
        atlantic_reachable = bfs(atlantic_queue)
        
        # Find all cells that can reach both oceans, and convert to list
        return list(pacific_reachable.intersection(atlantic_reachable))


# 297 297. Serialize and Deserialize Binary Tree
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

# 3 Sum 针对相同字母的处理是一个细节。
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        def helper(nums, i, res):
            lo, hi = i+1, len(nums)-1
            while lo < hi:
                total = nums[i] + nums[lo] + nums[hi]
                if total == 0:
                    res.append([nums[i], nums[lo], nums[hi]])
                    lo += 1
                    hi -= 1
                    while lo < hi and nums[lo] == nums[lo-1]:
                        lo += 1
                if total < 0:
                    lo += 1
                if total > 0:
                    hi -= 1
         
        res = []
        nums.sort()
        for i in range(len(nums)):
            if nums[i] > 0:
                break
            # 题意要求，不能包含重复的数字
            if i == 0 or nums[i] != nums[i-1]:
                helper(nums, i, res)
        return res

# 139. Word Break
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        def helper(s, wordDict, start):
            if start == len(s):
                return True
            for end in range(start + 1, len(s) + 1):
                if s[start:end] in wordDict and helper(s, wordDict, end):
                    return True
            return False
        return helper(s, set(wordDict), 0)
# BFS的方法。 每次进q都是end点，因为出现才会start 更新。
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        word_set = set(wordDict)
        q = deque()
        visited = set()

        q.append(0)
        while q:
            start = q.popleft()
            if start in visited:
                continue
            for end in range(start + 1, len(s) + 1):
                if s[start:end] in word_set:
                    q.append(end)
                    if end == len(s):
                        return True
            visited.add(start)
        return False
# DP的方法：针对每一个i就是看前面的能不能组成，如果可以，那么可达性分析就能做。
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        word_set = set(wordDict)
        dp = [False] * (len(s) + 1)
        dp[0] = True

        for i in range(1, len(s) + 1):
            for j in range(i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        return dp[len(s)]


# 268. Missing Number
# 或者使用数字匹配法
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        nums.sort()
        if nums[0] != 0:return 0
        
        for i in range(1, len(nums)):
            if nums[i] - nums[i-1] != 1:
                return nums[i] - 1
        return nums[-1] + 1

# 98. Validate Binary Search Tree
# 这一题很值得反思呀，一般来讲我的dfs思路总是bottom up的，但是这一题是top down，所以才会直接return 两个helper，思考到了root.val和左右子树的差别。
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def helper(root, low =-math.inf, high=math.inf):
            if not root:
                return True
            if root.val <= low or root.val >= high:
                return False
            return helper(root.right, root.val, high) and helper(root.left, low, root.val)
        return helper(root)


# 662. Maximum Width of Binary Tree
# 难点在于如何计算width：
# 根据树的特性，2n/2n+1, 把值与节点一起考虑
class Solution:
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        result = 0
        q = [(root,0)]
        while q:
            n = len(q)
            head = q[0][1]
            for _ in range(n):
                node, index = q.pop(0)
                if node.left:
                    q.append((node.left, 2*index))
                if node.right:
                    q.append((node.right, 2*index + 1))
            
            result = max(result, index - head + 1)
        return result
        


# 114. Flatten Binary Tree to Linked List
# 整体的想法就是当 node.right = node.left，prev in left-subtree prev.next = 原来的node.right
# 怎么实现没有想好。in order？ post order? 这里其实应该用post orde，why？错了，这题和什么顺序没有关系。
# 这题要想理解清楚，要知道代码做了什么。if l是关键，我们从leaf nod开始往上走
# 我们分别把左右两个子树变形成为linkedlist形状，那么如果左右子树多了怎么半？ 观察return，return的其实是node，是最右侧的node，或者说是应该和右子树链接的node

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

# 199. Binary Tree Right Side View
# we are going to use bfs
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root: return []
        res = []
        q = [root]
        while q:
            for  _ in range(len(q)):
                node = q.pop(0)
                if node.left: q.append(node.left)
                if node.right: q.append(node.right)
            res.append(node.val)
                
        return res



# 116. Populating Next Right Pointers in Each Node
# 我写的答案，有点繁琐
class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root: return None
        q = [root]
        while q:
            temp = []
            for j in range(len(q)):
                node = q[j]
                if node.left: temp.append(node.left)
                if node.right: temp.append(node.right)
            for i in range(1, len(q)):
                prev, curr = q[i-1], q[i]
                prev.next = curr
                
            q = temp
        return root
# solution的答案，虽然跟我的思路一样，但是针对首尾的处理比我好多了～
import collections 
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return root
        Q = collections.deque([root])
        while Q:
            size = len(Q)
            for i in range(size):
                node = Q.popleft()
                # 点睛之笔
                if i < size - 1:
                    node.next = Q[0]
                if node.left:
                    Q.append(node.left)
                if node.right:
                    Q.append(node.right)
        
        return root


# 515. Find Largest Value in Each Tree Row
class Solution(object):
    def largestValues(self, root):
        ans = []
        if root is None:
            return ans
        queue  = [root]
        while queue:
            # python的骚操作
            ans.append(max(x.val for x in queue))
            new_queue = []
            for node in queue:
                if node.left:
                    new_queue.append(node.left)
                if node.right:
                    new_queue.append(node.right)
            queue = new_queue
        return ans

# 周赛第三题
# 思路很巧妙，从头向后遍历，如果遇到多的情况下，就break，遍历过的，都是subarray
class Solution:
    def countDistinct(self, nums: List[int], k: int, p: int) -> int:
        s = set()
        n = len(nums)
        for i in range(n):
            t = 0
            z = []
            for j in range(i,n):
                if nums[j] % p ==0:
                    t += 1
                if t>k:
                    break
                z.append(nums[j])
                # 把list换成tuple，然后入set里面
                s.add(tuple(z))
        return len(s)

# 15. 3Sum
# 勉强算自己写，针对重复答案有两个控制集注意一下：一个是helper中的while，一个是main中的nums[i] != nums[i-1]
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        def helper(nums,i,res):
            lo, hi = i+1, len(nums) - 1
            while lo < hi:
                total = nums[i] + nums[lo] + nums[hi]
                if total == 0:
                    res.append([nums[i], nums[lo], nums[hi]])
                    lo += 1
                    hi -= 1
                    while lo<hi and nums[lo] == nums[lo-1]:
                        lo += 1
                elif total > 0:
                    hi -=1
                else:
                    lo += 1
        res = []
        nums.sort()
        for i in range(len(nums)):
            if nums[i] > 0: break
            if i == 0 or nums[i] != nums[i-1]:
                helper(nums, i, res)
        return res

# 2266. Count Number of Texts
class Solution:
    def countTexts(self, pressedKeys: str) -> int:
        MOD=pow(10,9)+7
        # dp多一位很重要
        dp=[0]*(len(pressedKeys)+1)
        dp[0]=1
        # 这个有点类似跳楼梯
        for i in range(1,len(pressedKeys)+1):
            # 从上一层继承，相当于只考虑本身这一位，case1。
            dp[i]=dp[i-1]%MOD
            # Case2，就是两个按键的情况
            if(i-2>=0 and pressedKeys[i-1]==pressedKeys[i-2]):
                dp[i]=(dp[i]+dp[i-2])%MOD
                # Case3，就是按三个键是什么情况
                if(i-3>=0 and pressedKeys[i-1]==pressedKeys[i-3]):
                    dp[i]=(dp[i]+dp[i-3])%MOD
                    # Case4, 如果按键是79的话，那么4个按键也可以接受哈哈哈！
                    if(pressedKeys[i-1] in "79" and i-4>=0 and pressedKeys[i-1]==pressedKeys[i-4]):
                        dp[i]=(dp[i]+dp[i-4])%MOD
        return dp[-1]



# 417. Pacific Atlantic Water Flow
class Solution:
    def pacificAtlantic(self, matrix: List[List[int]]) -> List[List[int]]:
        if not matrix or not matrix[0]: return []
        num_rows, num_cols = len(matrix), len(matrix[0])
        pacific_q = deque()
        atlantic_q = deque()
        
        # 把周围接壤的地方围上
        for i in range(num_rows):
            pacific_q.append((i,0))
            atlantic_q.append((i, num_cols-1))
        for i in range(num_cols):
            pacific_q.append((0, i))
            atlantic_q.append((num_rows - 1, i))
        
        def bfs(queue):
            reachable = set()
            while queue:
                (row, col) = queue.popleft()
                reachable.add((row, col))
                for (x, y) in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    new_row, new_col = row + x, col + y
                    if new_row < 0 or new_row >= num_rows or new_col < 0 or new_col >= num_cols:
                        continue
                    if (new_row, new_col) in reachable:
                        continue
                    if matrix[new_row][new_col] < matrix[row][col]:
                        continue
                    queue.append((new_row, new_col))
            return reachable
        
        pacific_reachable = bfs(pacific_q)
        atlantic_reachable = bfs(atlantic_q)
        
        # Find all cells that can reach both oceans, and convert to list
        return list(pacific_reachable.intersection(atlantic_reachable))


# 139. Word Break
"""
这一题其实是dfs，又有点像树的结构思维。很厉害。
难点在于理解dfs是如何通过递归思维解决这道题，其实都是针对某一段字符串看其能否满足一个特性。最终可以确保每一个ch都能满足。
"""

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        def dfs(s, wordDict, start):
            # base case就是我们看的start index已经==len(s)，如果能遍历到这一步，那肯定为True
            if start == len(s):
                return True
            # 这里很有趣，针对每一个开始的start，都去遍历所有，就是看所有的可能性。只要其中有一个可能性成功了，那就是return True
            # 如果没有成功，不会做什么，如果所有end循环都没有成功才会返回false
            # 这里的end要取到到len(s)，因为是要给切片器使用。
            for end in range(start + 1, len(s) + 1):
                if s[start:end] in wordDict and dfs(s, wordDict, end):
                    return True
            return False
        return dfs(s, set(wordDict), 0)
# time complexity -> 2^n // 一共n个元素，有n+1种方法切分两个list，2*2*2*2..... = 2**n 
# space complexity -> n // recursion tree depth


# 139. Word Break
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        word_set = set(wordDict)
        dp = [False] * (len(s) + 1)
        dp[0] = True
    
        # dp为什么要len(s) + 1，一般是初始化需要。index可能也需要
        # 这里两个for是，i是end_index，j是start_index；
        for i in range(1, len(s) + 1):
            for j in range(i):
                # 如果dp[j]能组成，s[j:i]也能组成，那么我们就可以说dp[i]是没问题的！
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        return dp[len(s)]
# Time complexity : O(n^3) There are two nested loops, and substring computation at each iteration. Overall that results in O(n^3)
# Space complexity : O(n). Length of p array is n+1.

# 62. Unique Paths
# 我自己写的暴力解法，太丑了！
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        count = 0
        def dfs(x,y):
            nonlocal count
            if x == m and y == n:
                count += 1
                return 
            if x > m or y > n:
                return 
            dfs(x+1,y)
            dfs(x,y+1)
        dfs(1,1)
        return count
# solution的暴力解法，只要接触到m==1||n==1那么只剩下一条路了。
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        if m == 1 or n == 1:
            return 1
        
        return self.uniquePaths(m - 1, n) + self.uniquePaths(m, n - 1)

# 想到用DP的话，说实在是比较困难的点。特别是当前格子的value是由其旁边两个格子决定的。
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        d = [[1] * n for _ in range(m)]
        # 虽然我们d都赋值了，但是只用到第一行和第一列。
        for col in range(1, m):
            for row in range(1, n):
                d[col][row] = d[col - 1][row] + d[col][row - 1]

        return d[m - 1][n - 1]



# 323. Number of Connected Components in an Undirected Graph

class Solution:
#     # DFS的思路好厉害呀，通过dict存值得到点与点的direct关系，就可以用dfs了，dfs遍历一次，result+=1
#     def countComponents(self, n, edges):                
#         def dfs(n, g, visited):
#             if visited[n]:
#                 return
#             visited[n] = 1
#             for x in g[n]:
#                 dfs(x, g, visited)
        
#         visited = [0] * n
#         g = {x:[] for x in range(n)}
#         for x, y in edges:
#             g[x].append(y)
#             g[y].append(x)
#         ret = 0
#         for i in range(n):
#             if not visited[i]:
#                 dfs(i, g, visited)
#                 ret += 1
#         return ret
        
    
#     # 全新用法
#     # E = numbers of edges, V = numbers of vertices.
#     # O(E+V), space is the same.
#     def countComponents(self, n, edges):
#         g = {x:[] for x in range(n)}
#         for x, y in edges:
#             g[x].append(y)
#             g[y].append(x)

#         ret = 0
#         for i in range(n):
#             queue = [i]
#             ret += 1 if i in g else 0
#             # 这里面queue是会变的，当我们第一次进入for的时候，queue只有一位，但是我们的操作会改变queue
#             # for循环是会考虑改变的，而非在最开始的就决定了循环几次。
#             # 那这个for就相当于把一个集群全部修改了。
#             for j in queue:
#                 if j in g:
#                     queue += g[j]
#                     del g[j]

#         return ret

    # E = numbers of edges, V = numbers of vertices.
    # O(E*a) a is union ; Space: V 
    # 这道题的思路是怎么样的。union的input是edge，然后操作parent。最后通过find parent中的所有node，看有几个root node，就相当于有几个group
    def countComponents(self, n, edges):
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(xy):
            x, y = list(map(find, xy))
            if rank[x] < rank[y]:
                parent[x] = y
            else:
                parent[y] = x
                if rank[x] == rank[y]:
                    rank[x] += 1

        parent, rank = list(range(n)), [0] * n
        # 跟下面是一样的效果。
        # for x, y in edges:
        #     union(x, y)
        # 如果直接写map(union, edges)，这是python2的用法；它会返回一个map iterator，无法达到我们更改parent的目的。
        list(map(union, edges))
        return len({find(x) for x in parent})

# 190. Reverse Bits
# 还是对位运算的不熟悉
class Solution:
    def reverseBits(self, n: int) -> int:
        result, power = 0, 31
        while n:
            # 因为input给的是32位，所以我们通过& 1 得到最后一位时，直接对result进行移位处理，直接reverse，这里的power很奇妙
            result += (n & 1) << power
            # &1取得最后一位，但是n还是原来的，因此需要做处理
            n = n >> 1
            # 下一次就需要变了
            power -= 1
        return result



# 20. Valid Parentheses
# 简单题。我的思路和答案的思路一样，但是我的代码好丑。
class Solution:
    def isValid(self, s: str) -> bool:
        m = {
            ')': '(',
            ']': '[',
            '}': '{'
        }
        stack = []
        # 优化点一：for char in s
        for i in range(len(s)):
            # 优化点二：if的逻辑没想有想清楚
                # 只要碰到了m里面的key，那肯定意味着需要去匹配了，如果这个时候没有匹配到，或者匹配失败，直接False就成了
            if not stack or s[i] in '([{'or stack[-1] != m[s[i]]:
                stack.append(s[i])
            else:
                stack.pop()
        
        return len(stack) == 0
    
    
class Solution(object):
    def isValid(self, s):    
        stack = []
        mapping = {")": "(", "}": "{", "]": "["}
        for char in s:
            if char in mapping:
                top_element = stack.pop() if stack else '#'
                if mapping[char] != top_element:
                    return False
            else:
                stack.append(char)
        return not stack


# 572. Subtree of Another Tree
# 这一题的t/f是通过return传递的，总是没有办法灵活运用。
class Solution:
    def isMatch(self, s, t):
        # 有一方为空的话，就去判断是否两方都为空。
        if not(s and t):
            return s is t
        # 然后去判断node值，和左右两边的！
        return (s.val == t.val and 
                self.isMatch(s.left, t.left) and 
                self.isMatch(s.right, t.right))
    
    # 这是主方程，也有base case
    def isSubtree(self, s, t):
        # 第一个if，就针对每个node，我们都进去看一下是否成立，如果成立，那就最好了！
        if self.isMatch(s, t): return True
        # 如果s没有了，那肯定是false呀
        if not s: return False
        # 如果本node不成立，那我们就看子node，看到什么地方呢？看到leaf，也就base case
        return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)
        
    
# 1143. Longest Common Subsequence
# 首先我们需要注意到dp是n+1；
class Solution:
    def longestCommonSubsequence(self, t1: str, t2: str) -> int:
        dp = [[0]*(len(t2)+1) for _ in range(len(t1)+1)]
        
        # 为什么我们需要dp.size = len()+1? 因为针对首行首列我们也需要进行状态转移。
        # 但是需要注意到状态转移index的转变。i+1，i-1, i的选择需要仔细考究
        for i in range(len(t1)):
            for j in range(len(t2)):
                # print(dp)
                if t1[i] == t2[j]:
                    dp[i+1][j+1] = 1 + dp[i][j]
                else:
                    dp[i+1][j+1] = max(dp[i+1][j],dp[i][j+1])
        return dp[-1][-1]
               
    

# 200. Number of Islands
# 经典题目可以用dfs/bfs/union find做
# 这一题的难点在于index和为什么只用两个if，因为我们的for循环顺序，然后只用合并下一个就成了。
class Solution(object):
    def numIslands(self, grid):    
        if len(grid) == 0: return 0
        row = len(grid), col = len(grid[0])
        self.count = sum(grid[i][j] == '1' for i in range(row) for j in range(col))
        # 把所有的岛屿加起来，后续如果合并的话，就-1，最终就是有多少独立的岛屿。
        parent = [i for i in range(row*col)]
        def find(x):
            if parent[x] != x:
                return find(parent[x])
            return parent[x]
        
        def union(x,y):
            xroot, yroot = find(x), find(y)
            if xroot == yroot: return
            parent[xroot] = yroot
            self.count -= 1
        
        
        for i in range(row):
            for j in range(col):
                print(parent)
                if grid[i][j] == '0':
                    continue
                index = i*col + j
                if j < col-1 and grid[i][j+1] == '1':
                    union(index, index+1)
                if i < row-1 and grid[i+1][j] == '1':
                    union(index, index+col)
        return self.count

class Solution:
    def numIslands(self, grid):
        if not grid:
            return 0
            
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    self.dfs(grid, i, j)
                    count += 1
        return count

    def dfs(self, grid, i, j):
        if i<0 or j<0 or i>=len(grid) or j>=len(grid[0]) or grid[i][j] != '1':
            return
        grid[i][j] = '#'
        self.dfs(grid, i+1, j)
        self.dfs(grid, i-1, j)
        self.dfs(grid, i, j+1)
        self.dfs(grid, i, j-1)

# 79. Word Search
# Take-away：虽然我们是针对每一个格子判断，只要有一个可能性满足就好了，那么在main function里面，用一个if就能达成这种效果！
class Solution(object):
    def exist(self, board, word):   
        if not board: return False
        for i in range(len(board)):
            for j in range(len(board[0])):
                if self.backt(board, i, j, word):
                    return True
        return False
            
    def backt(self, board, i, j, word):
        if len(word) == 0: return True
        if i<0 or i>=len(board) or j<0 or j>=len(board[0]) or word[0]!=board[i][j]:
            return False
        temp = board[i][j]
        board[i][j] = '#'
        
        res = self.backt(board, i+1, j, word[1:]) or self.backt(board, i-1, j, word[1:]) \
        or self.backt(board, i, j+1, word[1:]) or self.backt(board, i, j-1, word[1:])
        board[i][j] = temp
        return res
        
        
        
# 105. Construct Binary Tree from Preorder and Inorder Traversal
# 这题很有难度的！   
# 这题的集体关键是什么？
# preorder的顺序去recursion
# inorder的左右是他的sub-tree
# 理解了前面pre-order和in-order你还是没有办法很好的做题，因为有一个很重要的key在这，是left和right。
# pre_index很重要，虽然我们递归的顺序是pre-order 但是如果没有index，我们将无法得到值
# inorder也很重要，一个子树的左右两遍一定是它的子树，但是注意，由于我们是preorder递归的，左右两边的边界很重要。如果左右边界没有了，意味着该node将没有子树了，因此，这里的理解很重要。
# 那么怎么确保碰到leafnode的时候，left righ是刚好的呢？ 因为我们是从root下来的，碰到leaf时，该node的predecessor一定是处理过了，你好好考虑下。
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def helper(left, right):
            nonlocal pre_index
            if left > right:
                return None
            
            val = preorder[pre_index]
            root = TreeNode(val)
            pre_index += 1
            
            root.left = helper(left, in_map[val] - 1)
            root.right = helper(in_map[val]+1, right)
            
            return root
        pre_index = 0
        in_map = {}
        for i, v in enumerate(inorder):
            in_map[v] = i
        return helper(0, len(preorder) - 1)
        
        
        
# 269. Alien Dictionary
# 这一题的难点在于是topological，与题意的理解。
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        dic = {}
        indegree = {chr(x): 0 for x in range(ord('a'), ord('a') + 26)}
        
        # 初始化，dic中每个字母都用set() init一下
        for word in words:
            for ch in word:
                dic[ch] = set()
        # 开始遍历，得到level了，因为比较有不同之后，才知道优先级，才能确定level
        for i in range(len(words) - 1):
            word_1, word_2 = words[i], words[i + 1]
            # 开始iterate我们的w，挑选较小len的
            for j in range(min(len(word_1), len(word_2))):
                key_1, key_2 = word_1[j], word_2[j]
                # 如果不同就要有优先级顺序了
                if key_1 != key_2:
                    # k2是在k1后面的按照提议，如果dic里面没有的话，直接存入，并且indegree++；
                    if key_2 not in dic[key_1]:
                        dic[key_1].add(key_2)
                        indegree[key_2] += 1
                    # 碰到第一次不同后，之后的就不用比较了。
                    break
                # 如果j到尽头了，并且w1还能继续往下走，意味着w2不满足题意？-> 直接return “”
                elif j == min(len(word_1), len(word_2)) - 1 and len(word_1) > len(word_2):
                    return ""
        
        # 把0度的key入到queue里面，进行“bfs”
        queue = collections.deque([key for key in indegree if indegree[key] == 0 and key in dic])
        alien_dic = ''

        while queue:

            check = queue.popleft()
            alien_dic += check
            
            # 把当前char(check)的后续节点入度-1；如果发现有0的话，入q
            for ch in dic[check]:
                indegree[ch] -= 1
                if indegree[ch] == 0:
                    queue.append(ch)
        # 这个判断是怎么得出来的呢？并不是所有的element的indegree都hit到了0，因此一定存在冲突，比如本来a>b,肯定有一环b>a.所以错误！
        return alien_dic if len(alien_dic) == len(dic) else ''
       
        
        
        
# 230. Kth Smallest Element in a BST
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
# BFS的iterate的code挺有趣
    def kthSmallest(self, root, k):       
        stack = []        
        while True:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if not k:
                return root.val
            root = root.right


# 213. House Robber II
"""
Since House[1] and House[n] are adjacent, they cannot be robbed together. 
Therefore, the problem becomes to rob either House[1]-House[n-1] or House[2]-House[n], 
depending on which choice offers more money.
这是这道题的精髓
"""
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 0 or nums is None:
            return 0
        if len(nums) == 1:
            return nums[0]
        
        # 上面都是为了处理edge case 
        # 这里为什么用[1:], [:-1]
        return max(self.rob_simple(nums[:-1]), self.rob_simple(nums[1:]))

    # 利用两个variable而不是single list to store values
    # t1是偷，t2是不偷
    def rob_simple(self, nums: List[int]) -> int:
        t1 = 0
        t2 = 0
        for current in nums:
            t1, t2 = max(current + t2, t1), t1
        return t1

# 48. Rotate Image
# 旋转的图像大概就是从最外层开始变化，经历一次i，往内进一层
# j是指某层中具体哪个元素
# n-1类似长度->index; 二我们
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix[0])
        # rotate groups of four cells
        # 因为4条边相互牵制，因此只用遍历一半就好！但是需要n%2!外侧循环的意义是什么？是用来进不同的层！一圈圈的
        for i in range(n // 2 + n % 2):
            # 0到中点/左侧，因为4条边相互牵制，因此只用遍历一半就好！中点不用管，为什么？具体的坐标是由i和j一起定的！这个 %2放在哪里都可以，因为是正方形。
            for j in range(n // 2):
                # matrix[n - 1 - j][i], matrix[n - 1 - i][n - j - 1], matrix[j][n - 1 - i], matrix[i][j] = matrix[n - 1 - i][n - j - 1], matrix[j][n - 1 -i], matrix[i][j], matrix[n - 1 - j][i]
                tmp = matrix[n - 1 - j][i]
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - j - 1]
                matrix[n - 1 - i][n - j - 1] = matrix[j][n - 1 -i]
                matrix[j][n - 1 - i] = matrix[i][j]
                matrix[i][j] = tmp
# 这题理解起来还是有难度。不能将外层的i循环简单看作一层层。
# 而是首先要理解，坐标之间的相互关系，为什么n-1? 为什么有时候i在前面，有时候j在前面？
    # OK，n-1-i/j是为了获得坐标轴上相对称的坐标，前后交换是为了交换x/y的相对位置？
    # 为什么要获得对称？为什么要交换相对位置？这都是为了满足旋转后的坐标要求。
# 而我们提到的一层层进入，不能只看i，而是要看n-1与i/j的综合关系，相互作用下才能起到一层层进入的效果。
# 现在来解释为什么是n//2 + n % 2与 n//2，试想都取到最后一位是什么情况，就是图中的最中点，这个点我们是不用动的（odd）；
# 如果是even的情况，我的i还是要去到终点比如5中的3，但是j不用娶到5中的3，明白了吧。



# 295. Find Median from Data Stream
from heapq import *
class MedianFinder:
    def __init__(self):
        self.small = []  # the smaller half of the list, max heap (invert min-heap)
        self.large = []  # the larger half of the list, min heap

    def addNum(self, num):
        if len(self.small) == len(self.large):
            # 如果small = large，我们往large里面添加数据，但是我们需要确保中位数左右大小是一致的。
            # 因此，先把num->push->pop 到small里，然后取出来的数据此时是heap最小的，因为是负数，所以取反最大。
            heappush(self.large, -heappushpop(self.small, -num))
        else:
            heappush(self.small, -heappushpop(self.large, num))

    def findMedian(self):
        if len(self.small) == len(self.large):
            return float(self.large[0] - self.small[0]) / 2.0
        else:
            return float(self.large[0])


# 125. Valid Palindrome
class Solution:
    def isPalindrome(self, s: str) -> bool:
        i, j = 0, len(s)-1
        while i < j:
            # isalum只会判断数字和char
            while i < j and not s[i].isalnum():
                i += 1
            while i < j and not s[j].isalnum():
                j -= 1
            
            if s[i].lower() != s[j].lower():
                return False
            i += 1
            j -= 1
        return True
        

# 261. Graph Valid Tree
# 这一题的python技巧厉害呀！
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        parent = list(range(n))
        def find(x):
            if parent[x] == x:
                return x
            return find(parent[x])
        def union(xy):
            x, y = map(find, xy)
            parent[x] = y
            # 我们本是可以不用搞这个的。但是做了可以帮忙判别一下。
            # 如果是x != y 就是acyclic无环
            return x != y
        # 首先无环的话edges一定是n-1，并且判断了下每条边在union的时候是否碰到了自己。
        return len(edges) == n-1 and all(map(union, edges))
        
    # 这个DFS真是绝了，一般我们会构造图，然后seen记录，看是否存在环
    # 首先node的数量和edge的数量这对于无环来说是一个很强的约束；那我们大费周章，给那么多visit是干什么用的？是用来确保node是连接起来的，这是构成树的关键！
    def validTree_DFS(self, n, edges):
        if len(edges) != n - 1:
            return False
        neighbors = {i: [] for i in range(n)}
        for v, w in edges:
            neighbors[v] += w,
            neighbors[w] += v,
        def visit(v):
            map(visit, neighbors.pop(v, []))
        visit(0)
        return not neighbors

    # BFS
    queue = [0]
    for v in queue:
        queue += neighbors.pop(v, [])


# 133. Clone Graph

"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

class Solution:
    def __init__(self):
        self.visited = {}
    
    # 无论是最终的return/edge case/special case都是return node
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return node
        if node in self.visited:
            return self.visited[node]
        # copy一个node出来
        clone_node = Node(node.val, [])
        # 把当前node放在visited里面，value就是它的copyNode
        self.visited[node] = clone_node
        
        # 如果node有neighbors的话，那么我们就将其recurse掉。
        # 这一题的逻辑很厉害呀。
        if node.neighbors:
            clone_node.neighbors = [self.cloneGraph(n) for n in node.neighbors]
        return clone_node

# 377. Combination Sum IV
# 背包掌握不好耶，针对每一层target的情况，我们的num都可以更新一次
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [0 for _ in range(target+1)]
        dp[0] = 1
        
        for t in range(target + 1):
            for num in nums:
                if t-num >= 0:
                    dp[t] += dp[t-num]
        return dp[-1]
# 如果是1/0背包呢？就要考虑i-1了，

class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        # potential optimization
        # nums.sort()

        @functools.lru_cache(maxsize = None)
        def combs(remain):
            if remain == 0:
                return 1

            result = 0
            # 如果还有remain的话，去看当前层的所有可能元素，因此可以满足穷尽。
            for num in nums:
                if remain - num >= 0:
                    result += combs(remain - num)
                # potential optimization
                # else:
                #     break

            return result

        return combs(target)

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
            if not node:
                return 0 
            left  = max(0, helper(node.left)) 
            right = max(0, helper(node.right))
            path_sum = left + right + node.val
            target = max(target, path_sum)
            return node.val + max(left, right)
            
        target = float('-inf')
        helper(root)
        return target


# 这题的思路是check each bit；
# combination无所谓顺序，当前位有1就成。
# 首先确认bg: bitwise AND操作的话只要位上有1就可以满足>0， 否则 == 0；
class Solution:
    def largestCombination(self, arr: List[int]) -> int:
        # 2 ** 30 > 10^7； 其实这题24就成。
        # cnt存的是所有的数字有多少位是1
        cnt = [0] * 30
        
        for x in arr:
            for i in range(30):
                if x & (1 << i):
                    # 如果i位上是1，我们就在cnt里记录下来。
                    cnt[i] += 1
        return max(cnt)

    