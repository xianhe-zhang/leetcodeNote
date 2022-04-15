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