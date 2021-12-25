# 283 Move zeros 移动零
# 这一题的难度在于case的不同，有可能只有一个0，也有可能连续的0，更有可能都不是0，因此你的算法需要cover all posibilities。
# 我的想法是，每一次遍历到非0的数，让他移动到list的前面，但是这个想法不太成熟，会导致冲突（因为在原list上直接修改，还要记录0的位置，所以很麻烦），最终修代码修成狗屎。
# 答案思路：利用两个指针，第一个指针永远指着第一位0，第二个指针用于遍历数组。有点类似quick sort，将比0小的数直接放在左边
from typing import List


class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        if not nums:
            return 0
        j = 0 
        for i in xrange(len(nums)): #xrange和range不同的是，xrange生成的是生成器，range生成的是list，如果大数据录入情况下，前者可能性能更好，但是leetcode不支持，需要引包
            # 非0也可以直接用这种boolean判断，长见识了
            if nums[i]:
                # 如果没有遇到0的时候，一直进来，含义为：自身交换；
                # 如果遇到0了进来，那么就是0与其交换
                nums[i], nums[j] = nums[j], nums[i]
                # j是指向第一位0的，如果遇到0，那么j与i之间就会有差，就可以达到0与非0的交换，而非自身交换
                j += 1
# 好巧妙， 一般的做法会是两次遍历，第一次遍历填补非0，并且记录0的个数，第二次遍历直接将list后面几位变为0
class Solution(object):
	def moveZeroes(self, nums):
		if not nums:
			return 0
		# 第一次遍历的时候，j指针记录非0的个数，只要是非0的统统都赋给nums[j]	
		j = 0
		for i in xrange(len(nums)): 
			if nums[i]:
				nums[j] = nums[i]
				j += 1
		# 非0元素统计完了，剩下的都是0了
		# 所以第二次遍历把末尾的元素都赋为0即可
		for i in xrange(j,len(nums)):
			nums[i] = 0
# 🌟Take-away：for循环 + if条件判断 -> 实现j与i的隔离 -> 原数组满足条件的replace



# 566 Reshape the Matrix 重塑矩阵
# 这题难点在于思想，重塑矩阵元素的位置前后变化其实是有特定的映射关系的，找到这个就可以找到解题方法了。
class Solution:
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        m = len(mat)
        n = len(mat[0])
        if m * n != r * c:
            return mat
        
        # 先把答案构造出来
        res = [[0] * c for _ in range(r)]
        for x in range(m * n): 
            # 🌟这个映射关系是关键
            res[x // c][x % c] = mat[x // n][x % n]
        return res


# 485 Max consecutive ones 最大连续1的个数
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        temp = 0
        res = 0
        for num in nums:
            if num == 1:
                temp +=  1
            else: 
                temp = 0
            # 🌟这里的max判断不能放在if语句里面，因为如果遇不到非1的数，那么就没有办法把当前的temp转移到res中
            # 如果要放，那么还要在for循环外面添加一次这个语句
            
            res = max(temp, res)
        return res
#复杂度为n, 遍历一次

class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        temp = 0
        res = 0
        for num in nums:
            if num == 1:
                temp +=  1
            else: 
                res = max(temp, res)
                temp = 0
        res = max(temp, res)
        return res



# 240 Search a 2D Matrix II
# 1. 暴力查找(复杂度为mn) 2.二分查找（因为每一行都是升序排列） 3.抽象BSt
# 我的思路是第三种，也是最优的算法，但是究竟怎么写，我不太清楚
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        for row in matrix:
            for element in row:
                if element == target:
                    return True
        return False

# 用了API
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        for row in matrix:
            # 在row行，用二分查找target，这是个API
            index = bisect.bisect_left(row, target)
            if index < len(row) and row[index] == target:
                return True
        return False

# 原生二分  （复杂度m * log n)
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        for i in range(n):
            l, r = 0, m - 1
            while l < r:
                # 这里还能用mid = l + r + 1 >> 1
                # 这是二进制下的位运算，相当于✖️2 或者 ➗2
                mid = (l + r + 1) // 2
                # 🌟为什么我们要+1呢？ 这是最关键的，因为我们想让right指的是我们能取到的值，而且+1 可以帮助我们上一位而不是下移一位。
                if matrix[mid][i] <= target:
                    l = mid
                else:
                    r = mid - 1
            if matrix[r][i] == target:
                return True
        return False

# 模拟BST
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        row, col = 0, n - 1
        while row < m and col >= 0:
            if matrix[row][col] < target: 
                row += 1
            elif matrix[row][col] > target:
                col -= 1
            else:
                return True
        return False

# 妙呀！从图的右上角出发，如果大了，对应的就是row，小了对应的就是col！
# 那么可以从[0, 0]出发么？可以是可以，但是逻辑会比较复杂吧。要写两个while相当于，太麻烦。因为数字小了，你不知道该去弥补横轴还是纵轴。



# 74 
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        row, col = 0, n - 1
        while row < m and col >= 0:
            if matrix[row][col] < target: 
                row += 1
            elif matrix[row][col] > target:
                col -= 1
            else:
                return True
        return False
# 与240题目解法一样




# 378 Kth Smallest element in a Sorwted Matrix
# 直接排序/归并排序/二分查找（因为纵向是有序的）
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        rec = sorted(sum(matrix, []))  #sum的奇技淫巧，可以用来拼接[]
        return rec[k - 1]
# 时间复杂度为n2 logn 即为n2个数进行排序； 空间复杂度为n

# 归并排序 （时间：klogn 归并k次，然后排序是logn； 空间为n）
# 首先核心思想就是读懂题，然后往右看，一次比较n排的首位，相当于对n排的首位进行排序，所以是归并排序
# 实现有序链表的归并我们可以用到heapq，就是用来添加/去除，然后得到想要的堆顶。
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        n = len(matrix)
        # priority queue
        # 添加每一行的队首的人
        pq = [(matrix[i][0], i, 0) for i in range(n)]

        # 创建一个heap
        heapq.heapify(pq)

        ret = 0
        for i in range(k - 1):
            num, x, y = heapq.heappop(pq) #弹出顶值
            if y != n - 1: # 用来判断一行是否被弹完
                heapq.heappush(pq, (matrix[x][y + 1], x, y + 1)) #加入heap值
        
        return heapq.heappop(pq)[0] #执行道第k-1次停下， 那么k次pop的数字就是我们要的数字

#但注意，这一题忽略了升序这个题目中的要求，因此使用方法三二分更好。

#二分🌟🌟🌟
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        n = len(matrix)

        # 返回我们给到的mid值，处于k的右边，即还有超过k的数比mid小，因此mid不是我们要找的值
        def check(mid):
            i, j = n - 1, 0
            num = 0
            while i >= 0 and j < n:
                if matrix[i][j] <= mid:
                    num += i + 1
                    j += 1
                else:
                    i -= 1
            return num >= k

        left, right = matrix[0][0], matrix[-1][-1]
        while left < right:
            mid = (left + right) // 2
            if check(mid):
                # 如果true的话，right = mid就好，因为mid这个时候有可能是我们要的值
                right = mid
            else:
                # 此时mid肯定不是我们的值，所以大胆+1
                left = mid + 1 
            # 最终停下来肯定是left == right了，因为每次都只
        return left
# 🌟该题的思路很巧妙：在本题矩阵（仅仅没行递增，行头递增）中我们清楚：可以把矩阵中的所有值看成一个list
# >>> 那么其中肯定有mid，我们可以使用二分去查找我们的值。此时单个list与矩阵的映射关系比较难想到。
# >>> 这里应该考虑的是，我们找到的mid是否应该继续二分？ 答案是利用了一个check helper method。
# >>> 通过判断是否还有比mid小足够多的数，来判断mid是否是我们需要找的。
# 那么如何判断是否有足够小的？想象矩阵为一个平面，你其实可以画出来一个线，先左边的全部小于特定值！
# 本题就是那么奇妙

"""     关于二分的left 和 right
1. 
    mid = (left + right) // 2 容易溢出
    mid = left + (right - left) // 2 不容易溢出
    但其实在python当中都还好，因为//2一定是降位的。不能这么理解 
    ->left+right 当left和right都很大的时候，可能会造成越界。

2.  🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟
    left < right : left = mid + 1; right = mid      #最后应该return 
    left <= right: left = mid + 1; right = mid - 1  #最后应该return mid
    # 同时需要考虑mid是否要被纳入下一次二分搜索的范围内
    # 没有等号的终止条件是 left == right      即区间只有一个元素left=right, 此时还需要在while循环外再判断是否这一个元素满足题意
    # 有等号的种植条件是  left == right + 1  即区间没有元素，条件判断直接在while中
    
3. 我的疑问是在于left最后是大于还是等于right？
    上述搜索区间已经帮忙阐述过了。

4. 二分的寻找左右侧边界会有点复杂，具体遇到再进行操练。
"""

# 下面代码是用来求出kth smallest number，而且这个number是去重的
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        m, n = len(matrix), len(matrix[0])
        last = matrix[0][0]
        if k == 1: return last
        count = 1
        for r in range(m):
            for c in range(n):
                if matrix[r][c] == last:
                    continue
                else:
                    last = matrix[r][c]
                    count += 1
                if count == k:
                    return matrix[r][c]









# 23 Merge K Sorted Lists 合并k个升序链表(困难) 

# from queue import PriorityQueue
# class Solution:
#     def mergeKLists(self, lists: List[ListNode]) -> ListNode:
#         head = point = ListNode(0)
#         q = PriorityQueue()
#         for list in lists:
#             if list:
#                 q.put((list.val, list))
#         while not q.empty():
#             val, node = q.get()
#             point.next = ListNode(val)
#             point = point.next
#             node = node.next
#             if node:
#                 q.put((node.val, node))
#         return head.next
"""上述用了priority queue模块，事实证明不好用！"""
# K指针：K 个指针分别指向 K 条链表，然后比较头节点
# 复杂度：时间（NK：N为每个链表中的节点数，有K个链表）空间（1+N：新建了一个链表，1是遍历原来的list）
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        # 新建Linked-List, 两个节点，一个用于最后的答案展示，一个用于处理逻辑深入
        tail = dummyHead = ListNode(0)
        # 比较k个首节点，每一次        
        k = len(lists)

        while True:
            # minNode是每次队首比较的最小节点
            minNode = None
            # minPointer是指向第几个队列是当前循环最小值
            minPointer = -1 
            # 比较队首，比较n次
            for i in range(k):
                # 如果该列没有了，继续比较其他列
                if not lists[i]:
                    continue
                # ==None是为了判断首次，或者如果当前遍历的list小于我们的minNode，那么将其更新
                if minNode == None or lists[i].val < minNode.val:
                    minNode = lists[i]
                    minPointer = i
            # 最后minPointer肯定为-1，为什么呢？因为遍历到最后没有元素后是没办法进入for循环给变量赋值的，所以就是默认值，所以break
            if minPointer == -1:
                break
            # 每一次找到一个节点就更新答案tail，并且更新原来列表中的顺序
            tail.next = minNode
            tail = tail.next
            lists[minPointer] = lists[minPointer].next
        return dummyHead.next

# 两两合并
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists:
            return None
        res = None
        for listi in lists:
            res = self.mergeTwoLists(res, listi)
        return res
    
    def mergeTwoLists(self, list1: ListNode, list2:ListNode):
        dummyHead = move = ListNode(0)
        while list1 and list2:
            if list1.val < list2.val:
                move.next = list1
                list1 = list1.next
            else:
                move.next = list2
                list2 = list2.val
            move = move.next
        move.next = list1 if not list2 else list2
        return dummyHead.next


# 归并排序
class Solution:
    """第一个方法：需要传参给MergeSort"""
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists: return None
        n = len(lists) #记录子链表数量
        return self.mergeSort(lists, 0, n - 1) #调用归并排序函数

    """Merge Sort的Main method，将l,r按照分治的方法处理，分别将左右拆分，然后放入最终"""
    def mergeSort(self, lists: List[ListNode], l: int, r: int) -> ListNode:
        if l == r:
            return lists[l]
        m = (l + r) // 2 
        L = self.mergeSort(lists, l, m) #循环的递归部分 # 递归就是拆分，颗粒度为每一个子链表
        R = self.mergeSort(lists, m + 1, r)
        return self.mergeTwoLists(L, R) #调用两链表合并函数 # 每一次我们mergesort返回的是排序后的一个链表
    """方法三：实现链表排序"""
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummy = ListNode(0) #构造虚节点
        move = dummy #设置移动节点等于虚节点
        while l1 and l2: #都不空时
            if l1.val < l2.val:
                move.next = l1 #移动节点指向数小的链表
                l1 = l1.next
            else:
                move.next = l2
                l2 = l2.next
            move = move.next
        move.next = l1 if l1 else l2 #连接后续非空链表
        return dummy.next #虚节点仍在开头

# 最小堆/小根堆
# 调包排序 -> 无聊: 首先将所有值放在heap里，然后弹出。
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        import heapq
        minHeap = []
        for listi in lists:
            while listi:
                heapq.heappush(minHeap, listi.val)
                listi = listi.next
        move = dummy = ListNode(0)
        while minHeap:
            move.next = ListNode(heapq.heappop(minHeap))
            move = move.next
        return dummy.next


# 645 Set Mismatch 错误的集合
# 4个解法：暴力、数学求差、计数、桶排序
# 暴力：无法通过全部case，难得管了 复杂度时间为N
class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:
        nums.sort()
        left, right = 0, 1
        while nums[left] != nums[right]:
            left += 1
            right += 1
        return [left+1, left+2] 


# Hash计数 复杂度：时间N， 空间N

class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:
        n = len(nums)
        cnts = collections.Counter(nums) #直接用这个也行
        ans = [0, 0]
        for i in range(1, n+1):
            if not cnts[i]:
                ans[1] = i
            if cnts[i] == 2:
                ans[0] = i
        return ans

class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:
        n = len(nums)
        cnts = [0 for _ in range(n + 1)] #命名方式
        for num in nums:
            cnts[num] += 1
        ans = [0,0]    #命名方式
        for i in range(n + 1):
            if cnts[i] == 0: 
                ans[1] = i
            if cnts[i] == 2:
                ans[0] = i
        return ans

# 数学求差 复杂度时间空间都为N，空间是因为都sum了么？
class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:
        n = len(nums)
        sum_set = sum(set(nums)) #去重后的求和
        tot = ( n * (n + 1) ) >> 1  #利用等差数列的求和公式
        # 前者重复的项，后者缺失的项，秒啊！
        return [sum(nums) - sum_set, tot - sum_set] 

# 桶排序 时间复杂度n，空间复杂度为1，因为没有新建什么东西
# 这个可以不用看
class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:
        # 交换nums中的i和j项
        def swap(nums, i, j):
            tmp = nums[i]
            nums[i] = nums[j]
            nums[j] = tmp
        
        # 遍历一遍，将顺序还原成递增
        n = len(nums)
        for i in range(n):
            while nums[i] != i + 1 and nums[nums[i]-1]!=nums[i]:
                swap(nums, i, nums[i] - 1)

        a = b = -1
        for i in range(n):
            if nums[i] != i + 1:
                a = nums[i]
                if not i:
                    b = 1
                else:
                    b = nums[i-1]+1
        return [a,b]


# 287 Find the duplicate number 寻找重复数
# 二分 特殊用法
class Solution:
    def findDuplicate(self, nums):
        n = len(nums)
        left = 1
        right = n - 1
        
        # 首先本题明确了取值范围为1~N，数组里有N+1个数
        while left < right:
            # 根据取值范围获得mid
            mid = left + (right - left) // 2 
            cnt = 0
            for num in nums:
                if num <= mid:
                    cnt += 1
            
            # cnt的妙用。如果小于mid的数要比mid大，也就是说在1～mid区间里有重复的数字！
            # 抽屉原理：放10个苹果在9个抽屉里，至少有一个抽屉的苹果量至少为2
        
            if cnt > mid:
                right = mid
            # 如果cnt <= mid 意味着是后半部分有问题，=mid意味着mid没问题
            else: 
                left = mid + 1
        return left

# 快慢指针
# 预备知识：如果数组中存在重复的数，如果将index与nums[i]建立映射关系的话，就可以发现一个环
# 利用快慢指针，最终会在快慢指针出处汇合。
class Solution:
    def findDulicate(self, nums):
        slow, fast = 0, 0
        slow = nums[slow]           #nums[0]
        fast = nums[nums[fast]]     #nums[nums[0]]

        # 🌟本题数据限制比较特殊，所以可以利用二分/快慢指针，如果从0开始就很有可能快慢指针就不能用了！但本题是从1～n，所以没关系，可以用！
        # 通过while建立映射关系
        while slow != fast:
            slow = nums[slow]           #因为nums的index与val相差1，所以每一次slow映射时都可以前进一位，这种前进跟原本nums的顺序无关
            fast = nums[nums[fast]]     #如同slow一样，只不过每一次跳2次 .next.next
        
        # pre1从初始位置出发，pre2与slow指针一起出发
        """
        起点到环的入口长度为m，环的周长为c，在fast和slow相遇时slow走了n步。则fast走了2n步，fast比slow多走了n步，而这n步全用在了在环里循环（n%c==0）。
        当fast和last相遇之后，我们设置第三个指针finder，它从起点开始和slow(在fast和slow相遇处)同步前进，
        当finder和slow相遇时，就是在环的入口处相遇，也就是重复的那个数字相遇。

                                        *** 为什么 finder 和 slow 相遇在入口? *** 🌟好牛逼的数学证明
        fast 和 slow 相遇时，slow 在环中行进的距离是n-m，其中 n%c==0，可以推算出来。这时我们再让 slow 前进 m 步——也就是在环中走了 n 步了。
        而 n%c==0 即 slow 在环里面走的距离是环的周长的整数倍，就回到了环的入口了，而入口就是重复的数字。
        我们不知道起点到入口的长度m，所以弄个 finder 和 slow 一起走，他们必定会在入口处相遇。
        """
        pre1 = 0
        pre2 = slow
        while pre1 != pre2:
            pre1 = nums[pre1]
            pre2 = nums[pre2]
        return pre1



# 667 Beautiful Arrangement II 优美的排列II
# 题意理解：1～n的list，要求相邻两元素差有k个不同的值，那么在不清楚有多少位n的时候，我们需要格外控制“不同的值”这一变量
# 那么我们可以利用等差为1的等差数列，去控制这个值只有一个。
class Solution:
  def constructArray(self, n: int, k: int) -> List[int]:

    res= [0 for _ in range(n)]
    # 首先construct 等差数列Arithmetic sequence
    # n-k-1 ~ n-1 有 (n-k-1 -n+1 +1) = k+1位数，但是不要紧，在下面的循环中(n-k-1)还是会和前面的list一样属于等差数列。
    for i in range(n - k - 1):
      res[i] = i + 1
    
    # 如果一个sorted list要满足题意，只需要按照有规律的排序就好了。
    j = 0 
    left = n - k
    right = n
    for i in range(n - k - 1, n):
      if j % 2 == 0:    #区分基数偶数
        res[i] = left
        left += 1
      else:
        res[i] = right
        right -= 1
      j += 1
    return res
# 复杂度为n
"""
首先，写到n-k-1位数字，但index=n-k-1并没有写；留下了n-k-1~n-1一共k+1个数字，共有k个差。有两个额外情况需要考虑。
1. k个差 + 原先的队列差为1，最后有k+1个值，怎么求解？ --> 最后一对数是相同的，差也为1，解释在下面；
2. 在最开始的时候，第一位跟之前的队列也会有差，怎么弄？那就保证n-k-1这个位置与之前的差也为1就好
[总结] 在排除上面两个点后，差值可以刚好有k个不同的值
[解释] 插排n个数即[1,n,2,n-1...]则k为n-1，且最后一对的差值为1。 则可以利用这个性质先顺排再插排来构造，其中插排需要的元素个数为k+1即可构造k，其余的n-(k+1)的数顺排。
[反思] 我自己写过一版本，主要是纠结在index上，也没有办法跑过case，主要的原因是在于没有完全理解插排的过程
"""


# 697 Degree of Array 数组的度
# 我自己的思路是两个method：找到degree的num，然后去求最小的index diff
# 别人的思路：利用两个dict/hash去记录一个数字的首尾index，然后找到degree的数字，去求最小的diff.
class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        left, right = dict(), dict()
        res = len(nums)
        counter = collections.Counter(nums) 
        for i, num in enumerate(nums):
            if num not in left:
                left[num] = i
            right[num] = i
            
        degree = max(counter.values())
        for k, v in counter.items():
            if v == degree:
                res = min(res, right[k] - left[k] + 1)
        return res
# 时间复杂度为n，因为要遍历所有元素
# 空间复杂度为n，因为最差情况下，要为每一个num都要新建
"""
API: 
collections.Counter()
enumerate()
.values()
.keys()
.items()
"""



# 766 Toeplitz Matrix 
# 因为是matrix，所以每一行用切片器，然后对角线去下一行去进行比对！
class Solution:
    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        n = len(matrix[0])
        for i in range(len(matrix) - 1):
            if matrix[i][0 : n-1] != matrix[i+1][1:n]:
                return False
        return True

#复杂度为n，空间为1，毕竟没有new

# 565 Array Nesting 数组嵌套 #permutation 排列/组合/置换
# 理解题意最重要，这一题是看每一个element可以组成的list，而非只有从头开始。并且从题意当中可以得知，最终一定是个环！每一个elemet都是某个环的一部分。
# 搞清楚这一题的数据结构后，就很简单啦！这一题的众多element都会组成自己的环，找出最长的那个环就行了。
class Solution:
    def arrayNesting(self, nums: List[int]) -> int:
        res = 0 
        for i in range(len(nums)):
            # 如果==-1，意味着我们已经遍历过该ele，那么就不用再遍历这个了。
            if nums[i] == -1:
                continue

            temp = 1
            path_i = i
            
            # 我们下一个将要遍历的index 不等于我们的开头进来的index
            while nums[path_i] != i:
                # 遍历过的元素更新，然后更新path_i
                nums[path_i], path_i = -1, nums[path_i]
                # 满足题意之后temp + 1
                temp += 1
            nums[path_i] = -1
            # 因为所有element都有自己属于自己的环，因此只用遍历一次，而且只用取temp的最大值就可以了！
            res = max(temp, res)
        return res
# 弄清楚你在打交道的数据结构很重要。这一题也可以通过维护一个visit[]判断是否已经遍历过

# 769 Max chunks to make sorted
# 暴力解法、复杂度为n
# 这一题就是找规律，如果最后的一位数是已经遍历过的最大数，就可以split，然后ans+1.
class Solution:
    def maxChunksToSorted(self, arr: List[int]) -> int:
        ma = ans = 0
        for i, num in enumerate(arr):
            ma = max(ma, num)
            if ma == i: ans += 1
        return ans


