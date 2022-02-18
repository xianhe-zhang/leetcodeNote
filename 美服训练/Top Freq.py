# 206 反转链表
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        pre = None
        cur = head
        while cur:
            ntemp = cur.next    # ntemp为cur next指针指向的数字
            cur.next = pre      # cur的指针转向
            pre = cur           # pre指向cur
            cur = ntemp         # 现在的cur下一位迭代的数字
        return pre

# 现在看也还是好优美呀，递归的方法
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next: return head
        p = self.reverseList(head.next)
        # 通过上面两行就可以从最后一位开始进行

        head.next.next = head # head下一位的指针指向自己，
        head.next = None      # 取消自己的指针
        return p              # p这里也好美，每一次处理完递归的结果。
"""
反转链表的解法：通过callback self先行进入递归，然后处理逻辑在后面，表明从后往前走。漂亮
"""

#  Linked List Cycle
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        visited = set()
        
        while head:
            if head in visited:
                return True
            visited.add(head)
            head = head.next
        return False

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        # 不要忘记极限情况
        if not head: return False
        slow = head
        fast = head.next
        while slow != fast: # 直接用限制条件
            if not fast or not fast.next:
                return False
            slow = slow.next
            fast = fast.next.next
        return True

# 83 Remove Duplicates from Sorted List
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head: return None
        cur = head
        while cur.next is not None:
            if cur.val == cur.next.val:
                cur.next  = cur.next.next
            else: 
                cur = cur.next
        return head
# 重点：is not None和 != None一样
# 在python中 None, False, 空字符串"", 0, 空列表[], 空字典{}, 空元组()都相当于False;
"""
is, is not 对比的是两个变量的内存地址
==, != 对比的是两个变量的值
比较的两个变量，指向的都是地址不可变的类型（str等），那么is，is not 和 ==，!= 是完全等价的。
对比的两个变量，指向的是地址可变的类型（list，dict，tuple等），则两者是有区别的。
总结：最好不要使用not这种逻辑判断，而使用is not none/ != None
"""

          
# 234. Palindrome Linked List
# 把链表转化成list，然后判断回文，记得有很好的api可以用，不需要用双指针
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        vals = []
        cur = head
        while cur:
            vals.append(cur.val)
            cur = cur.next
        return vals == vals[::-1]        
# 如果想要链表倒叙，那么需要利用到recursion
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        self.front = head
        def helper(cur = head): # 全新的传参方式
            if cur:
                # 这个确保如果之后的递归中之要有一个不相等，不满足 -> 都直接false一路传回去
                if not helper(cur.next):    
                    return False
                if self.front.val != cur.val:
                    return False
                self.front = self.front.next
            # 如果最后所有的递归都跑完没问题，一定是要return True的
            return True
        return helper()


# 203. Remove Linked List Elements
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
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
# 如何创建出落差，dummy不直接等于head，而是声明一个List


# 237 delete a node 笑死这道题，不用看
class Solution:
    def deleteNode(self, node):
        node.val = node.next.val
        node.next = node.next.next


# 876 Middle of the Linked List
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

# 变态写法，真无语
class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        arr = [head]
        while arr[-1].next:
            arr.append(arr[-1].next)
        return arr[len(arr) // 2]

# 240. Search a 2D Matrix II
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # an empty matrix obviously does not contain `target`
        if not matrix:
            return False

        def search_rec(left, up, right, down):
            # this submatrix has no height or no width.
            if left > right or up > down:
                return False
            # `target` is already larger than the largest element or smaller
            # than the smallest element in this submatrix.
            elif target < matrix[up][left] or target > matrix[down][right]:
                return False

            mid = left + (right-left) // 2

            # Locate `row` such that matrix[row-1][mid] < target < matrix[row][mid]
            row = up
            while row <= down and matrix[row][mid] <= target:
                if matrix[row][mid] == target:
                    return True
                row += 1
            
            return search_rec(left, row, mid - 1, down) or \
                   search_rec(mid + 1, up, right, row - 1)

        return search_rec(0, 0, len(matrix[0]) - 1, len(matrix) - 1)


# 84. Largest Rectangle in Histogram
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        def calculateArea(heights: List[int], start: int, end: int) -> int:
            if start > end:
                return 0
            min_index = start
            for i in range(start, end + 1):
                if heights[min_index] > heights[i]:
                    min_index = i
            return max(
                heights[min_index] * (end - start + 1),
                calculateArea(heights, start, min_index - 1),
                calculateArea(heights, min_index + 1, end),
            )

        return calculateArea(heights, 0, len(heights) - 1)
# 这一题的分治写的好经典呀。


# 287. Find the Duplicate Number
# 这一题也挺不错的
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # 'low' and 'high' represent the range of values of the target
        low = 1
        high = len(nums) - 1
        
        while low <= high:
            cur = (low + high) // 2
            count = 0

            # Count how many numbers are less than or equal to 'cur'
            count = sum(num <= cur for num in nums)
            if count > cur:
                duplicate = cur
                high = cur - 1
            else:
                low = cur + 1
        return duplicate
# 抽象的二分查找，利用了数字出现次数的特性
# 时间 nlogn 空间1

# 92. Reverse Linked List II
# 第一种的方式是递归，所以这一题找left，right的在递归中的操作时亮点
class Solution:
    def reverseBetween(self, head, m, n):
        if not head:
            return None
        left, right = head, head
        stop = False
        
        def recurseAndReverse(right, m, n):
            nonlocal left, stop             #left，stop直接放在外部的作用域context进行更改。
            if n == 1:
                return 
            
            right = right.next
            
            if m > 1:                       # 通过if判断是否继续处理左边的
                left = left.next
            
            recurseAndReverse(right, m - 1, n - 1)  #此时已经全部递归完成，然后继续接下来的操作，之前我的思路是迭代的思路。

            if left == right or right.next == left:
                stop = True
               
            if not stop: 
                left.val, right.val = right.val, left.val
                left = left.next           

        recurseAndReverse(right, m, n)
        return head
# 这一题的递归还有一点很厉害的是什么？我们的right最开始是指向最右边的，因为递归完，但是我们递归是倒叙处理的，因此right会一次向左移动
# 这时，有一个问题，就是我们的left也会依次向左移动呀，那怎么办？🌟亮点就是我们的left不是该函数的作用域，而是全局的作用域，而且每一次交换后，left就next一次！完美！
# 复杂度都是O(N)

#下面介绍iterative的方法，跟recursion的方法还不太一样，recursion是直接swap值，而迭代是一个个改指针...
class Solution:
    def reverseBetween(self, head, m, n):
        if not head:
            return None

        cur, prev = head, None
        while m > 1:
            prev = cur
            cur = cur.next
            m, n = m - 1, n - 1

        # tail是处理前链表的第一个node，con前面未处理链表的结尾
        tail, con = cur, prev

        while n:
            tempNode = cur.next
            cur.next = prev
            prev = cur
            cur = tempNode
            n -= 1
        
        # 等到这个while技术后，我们的prev处于处理link的最后一位，也就是reverse之后的头部；
        # 我们的cur在之后链表中的第一位，因此tail.next=cur就可以🔗上。
        # 害怕我们最开始的con有可能是None，也有可能是从任一Node
        if con:
            con.next = prev
        else:
            head = prev
        tail.next = cur
        return head


# 143 143. Reorder List这道题融合的非常好呀
class Solution:
    def reorderList(self, head: ListNode) -> None:
        if not head:
            return 
        
        # find the middle of linked list [Problem 876]
        # in 1->2->3->4->5->6 find 4 
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next 
            
        # reverse the second part of the list [Problem 206]
        # convert 1->2->3->4->5->6 into 1->2->3->4 and 6->5->4
        # reverse the second half in-place
        prev, curr = None, slow
        while curr:
            curr.next, prev, curr = prev, curr, curr.next 
            """ 原理是一样的
            temp = cur.next
            cur.next = prev
            prev = cur
            cur = temp
            """

        # merge two sorted linked lists [Problem 21]
        # merge 1->2->3->4 and 6->5->4 into 1->6->2->5->3->4
        first, second = head, prev
        while second.next:
            first.next, first = second, first.next
            second.next, second = first, second.next
        


# 82. Remove Duplicates from Sorted List II
# 典型的三node解法
# 1. 最开始有个dummy node，用来最后的return next用
# 2. head Node，扮演处理/跳过某些Node的作用
# 3. pred Node，用来连接需要的Node
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        sentinel = ListNode(0, head)
        pred = sentinel
        
        while head:
            # 碰到有相同的node           
            if head.next and head.val == head.next.val:
                # 这里用来跳过所有相同的node
                while head.next and head.val == head.next.val:
                    head = head.next
                # 使prev_node的next指向最终head的next
                pred.next = head.next 
            # 遍历成功，没有遇到
            else:
                pred = pred.next 
            # 继续遍历head
            head = head.next # 别忘记
            
        return sentinel.next
    
# 19. Remove Nth Node From End of List
