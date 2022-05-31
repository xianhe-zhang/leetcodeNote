# 206 反转链表
from platform import node


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
# 自己写的，踩到坑了。就是两个while不能嵌套起来，否则容易出现head的边界问题！！！不能在同一个while中同时处理两次head的next
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
# 这一题的思路还是两指针，三指针的写法；答案给的更清晰，遍历两次2N
"""
public ListNode removeNthFromEnd(ListNode head, int n) {
    ListNode dummy = new ListNode(0);
    dummy.next = head;
    int length  = 0;
    ListNode first = head;
    while (first != null) {
        length++;
        first = first.next;
    }
    length -= n;
    first = dummy;
    while (length > 0) {
        length--;
        first = first.next;
    }
    first.next = first.next.next;
    return dummy.next;
}
"""

# 148. Sort List
class Solution(object):
    def merge(self, h1, h2):
        # 首先我们需要dummy node技术
        # 这里也很明确，tail去充当合并时的index，而dummy充当最后return时的功能性head node
        dummy = tail = ListNode(None)
        while h1 and h2:
            if h1.val < h2.val:
                tail.next, h1 = h1, h1.next
            else:
                tail.next, h2 = h2, h2.next
            tail = tail.next
    
        tail.next = h1 or h2
        return dummy.next
    
    # 当我们把linkedlist拆到不能拆的地步的时候，我们开始merge最小的。
    # 什么时候不能拆呢？slow进入的下一个递归当head时，下一位没有了。
    def sortList(self, head):
        if not head or not head.next:
            return head
    
        # 很明确哈：pre是用来断开连接的；slow是用来铆钉sub-list的head的；fast是用来暂停while的
        pre, slow, fast = None, head, head
        while fast and fast.next:
            pre, slow, fast = slow, slow.next, fast.next.next
        pre.next = None
        
        """
        *的用法
        f(*[1,2,...]) = f(1,2,...)
        self.merge(*map(self.sortList, (head, slow)))
        equals
        self.merge(self.sortList(head),self.sortList(slow))
        """
        return self.merge(*map(self.sortList, (head, slow)))
# 自己写的方法
class Solution(object):
    def sortList(self, head):
        helperList = list()
        res = ListNode(0)
        cur = res
        while head:
            helperList.append(head.val)
            head = head.next
        helperList.sort()
        while helperList:
            cur.next = ListNode(helperList.pop(0))
            cur = cur.next
        return res.next


# 86. Partition List
# 牛逼，收下小弟膝盖
# 这题的想法是直接new 两个list出来，一个用来存放不经修改的值，一个用来存放修改的值。
# 并不是我的思路，我的思路是，只保存一个我修改的值，之前的值直接串联起来，这样逻辑上有点复杂...所以没写出来
class Solution(object):
    def partition(self, head, x):
        before = before_head = ListNode(0)
        after = after_head = ListNode(0)

        while head:
            if head.val < x:
                before.next = head
                before = before.next
            else:
                # If the original list node is greater or equal to the given x,
                # assign it to the after list.
                after.next = head
                after = after.next

            # move ahead in the original list
            head = head.next

        # Last node of "after" list would also be ending node of the reformed list
        after.next = None
        # Once all the nodes are correctly assigned to the two lists,
        # combine them to form a single list which would be returned.
        before.next = after_head.next

        return before_head.next

# 61. Rotate List
# 这一题的思路相当于遍历两次链表
    # 第一次将链表连成一个环，并且数有多少个node
    # 第二次数清楚应该走到哪，然后断开连接，简单直接！
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head: return None
        dummy = ListNode(0)
        dummy.next = head
        count = 1
        while head.next:
            count += 1
            head = head.next
        head.next = dummy.next
        
        steps = count - k%count
        cur = dummy
        while steps > 0:
            cur = cur.next
            steps -=1
        newHead = cur.next
        cur.next = None
        return newHead



# 147. Insertion Sort List
# 🌟很巧妙
# 领悟到了两点：
        # 1. 并不是所有的dummy都需要和head连接起来
        # 2. 修改node的pointer，即修改原有的linked list，其实并不像是在原有的基础上连接，那样太麻烦了。🌟而是给一个新的linked list node，依次添加起来，因为只操作指针，所以空间复杂度为constant
class Solution:
    def insertionSortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        cur = head
        while cur:
            pre = dummy 
            while pre.next and pre.next.val < cur.val:
                pre = pre.next
            
            next = cur.next 
            cur.next = pre.next
            pre.next = cur 
            cur = next     
        return dummy.next
        

# 138. Copy List with Random Pointer
# 这一题的deepcopy其实是用了dict暂时保存。🌟
# 一样的，首先都是用边界条件/返回条件，然后把当前值处理，保存值/判断之类的，然后进入下一个递归。递归的代码就是优雅哈
"""
iterate的思路：差不多一致。最外层traverse all the nodes。每一次遍历的时候，我们用将我们新创立的nodes连接起来，如果已经联结起来就算，如果没有，就修改连接。
当然了以上的修改都是在我们的helper dict中，然后返回head就成。
"""
class Solution(object):
    def __init__(self):
        self.visited = {}
    def copyRandomList(self, head):
        if not head: return None
        if head in self.visited:
            return self.visited[head]
        node = ListNode(head.val, None)
        # 将key:value都存入node，key是原来的值，value是deep copy的值
        self.visited[head] = node
        node.next = self.copyRandomList(head.next)
        node.random = self.copyRandomList(head.random)
        return node

# 24. Swap Nodes in Pairs
# 自己写的，写了20分钟...主要是这些指针太麻烦了。烦人。
# 原地改指针，之前我记得写的有，可以直接往新的listNode添加指针，倒是个不错的选择。
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next: return head
        
        dummy = ListNode(0)
        dummy.next = head
        pre, cur = dummy, head
        
        while cur and cur.next:
            nex = cur.next 
            pre.next = nex
            pre = cur
            cur.next = nex.next
            nex.next = cur
            cur = cur.next
        return dummy.next

# 学习recursion的写法！自己总是想不到递归的写法，愚蠢！
# recursion的思路：把second.next -> first，然后firse.next 指向下一层的second
# 太优雅了。而且很直观，简单。
class Solution(object):
    def swapPairs(self, head: ListNode) -> ListNode:

        # 边界条件
        if not head or not head.next:
            return head

        # Nodes to be swapped
        first_node = head
        second_node = head.next

        # Swapping
        first_node.next  = self.swapPairs(second_node.next)
        second_node.next = first_node

        # Now the head is the second node
        return second_node
# 接下来的链表题，主要就是学习recursion和指针操作咯～

# 328. Odd Even Linked List
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return head
        
        odd = head
        even = head.next
        eHead = even
        while even and even.next:
            odd.next = odd.next.next
            even.next = even.next.next
            odd = odd.next
            even = even.next
        
        
        odd.next = eHead
        return head


# 109. Convert Sorted List to Binary Search Tree
# 这一种是模仿preoder的递归。
class Solution:
    def size(self, head):
        ptr = head
        c = 0
        while ptr:
            ptr = ptr.next
            c += 1
        return c

    def sortedListToBST(self, head):
        s = self.size(head)
        # 首先明确helper返回的是当前层的node
        def preorder_helper(l, r):
            nonlocal head
            if l > r:
                return None
            # 为什么我们需要mid，因为我们需要将链表从中间拆开？
            # 为什么需要从中间拆开？因为中间值就是我们当下root node的值
            mid = l + (r-l)//2
            left = preorder_helper(l, mid  - 1)
            node = TreeNode(head.val)
            node.left = left
            head = head.next
            node.right = preorder_helper(mid + 1, r)
            return node
        return preorder_helper(0, s - 1)
        
  # 递归的写法
class Solution:
    def findMiddle(self, head):

        # The pointer used to disconnect the left half from the mid node.
        prevPtr = None
        slowPtr = head
        fastPtr = head
        while fastPtr and fastPtr.next:
            prevPtr, slowPtr = slowPtr, slowPtr.next
            fastPtr = fastPtr.next.next
        if prevPtr:
            prevPtr.next = None
        return slowPtr

    def sortedListToBST(self, head):
        if not head: return head
        mid = self.findMiddle(head)
        node = TreeNode(mid.val)
        # 这里又写错了一次😮‍💨
        if mid == head:
            return node
        node.left = self.sortedListToBST(head)
        node.right = self.sortedListToBST(mid.next)
        return node

# 430. Flatten a Multilevel Doubly Linked List
class Solution(object):
    def flatten(self, head):
        if not head:
            return

        dummy = Node(0,None,head,None)
    
        prev = dummy

        stack = []
        stack.append(head)

        # 非常巧妙，本体利用了stack的标准，拆成一个个node看待
        # 当碰到child时，一定优先处理child，所以利用两个if和stack的特性，把这点很好的结合起来。
        while stack:
            curr = stack.pop()
            # 双向指针
            prev.next = curr
            curr.prev = prev

            if curr.next:
                stack.append(curr.next)
 
            if curr.child:
                stack.append(curr.child)
                # don't forget to remove all child pointers.
                curr.child = None

            prev = curr
        
        dummy.next.prev = None
        return dummy.next

# 虽然也是dfs，但是用了recursion的方法。
class Solution(object):
    def flatten(self, head):
        if not head:
            return head
        # 这里的main function只把代码的主逻辑放出来，其他的细节/重复操作都放在recursion的function中国呢
        pseudoHead = Node(None, None, head, None)
        self.flatten_dfs(pseudoHead, head)

        pseudoHead.next.prev = None
        return pseudoHead.next


    # 其实recursion也像是一种“分治”，如同greedy、dp、merge sort一般，都是把大问题化解成小问题，把宏观局限在微观
    # 链表的recursion的主题只有node，所以千万别把linked NODES考虑进来
    # 主要有两个任务：pointer连接prev和cur；判断child与next
    def flatten_dfs(self, prev, cur):
        # base case.
        if not cur:
            return prev
        
        prev.next = cur 
        cur.prev = prev
        
        temp = cur.next
        tail = self.flatten_dfs(cur, cur.child)
        cur.child = None
        return self.flatten_dfs(tail,temp)

# 725. Split Linked List in Parts 好题哦～
class Solution(object):
    def splitListToParts(self, root, k):
        cur = root
        # 用这种方式求出N也是够别致的。
        for N in range(1001):
            if not cur: break
            cur = cur.next
        width, remainder = divmod(N, k)

        ans = []
        cur = root
        for i in range(k):
            head = write = ListNode(None)
            # 利用(i < remainder) #太赞了
            for j in range(width + (i < remainder)):
                # 因为这个等式，让write无中生有了。并且每次循环都更新。
                write.next = write = ListNode(cur.val)
                # 如果满足的话就继续，如果不满足的话就是None，也棒。
                if cur: cur = cur.next
            ans.append(head.next)
        return ans

# 还是要想清楚处理的逻辑，所有的代码都是跟处理的逻辑走，但说实话，这一题很值得品味。
# 包括状态的处理，好复杂。
# 25. Reverse Nodes in k-Group
class Solution:
    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        
        pointer = head
        new_head = None
        ktail = None
        
        while pointer:
            count = 0
            
            while pointer and count < k:
                pointer = pointer.next
                count += 1

            if count == k:
                revHead = self.reverseLinkedList(head, k)
            
                if not new_head:
                    new_head = revHead
                if ktail:
                    ktail.next = revHead
  
                ktail = head
                head = pointer

        if ktail:
            ktail.next = head
        return new_head if new_head else head
    
    def reverseLinkedList(self, head, k):
        pointer, new_head = head, None
        while k:
            # pointer.next, new_head, pointer = new_head, pointer, pointer.next
            temp = pointer.next
            pointer.next = new_head
            new_head = pointer
            pointer = temp
            k -= 1
        return new_head
        
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
            x, v = divmod(x, 10)
            # 这里的指针操作太巧妙了
            # head.next -> ListNode(v)
            # head.next.next = 之前head.next
            # 怎么理解，等号前面的是顺序；等号后面的是局部变量，要等着。
            head.next, head.next.next = ListNode(v), head.next

        return head.next

# 2. Add Two Numbers
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        head = dummy 
        carry = 0
        
        while l1 or l2 or carry:
            v1 = l1.val if l1 else 0 
            v2 = l2.val if l2 else 0 
            carry, out = divmod(v1 + v2 + carry, 10)
            head.next = ListNode(out)
            head = head.next
            
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
            
        return dummy.next

# 160. Intersection of Two Linked Lists
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        pa = headA
        pb = headB
        
        while pa!=pb:
            pa = headB if not pa else pa.next
            pb = headA if not pb else pb.next
            
        return pa
            

# 21. Merge Two Sorted Lists
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        head = dummy 
        l1 = list1
        l2 = list2
        while l1 or l2:
            if not l1:
                head.next = l2
                break
            if not l2:
                head.next = l1
                break
            
            if l1.val < l2.val:
                head.next = l1
                head = head.next
                l1 = l1.next
            else:
                head.next = l2
                head= head.next
                l2 = l2.next
        
        return dummy.next
                    


# 1669. Merge In Between Linked Lists
class Solution:
    def mergeInBetween(self, list1: ListNode, a: int, b: int, list2: ListNode) -> ListNode:
        start, end = None, list1
        # 这个双指针的前进方法还是挺好的。
        for i in range(b):
            if i == a - 1:
                start = end
            end = end.next
        start.next = list2
        while list2.next:
            list2 = list2.next
        list2.next = end.next
        end.next = None
        return list1