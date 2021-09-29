/*
 * @Author: mario.zhangxianhe 
 * @Date: 2021-09-20 16:30:06 
 * @Last Modified by: mario.zhangxianhe
 * @Last Modified time: 2021-09-20 16:42:59
 */
from typing import List


leetcode-160 相交链表
#这一题有两个解法：记录、双指针走路
#记录
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if not headA or not headB:
            return False

        visited = set()
        while headA:
            visited.add(headA)
            headA = headA.next
        while headB:
            if headB in visited:
                return headB
            headB = headB.next
        return None
#自己写的，复杂度为mn；空间为m，用于储存headA的节点

#双指针-你走过我来时的路
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if not headA or not headB:
            return False
        A, B = headA, headB
        while A != B:   #如果没有交点，那么最后都为None，返回的也为None，因此不想交的情况不影响
            A = A.next if A else headB
            B = B.next if B else headA
        return A

        
leetcode-106 反转链表
#常规方法
class Solution(object):
	def reverseList(self, head):
		# 申请两个节点，pre和 cur，pre指向None
		pre = None
		cur = head
		while cur:                  #经典套路
			# 记录当前节点的下一个节点
			tmp = cur.next
			# 然后将当前节点指向pre
			cur.next = pre
			# pre和cur节点都前进一位
			pre = cur
			cur = tmp
		return pre	

#递归
class Solution(object):
	def reverseList(self, head):
		# 递归终止条件是当前为空，或者下一个节点为空
		if(head==None or head.next==None):
			return head
		# 这里的cur就是最后一个节点
		cur = self.reverseList(head.next)
		# 这里请配合动画演示理解
		# 如果链表是 1->2->3->4->5，那么此时的cur就是5
		# 而head是4，head的下一个是5，下下一个是空
		# 所以head.next.next 就是5->4
		head.next.next = head
		# 防止链表循环，需要将head.next设置为空
		head.next = None
		# 每层递归函数都返回cur，也就是最后一个节点
		return cur
#这个递归的套路好有趣！

leetcode-21 合并两个有序链表
#两种套路：递归和迭代；迭代非常容易想到，递归稍难
#迭代
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        prehead = ListNode(-1)

        prev = prehead
        while l1 and l2:
            if l1.val <= l2.val:
                prev.next = l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next            
            prev = prev.next

        # 合并后 l1 和 l2 最多只有一个还未被合并完，我们直接将链表末尾指向未合并完的链表即可
        prev.next = l1 if l1 is not None else l2

        return prehead.next
#就是咱们说的双指针。

#递归
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1: return l2  # 终止条件，直到两个链表都空
        if not l2: return l1
        if l1.val <= l2.val:  # 递归调用
            l1.next = self.mergeTwoLists(l1.next,l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1,l2.next)
            return l2
#这一题的思路就是通过递归跳转track从而达到有序排列


leetcode-83 删除排序链表中的重复元素
# 解法一：递归
class Solution:
    def deleteDuplicates(self, head) -> ListNode:
        if not head or not head.next: return head
        head.next=self.deleteDuplicates(head.next)
        if head.val==head.next.val:
            head.next = head.next.next
        return head

 # 解法二：遍历        
class Solution:
    def deleteDuplicates(self, head) -> ListNode:
        dummy = ListNode(next=head)     #dummy头技巧写法之一，我之前记得是用None连接之类的
        while head:
            while head.next and head.val==head.next.val:
                head.next = head.next.next
            head = head.next
        return dummy.next

leetcode-19 删除链表的倒数第N个节点

#此题有3种解法：1. 计算链表长度 2. 双指针 3.栈
#1.计算链表长度
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        def getLength(head: ListNode) -> int:
            length = 0
            while head:
                length += 1
                head = head.next
            return length
        
        dummy = ListNode(0, head)   #直接把头节点0 和 head连接起来了。
        length = getLength(head)
        cur = dummy
        for i in range(1, length - n + 1):
            cur = cur.next
        cur.next = cur.next.next
        return dummy.next
#2.栈
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(0, head)
        stack = list()
        cur = dummy
        while cur:
            stack.append(cur)
            cur = cur.next
        
        for i in range(n):
            stack.pop()

        prev = stack[-1]    #这里的是倒数n+1个node，那么prev.next本来连接的是我们要删除的元素，这下直接在原来的链表中.next.next就好。
        prev.next = prev.next.next
        return dummy.next

#3.双指针
#双指针作sliding window也有个好处，就是窗口的长度为n，当right
#这里的说是sliding window其实有些勉强，因为两个指针并不是同步变化的。
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        dummy = ListNode(0, head)
        left, right = dummy, head
        for _ in range(n):
            right = right.next

        while right:
            left = left.next
            right = right.next

        left.next = left.next.next
        return dummy.next



leetcode-24 两两交换链表中的节点
#两种方法
#递归
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head

        next = head.next
        #这里是next.next保证了两两处理 #考虑的时候要分层考虑，因为每一层我们的head和next指代对象都不一样（会有初始化的作用）
        head.next = self.swapPairs(next.next) 
        next.next = head
        return next
#这一题好难。 1.首先需要明白，每一层递归回归后的head，next可能并不相同。这里可以看作是每一层递归都会对我们的部分指代对象进行初始化的作用
#2. 其次意识到，递归进入的条件。是next.next，进入后这个将作为head, 也就是并不是每一个元素都会进入递归的。这也就形成了我们想要的两两交换的作用。
#3. 需要意识head.next, next.next是改变指针，而非改变以此为节点的所有链条。

leetcode-445 两数相加II
#用栈可以解决这个问题
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        s1, s2 = [], []
        while l1:
            s1.append(l1.val)
            l1 = l1.next
        while l2:
            s2.append(l2.val)
            l2 = l2.next
        ans = None
        carry = 0
        while s1 or s2 or carry != 0:
            a = 0 if not s1 else s1.pop()
            b = 0 if not s2 else s2.pop()
            cur = a + b + carry     #这里的carry就是可以进位的那个数字。
            carry = cur // 10
            cur %= 10
            curnode = ListNode(cur)
            curnode.next = ans  #先将这一轮的节点与之前的ans联合起来
            ans = curnode       #然后将整个链条更新为ans
        return ans


leetcode-234 回文链表
#双指针方法：快慢指针；当快指针走到头的时候，慢指针继续往前走，同时，pre也同步走；如果pre跟slow对应不上，则不为回文。
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        slow = head
        fast = head
        pre = head
        prepre = None
        while fast and fast.next:
            #pre记录反转的前半个列表，slow一直是原表一步步走
            pre = slow
            slow = slow.next
            fast = fast.next.next

            pre.next = prepre
            prepre = pre

        if fast:#长度是奇数还是偶数对应不同情况
            slow = slow.next  
        
        while slow and pre:
            if slow.val != pre.val:
                return False
            slow = slow.next
            pre = pre.next
        return True 

#复制值到数组中
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        vals = []
        current_node = head
        while current_node is not None:
            vals.append(current_node.val)
            current_node = current_node.next
        return vals == vals[::-1]


leetcode-725 分割链表
#这一题控制宽度的亮点就是a//b 和 a%B
#创建新链表 复杂度为N+K ；首先K循环没得跑，那么N比较灵活，最差情况就是所有对的node都要遍历。
class Solution:
    def splitListToParts(self, head: ListNode, k: int) -> List[ListNode]:
        cur = head
        for N in range(1001):   #利用中断for循环求出链表的长度。
            if not cur: break
            cur = cur.next
        width, remainder = divmod(N, k) #divmod() 函数把除数和余数运算结果结合起来，返回一个包含商和余数的元组(a // b, a % b)

        ans = []
        cur = head
        for i in range(k):                                  #决定有多少个子集
            head = write = ListNode(None)                   #head是None开始的链表；write用来执行操作；
            for j in range(width + (i < remainder)):        #决定每个子集内有多少个Node #“i < reminder”表明 i < reminder ? 1 : 0
                                                            #why需要判断？ 因为根据题意前n%k部分会有一个额外的节点。remainder就是留下来的，因此平摊到前面几个部分。
                write.next = write = ListNode(cur.val)      #相当于 write.next = ListNode(cur.value), wirte = write.next🌟
                if cur: cur = cur.next                      #放进去一个元素，那么我们的主序列cur应该继续前往下一节点
            ans.append(head.next)
        return ans

#分割原有链表  复杂度为N+K 
class Solution(object):
    def splitListToParts(self, head, k):
        cur = head
        for N in range(1001):
            if not cur: break
            cur = cur.next
        width, remainder = divmod(N, k)

        ans = []
        cur = head
        for i in range(k):
            head = cur
            for j in range(width + (i < remainder) - 1):
                if cur: cur = cur.next
            if cur:
                cur.next, cur = None, cur.next
            ans.append(head)
        return ans
#command + K + J （按照顺序来，用来拆解折叠函数）
#shift + command + L  统一选择变量修改；多光标按住option就行了。


leetcode-328 奇偶链表
#这题理解题意是关键。这里的奇偶是指index而非是node.val
class Solution:
    def oddEvenList(self, head: ListNode) -> ListNode:
        if not head:
            return head
        
        evenHead = head.next        #偶数Head
        odd, even = head, evenHead
        while even and even.next:
            odd.next = even.next #将odd的next指针指向even.next指向的对象，比如1->3
            odd = odd.next  #更新odd链表
            even.next = odd.next #同上
            even = even.next
        odd.next = evenHead #将两个链表连接起来
        return head