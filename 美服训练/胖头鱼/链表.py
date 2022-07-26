# 206. Reverse Linked List
class Solution:
    def reverseList1(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # base case不能丢
        if not head or not head.next:
            return head
        prev = self.reverseList(head.next)  # prev一开始是最后一个结点，也就是我们的base case
        head.next.next = head   # 第2层recursion才会运行这一行，让自己的next节点指向自己，这是修改上一个node的操作
        head.next = None    # 将自己的next放空，交给下一个recursion组合
        return prev # 其实所有recursion中，我们的prev是不变的。但是最开的作用就是把运行带入到递归中去。

    # 迭代的方法不存在head.next.next，而使用prev当作绣花针将其串联起来。
    def reverseList1(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        while head:
            head.next, head, prev = prev, head.next, head
        return prev


# 这道题有几种写法：
    # 1- 利用额外的数据结构
    # 2- 反转链表进行判断 ✅
    # 3- 利用递归 ✅
    # 4- 直接反转一半 相当于先找中点了，
# 234. Palindrome Linked List
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        self.front = head
        
        def dfs(cur = head):
            if cur:
                if not dfs(cur.next):
                    return False
                if self.front.val != cur.val:
                    return False
                self.front = self.front.next
            return True
        return dfs()
