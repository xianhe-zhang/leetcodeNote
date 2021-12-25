#剑指 Offer 09. 用两个栈实现队列
class CQueue:
    def __init__(self):
        self.A, self.B = [], []

    def appendTail(self, value: int) -> None:
        self.A.append(value)

    def deleteHead(self) -> int:
        if self.B: return self.B.pop()
        # 如果能执行到这里，意味着B空，就看A空不空。
        if not self.A: return -1
        while self.A:
            self.B.append(self.A.pop())
        return self.B.pop()
# 注意A和B的stack不一定一侧必为空，可以在while遍历的时候再全部执行到一侧
# A = in stack ； B = out stack


# 剑指 Offer 30. 包含min函数的栈
# 这种方法和用辅助栈一样，不过使用了二维列表维护。
class MinStack(object):
    def __init__(self):
        self.stack = []
        
    def push(self, x):
        if not self.stack:
            self.stack.append((x, x))
        else:
            self.stack.append((x, min(x, self.stack[-1][1])))

    def pop(self):
        self.stack.pop()
        
    def top(self):
        return self.stack[-1][0]
        
    def getMin(self):
        return self.stack[-1][1]

# 利用辅助栈
# 辅助栈的作用就是：如果遇到比当前最小值还要小的数字，入辅助栈，出栈的话同时出；这样辅助栈的栈顶保存的就是当前元素对应的最小值。
class MinStack:
    def __init__(self):
        self.A, self.B = [], []

    def push(self, x: int) -> None:
        self.A.append(x)
        if not self.B or self.B[-1] >= x:
            self.B.append(x)

    def pop(self) -> None:
        if self.A.pop() == self.B[-1]: # 这里其实A已经pop了
            self.B.pop()

    def top(self) -> int:   # 看题理解
        return self.A[-1]

    def min(self) -> int:   # 栈B是栈顶越来越小
        return self.B[-1]


# 剑指offer 06 从头到尾打印链表
class Solution:
    def reversePrint(self, head: ListNode) -> List[int]:
        stack = []
        while head:
            stack.append(head.val)
            head = head.next
        return stack[::-1]
"""
class Solution {
    ArrayList<Integer> tmp = new ArrayList<Integer>();
    public int[] reversePrint(ListNode head) {
        recur(head);
        int[] res = new int[tmp.size()];
        for (int i = 0; i < res.length; i++) {
            res[i] = tmp.get(i);
        }
        return res;
    }

    void recur(ListNode head){
        if(head == null) return;
        recur(head.next);
        tmp.add(head.val);
    }
}

"""

# 剑指offer 24 反转链表
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        cur, pre = head, None
        while cur:
            temp = cur.next         # let temp point to next
            cur.next = pre          # let cur -> pre
            pre = cur               # let pre == cur 
            cur = temp              # move cur-> to next
        return pre
# 这个版本是可以背的 时间N，空间1

class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        def recur(cur, pre):
            if not cur:
                return pre
            res = recur(cur.next, cur)
            cur.next = pre
            return res
            
        return recur(head, None)
# 说实话，在链表里面玩递归是一件很酷的事情。双N