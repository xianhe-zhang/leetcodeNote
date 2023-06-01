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
            # 这里的if else 相当于base case 用来终止recursion的扩展
            if cur:
                # 用作recursion链条上的terminator，
                # 当前的条件判断可能通过，但是子recur可能无法通过难过，因此将通过这个条件一路false
                if not dfs(cur.next):  return False
                # Core Divider
                if self.front.val != cur.val: return False
                self.front = self.front.next
            return True 
        return dfs()

# 160. Intersection of Two Linked Lists
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        pa = headA
        pb = headB
        
        while pa!=pb:
            pa = headB if not pa else pa.next   # 看仔细咯这是a 和 headBBBB
            pb = headA if not pb else pb.next
        return pa

        # 如果不相交，最后肯定会为0，会==，然后return null
        # 如果相交，走过相同的路，两者肯定会相交于交点。


# 253. Meeting Rooms II
import heapq
class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        # 这里用到了heap的结构，不得不说这种精妙的题真的都是好题
        # 我们把所有的endTime都push进堆; python中的堆是小根堆MinHeap！
        # 队首元素都是最小的。
        # 那么弹出与pushin的平衡在哪里，每次弹出最小值，就意味着一个meeting room用完了，可以用这个。最大值就不用增加了，最大值是同时room的一个量
        if not intervals:
            return 0
        rooms = []
        intervals.sort(key = lambda x: x[0])
        heapq.heappush(rooms, intervals[0][1])
        for i in intervals[1:]:
            if rooms[0] <= i[0]:
                heapq.heappop(rooms)
            heapq.heappush(rooms, i[1])
        return len(rooms)
        
# 也是面试中经常会出现的问题。先排序->利用了heap->依次判断i[1]

class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0

        used_rooms = 0

        # 太胆大了，太巧妙了
        # 将start/end拆分开。
        start_timings = sorted([i[0] for i in intervals]) 
        end_timings = sorted(i[1] for i in intervals)
        L = len(intervals)
        end_pointer = 0
        start_pointer = 0

        # start_pointer可以看作正常loop的标记器
        # used_rooms也很灵性，可以满足sliding window，只增大不缩小。
        while start_pointer < L:
            # 每次只操控一个房间，如果满足就dynamic维持rooms不变，如果不满足直接upper limit +1
            if start_timings[start_pointer] >= end_timings[end_pointer]:
                used_rooms -= 1
                # 不用遍历，用end ptr的目的是因为end的顺序和main loop是async的。
                end_pointer += 1
            used_rooms += 1    
            start_pointer += 1   

        return used_rooms


    