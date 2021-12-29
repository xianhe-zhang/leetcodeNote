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

# 剑指offer 35 复杂链表的复制
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if not head: return 
        dic = {}
        # 第一遍遍历
        cur = head
        while cur:
            dic[cur] = Node(cur.val)
            cur = cur.next
        
        # 第二次遍历    
        cur = head
        while cur:
            dic[cur].next = dic.get(cur.next)
            dic[cur].random = dic.get(cur.random)
            cur = cur.next
        
        return dic[head]


# 剑指offer 05 替换空格
class Solution:
    def replaceSpace(self, s: str) -> str:
        res = []
        for c in s:
            if c == " ":
                res.append("%20")
                continue
            res.append(c)
        return "".join(res)
"""
if...else...的简略写法
if(c == ' ') res.append("%20");
else res.append(c);
     
class Solution {
    public String replaceSpace(String s) {
        StringBuilder res = new StringBuilder();
        for(Character c: s.toCharArray()) {
            if (c == ' ') {
                res.append("%20");
            } else {
                res.append(c);
            }
            return res.toString();
        }
    }
}
"""
# 剑指 Offer 58 - II. 左旋转字符串 
class Solution:
    def reverseLeftWords(self, s: str, n: int) -> str:
        return s[n:] + s[:n]
        # 切片器 前闭后开 即[:n] == 0～n（不包含n）；[n:] n到len() - 1


# 剑指 Offer 03. 数组中重复的数字
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        dic = set()
        for num in nums:
            if num in dic:
                return num
            dic.add(num)
        return -1

# 第二种解法，有点费脑子
class Solution:
    def findRepeatNumber(self, nums: List[int]) -> int:
        i = 0
        while i < len(nums):
            if nums[i] == i:
                i += 1
                continue
            # 根据题意，这种解法的切入点在于：index和数字是一一对应的关系
            # nums[i]就是i对应的数字，并且将其作为index
            # 由于上面if存在，那么到这里i和nums[i]肯定不相同，但是他们指向的值相同，意味着找到了一组重复值
            if nums[nums[i]] == nums[i]:
                return nums[i]
            """
            Python中 a,b=c,d 操作的原理是先暂存元组 (c,d) 然后 “按左右顺序” 赋值给a和b 。
            因此，若写为 nums[i], nums[nums[i]] = nums[nums[i]], nums[i]
            则 nums[i] 会先被赋值，之后 nums[nums[i]] 指向的元素则会出错。
            """
            # 交换索引为 ii 和 nums[i]nums[i] 的元素值，将此数字交换至对应索引位置。
            nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
        return -1

# 剑指 Offer 53 - I. 在排序数组中查找数字 I
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        dic = collections.Counter(nums)
        res = dic.get(target)
        return res if res else 0
# 用API有点胜之不武，其他类似API有enumerate
# 这题是排序的，所以从时间上来说，最优的可能是二分
# 左右边界分别是target紧邻的，并非target本身
class Solution:
    def search(self, nums: [int], target: int) -> int:
        # 搜索右边界 right
        i, j = 0, len(nums) - 1
        while i <= j:
            m = (i + j) // 2
            if nums[m] <= target: i = m + 1
            else: j = m - 1
        right = i
        # 若数组中无 target ，则提前返回
        # 如何理解？因为跳出第一个while的条件就是j比i小1，那么既然i为右边界，那么j肯定指向target
        # 如果不等于target，那等于找了个寂寞，即数组里本就没有target
        if j >= 0 and nums[j] != target: return 0
        # 搜索左边界 left
        i = 0
        while i <= j:
            m = (i + j) // 2
            if nums[m] < target: i = m + 1
            else: j = m - 1
        left = j
        return right - left - 1
# 封装到helper里面
class Solution:
    def search(self, nums: [int], target: int) -> int:
        def helper(tar):
            i, j = 0, len(nums) - 1
            while i <= j:
                m = (i + j) // 2
                if nums[m] <= tar: i = m + 1
                else: j = m - 1
            return i
        return helper(target) - helper(target - 1)


# 剑指 Offer 53 - II. 0～n-1中缺失的数字
# 排序数组中的搜索问题，记得用二分，这样complexity好看
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        i, j = 0, len(nums) - 1
        # 跳出的条件是i > j，即第一个对应不上的index
        while i <= j:
            m = (i + j) // 2
            # 如何理解+1，-1？十分重要！
            # 最终我们找的值是第一个对应不上的index，所以如果==，那么index至少在m+1处
            # 如果不相同index必然在m以及m之前，那为什么还要-1呢？因为最终跳出的时候i在j的右边第一位！
            if nums[m] == m: i = m + 1
            else: j = m - 1
        return i


# 剑指 Offer 04. 二维数组中的查找
# 二维排列数组有规律，所以一开始不从(0,0)处开始遍历，换一个对角，这样，横竖分别对应大小不同的关系，可以看作一个二叉树。
class Solution:
    def findNumberIn2DArray(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix: return False
        r, c = 0, len(matrix[0]) - 1
        while r < len(matrix) and c >= 0:
            if matrix[r][c] == target:return True
            elif matrix[r][c] < target:r += 1
            else:c -= 1
        # 如果没有找到，那么过了临界点我理解的是删除根据格子的值，去不断改变r/c直到边界，最终跳出while，返回false
        return False
