# 232 用栈实现队列
class MyQueue:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.s1 = []
        self.s2 = []
        self.front = None


    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        if not self.s1: self.front = x
        self.s1.append(x)
        


    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
            self.front = None
        return self.s2.pop() #这里的pop是stack中的pop

    def peek(self) -> int:
        """
        Get the front element.
        """
        if self.s2: 
            return self.s2[-1]
        return self.front


    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        if not self.s1 and not self.s2:
            return True
        return False



# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()


"""
class MyQueue {
    Deque<Integer> inStack;
    Deque<Integer> outStack;

    public MyQueue() {
        inStack = new LinkedList<Integer>();
        outStack = new LinkedList<Integer>();
    }
    
    public void push(int x) {
        inStack.push(x);
    }
    
    public int pop() {
        if (outStack.isEmpty()) {
            in2out();
        }
        return outStack.pop();
    }
    
    public int peek() {
        if (outStack.isEmpty()) {
            in2out();
        }
        return outStack.peek();
    }
    
    public boolean empty() {
        return inStack.isEmpty() && outStack.isEmpty();
    }

    private void in2out() {
        while (!inStack.isEmpty()) {
            outStack.push(inStack.pop());
        }
    }
}

"""

# 225 用队列实现栈
class MyStack:

    def __init__(self):
        self.qIn = collections.deque()
        self.qOut = collections.deque()
        
    def push(self, x: int) -> None:
        self.qOut.append(x)
        while self.qIn:
            self.qOut.append(self.qIn.popleft())
        self.qIn, self.qOut = self.qOut, self.qIn

    def pop(self) -> int:
        return self.qIn.popleft()

    def top(self) -> int:
        # return self.top
        return self.qIn[0]

    def empty(self) -> bool:
        return not self.qIn

#这里的pIn和pOut命名其实不是特别合适。
#pOut更像是一个helper，而不是指定了输入/输出queue

# 155 最小栈
"""
🌟 辅助栈算法，典型的“空间换时间”的做法.
但是在具体的实现方法上，其实会有一些差异，主要集中在helper stack上，差异点在于：是否入栈/出栈等操作与正常栈同步/或者异步。坑在于一些边界的处理上。
总结起来就是：出栈时，最小值出栈才同步；入栈时，最小值入栈才同步。
"""
class MinStack:
    def __init__(self):
        self.stack = []
        self.stackHelper = []

    def push(self, val):
        self.stack.append(val)
        if not self.stackHelper or val <= self.stackHelper[-1]:
            self.stackHelper.append(val)
    
    def pop(self):
        top = self.stack.pop()
        if self.stackHelper and top == self.stackHelper[-1]:
            self.stackHelper.pop()
        return top
    
    def top(self):
        if self.stack:
            return self.stack[-1]
        
    def getMin(self):
        if self.stackHelper:
            return self.stackHelper[-1]
#时间复杂度：O(1)，因为每次操作就两步，总是常量
#时间复杂度：O(n)

#还会有不利用额外空间的做法，维护差值
#Copy来的。解析：stack存差值，然后及时更新min_value，从而实现每次pop的就是当前最小值。最小值pop出去之后，min_value + stack就是自动更新到的下一个最小值
#这里我很迷糊的一个点，就是我们按照顺序append，那么怎么就能确认pop出来的就是最小呢？stack里面储存的到底是什么？是如何做到保证pop最小值的呢？
#答：因为算法的机制，每一次我们append的时候都是围绕‘当前’最小值，这是一个自动更新机制，所以把眼光局限在当前值上，每一次都是对应当前min_value就容易理解了，其实就是decode和uncode的机制。
class MinStack:
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.min_value = -1

    def push(self, x: int) -> None:
        if not self.stack:
            self.stack.append(0)
            self.min_value = x
        else:
            diff = x-self.min_value
            self.stack.append(diff)
            self.min_value = self.min_value if diff > 0 else x

    def pop(self) -> None:
        if self.stack:
            diff = self.stack.pop()
            if diff < 0:
                top = self.min_value
                self.min_value = top - diff
            else:
                top = self.min_value + diff
            return top

    def top(self) -> int:
        return self.min_value if self.stack[-1] < 0 else self.stack[-1] + self.min_value

    def getMin(self) -> int:
        return self.min_value if self.stack else -1
# 20  Valid Parentheses 有效的括号
class Solution:
    def isValid(self, s: str) -> bool:
        dic = {
            '{': '}',  
            '[': ']', 
            '(': ')',
            '?': '?'
        }
        #✨这里的’？‘不能省略，因为如果stack为空的话，输入invalid的话，比如输入为空，那么pop会报错！
        stack = ['?'] 
        for character in s:
            if character in dic:
                stack.append(character)
            elif dic[stack.pop()] != character:
                return False
        return len(stack) == 1


#cmd + k， cmd + j 展开
#shift + cmd + k 全选变量

# 739 Daily temperature
#不要用暴力遍历。采用递减栈，我们将温度val与index作为一个对象同时入栈，没遍历一个新的温度val时，与栈内的进行匹配，如果满足题意，就出栈，然后拿到之前入栈的index，从而修改index处的值，放入我们的结果值。
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        #因为0是默认不处理的，可以直接用命名进行自动填充
        res = [0] * len(temperatures) 
        stack = []

        for i in range(len(temperatures)):
            #这里入栈的可以直接是index
            while stack and temperatures[i] > temperatures[stack[-1]]:
                small = stack.pop()
                #i是目前遍历到的，small是可以匹配到的最近的值（因为是stack所以是最近）
                res[small] = i - small 
            stack.append(i)
        #最终没出栈的/没处理的值都为0
        return res



# 503 下一个更大元素 II Next Greater Element II 
# @单调栈: 从栈顶到栈底是单调递增/单调递减的
# 整体思路：建立单调栈，每一次遍历的元素，与栈顶元素对比。
# 如果当前元素比栈顶元素大：说明当前元素是前面一些元素的「下一个更大元素」，则逐个弹出栈顶元素，直到当前元素比栈顶元素小为止。
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        N = len(nums)
        res = [-1] * N
        stack = []      #入stack的是原数组的index，而非值

        # 1. 利用N * 2模拟循环数组，另一种方式是直接在原数组后面复制一份？
        # 2. 为什么需要循环数组，第一次循环建立单调栈（并针对部分进行处理）；第二次循环进行逻辑判断
        for i in range(N * 2):
            # 当stack有值，并且满足当前元素大于栈顶元素时，出栈并且修改相应的值。
            while stack and nums[stack[-1]] < nums[i % N]:
                res[stack.pop()] = nums[i % N]
            # 入栈
            stack.append(i % N)
        return res