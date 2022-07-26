# 1021. Remove Outermost Parentheses
# 这题的思路牛逼啊
class Solution:
    def removeOuterParentheses(self, s: str) -> str:
        stack = []
        flag = 0
        for c in s:
            if c == '(' and flag > 0: stack.append(c)
            if c == ')' and flag > 1: stack.append(c)
            flag += 1 if c == '(' else -1
        return ''.join(stack)
            
            
        
# 682. Baseball Game
# 我最开始写的是isdigit()放在第一个if里面，但是这个function只能用于正整数！其他都会return False
class Solution:
    def calPoints(self, ops: List[str]) -> int:
        stack = []
        for o in ops:
            if o == '+': stack.append(stack[-1]+stack[-2])
            elif o == 'C':stack.pop()
            elif o == 'D':stack.append(stack[-1]*2)
            else:
                stack.append(int(o))
                
        
        return sum(stack)



# 844. Backspace String Compare
# 这一题给到的if条件有点tricky，其他还好。
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        
        def helper(i):
            stack = []
            for c in i:
                if c == '#':
                    if stack:
                        stack.pop()
                    else:
                        continue
                else:
                    stack.append(c)
            return stack
            
        return helper(s) == helper(t)



# 1190. Reverse Substrings Between Each Pair of Parentheses
class Solution:
    def reverseParentheses(self, s: str) -> str:
        res = ['']
        for c in s:
            if c == '(':
                res.append('')
            elif c == ')':
                # 这里很重要，如果直接写-2，那么这个index将会是在其赋值的时候寻找的
                # 但如果写-1，那么-1的值又会保留，无法pop
                res[len(res)-2] += res.pop()[::-1]
            else:
                res[-1] += c
            
        return ''.join(res)
    

# 394. Decode String
# 这一题的核心点就在入stack的是什么数据？它不是单一数据而是在组合数据
# 抽象点没有用一个res的list额外保存数据，节省了一点空间。这题直接在现有答案的基础上进行更改，有点抽象。
class Solution(object):
    def decodeString(self, s):
        # curString用的很巧妙，一方面当成变量暂时存值，另一方面
        stack = []; curNum = 0; curString = ''
        # 顺序遍历没跑
        for c in s:    
            # 这里注意了，如果遇到[，每次update的是curString，是已经暂存的结果。
            # 几个坑注意一下：
                # 为什么要用stack，而不用一个变量存？因为我们的括号可能是nested，单个变量只能存放一个值，遇到nested的情况就白瞎了。这是对input的理解不够呀。
                # 为什么要选择在这里初始化？同nested道理，如果放在]的情况下，我们遇到的数字和字符串就不一致了。这里init表示一旦进入括号，前面的数字和str就要先敲定下来了。
            if c == '[':
                stack.append(curString)
                stack.append(curNum)
                curString = ''
                curNum = 0
                
            # Case 2 --- ]
                # 因为数字和括号是紧跟着的，因此遇到这个括号的时候
                # 要把括号内的字符串*num，然后加上括号外的num，因为这个时候已经pop出来了。
            elif c == ']':
                num = stack.pop()
                prevString = stack.pop()
                curString = prevString + num*curString
            
            # Case 3  --- number
                # 主要针对两位数以上
            elif c.isdigit():
                curNum = curNum*10 + int(c)
                
            # Case 4  --- char
            else:
                curString += c
        return curString

# 456. 132 Pattern
# 时空为N
class Solution:
    def find132pattern(self, nums: List[int]) -> bool:
        if len(nums) < 3:
            return False
        stack = []
        min_array = [-1] * len(nums)
        min_array[0] = nums[0]
        
        # 找到前缀最小值
        # 同时要明白min_array中的值一定是呈现平稳/递减的趋势的。因为是一路min下去的
        for i in range(1, len(nums)):
            min_array[i] = min(min_array[i-1], nums[i])
            
        
        # 倒序遍历；针对每一个cur，我们看是否stack中存在比当前数还要小的值；如果存在，且当前值不等于最小值，那么就是Ture。
        for j in range(len(nums)-1, -1, -1):
            # 一般不会发生
            if nums[j] <= min_array[j]:
                continue
            # 如果发现stack[-1]比最小值小，那就pop掉，因为在数组中，某一个数的后几位确实存在比他大/小的情况。
            while stack and stack[-1] <= min_array[j]:
                stack.pop()   
            if stack and stack[-1] < nums[j]:
                return True
            stack.append(nums[j])
            
        return False


# 227. Basic Calculator II
class Solution:
    def calculate(self, s: str) -> int:
        if not s: return 0
        stack = []
        curNumber = 0
        operation = '+'
        
        for i in range(len(s)):
            curChar = s[i]
            if curChar.isdigit():
                # ord(curChar) - ord('0')
                curNumber = curNumber*10 + int(curChar)
            if not curChar.isdigit() and not curChar.isspace() or i == len(s) - 1:
                if operation == '-':
                    stack.append(-curNumber)
                elif operation == '+':
                    stack.append(curNumber)
                elif operation == '*':
                    stack.append(stack.pop() * curNumber)
                elif operation == '/':
                    temp = stack.pop()
                    # 题意要求向绝对值小的方向进位。 
                    # 第一个if判断看之前的值是正/负，如果是负且需要进位，那么计算后+1
                    if temp // curNumber < 0 and temp % curNumber != 0:
                        stack.append(temp//curNumber + 1)
                    # 其他情况向上进位就好。
                    else:
                        stack.append(temp//curNumber) # 这里注意改动
                # 注意operation是之前的loop循环下来的！过往变量保存，这一点值得学习
                operation = curChar
                curNumber = 0
    
        return sum(stack)


# 150. Evaluate Reverse Polish Notation
# 我自己的写法是通过if多个操作，但是不对为什么？因为isdigit()没有办法识别负数！
class Solution:   
    def evalRPN(self, tokens: List[str]) -> int:
        
        operations = {
            "+": lambda a, b: a + b,
            "-": lambda a, b: a - b,
            "/": lambda a, b: int(a / b),
            "*": lambda a, b: a * b
        }

        stack = []
        for token in tokens:
            if token in operations:
                number_2 = stack.pop()
                number_1 = stack.pop()
                operation = operations[token]
                stack.append(operation(number_1, number_2))
            else:
                stack.append(int(token))
        return stack.pop()
    

# 71. Simplify Path
# 这一题我们忘记了，可以直接用split这个API，而不用一个个ch去遍历了
class Solution:
    def simplifyPath(self, path: str) -> str:
        stack = []

        for portion in path.split("/"):
            if portion == "..":
                if stack:
                    stack.pop()
            elif portion == "." or not portion:
                continue
            else:
                stack.append(portion)
        
        return "/" + "/".join(stack)

# 856. Score of Parentheses
# 这已经不涉及算法了，而是套路
# 多层嵌套一定会分层；我们做的就是保证每层的分数是一致的
class Solution(object):
    def scoreOfParentheses(self, S):
        stack = [0] #The score of the current frame

        for x in S:
            # 如果发现(，那么我们就append(0), 相当于该层目前为0
            if x == '(':
                stack.append(0)
            # 如果碰到)，有两种情况，要么是自己组合了，还可能是nested的组合，因此要*2
            else:
                v = stack.pop()
                # 与此同时，我们在pop过后，进入到上一层的时候记得加上。
                stack[-1] += max(2 * v, 1)

        return stack.pop()

# 907. Sum of Subarray Minimums
# 天才般的思路：dp与单调栈stack的结合。详情见注解
class Solution:
    def sumSubarrayMins(self, A: List[int]) -> int:
        
        # 为了第一位能够顺利进行，所以之类也有新增，为什么新增，因为看下面有可能是它自己。
        A = [0]+A
        result = [0]*len(A)
        # 这里的处理就是为了能够取到index=0
        stack = [0]
        for i in range(len(A)):
            # 这里的stack[-1]是A里面的索引
            while A[stack[-1]] > A[i]:
                stack.pop() 
            # j就是当前位能取到的最小位的
            j = stack[-1]
            # dp的思路：每次增加一位，那么之前所有的组合都会增加一遍，再加上当前组合
            # confirm：result里面存的是各个位最小值的sum！不是单一的一个最小值
            # 怎么理解result？ 分成两部分
            # 第一部分与前j部分一样，是最小值；第二部分加上相对应i-j的前缀最小值A[i]
            result[i] = result[j] + (i-j)*A[i]
            # stack里存的是index
            stack.append(i)
        return sum(result) % (10**9+7)



# 1249. Minimum Remove to Make Valid Parentheses
# 这一题明显涉及到先遍历存值/后满足条件再更改，这种情况下利用split更改值
# 利用stack存什么？存index！
class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        stack = []
        split = list(s)
        
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(i)
            elif s[i] == ')':
                if stack:
                    stack.pop()
                else:
                    split[i] = ""
        for i in stack:
            split[i] = ""
        
        return "".join(split)
        

# 636. Exclusive Time of Functions
# 一个通用型技巧：如果入栈/出栈的数据要有ID，身份认证。那么我们可以入栈/出栈用index来表示
class Solution:
    def exclusiveTime(self, N, logs):
        ans = [0] * N
        stack = []
        prev_time = 0
        
        
        for log in logs:
            fn, typ, time = log.split(':')
            fn, time = int(fn), int(time)
            
            if typ == 'start':
                if stack:
                    ans[stack[-1]] += time - prev_time
                stack.append(fn)
                prev_time = time
            else:
                ans[stack.pop()] += time - prev_time +1
                prev_time = time + 1
        return ans


# 341. Flatten Nested List Iterator

class NestedIterator:
    
    def __init__(self, nestedList: [NestedInteger]):
        self.stack = list(reversed(nestedList))
        
        
    def next(self) -> int:
        self.make_stack_top_an_integer()
        return self.stack.pop().getInteger()
    
        
    def hasNext(self) -> bool:
        self.make_stack_top_an_integer()
        return len(self.stack) > 0
        
        
    def make_stack_top_an_integer(self):
        # While the stack contains a nested list at the top...
        while self.stack and not self.stack[-1].isInteger():
            # Unpack the list at the top by putting its items onto
            # the stack in reverse order.
            self.stack.extend(reversed(self.stack.pop().getList()))


# 224. Basic Calculator
"""
We push the elements of the expression one by one onto the stack until we get a closing bracket ). 
Then we pop the elements from the stack one by one and evaluate the expression on-the-go. 
This is done till we find the corresponding ( opening bracket. 
This kind of evaluation is very common when using the stack data structure.
但是需要注意，我们需要倒序，否则出栈后的计算顺序不对
"""
class Solution:

    def evaluate_expr(self, stack):
        
        # If stack is empty or the expression starts with
        # a symbol, then append 0 to the stack.
        # i.e. [1, '-', 2, '-'] becomes [1, '-', 2, '-', 0]
        if not stack or type(stack[-1]) == str:
            stack.append(0)
            
        res = stack.pop()

        # Evaluate the expression till we get corresponding ')'
        while stack and stack[-1] != ')':
            sign = stack.pop()
            if sign == '+':
                res += stack.pop()
            else:
                res -= stack.pop()
        return res       

    def calculate(self, s: str) -> int:

        stack = []
        n, operand = 0, 0

        for i in range(len(s) - 1, -1, -1):
            ch = s[i]

            if ch.isdigit():

                # Forming the operand - in reverse order.
                # 数字的倒序解法
                operand = (10**n * int(ch)) + operand
                n += 1
            
            # 不是digit/space，只能是括号/运算符
            elif ch != " ":
                # 看看我们之前是否遇见了digit
                if n:
                    # Save the operand on the stack
                    # As we encounter some non-digit.
                    stack.append(operand)
                    n, operand = 0, 0

                if ch == '(':         
                    res = self.evaluate_expr(stack)
                    # 看看stack是None还是)，不用担心外层一定会有操作符号的
                    stack.pop()        

                    # Append the evaluated result to the stack.
                    # This result could be of a sub-expression within the parenthesis.
                    stack.append(res)

                # For other non-digits just push onto the stack.
                else:
                    stack.append(ch)

        # Push the last operand to stack, if any.
        if n:
            stack.append(operand)

        # Evaluate any left overs in the stack.
        return self.evaluate_expr(stack)



"--------------------------------------------- Monotonic stack -------------------------------------------------"
# 496. Next Greater Element I
# 时空都为N
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        if not nums2:
            return None

        mapping = {}
        result = []
        stack = []
        stack.append(nums2[0])
        
        # 这个for+while的目的是，将所有val入map，目的是找到所有val的next greater
        # 首先遍历nums2，如果碰到greater，那么将cur:greater存入mapping
        for i in range(1, len(nums2)):
            # 这里注意了这里的stack是单调递减，因此新来的如果-1都比不过，那肯定不会是其他的greater了
            while stack and nums2[i] > stack[-1]:      
                mapping[stack[-1]] = nums2[i]          
                stack.pop()                            
            # 把greater存入stack
            stack.append(nums2[i])                      
        
        # 还在stack中的，就是没有next/greater
        for element in stack:                           
            mapping[element] = -1

        # 将nums1的答案存入
        for i in range(len(nums1)):
            result.append(mapping[nums1[i]])
        return result

# 739. Daily Temperatures
# monotonic stack 经典题
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        result = [0 for _ in range(len(temperatures))]
        stack = []
        for i in range(len(temperatures)):
            while stack and temperatures[i] > stack[-1][1]:
                index = stack.pop()[0]
                result[index] = i - index
            stack.append([i, temperatures[i]])
        return result


# 402. Remove K Digits
# 这个想法很妙啊。我的想法是stack里存的都是要删除的，而下面的solution是stack里面存的都是result的值
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        stack = list()
        for n in num:
            # 单调增stack
            while stack and k and stack[-1] > n:
                stack.pop()
                k -= 1
            # preventing leading 0s
            # 这里是如何通过stack和0的配合筛选出leading 0s的？
            # 如果是leading，那么经历了while后，stack一定为空，因为0是最小。所以如果没有stack碰到0，那么肯定不入；
            # 只要n不为0，都入。
            if stack or n != '0':
                stack.append(n)
        
        # 如果发现最后k有剩余，那么直接返回前面几位，因为是单调增。
        if k:
            stack = stack[0:-k]
        
        return ''.join(stack) or '0'
        
        
# 316. Remove Duplicate Letters

class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        stack, seen = [], set()
        # 单独开一张dict，用来维护每个字母最后出现的位置
        last_occurrence = {c: i for i, c in enumerate(s)}
        
        for i, c in enumerate(s):
            # 只去处理没有seen过的
            if c not in seen:
                # 1. char要是小的
                # 2. char出现的位置要比stack[-1]小？
                # 满足条件的话，就把[-1]从seen中删除，我们为什么这么做？
                # 因为我们要确定我们的stack严格满足我们的题意。上述两个条件表明，当我们遇到的c<[-1]时，并且[-1]在之后的位置还会再次出现，那我们就先暂时把它舍弃。
                while stack and c < stack[-1] and i < last_occurrence[stack[-1]]:
                    seen.discard(stack.pop())
                seen.add(c)
                stack.append(c)
        return ''.join(stack)
        # 这一题利用了比较多的space帮助判断，因为我们不仅要照顾顺序，还要照顾重复，而且顺序也不是稳定的，所以利用了比较局部的算法，同时加了一些限制
        
        
"""
注意反思，下面答案是高赞，但是不是很适合你，也并没有用到monotonic stack
很多fancy的答案是不是并不适合你现在的水平？
"""
# 1124. Longest Well-Performing Interval
# 这题思路也精彩
class Solution:
    def longestWPI(self, hours: List[int]) -> int:
        res = score = 0
        seen = {}
        for i, h in enumerate(hours):
            score += 1 if h > 8 else -1
            if score > 0:
                # 这里入栈的其实是实际的天数！而我们的i是index要对应上。
                res = i + 1 
            
            # 与seen[score] = i不同的是，setdefault先去寻找key=score，如果能找到直接返回，和i没关系了，如果找不到那么返回i，并且seen里添加
            seen.setdefault(score, i)
            
            # 如果当前分数遇见过，表明一定有增有减，那么 i-seen[score-1]是当前满足题意的interval
            if score - 1 in seen:
                res = max(res, i - seen[score-1])
        return res



# 2289. Steps to Make Array Non-decreasing
# 周赛第三题
"""
我用的是双指针；答案是dp+单调栈
为什么max？ 第一个max是为了求出delete的次数；第二个是为了update我们的res
为什么倒序？题意如果i-1>i的话，删除i；这种和stack的顺序不谋而合，元素入栈，新元素和栈顶比，可以pop栈顶
而且倒序可以实现什么？拿[5,1,1,1,6...]举例，当遍历刀5的时候，这个时候栈有3个1，可以不断pop，然后将cnt+1；符合操作风格

我用的双指针为什么不可？因为每一次删除是可以同时删除多个满足提议的要求，双指针线性遍历，没有办法存储状态，会使得答案重复计算

这一题入stack我们可以用个tuple，把dp与i一起放入，也可以像下面一样，分开更容易阅读一些。
"""
class Solution:
    def totalSteps(self, A: List[int]) -> int:
        n = len(A)
        dp = [0] * n
        res = 0
        stack = []
        # 为什么倒序？为什么max？
        for i in range(n-1, -1, -1):
            # 单调递增栈，
            # 发现比遍历过的还大就进入，意味着当前的是能够把之前比较小的给吃掉的。
            # 这个时候遇到一个选择：删除当前大值？删除之前小值？
            while stack and A[i] > A[stack[-1]]:
                # 需要注意！我们init是0的原因是，我们会出现很多次相同的问题，也就是every step我们针对所以满足题意的元素可以delete一个。
                # 因为我们是碰见比之前大的值才会入循环，在循环中我们会遇到3种情况：
                    # 第一种：前面没有出现删除的情况，所以跟的是stack.pop()，比较小
                    # 第二种：要pop元素找到要删除的元素数，没删除一次+1
                    # 比如[6,5,1,1,7]，为什么会有dp[stack.pop()]？ 而不用+1
                        # 这又碰到两种情况，一种是dp[i]=0，这就意味着i和上一位是挨着的，挨着的话满足题意是可以同时del掉的，比如[6,5]和[5,1]
                        # 另一种情况就是dp其实是有值的，但是没有大过stack.pop()，没大过意味着删除后可以合并成第一种情况。
                dp[i] = max(dp[i] + 1, dp[stack.pop()])
                # 更新res
                res = max(res, dp[i])
            stack.append(i)
        return res
                