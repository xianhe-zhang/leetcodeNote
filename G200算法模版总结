目录：
1- 回溯
2- DFS
3- BFS
4- 贪心
5- 双指针
6- 排序
7- 二分
8- 分治



=================================         1-回溯           =================================
# 一般需要helper方法去达到传递值/初始化，模版如下：
def solution(self,n):
    def dfs(n,size,depth,path,res):
        if 递归总结:
            XXXXX/if:
            res.append()

        for i in range(size):
            if XX:
                continue
            path.append('x')
            dfs(n,size,i+1,path,res)
            path.pop()

    if not n:
        return []
    size = len(n)
    path = []
    res = []
    dfs(n,size,0,path,res)
    return res
#有时候我们会根据题意灵活调整其中的一些条件，或者添加一些其他有的没有。也要考虑到剪枝的情况
#有时候需要在主方法中先搭建出框架，用来标记元素，这一点也很重要！🌟
#有一种比较特殊的路径选择：direcitons = [(0,1),(0,-1),(1,0),(-1,0)]
#剪枝条件：边界限制/相邻元素重复/已begin为元素向后开展（子序列）/在DFS中两个判断，如果一个子树不满足，直接跳出所有。
=================================         2-DFS          =================================
"""
整体思路与回溯相似，但是套路没有回溯强。
总体来说记住几个要素就行：
    0. 回溯是所有可能性，DFS是可达性，因此不需要剪枝/回溯
    1. 递归终结条件
    2. helper 方法
    3. 判断条件：如何继续进入递归？
    4. 标记：是否访问过
    5. 双循环遍历常用
    6. 如果有特殊值，先处理特殊值，从特殊值向中心靠拢，而不是单纯遍历遇到特殊值再处理
    7. 设计思想：两个方法做两件事：1.按条件遍历  2，针对遍历结果进行判断
"""
=================================         3-BFS          =================================
#通常用于最短路径，模版也比较类似：
def bfs(self, list):
    if not list or x: 
        return 0
    n = len(list)
    queue = []  #要记录走过的节点和层数

while len(queue): #处理队列，如果处理完了还没return，那么就return None
    queue.pop()
    处理
    if 判断条件:
        return 结果
    
    for i, j in list:
        if x: #判断是否往下走，如果往下走
            queue.append
            visited = []
#记住BFS是一层一层遍历的，虽然往queue里添加的元素也是一个一个的，但是因为遍历顺序所有一层紧邻的。
#利用set模仿hash查数据更快为O1
#deque是双向队列比列表复杂度低
#chr(ord('a') + k)用于操作字符串

=================================         4-贪心          =================================
#贪心主要涉及算法思想，因此在写代码前一定要想好，将问题抽象形成模型
#防御式编程，如何处理数列两端的数据，可以在首尾加上0
#禁止面向案例编程，贪心问题往往遇到list range的问题，3个以内的特例可以考虑单独处理
#如果存在i与i+1，那么一般我们会改动i而非改动i+1，为了方便理解，我们一般写成i与i-1
"""
.sort(key=lanmbda x:x[1]) 贪心时常需要按照需求排序
.sort(key=lambda x: (-x[0], x[1]))
low = float("inf")  #用来最开始设定边界用
dic = {s: index for index, s in enumerate(S)} #s为数值，index为s最后一次出现的index
"""
=================================         5-双指针          =================================
#同向指针，相向指针，快慢指针
#while start < end; start <= end； 这就看你要什么数据了
#不能对一个字符串针对其index进行操作
#模版还会考虑指针转移条件与终止条件
=================================         6-排序         =================================
"""
常用套路：
    1. .sort() 库函数
    2. 小顶堆
    3. 桶排序
    4. Max heap / Priority queue

"""
#小顶堆
def heap(self, nums, k):
    heapq.heapify(nums)
    size = len(nums)
    while size > k:
        heapq.heappop(nums) #删除最小值并且重新构建
        size -= 1
    return nums[0]


#桶排序
def bucket(self,nums,k):
    counts = collections.Counter(nums)
    bucket = [[] for i in range(len(nums) + 1)]
    for num in counts.key():
        bucket[counts[num]].append(num) #为什么下标为频次？
    ans = []
    for i in range(len(nums), 0, -1):      #这时候倒序非常高明，就是将频次高的num先放入ansf
        ans += bucket[i]
        if len(ans) == k: return ans    #本循环为一直添加每一个下标，但是如果出现频次为空，那么无法加入ans。比如没有出现频次为5的数字，那么ans在为五的时候是不变的
    return ans  

#Max heap/ Priority queue
def topKFrequent(self, nums, k):
    counts = collections.Counter(nums)
    h = []                              #这是我们的栈
    for num in counts.keys():           #.key()
        heapq.heappush(h, (counts[num], num)) #入h堆，入的对象是一个数组（频次，数字）
        if len(h) > k: heapq.heappop(h)       #pop返回最小值，那么最终heap里是最大的一些数字
    return [heapq.heappop(h)[1] for i in range(k)] #这里heapq.heappop(h)[1] ，所以总共返回了k个值，还是最大的。

"""
#哈希表统计,counter
先使用「哈希表」对词频进行统计；
遍历统计好词频的哈希表，将每个键值对以 {字符,词频} 的形式存储到「优先队列（堆）」中。并规定「优先队列（堆）」排序逻辑为：
如果 词频 不同，则按照 词频 倒序；
如果 词频 相同，则根据 字符字典序 升序（由于本题采用 Special Judge 机制，这个排序策略随意调整也可以。但通常为了确保排序逻辑满足「全序关系」，这个地方可以写正写反，但理论上不能不写，否则不能确保每次排序结果相同）；
从「优先队列（堆）」依次弹出，构造答案。
"""

#🌟总结一下：基本上实习遇到的面试排序都是快排，因此快排要特别熟悉；除此之外，其他排序算法锻炼的不够，针对高频算法要能够手撕。

=================================         7-二分         =================================
def erFen(self,x):
    l, r, ans = 0, x, -1
    while l <= r:
        #这其实是
        mid = (l + r) // 2  #//取整，都是取靠左的数字
        if mid * mid <= x:
            ans = mid    
            l = mid + 1
        else:
            r = mid - 1
    return ans
#二分的套路一定要在有序下进行，困难的不是二分思想，而是二分模版下，各个情况的判定应该怎么做？
"""
模版如下：<= 或者 < 以及mid + 1 和 mid
    while left <= right: #这里必须有等号，解释如下。
            mid = (left + right) // 2

            if nums[mid] > nums[-1]:
                left = mid + 1
            else:   #nums[mid] <= nums[-1]
                right = mid - 1
        return nums[left] #必须是left

上面循环判断条件如果是<=，那么mid + 1 和 mid - 1；如果是<， 那么是mid + 1 和 mid。

<= :
1. 循环退出后right在左，left在右；🌟
2. 没有<的思考路径直观
3. 果要找的问题性质复杂，可以在循环外进行判断返回
4. 最后left = target, right = target - 1 

<:
1. 值的区间为1，跳出后直接可以返回值，就是我们要的值
2. 思考路径简单
3. 可以直接在循环体里跳出循环
4. left = right = target

注意：两者不是100%互换的，需要根据题意及时进行调整。
"""

=================================         8-分治         =================================
分治可以理解为递归
