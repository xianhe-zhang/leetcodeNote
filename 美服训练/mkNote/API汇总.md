## Python

```python
divmod(a, b)
# 接收两个数字类型（非复数）参数，返回一个包含商和余数的元组(a // b, a % b)。

enumerate(nums)
# 用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
# key = index； value = data

注意dict是无序的，但是我么可以用dict.items()转换一下，从而可以排序了！
sorted(iterable, key=None, reverse=False)  
# iterable -- 可迭代对象。
# key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
# reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
# list 的 sort 方法返回的是对已经存在的列表进行操作，而内建函数 sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作。

a = [1,2,3,4]
b = [1,2,3,4]
zipped = zip(a,b)     # 返回一个对象
print(list(zipped)) = [(1,1), (2,2), (3,3), (4,4)]

round(56.659,1)
floor(x)

reversed(seq) # 返回一个反转的迭代器。


intervals.sort(key= lambda x: x[0]) # 这个用法你很清楚了
heapq.heappush(free_rooms, intervals[0][1])
heappush(heap,n) # 数据堆入
heappop(heap) # 将数组堆中的最小元素弹出
heapify(heap) # 将heap属性强制应用到任意一个列表
heapreplace(heap，n) # 弹出最小的元素被n替代
heappushpop(head, n) # 先push再pop
nlargest(k, nums)[-1]  # Return a list with the n largest elements
nlargest(k, count.keys(), key=count.get) 
# 注意heapq default是小顶堆

range()和xrange()都是在循环中使用，输出结果一样。 range()返回的是一个list对象，而xrange返回的是一个生成器对象(xrange object)。 xrange()则不会直接生成一个list，而是每次调用返回其中的一个值，内存空间使用极少。 因而性能非常好，所以尽量用xrange吧。


collections.defaultdict(list)
使用普通的字典时，用法一般是dict={},添加元素的只需要dict[element] = xxx,但前提是element字典里，如果不在字典里就会报错。
defaultdict的作用是在于，当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值，这个默认值是什么呢？ 
当key不存在时，返回的是工厂函数的默认值，比如list对应[ ]，str对应的是空字符串，set对应set( )，int对应0
** 也不会出现index out of range 的问题 **

from collections import Counter
Counter(nums) # 把nums这个list转化为dict，key=num for num in nums; value=出现过几次
# Counter之后生成的dict已经是自动“排序”过了，按照value从高到低。

dict.get(key, default=None)
返回指定键的值，如果值不在字典中返回default值

两者是一样的，第一个的3rd parameter我还不清楚是干嘛的
# 明白了，Tree/list 可以省略，overload嘛
node = TreeNode(head.val)
node = Node(head.val, None, None) 
node = ListNode(head.val, None)
node = Node(0,None,head,None) # 新建一个node，val为0，并且让这个node与head相连。


这些骚操作你要记住！
l = list(map(x+1, lists))
l = [i for i in range(n) if nums[i] == key]
if any(abs(i-j) <= k for j in l)
if all((r,c) in s for r in range(r1, r2) for c in range(c1, c2))
res = set((x,y) for x,y in dig)
ans.append(max(x.val for x in queue))
"""
*的用法
列表或元组前面加星号作用是将列表解开成两个独立的参数，传入函数；
字典前面加两个星号，是将字典的值解开成独立的元素作为形参。
f(*[1,2,...]) = f(1,2,...)
self.merge(*map(self.sortList, (head, slow)))
equals
self.merge(self.sortList(head),self.sortList(slow))
"""

itertools.accumulate(iterable[, func])
# accumulate函数的功能是对传进来的iterable对象逐个进行某个操作（默认是累加，如果传了某个fun就是应用此fun


删除新增的两大坑：
list不能在循环中正序删除，要倒序
set/dict 不能轻易使用add，会引起错误，因为set/dict本质都是hashtable，一些题意有可能先暂存相同的值，因此不能用这个套路
可以换个思路，不要暂存add/remove重复操作，如果满足题意的答案，都进行下一步操作，在结尾最后一步再判断，没有出现过的答案，进入最终的result

# 连续赋值
a = b = c = 1
a, b, c = 1, 1, 1
a = 3
a, b = 1, a # b = 3
# 涉及到连续赋值要记住，右侧其实都是局部变量，而非变量本身。
# 反转链表
while head:
	L.next, head.next, head = head, L.next, head.next
 return L.next
# 就不用在用一个temp = head.next之类的了。
head.next = head = ListNode(5) # 链表同理，这里只是因为在每一次循环中，我们要随时更新head的状态才用这种形式。

temp = collections.Counter()
print(temp.most_common(2)) #[(9, 3), (0, 2)]  统计出现次数最多个两个元素
# sorted的第一个参数一定是iterable，key就是iterable的值，如果还要进行处理就用lambda，然后怎么排序。
print(sorted(key_value.items(), key = lambda kv:(kv[1], kv[0])))   



# 除二是向零取整
# 右移一位是向下取整
也就是说正数没有影响，负数有影响


res = max(self.helper(s,i,i), self.helper(s,i,i+1), res, key=len)

# 可以执行字符串的表达式
eval() 
# 可以找到字符串中x的第一个index
s.index('x')
s.startswith()
s.endswith()
s.replace()

# 如果想看group的话，一定要用list()
# 和counter不同，groupby只会针对连续相同进行分类。
s = ''.join(str(len(list(group))) + digit for digit, group in itertools.groupby(s))
groupby(a_list, lambda x:x[0]) 
# iterable:Iterable可以是任何类型(列表，元组，字典)。
# key:为可迭代的每个元素计算键的函数。
# 返回类型：它从迭代器返回连续的键和组。如果未指定键函数或为“无”，则键默认为标识函数，并返回不变的元素。
itertools.groupby(iterable, key_func)

# intersection(set1,set2,set3)
list(pacific_reachable.intersection(atlantic_reachable))


@lru_cache
对，就相当于有个dict，储存了之前的结果，如果在dict里就直接返回了

"""
bisect_left()	查找 目标元素左侧插入点
bisect_right()	查找 目标元素右侧插入点
bisect()	同 bisect_right()
insort_left()	查找目标元素左侧插入点，并保序地 插入 元素
insort_right()	查找目标元素右侧插入点，并保序地 插入 元素
insort()	同 insort_right()

使用 bisect 模块的方法之前，须确保待操作对象是 有序序列，即元素已按 从大到小 / 从小到大 的顺序排列
参数是(a, x, [lo=0, hi=len(a)])
(a, x, lo=0, hi=len(a))
"""


ans.append(max(x.val for x in queue))

str.isalnum() # 方法检测字符串是否由字母和数字组成。
str.lower()

str.isdigit()    判断所有字符都是数字（整形）   
str.isalnum() 判断所有字符都是数字或者字母    
str.isalpha()  判断所有字符都是字母 
str.islower()  判断所有字符都是小写 
str.isupper() 判断所有字符都是大写
str.istitle()    判断所有单词都是首字母大写


collections.Orderdict() # 根据放入dict的顺序进行排序...

# >>> x & y # 交集  
set(['a', 'm'])  
  
# >>> x | y # 并集  
set(['a', 'p', 's', 'h', 'm'])  
  
# x - y # 差集  
set(['p', 's'])  


seek() # 默认种子是系统时钟
random() # 生成0到1的随机小数
uniform(a,b) # 生成a到b的随机小数
randint(a,b) # 生成一个a到b的随即整数
randrange(a,b,c) # 生成一个a到b，以c递增的数
choice(<list>) # 随机返回一个列表里面的元素
shuffle(<list>) # 将列表的元素随机打乱
sample(<list>,k) # 从列表中随机抽取k个元素


```



## Java

```java
```

