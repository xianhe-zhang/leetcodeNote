Leetcode-215
1- #sort函数-自己想的
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        nums.sort() #后面可以有参数
        return nums[len(nums) - k] #求第k大的数字
#剩下的就是排序：这一题涉及快排/堆排。
#思想：因为是找最值，可以看作动态变化的求值过程
#关于堆排序和快排的话还是去看印象笔记吧...

2-堆排序
def findKthLargest(self, nums: List[int], k: int) -> int:
        heapq.heapify(nums) #依据nums建立小顶堆
        amount = len(nums)  #长度
        while amount > k:   
            heapq.heappop(nums) #删除最小值
            amount -= 1
        return nums[0] #删完队首就是这个了



Leetcode-347
#这一题的难点，怎么得到数组里的元素以及其映射
#得到了上述条件，就可以用排序然后return 相应的值就可以了。
#python如何处理这种元素与出现次数的映射，作弊方法：sorted(d.items(), key=lambda x: x[1])[:-k]
#如何理解？items将dic展开，然后用sorted将其排序，排序的依据key来，但是这里的key又用了lambda，意味着利用d中的每一项[1]列，也就是按照出现次数排列元素。最后的[:-k]是取倒数k个数字

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        dic = {}
        for i in range(len(nums)):
            dic[nums[i]] = dic.get(nums[i],0) + 1 #dic中的特定get用法，取得key对应的value，0是default，如果去不到就取0
        #也可以用下面
        """
        for num in nums:
            if num not in dic:
                dic[num] = 1
            else:
                dic[num] += 1
        """
        #上述的方法甚至可以用API Counter() 替代。
        """
        counts = collections.Counter(nums) # key为num, value为频次
        """
1- 桶排序
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        counts = collections.Counter(nums) # key为num, value为频次
        bucket = [[] for i in range(len(nums)+1)] # 因为桶的频次是可能达到len(nums)，所以这里要用len + 1
        for num in counts.keys():
            bucket[counts[num]].append(num) #为什么下标为频次？
        ans = []
        for i in range(len(nums), 0, -1):      #这时候倒序非常高明，就是将频次高的num先放入ansf
            ans += bucket[i]
            if len(ans) == k: return ans    #本循环为一直添加每一个下标，但是如果出现频次为空，那么无法加入ans。比如没有出现频次为5的数字，那么ans在为五的时候是不变的
        return ans  
                
2- Max heap/ Priority queue
class Solution:
  def topKFrequent(self, nums, k):
    counts = collections.Counter(nums)
    h = []                              #这是我们的栈
    for num in counts.keys():           #.key()
      heapq.heappush(h, (counts[num], num)) #入h堆，入的对象是一个数组（频次，数字）
      if len(h) > k: heapq.heappop(h)       #pop返回最小值，那么最终heap里是最大的一些数字
    return [heapq.heappop(h)[1] for i in range(k)] #这里heapq.heappop(h)[1] ，所以总共返回了k个值，还是最大的。



leetcode-451
1- 桶思想
class Solution:
    def frequencySort(self, s: str) -> str:
        bucket=[[] for _ in range(len(s) + 1)]
        counts = collections.Counter(s)
        for letter in counts.keys():
            bucket[counts[letter]].append(letter)
        ans = []
        for i in range(len(s), 0, -1):
            ans += bucket[i] * i
        return "".join(ans)
#写不出来 - 因为一个桶当中可能存在多个字母，所以打出来的是cacaca而非cccaaa。
#from collections import Counter

class Solution:
    def frequencySort(self, s):
        li = []
        for i, j in Counter(s).items():             #items() 遍历键和值
            li.append([i, j])
        new_li = sorted(li, key=lambda x: -x[1])    #按照li的元素[1]的倒序排列。
        return ''.join([i * j for i,j in new_li])
#模版
"""
具体做法如下：

先使用「哈希表」对词频进行统计；
遍历统计好词频的哈希表，将每个键值对以 {字符,词频} 的形式存储到「优先队列（堆）」中。并规定「优先队列（堆）」排序逻辑为：
如果 词频 不同，则按照 词频 倒序；
如果 词频 相同，则根据 字符字典序 升序（由于本题采用 Special Judge 机制，这个排序策略随意调整也可以。但通常为了确保排序逻辑满足「全序关系」，这个地方可以写正写反，但理论上不能不写，否则不能确保每次排序结果相同）；
从「优先队列（堆）」依次弹出，构造答案。

会使用数组去模拟我们想要用的数据结构

"""

leetcode-75 
@荷兰国旗问题(three elements)
#除了排序的API，我能想到的也只有双指针了，一般来说排序都可以用排序去解决，但是这道题没必要不是。
#用双指针写吧：这题tricky的点，一个个遍历交换没有办法解决这个问题，因为存在跳过1个或多个值的情况，这里的指针就用来判别应该跳到那个地方的。
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        n = len(nums)
        p0 = p1 = 0
        for i in range(n):
            if nums[i] == 1:    #如果是1，将1放在p1的指针处
                nums[i], nums[p1] = nums[p1], nums[i]
                p1 += 1
            elif nums[i] == 0:  #如果是0，将1放在p0的指针处
                nums[i], nums[p0] = nums[p0], nums[i]
                if p0 < p1:     #这个理解起来有趣，详情见下面
                    nums[i], nums[p1] = nums[p1], nums[i]
                p0 += 1
                p1 += 1
#p0，p1分别指的是0和1的下一位；除了刚开始的阶段，后面p0指的就是1，p1指的就是2，那么遇到0时，将1与0互换，那么这个1很有可能在2后面，因为原来的0位置不确定，所以要跟p1进行二次交换
#使用双指针，如果遍历对象不是目标不进行处理时，就可以跳过这群不是目标的对象从而针对目标对象进行集中处理。（脑中有画面很重要
#这一题用单指针也可以，两个循环，分别处理0和1.




"""
套路复盘：
1.桶排序        @46~56
2.大根堆        @12~18
3.优先队列      @60~66
4.哈希表记录     @72~92
5.复习复杂度    
6.复习手撕排序
"""