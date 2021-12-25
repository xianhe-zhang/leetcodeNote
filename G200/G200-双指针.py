

Leetcode- 167
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        start, end = 0, len(numbers) - 1
        
        while start < end:
            total = numbers[start] + numbers[end]
            if total > target:
                end -= 1
            elif total < target:
                start += 1
            else: 
                return [start+1, end+1]

#要考虑两个指针移动的条件与边界
#因为是有序数组，所以暴力解体太危险
#这一题应该使用双向双指针，而非同向；因为同向没办法避免移动指针可能出现的两种可能性


Leetcode-633
class Solution:
    def judgeSquareSum(self, c: int) -> bool:
        n = 1
        while n*n < c:
            n += 1
        start = 0 
        end = n 
        while start <= end:
            total = start*start + end*end
            if total == c:
                return True
            if total > c:
                end -= 1
            if total < c:
                start += 1
        return False
#点评
#双指针利用没错，与标准答案不同的是如何去寻找n这个临界值；
#std solution: import Math   n = Math.sqrt(c) // 1 #sqrt() 取平方根


Leetcode-345
class Solution:
    def reverseVowels(self, s: str) -> str:
        vowels = ["a","A","E","I","O","U","e","i","o","u"]
        left = 0
        right = len(s) - 1
        while left < right:
            if s[left] in vowels and s[right] in vowels:
                temp = s[left]
                s[left] = s[right]
                s[right] = temp
                left += 1
                right -= 1
            elif s[left] in vowels:
                right -= 1
            elif s[right] in vowels:
                left += 1
            elif s[left] not in vowels and s[right] not in vowels:
                left += 1
                right -= 1
        return s
#错误！🙅
#字符串，数字之类的在python中属于不可变对象，因此无法直接通过下标/index直接对其进行修改/赋值
优化版本如下：
class Solution:
    def reverseVowels(self, s: str) -> str:
        temp = list(s)
        vowels = ["a","A","E","I","O","U","e","i","o","u"]
        left ,right = 0, len(s) - 1
        while left < right:
            if temp[left] not in vowels:
                left += 1
                continue
            if temp[right] not in vowels:
                right -= 1
                continue    #注意break与continue的区别
            temp[left],temp[right] = temp[right],temp[left]
            left += 1
            right -= 1
        return "".join(temp)

#这一题的take-away就是continue，list，操作运算

Leetcode-650
❌❌
class Solution:
    def validPalindrome(self, s: str) -> bool:
        left = 0 
        right= len(s) - 1
        counter = 0
        while left < right:

            if s[left] != s[right]:
                counter += 1
                if s[left + 1] == s[right]:
                    left += 1
                elif s[left] == s[right - 1]:
                    right -= 1
                elif s[left + 1] != s[right] and s[left] != s[right - 1]:
                    return False
            left += 1
            right -= 1
            if counter == 2:
                return False
        return True
❌❌      
#点评：有些思考不错，但是总体思路不对。之所以无法通过的原因是没有考虑到所有可能性。
#值得肯定的点：双指针模版清晰，学会用counter处理出现二次异常，如果是只用处理一次异常的情况，我们也可以用另写一个方法去判断～
#缺点：没有想到抓手，无法确定是从左还有从右删除/跳过 ——》此时的思考路径应该为，如果有两种可能性，任意一个满足回文串即满足题意，那么我们就去同时看两个可能性，而非想尽办法排除其中一个。
✅
class Solution:
    def validPalindrome(self, s: str) -> bool:
        def checkPalindrome(left, right):
            while left<right:
                if s[left] != s[right]:
                    return False
                left += 1
                right -= 1
            return True
            
        left ,right = 0, len(s) - 1
        while left < right:
            if s[left] != s[right]:
                return checkPalindrome(left + 1, right) or checkPalindrome(left, right - 1)
            left += 1
            right -= 1
        return True
✅
#用这种“递归/dp”的方式理解起来格外容易

Leetcode-88 
#三种思路：额外空间-双指针；直接操作但需要用到sort接口；逆向双指针，因为index会变动。
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        res = []
        i, j = 0,0
        while i < m or j <n: #同向双指针条件
            if i == m  :
                res.append(nums2[j])
                j += 1
            elif j == n : #这一题很好的说明了 None可以跟0去对比，两者相同
                res.append(nums1[i])
                i += 1
            elif nums1[i] < nums2[j] :
                res.append(nums1[i])
                i += 1
            elif nums1[i] >= nums2[j]:
                res.append(nums2[j])
                j += 1
        nums1[:] = res


🀄️🀄️🀄️记得去计算时间复杂度
Leetcode-141
#环形列表两种解答方式：1.双指针 2.Hash表
❌❌
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if not head or not head.next:   #❌这个忘掉了
            return False
        slow, fast = head, head
        while fast or fast.next:    #❌这个不对，ListNode不能用.next去判断是否为空，因为这个时候仅仅只是判断，CPU并不会计算。所以要换个思路，用确定的值当判断条件
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                return True
        return False

✅✅
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if not head or not head.next:
            return False
        
        slow,fast =head, head.next

        while slow != fast:
            if not fast or not fast.next:
                return False
            slow = slow.next
            fast = fast.next.next
        
        return True

#点评：这一题很好地暴露了自己针对特定题型模版的不熟悉，以及控制循环流的不熟悉

2-Hash解法
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        seen = set()    #用set代替哈希表
        while head:
            if head in seen:
                return True
            seen.add(head)  #add这个用法，head与head.val不同
            head = head.next
        return False
#我去，这一题。

Leetcode-524
❌❌
class Solution:
    def findLongestWord(self, s: str, dictionary: List[str]) -> str:
        queue = ["a"]

        for i in range(len(dictionary)):#总共遍历n个元素
            for j in range(len(dictionary[i])):#遍历每个字符串中的所有字母
                p0 = 0      #指针不能放在循环外，因为无法初始化
                p1 = 0
                if len(dictionary[i]) > len(s):#安全机制：如果dic字符串大，没必要再遍历了
                    continue

                while p0 < len(s):
                    if s[p0] == dictionary[i][p1]:
                        p0 += 1
                        p1 += 1
                    if p1 == len(dictionary[i]):
                        if len(queue[0]) < len(dictionary[i]) :
                            if not queue:
                                queue.append(dictionary[i])
                            else:
                                queue.pop(0)
                                queue.append(dictionary[i])
                        break
                    if s[p0] != dictionary[i][p1]:
                        p0+=1

                j+=1
            i+=1
        return queue[0]
#自己写的，没有满足题意，没有解决两个问题：1.同等长度的子字符串没有删除的的优先 2.queue在最开始如果没有字符串的边界问题，因为无法使用0，上述版本用if修改了
#太臃肿。可取的地方在于：1.想到了遍历的思路，自己也写出来了。 2.关于list边界问题，自己处理的还可以。通过if判断的位置。
#自己的其他思路，写help func/利用额外的表求，这两个都能解决。这也给自己提个醒，利用额外的帮助，说不定比自己在一个方法里写出来更牛逼！因为工程性/可阅读性

#两个methods的写法 -判断是否为最长，判断是否位子字符串
class Solution:
    def findLongestWord(self, s: str, dictionary: List[str]) -> str:
        #❌         lstword = []
        #原因：这事数组的写法，如果你只求一个值，可以直接声明一个空的变量
        lstword = ""
        #           p0,p1 = 0, 0
        #❌         while p1 < len(dictionary):
        #               if self.isSubstring(s, dictionary[p1]):
        #           不要这么写去遍历数组中的字符串，可以直接按照下面的写                   
        for target in dictionary:
            if len(lstword) > len(target) or (len(lstword) == len(target) and lstword < target):
                continue
            if self.isSubstring(s,target):
                lstword = target
        return lstword
    

    def isSubstring(s: str, sub: str) -> bool:  #判断是否为子字符串
        j = 0
        for i in range(len(s)):
            if j < len(sub) and s[i] == sub[j]:   #迭代条件
                j += 1
            return j == len(sub)                #❌🙅被这个缩进气死了
✅✅✅
class Solution:
    def findLongestWord(self, s: str, dictionary: List[str]) -> str:
        res = ""            
        for target in dictionary:
            if len(res) > len(target) or (len(res) == len(target) and res < target): #字符串比较是比较asin码
                continue
            if self.isSubstring(s,target):
                res = target

        return res
    
    def isSubstring(self,s: str, sub: str):  #判断是否为子字符串
        j = 0
        for i in range(len(s)):
            if j < len(sub) and sub[j] == s[i]:   #迭代条件
                j += 1
        return j == len(sub)                #判断！！！！！
#被最后一个缩进气死了
#这一题的难点在于如何判断子字符串，然后是这个工程思想，其他的一点都不难，但是被代码气到了。


