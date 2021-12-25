leetcode-204 计数质数
#数质数有只有两种思路：1. 暴力 2. 埃氏筛（类似剪枝）3.线性筛选（更好，但是笔试不考）
#1- 暴力
class Solution:
    def countPrimes(self, n: int) -> int:
        cnt = 0                         
        for i in range(2, n):
            if self.isPrimes(i):
                cnt += 1
        return cnt

    def isPrimes(self, num):
        #因为如果一个数不是素数是合数， 那么一定可以由两个自然数相乘得到， 其中一个大于或等于它的平方根，一个小于或等于它的平方根，并且成对出现。
        border = int(sqrt(num))     
        for i in range(i, border + 1):
            if num % 2 == 0:
                return False
        return True
#这种方法会超时
#光标选中想要注释的所有代码，ctrl+/，取消同理。
#多光标选中变量，然后command+shift+l

#2- 埃氏筛
class Solution:
    def countPrimes(self, n: int) -> int:      
        ans = [True] * n
        for i in range(2, int(sqrt(n)) + 1):
            if ans[i]:
                for j in range(i*i, n, i):  #埃氏筛的精髓！🌟
                    ans[j] = False
                
        cnt = 0
        for i in range(2, n):
            if ans[i] == True:
                cnt += 1
        return cnt

leetcode-504 七进制数
class Solution:
    def convertToBase7(self, num: int) -> str:
        if not num:
            return "0"
        
        res = ""
        temp = abs(num)

        while temp:
            q, r = temp // 7, temp % 7  #divmod(temp, 7)
            res += str(r)
            temp = q

        if num < 0:
            res += "-"
        return res[::-1]
#七进制的题目呢就用  除取余数取模计算好了。
    
leetcode- 168 excel表列名称
class Solution:
    def convertToTitle(self, columnNumber: int) -> str:
        ans = list()
        while columnNumber > 0:
            columnNumber -= 1       #为什么要减一？ 因为下面ordA本身就占据一个A，所以实际可以操作的情况就要 -1
            ans.append(chr(columnNumber % 26 + ord("A")))
            columnNumber //= 26
        return "".join(ans[::-1])
#数学题的难点就在这，对于一些细节的把握


leetcode - 172 阶乘后的零
class Solution:
    def trailingZeroes(self, n: int) -> int:
        cnt = 0
        while n > 0:
            cnt += n // 5
            n //= 5
        return cnt
#规律：每隔25出现两个5；每隔125出现3个5 ...所以最后5的个数=n/5 + n/25 + n/125 + n/625 ...
#纯粹数学题

leetcode-67 二进制求和
#这一题题意理解错了 
#主要涉及二进制的处理方式
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        ans, extra = '',0 
        i,j=len(a)-1,len(b)-1
        while i>=0 or j>=0:
            if i >= 0:
                extra += ord(a[i]) - ord('0')
            if j >= 0:
                extra += ord(b[j]) - ord('0')
            ans += str(extra % 2)
            extra //= 2
            i,j = i-1,j-1
        if extra == 1:
            ans += '1'
        return ans[::-1]
#这个二进制的处理思路太新奇了！因为两个字符串，因此从尾巴开始就针对两个单位的数字进行操作
#思路：extra在这里起到进位，保留原位的作用，给个赞，而且因为循环没有初始化，就是可以进位运算！太棒了！

leetcode-415 字符串相加
#利用的双指针
class Solution:
    def addStrings(self, num1: str, num2: str):
        res = ""
        i, j, carry = len(num1) - 1, len(num2) - 1, 0
        while i >= 0 or j >= 0:
            n1 = int(num1[i]) if i >= 0 else 0
            n2 = int(num2[j]) if j >= 0 else 0
            tmp = n1 + n2 + carry
            carry = tmp // 10
            res = str(tmp % 10) + res
            i, j = i - 1, j - 1
        return "1" + res if carry else res  #if carry是什么意思呢？ = carry > 0, carry 就是进位了，如果有进位就加一
#take-away在于同时针对两个字符串的处理。其实可以用ord 和 chr的函数
 
leetcode-462  最少移动次数使数组元素相等 II
class Solution:
    def minMoves2(self, nums: List[int]):
        nums.sort()
        left = 0
        right = len(nums) - 1
        steps = 0
        while left < right:
            steps += (nums[right] - nums[left])
            left += 1
            right -= 1
        return steps
#这题重要的是数学思想，重数和平均数都没有办法解决，中位数才能解决。然后利用双指针解决。


leetcode-169 多数元素
#1- 计数  2- 排序 3- 摩尔投票法
#计数
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        counts = collections.Counter(nums) 
        return max(counts.keys(), key=counts.get)
#collections.Counter()可以查找出元素与出现的次数；key为元素，value为出现的次数
#counts这里就是一个对象。counts.keys()是求出counts中的键； key=counts.get是按照counts的键值value查询。
#传统思想：计数，然后找到最大的那个

#排序
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        nums.sort()
        return nums[len(nums) // 2]
#排序过后，因为多数元素肯定会占据超过一半的值，因此。直接返回中位数就行。

#摩尔投票法
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        count = 0
        candidate = None

        for num in nums:
            if count == 0:
                candidate = num
            count += (1 if num == candidate else -1)

        return candidate
# 为何这行得通呢？
# 投票法是遇到相同的则票数 + 1，遇到不同的则票数 - 1。
# 且“多数元素”的个数> ⌊ n/2 ⌋，其余元素的个数总和<= ⌊ n/2 ⌋。
# 因此“多数元素”的个数 - 其余元素的个数总和 的结果 肯定 >= 1。
# 这就相当于每个“多数元素”和其他元素 两两相互抵消，抵消到最后肯定还剩余至少1个“多数元素”。
# 无论数组是1 2 1 2 1，亦或是1 2 2 1 1，总能得到正确的候选人。

leetcode-326 #3的幂
#常规迭代
class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        if n == 0:
            return False
        while n % 3 == 0:
            n = n//3
        return n == 1
#其他都是用数学方法写出来的。

leetcode-367 有效的完全平方数
#思路：1.调库sqrt 2.二分查找 3.牛顿迭代 4.数学
#二分查找
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        if num < 2:
            return True
        
        left, right = 2, num // 2#这里只是处理了一下平方根的边界问题，并没有真正开始二分；
        
        while left <= right:
            x = left + (right - left) // 2
            guess_squared = x * x
            if guess_squared == num:
                return True
            if guess_squared > num:
                right = x - 1
            else:
                left = x + 1
        return False
#LogN
#双指针的left 与 right <= 与 < 一直都很有趣。这一题left，right都有可能取到，所以利用<=

#牛顿迭代
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        if num < 2:
            return True
        
        x = num // 2
        while x * x > num:
            x = (x + num // x) // 2
        return x * x == num
#LogN
#不会先不看了 

#数学
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        num1 = 1
        while num > 0:
            num -= num1
            num1 += 2
        return num == 0
#规律：1 4=1+3 9=1+3+5 16=1+3+5+7以此类推，模仿它可以使用一个while循环，不断减去一个从1开始不断增大的奇数，若最终减成了0，说明是完全平方数，否则，不是。
#Log N

leetcode-238 除自身以外数组的乘积
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [0] * n

        k = 1
        for i in range(n):
            res[i] = k
            k *= nums[i]
        k = 1
        for i in range(n - 1, -1, -1):
            res[i] *= k     #因为是第二遍遍历，因此不能再直接=了
            k *= nums[i]
        return res
#这一题的思路与解法非常有意思。左边累乘+右边累乘
#注意两个for并列的时候是O2n = On，而不是On2


leetcode-628 三个数的最大乘积
class Solution:
    def maximumProduct(self, nums):
        nums.sort()
        return max(nums[-1]*nums[-2]*nums[-3],nums[0]*nums[1]*nums[-1])
# 如果数组中全是非负数，则排序后最大的三个数相乘即为最大乘积；如果全是非正数，则最大的三个数相乘同样也为最大乘积。
# 如果数组中有正数有负数，则最大乘积既可能是三个最大正数的乘积，也可能是两个最小负数（即绝对值最大）与最大正数的乘积。

leetcode-7 整数反转
class Solution:
    def reverse(self, x: int) -> int:
        res = 0
        x1 = abs(x)
        while(x1!=0):
            temp = x1%10
            if res > 214748364 or (res==214748364 and temp>7):
                return 0
            if res<-214748364 or (res==-214748364 and temp<-8):
                return 0
            res = res*10 +temp
            x1 //=10
        return res if x >0 else -res