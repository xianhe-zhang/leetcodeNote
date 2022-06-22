# 136. Single Number
# ^异或运算，相异时为1，一样的话会归0
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        number = 0
        for n in nums:
            number ^= n
        return number


# 191. Number of 1 Bits
# API熟记，bin和count
class Solution:
    def hammingWeight(self, n: int) -> int:
        return bin(n).count("1")

# & 位与运算，相同时为1，Trick: x&1==1？ 其实就是判断x的末位是不是1，然后接下来>>删除末位。
# >>1 其实也相当于 /2
class Solution:
    def hammingWeight(self, n: int) -> int:
        ct = 0
        for i in range(1,33):
            ct += n & 1
            n = n >> 1
        return ct


# 169. Majority Element
# 三种解法：bit manipulation/ divide & conquer/ boyer-moore voting
class Solution:
    def majorityElement(self, nums, lo=0, hi=None):
        def majority_element_rec(lo, hi):
            if lo == hi:
                return nums[lo]
            # recurse on left and right halves of this slice.
            mid = (hi-lo)//2 + lo
            left = majority_element_rec(lo, mid)
            right = majority_element_rec(mid+1, hi)
            # if the two halves agree on the majority element, return it.
            if left == right:
                return left

            # otherwise, count each element and return the "winner".
            left_count = sum(1 for i in range(lo, hi+1) if nums[i] == left)
            right_count = sum(1 for i in range(lo, hi+1) if nums[i] == right)

            return left if left_count > right_count else right

        return majority_element_rec(0, len(nums)-1)
# 分治的解法主要是理解思想：左右两边不断拆开，取出左边较大的，右边较大的，如果两个值不一样，就比较左右两个区间内各自出现了多少次，然后更新。

# Boyer-Moore的算法有点意思
class Solution:
    def majorityElement(self, nums):
        count = 0
        candidate = None

        for num in nums:
            if count == 0:
                candidate = num
            count += (1 if num == candidate else -1)

        return candidate

# bit的算法，有点意思 
class Solution:
    def majorityElement(self, nums):
        result = 0
        # 去看所有num每一位上有多少1
        for i in range(32):
            cnt = 0
            for j in range(len(nums)):
                # 实现的关键
                if nums[j] & 1<<i:
                    cnt += 1
            # 如果该位上1有一半以上，利用位或运算将1落进来
            if cnt > len(nums)/2:
                result |= 1<<i
        return result
            

# 190. Reverse Bits
# 通过对位的调整，一个1一个1的“对齐”
# 还是对位运算的不熟悉
class Solution:
    def reverseBits(self, n: int) -> int:
        result, power = 0, 31
        while n:
            # 因为input给的是32位，所以我们通过& 1 得到最后一位时，直接对result进行移位处理，直接reverse，这里的power很奇妙
            result += (n & 1) << power
            # &1取得最后一位，但是n还是原来的，因此需要做处理
            # n的最后一位在上一行处理完毕，因此需要同步起来。
            n = n >> 1
            # 下一次就需要变了
            power -= 1
        return result
    
# 231. Power of Two
# 位与
# 虽然这题普通的运算也能做，但是复杂度做到了logN
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n == 0:
            return False
        # -x = ~x + 1  -> ~x表示二进制位下取反，即0变1，1变0
        return n & (-n) == n
# 理解这个操作需要画图，拿4和6来举例
# 4    : 0 0 0 0 0 1 0 0 
# -x   : 1 1 1 1 1 0 1 1 + 1
# -x   : 1 1 1 1 1 1 0 0
# x&-x : 0 0 0 0 0 1 0 0
# ---------------------------
# 6    : 0 0 0 0 0 1 1 0
# -x   : 1 1 1 1 1 0 1 0
# x&-x : 0 0 0 0 0 0 1 0

# 389. Find the Difference
# ^异或运算；bit manipulation就7中: & | ^ ~ >> <<  分别是位与，位或，异或，取反，左移，右移
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:

        # Initialize ch with 0, because 0 ^ X = X
        # 0 when XORed with any bit would not change the bits value.
        ch = 0

        # XOR all the characters of both s and t.
        for char_ in s:
            ch ^= ord(char_)

        for char_ in t:
            ch ^= ord(char_)

        # What is left after XORing everything is the difference.
        return chr(ch)



# 268. Missing Number
# sorting和bit的解法，复杂度分别为logN和1
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
#         nums.sort()
#         if nums[0] != 0:return 0
        
#         for i in range(1, len(nums)):
#             if nums[i] - nums[i-1] != 1:
#                 return nums[i] - 1
#         return nums[-1] + 1

        missing = len(nums)
        # missing给了初始化赋值与i刚好组成0～n的完全图谱，然后与nums中实际的val进行对对碰，留下来的就是missing。
        for i, val in enumerate(nums):
            missing ^= i ^ val
        return missing
        
# 78. Subsets
# 这里用的是backtracking，复杂度是N*2^N 空间是N
"""
cur[:] 和 cur.copy() 都是shallow copy 不是deep copy
shallow coyp就是新建一个对象 然后把pointer指向原来的
deep copy是ptr指向所有的new出来的
这题用shallow copy就是因为我们的cur在每次的dfs中是会被初始化的。
"""
# 利用k控制答案的长度，利用first和i+1控制subset的选择。
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        # 方法内初始化需要不进行传参才可以哦～
        def dfs(first = 0, cur = []):
            if first == k:
                output.append(cur[:])
                return 
            for i in range(first, n):
                cur.append(nums[i])
                dfs(i + 1, cur) 
                cur.pop()
        
        output = []
        n = len(nums)
        for k in range(n + 1):  # 复杂度N
            dfs()
        return output

# 338. Counting Bits
# 利用API很简单，用BIT的话要与DP联系起来
class Solution:
    def countBits(self, n: int) -> List[int]:
        res = []
        for i in range(n+1):
            res.append(str(bin(i)).count('1'))
        return res
# ans[x]与ans[x//2]的关系其实就是前者和后者大数部分的1肯定相同，小数部分的就是余数部分有没有相等的。
class Solution:
    def countBits(self, n: int) -> List[int]:
        ans = [0] * (n + 1)
        for x in range(1, n + 1):
            # x // 2 is x >> 1 and x % 2 is x & 1
            ans[x] = ans[x >> 1] + (x & 1) 
        return ans 
    

class Solution:
    def countBits(self, n: int) -> List[int]:
        ans = [0] * (n + 1)
        for x in range(1, n + 1):
            ans[x] = ans[x & (x - 1)] + 1
        return ans 


# 1318. Minimum Flips to Make a OR b Equal to c
# 自己写的
class Solution:
    def minFlips(self, a: int, b: int, c: int) -> int:
        flips = 0
        for i in range(32):
            ta, tb, tc = a>>i & 1, b>>i & 1, c>>i & 1
            if tc == 1:
                if ta == 0 and tb == 0:
                    flips += 1
            else:
                if ta == 1 and tb == 1:
                    flips += 2
                elif ta == 1 or tb == 1:
                    flips += 1
        return flips

# 89. Gray Code
# 我觉得这一题可以跳过了，因为如果没有pre graycode knowledge没有发现它的规律，是很难想出来的
# 时间复杂度为2^n，空间复杂度为n
class Solution:
    def grayCode(self, n: int) -> List[int]:
        res = [0]
        for i in range(1, 2**n):
            # i & -i 是取i最后一个1位。
            res.append(res[-1] ^ (i & -i))
        return res


# 2293. Min Max Game 周赛第一题我的写法
class Solution:
    def minMaxGame(self, nums: List[int]) -> int:
        while len(nums) > 1:
            temp = []
            n = int(len(nums) //2)
            for i in range(n):
                if i % 2 == 0:
                    temp.append(min(nums[2*i], nums[2*i + 1]))
                else:
                    temp.append(max(nums[2*i], nums[2*i + 1]))
                    
            nums = temp
        return nums[0]
# 首先我的时空复杂度为N，我是用len(nums)作信号
# 大神们的写法是什么样的呢？用n = len(nums)作信号，但是不用temp做辅助而是直接在nums中进行操作，在结尾n//2

class Solution:
    def partitionArray(self, nums: List[int], k: int) -> int:
        nums.sort()
        i = 0 
        result = 0
        while i < len(nums):
            j = i
            while j < len(nums) and nums[i] + k >= nums[j]:
                j += 1
            i = j
            result += 1
        return result
    
class Solution:
    def partitionArray(self, nums: List[int], k: int) -> int:
        nums.sort()
        a = [nums[0]]
        for i in nums[1:]:
            if i - a[-1] > k:
                a.append(i)
        return len(a)


# 2294. Partition Array Such That Maximum Difference Is K
class Solution:
    def partitionArray(self, nums: List[int], k: int) -> int:
        nums.sort()
        i = 0 
        result = 0
        while i < len(nums):
            j = i
            while j < len(nums) and nums[i] + k >= nums[j]:
                j += 1
            i = j
            result += 1
        return result
    
class Solution:
    def partitionArray(self, nums: List[int], k: int) -> int:
        nums.sort()
        a = [nums[0]]
        for i in nums[1:]:
            if i - a[-1] > k:
                a.append(i)
        return len(a)
# 从复杂度上来讲，我的更好一点，但从算法设计上讲大神还是大神，我的步骤太多余了。
# 大神的思路：遍历每一个元素只计算区间开头的元素，这是我没想到的。

# 2295. Replace Elements in an Array
class Solution:
    def arrayChange(self, nums: List[int], operations: List[List[int]]) -> List[int]:
        pos = {}
        n = len(nums)
        for i in range(n):
            pos[nums[i]] = i
        for i, j in operations:
            pos[j] = pos[i]
            del pos[i]
        res = [0] * n
        for i, j in pos.items():
            res[j] = i
        return res




# 371. Sum of Two Integers
# ^ XOR x^0=x; x^x = 0
class Solution:
    def getSum(self, a: int, b: int) -> int:      
        return sum([a,b]) 
    
# 搞清楚x^y, (~x&y) << 1的关系就好了
# x^y都是1的时候为0，x&y都是1的时候为1， (x&y)<<1 = carry，carry就是进位，x&y找到哪一位是两个1，<<1进一位
# 这下就明白了两者的关系了，到题目中去接是吧。
class Solution:
    def getSum(self, a: int, b: int) -> int:
        x, y = abs(a), abs(b)
        # ensure x >= y
        if x < y:
            return self.getSum(b, a)  
        sign = 1 if a > 0 else -1
        
        if a * b >= 0:
            # sum of two positive integers
            # 首先y是小的，x是大的，x看作base
            # x=x^y 将所有x为0y为1的地方换做1；将所有两者为1的地方换做0，然后同时用y来记录更新进位
            # 上面这个步骤实现了x与y的不进位情况下的加法
            # 出现进位后，y此时已经更新了，然后再次执行上述操作，直到y没有，从而实现y一位位地加入到x中
            while y:
                x, y = x ^ y, (x & y) << 1
        else:
            # difference of two positive integers
            # 相反的情况要用到～取反，具体逻辑我没想
            while y:
                x, y = x ^ y, ((~x) & y) << 1
        
        return x * sign


