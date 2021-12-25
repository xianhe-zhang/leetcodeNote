'''======================== 哈希表 ======================='''
# 1 Two Sum
from typing import Collection


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        #enumerate这个API经常用
        for index, num in enumerate(nums):      
            diff = target - num
            if diff in dic:
                return [dic[diff],  index]
            else:
                dic[num] = index


# 217 Contains Duplicates
# 这种题直接用set特性，它不想么
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        doneNums = set(nums)
        return len(nums) != len(doneNums)

# class Solution:
#     def containsDuplicate(self, nums: List[int]) -> bool:
#         done = set()
#         for num in nums:
#             if not done.add():
#                 return True
#         return False
# 不能这么写，因为python集合set()，add失败后不会有任何notification


# 594 Longest Harmonious Subsequence 最长和谐子序列
class Solution(object):
    def findLHS(self, nums):
        ans = 0
        #返回的是集合 Counter({2: 3, 3: 2, 1: 1, 5: 1, 7: 1}) key为nums里的值, value为出现的次数
        d = collections.Counter(nums)
        for num in nums:
            if num + 1 in d: #关键
                ans = max(ans, d[num] + d[num + 1])
        return ans

# 128 Longest Consecutive Sequence 最长连续序列
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        resStreak = 0
        numsSet = set(nums)
        for num in numsSet:
            # 只有num是序列的开始时，才会进入逻辑判断
            if num - 1 not in numsSet:
                curStreak = 1   

                # 如果num+1 在set中，意味着序列可以往后扩展。
                while num + 1 in numsSet:
                    curStreak += 1
                    num += 1
                resStreak = max(resStreak, curStreak)

        return resStreak

#这道题用set来说其实不难，自己也能写，但是一些细节估计实现起来还是不那么熟练



'''======================== 字符串 ======================='''
# 242 Valid Anagram 有效的字母异位符
# 好几种方法，一个个来。
# str -> list -> sort. 注意sorted VS sort的区别，前者是返回一个新的list，并且可以针对多种数据类型的list；后者是直接在原来基础上排列
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        sl = sorted(list(s))
        tl = sorted(list(t))
        return sl == tl
#因为涉及到排序，所以复杂度时n*logn

#利用Counter API，也是hashTable
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return collections.Counter(s) == collections.Counter(t)


# 409 Longest Palindrome 最长回文串
# dictionary的api： .key(), .value() .keys(), .values()
class Solution:
    def longestPalindrome(self, s: str) -> int:
        ans = 0
        #这个是降序排列的
        d = collections.Counter(s)
        # flag是为了判断
        flag = 0
        for i in d:
            # 如果i出现过偶数次，那么满足，添加就好
            if d[i] % 2 == 0:
                ans += d[i]
            # 如果flag为0，意味着该奇数次没有添加过
            elif flag == 0:
                ans += d[i]
                flag = 1
            # 遇到出现奇数次的i，我们可以往我们的回文串中添加d[i] - 1次
            else:
                ans += d[i] - 1
        return ans
# 优化方法：这里的判断其实有点多余，我们可以用多种方法（比如我们可以直接判断ans是奇数还是偶数，如果是偶数，就意味着还没有添加奇数项，那么直接添加就好了。



# 205  Isomorphic Strings 同构字符串
# 自己的思路转化成 int list进行比较； 也可以据此通过项项比较。
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        diff = ord(s[0]) - ord(t[0])
        for i in range(1,len(s)):
            if ord(s[i]) - ord(t[i]) != diff :
                return False 
        return True
# ❌，因为diff不一定相同就可以满足提议，你好愚蠢。
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        #helper来判断字符是否不同，但是这种思路没有办法解决全是不同字母的str...头秃
        def helper(s:str):
            ans = 1
            temp = 1
            for i in range(1, len(s)):
                if s[i] != s[i - 1]:
                    temp += 1
                ans +=  temp
            return ans
        return helper(s) == helper(t)
# ❌，没有理解题意，同构的意味着字母之间拥有着映射关系，而非string的结构类似就可以了。

#正确答案，思路：通过记录map中的映射关系，如果在之后的string中映射关系有冲突，那么就不是同构了。
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        mapS = dict()
        mapT = dict()
        for i in range(len(s)):
            cS, cT = s[i], t[i]
            if cS not in mapS and cT not in mapT:
                mapS[cS] = i + 1
                mapT[cT] = i + 1
            elif cS not in mapS or cT not in mapT or mapS[cS] != mapT[cT]:
                return False
        return True
#这里也可以写helper


# 647 Palindromic Substrings 回文子串
# 两种解法：1.DP 2.中心扩展法，不过复杂度都是N2，这好像是回文cannot avoid 的 complexity
# 状态转移的理解：因为回文的判断涉及到每一个字符，因此smaller的substring为palindromic时，大的回文串才可能是
class Solution:
    def countSubstrings(self, s: str) -> int:
        ans = 0
        dp = [ [False for _ in range(len(s))] for _ in range(len(s))]
        for j in range(len(s)):
            for i in range(j + 1): #我们想让i取到j，所以这里放了j+1
                if s[j] == s[i] and (j - i < 2 or dp[i+1][j-1]): #要么j~i的间距最多为3位数；要么子字符串也为回文。  
                    dp[i][j] = True
                    ans += 1
        return ans


#中心扩展法
class Solution:
    def countSubstrings(self, s: str) -> int:
        ans = 0
        # center 可以取一位或者两位数，共有这么多种取法
        for center in range(2 * len(s) - 1):
            #left 和 right是根据中心点center开始的地方，理想状态下指向同一个点center，或者紧密相连的两个点
            left = center // 2
            right = left + center % 2

            while left >=0 and right < len(s) and s[left] == s[right]:
                ans += 1
                right += 1
                left -= 1
        return ans
# 核心思路就是找到所有可以组成回文substring的中点，可能是一个数，也可能是两位数，然后依次往左右扩展
# 精彩的点在于如何找center中心点
# 复杂度难以避免为N2


# 9 Panlindrome number 回文数
# 当然了，这里也可以用运算规则将int -> string
class Solution:
    def isPalindrome(self, x: int) -> bool:
        s = str(x)
        left = 0 
        right = len(s) - 1
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True

class Solution:
    def isPalindrome(self, x: int) -> bool:
        return str(x) == str(x)[::-1]

"""
反转字符串的方法：
# 切片器
def string_reverse1(string):
    return string[::-1]

# 也是Join的方法
def string_reverse2(string):
    t = list(string)
    l = len(t)
    for i,j in zip(range(l-1, 0, -1), range(l//2)):
        t[i], t[j] = t[j], t[i]
    return "".join(t)
 
# DP的方法
def string_reverse3(string):
    if len(string) <= 1:
        return string
    return string_reverse3(string[1:]) + string[0]
 
# JOIN的方法 
def string_reverse5(string):
    #return ''.join(string[len(string) - i] for i in range(1, len(string)+1))
    return ''.join(string[i] for i in range(len(string)-1, -1, -1))
"""


# 696 Count Binary Substrings 计数二进制子字符串
class Solution:
    def countBinarySubstrings(self, s: str) -> int:
        ans = 0 
        ptr = 0 #bref. for pointer
        last = 0
        while ptr < len(s):
            count = 0
            char = s[ptr]
            while ptr < len(s) and char == s[ptr]:
                ptr += 1
                count += 1
            # 0和1分组，答案只与最小的值有关。
            ans += min(last, count)
            last = count
        return ans
# 这题重要的是思路，要将0和1分组计数，然后组组进行比较，将结果放进去。
# 这题的take-away: 因为要遍历整个list，但是其中有一些index不需要进行处理，只是计数，就可以用到这题中的双while结构

