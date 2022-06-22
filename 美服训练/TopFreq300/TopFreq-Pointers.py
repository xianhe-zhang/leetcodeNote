# 387. First Unique Character in a String
class Solution:
    def firstUniqChar(self, s: str) -> int:
        # build hash map : character and how often it appears
        count = collections.Counter(s)
        
        # find the index
        for idx, ch in enumerate(s):
            if count[ch] == 1:
                return idx     
        return -1



# 349. Intersection of Two Arrays
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        # return list(set(nums1).intersection(set(nums2)))
        # return list(set1 & set2)
 
        # if the lists are already sorted and you're told to solve in O(n) time and O(1) space:
        nums1.sort() # assume sorted
        nums2.sort() # assume sorted

        # iterate both nums backwards till at least 1 is empty
        # if num2[j] > num1[i], pop num2
        # if num2[j] < num1[i], pop num1
        # if equal and num not last appended to result, append to result and pop both nums

        result = []

        while nums1 and nums2:
            if nums2[-1] > nums1[-1]:
                nums2.pop()
            elif nums2[-1] < nums1[-1]:
                nums1.pop()
            else:
                # to avoid duplicates
                if not result or nums1[-1] != result[-1]:
                    result.append(nums1[-1])
                nums1.pop()
                nums2.pop()

        return result

# 409. Longest Palindrome

class Solution:
    def longestPalindrome(self, s: str) -> int:
    
        res = 0
        # 这个小细节挺好的，.items(), //2*2
        for ch,val in collections.Counter(s).items():
            res += val // 2 * 2
            if res % 2 == 0 and val % 2 == 1:
                res += 1
            
        return res
            
            
            

# 219. Contains Duplicate II

class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        # temp = [k for k,v in collections.Counter(nums).items() if v > 1]
        # return len(temp) >= 2 and (max(temp) - min(temp)) <= k
        # 理解错题意了
        dic = dict()
        for i, v in enumerate(nums):
            if v in dic and i - dic[v] <= k:
                return True
            dic[v] = i
        return False
            

# 88. Merge Sorted Array
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:

        for i in range(n):
            nums1[i + m] = nums2[i]
        
        # Sort nums1 list in-place.
        nums1.sort()
        
        
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    
        nums1_copy = nums1[:m] 
        
        # Read pointers for nums1Copy and nums2 respectively.
        p1 = 0
        p2 = 0
        
        # Compare elements from nums1Copy and nums2 and write the smallest to nums1.
        for p in range(n + m):
            # We also need to ensure that p1 and p2 aren't over the boundaries
            # of their respective arrays.
            # 这个很精妙，p2便利完的话只剩下p1，直接冲；
            # 或者 p1没遍历完的情况下，满足p1
            if p2 >= n or (p1 < m and nums1_copy[p1] <= nums2[p2]):
                nums1[p] = nums1_copy[p1] 
                p1 += 1
            # 一个else其实包含了两个情况：p1遍历完了/p1小于p2
            else:
                nums1[p] = nums2[p2]
                p2 += 1

# 283. Move Zeroes
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        flag = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[flag] = nums[flag], nums[i]
                flag += 1
        
                
        
# 125. Valid Palindrome
class Solution:
    def isPalindrome(self, s: str) -> bool:
        i, j = 0, len(s)-1
        while i < j:
            # isalum只会判断数字和char
            while i < j and not s[i].isalnum():
                i += 1
            while i < j and not s[j].isalnum():
                j -= 1
            
            if s[i].lower() != s[j].lower():
                return False      
            i += 1
            j -= 1
        return True
        

# 27. Remove Element
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        i = 0
        for x in nums:
            if x != val:
                nums[i] = x
                i += 1
        return i


# 977. Squares of a Sorted Array
# 这个题有个特别有意思的点注意到之后就很简单，O(n)的复杂度肯定不能用sort了，只能遍历，这个时候线性遍历而且要比较大小一般就是双指针
# 这题的规律就是在于两头是极大值，最小值肯定在中间，所以两头往中间缩减，就很容易写出来了。
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        left, right = 0, len(nums) - 1
        result = [0] * len(nums)
        for i in range(len(nums)-1, -1, -1):
            if abs(nums[left]) > abs(nums[right]):
                temp = nums[left]
                left += 1
            else:
                temp = nums[right]
                right -= 1
            result[i] = temp ** 2
        return result
            


# 3. Longest Substring Without Repeating Characters
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        """      
        if not s: return 0
        if len(s) == 1: return 1
        
        seen = dict()
        res = 0
        
        for i in range(len(s)): 
            if s[i] not in seen:
                res += 1
            else: 
                res = max(res, i-seen[s[i]])
                
            seen[s[i]] = i
        return res 
"""
# 错误思路：本题入seen的变量有是第一次进的，也有重复进的，如果是新进的，我可以直接+1，但这里就出错了。
# 因为res是max出来的，是会保留历史记录，但我们求当前记录的时候不能直接在历史top上加，所以会出错。
# 下面是正确思路，用到变量的。
        i = 0
        res = 0
        seen = {}
        for j in range(len(s)):
            if s[j] in seen:
                # 因为碰到的ch不同，所以seen中的index可能更小
                i = max(seen[s[j]], i)
            res = max(res, j - i + 1)
            # 这里为什么用j+1不用j？因为我们想要1-indexed 这样的话开头也能处理！
            seen[s[j]] = j + 1
        return res
# 双指针/sliding window


 
# 781. Rabbits in Forest
class Solution:
    def numRabbits(self, nums: List[int]) -> int:
        count = collections.Counter(nums).items()
        return sum(ceil(v/(k+1))*(k+1) for k, v in count)
# 数学题

# 49. Group Anagrams
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # 利用了每个string里的元素(无关顺序)当作index进行归类
        ans = collections.defaultdict(list)
        for s in strs:
            ans[tuple(sorted(s))].append(s)
        return ans.values()
# 因为tuple是可以被hash的


# prefix sum 前缀和
# 560. Subarray Sum Equals K
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        result = presum = 0
        dic = {0:1}
        
        for n in nums:
            presum += n
            
            # 其实就是2-SUM，如果发现能够组成k了，就加入到result中
            if (presum-k) in dic:
                result += dic[presum-k]
        
            # 更新图谱
            if presum in dic:
                dic[presum] += 1
            else:
                dic[presum] = 1
            
        return result



# 18. 4Sum
# 思路要想好，核心还是利用双指针，在指定区间找到两个sum=target的数字。
# kSum的作用就是将情况分流，分成4-3-2的情况，target也是找到不同的target
# 但是target在缩减的时候需要注意，只能往后缩减[i+1:]，是因为如果考了i，还考虑[:i]的话，其实和之前的重复了。
class Solution:
    def fourSum(self, nums:List[int], target: int) -> List[List[int]]:
        def kSum(nums: List[int], target: int, k: int) -> List[List[int]]:
            res = []
            if not nums: return res

            avg_mark = target // k
            # 首先我们的nums是排过序的，所以目前可以利用avg_mark去判断，我们目前传入该方法是否有必要继续下去。
            if avg_mark < nums[0] or nums[-1] < avg_mark:   #表示目前nums里找不到满足题意是数，所以可以不用再继续下去。
                return res
            if k == 2: return twoSum(nums,target)
            
            for i in range(len(nums)):
                if i == 0 or nums[i - 1] != nums[i]:
                    for subset in kSum(nums[i+1:], target - nums[i], k - 1):
                        res.append([nums[i]] + subset)

            return res
        
        def twoSum(nums: List[int], target: int) -> List[List[int]]:
            res = []
            lo, hi = 0, len(nums) - 1
    
            while lo < hi:
                cur = nums[lo] + nums[hi]
                if cur < target or (lo > 0 and nums[lo] == nums[lo - 1]):
                    lo += 1
                elif cur > target or (hi < len(nums) - 1 and nums[hi] == nums[hi + 1]):
                    hi -= 1
                else:
                    res.append([nums[lo], nums[hi]])
                    lo += 1
                    hi -= 1
            return res

        nums.sort()
        return kSum(nums, target, 4)


# 454. 4Sum II
class Solution:
    def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
        count = 0
        m = collections.defaultdict(int)
        for a in A:
            for b in B:
                m[a+b] += 1
        
        for c in C:
            for d in D:
                count += m[-(c+d)]
        return count
# 这个点子太棒了，其实本质还是将多个数转化成2个sum。不同于4Sum I，这一题有4个nums，而且有隔离，因此允许用n^2的方法。至于4SumI，要一个个遍历了...


# 16. 3Sum Closest
# 这一题如何用diff来判断卡住我了一下。
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        diff = float('inf')
        
        for i in range(len(nums)-2):
            cur = nums[i]
            temp_target = target - cur
            lo, hi = i+1, len(nums)-1
            
            while lo < hi:
                t = nums[lo] + nums[hi]
                temp = abs(temp_target - t)
                # 判断diff，用于更新res
                if temp < diff:
                    diff =temp
                    res = cur + t
                # 正常的双指针遍历操作。
                if t > temp_target:
                    hi -= 1
                elif t < temp_target:
                    lo += 1
                else:
                    return cur + t
            
        return res
        
        
# 424. Longest Repeating Character Replacement
# sliding window类的题，为什么我没看出来。
class Solution:
     def characterReplacement(self, s, k):
        maxf = res = 0
        count = collections.Counter()
        
        for i in range(len(s)):
            # 更新counter
            # ch = s[i]
            count[s[i]] += 1
            # maxf是指窗口内目前有的字母，最高的数
            maxf = max(maxf, count[s[i]])
            # res-maxf也就是剩下的不一样的字母如果小于k的话，那么当前的ch就能到window中，同步更新res
            if res - maxf < k:
                res += 1
            # 否则不更新res，并且删除头部，res其实是窗口的长度。如果碰到合适的就扩展窗口，但如果不满足并不会缩小窗口。！
            else:
                count[s[i - res]] -= 1
        return res


# A subarray or substring will always be contiguous, but a subsequence need not be contiguous. 


# 713. Subarray Product Less Than K
# sliding window
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        left, prod, count = 0, 1, 0

        for right in range(len(nums)):
            prod *= nums[right]            

            while prod >= k and left <= right:                    
                prod /= nums[left]
                left += 1                     
            # 这个怎么理解？为什么right-left+1可以表示拥有的subarray数量？
            # intuition way：left，right分别是window的两侧，right-left+1 == 从left到right所有以right为结点的subarray的数量
            # [1,2,3,4],[2,3,4],[3,4],[4]
            count += right - left + 1                
        return count
        

# 992. Subarrays with K Different Integers
# 这一题的难点在于两个helper相减，这一步都很难想了。
# 那helper是做什么的？找到一共有多少个subarray满足少于k个不同的integer。为什么不能直接一步到位？也可以怎么做呢？遍历所有长度的，针对每一个进行counter，太复杂了
# 那么本题的helper做了什么？遍历一遍，直接记录并且维护window，每一次循环满足条件都记录下来，去看所有的可能性？那么为什么不直接看一种可能性？ 
# 因为会有问题，你的抓手是什么？如果你碰到满足题意左边界停下来了，res+=1，但这个时候我们就忽略了，窗口内的子窗口可能也满足提议，而是直接扩展右窗口，就会跳过一些可能性。
class Solution:
    def subarraysWithKDistinct(self, A, K):
        # k - (k-1)意味着只有k个different number的情况
        # 只用相减，就能得到只为K的结果了
        return self.atMostK(A, K) - self.atMostK(A, K - 1)
    
    # 这里是如果如果最多为k个数字，那么有多少种可能
    def atMostK(self, A, K):
        count = collections.Counter()
        res = i = 0
        for j in range(len(A)):
            # 如果遇到新的数字，那么K减去一
            if count[A[j]] == 0: K -= 1
            # 记录遇到过的J数字
            count[A[j]] += 1
            # 已经碰到满足的sliding window，要左移了！之前是不满足的话左移，这里是满足的话左移动
            while K < 0:
                count[A[i]] -= 1
                if count[A[i]] == 0: K += 1
                i += 1
            # 就是从right point开始，到左边pointer，每一个组合都是OK的，那么在i~j这个窗口中，一共有j-i+1个组合，而且不会重复。
            res += j - i + 1
        return res
    
    
    
# 76. Minimum Window Substring
# 这题的难点在哪？
# sliding window不难，两个while/或者一个for 一个while
# 难在左移右移的条件是如何判断的？
# 1. 利用两个dict去判断
# 2. 利用一个变量保存
# 3. ans利用tuple存起来。
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # 额外变量的使用，两个counter记录出现次数；两个独立变量，一个记录满足了几个字母，一个记录一共需要满足多少个字母，通过比对判断已经满足题意。
        # ans也比较特殊，用了一个元组记录了3个值，长度，左/右；因为最终return的是一个clip
        if not s or not t: return ""
        dict_t = collections.Counter(t)
        required = len(dict_t)
        l = r = 0
        formed = 0  # window中有几个字母满足要求了。
        window_counts = dict()
        ans = (float("inf"), None, None)
        
        while r < len(s):
            character = s[r]    # 右边届的ch
            window_counts[character] = window_counts.get(character, 0) + 1 # 将window Counter更新
            
            # 用两个dict分别保存窗口和t的情况，如果发现窗口的元素满足了，用一个变量保存。
            if character in dict_t and window_counts[character] == dict_t[character]:
                formed += 1
            
            # 满足condition的情况下，我们开始缩小左边届
            while l <= r and formed == required:
                character = s[l]
                if r - l + 1 < ans[0]:
                    ans = (r - l + 1, l, r)
                
                window_counts[character] -= 1
                if character in dict_t and window_counts[character] < dict_t[character]:
                    formed -= 1
                    
                l += 1
            r += 1
        return "" if ans[0] == float("inf") else s[ans[1] : ans[2] + 1]
   
        
