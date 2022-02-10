# 1 - Two Sum
import re


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap = dict()
        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in hashmap:
                return [i, hashmap[complement]]
            hashmap[nums[i]] = i
"""
class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> hashmap = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (hashmap.containsKey(complement)) {
                return new int[] {hashmap.get(complement), i};
            }
            hashmap.put(nums[i], i);
        }
        return null; 
    }
 }"""

 # 15 - Three Sum
 # 先说思路：1. Two Sum进化 2. Two pointers
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort() # Solution非要加sort，有点强迫症了
        for i in range(len(nums)):
            if nums[i] > 0: 
                break
            # 这个理解很关键，答案是可以出现相同的数字的，但是不能出现相同的答案。
            if i == 0 or nums[i - 1] != nums[i]:
                self.twoSum(nums, i, res)
        return res
    def twoSum(self, nums: List[int], i: int, res: List[List[int]]):
        lo, hi = i + 1, len(nums) - 1
        while lo < hi:
            sum = nums[i] + nums[lo] + nums[hi]
            if sum > 0:
                hi -= 1
            elif sum < 0:
                lo += 1
            else:
                res.append([nums[i], nums[lo], nums[hi]])
                hi -= 1
                lo += 1
                # 这个while也很有灵性，是为了防止出现相同答案。main中的限制i的，这里的是用来限制lo的。
                # 我们固定了左侧的i，然后移动lo和hi；其实固定右侧hi，移动i和low也是可以实现的，一样的道理，只是方向不一样。
                while lo < hi and nums[lo] == nums[lo - 1]:
                    lo += 1
# 本体思路就是2pointer，首先根据条件锚定i，然后觉得lo和hi的取值范围，通过判断三个数的和去判断是否最终入res
"""
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        Array.sort(nums);
        List<List<Integer>> res = new ArrayList<>()；
        for (int i = 0; i < nums.length; i++) {
            if ( i == 0 || nums[i - 1] != nums[i]) {
                twoSum(nums, i ,res);
            }
        } 
        return res
    }
    void twoSum(int[] nums, int i, List<List<Integer>> res) {
        int lo = i + 1, hi = nums.length - 1;
        while (lo < hi) {
            int sum = nums[i] + nums[lo] + nums[hi];
            if (sum < 0) {
                ++lo;
            } else if (sum > 0) {
                --hi;
            } else {
                res.add(Arrays.asList(nums[i], nums[lo++], nums[hi--]));
                while (lo < hi && nums[lo] == nums[lo - 1])
                    ++lo;
            } 
        }
    }
}
"""
# 再来个2 Sum进化版本
# 我们知道了一个数，就知道了target，那么从剩下的数字中，找到两个数的sum为target就成，组成res的一个组合
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort() # Solution非要加sort，有点强迫症了
        for i in range(len(nums)):
            if nums[i] > 0: 
                break
            # 这个理解很关键，答案是可以出现相同的数字的，但是不能出现相同的答案。
            if i == 0 or nums[i - 1] != nums[i]:
                self.twoSum(nums, i, res)
        return res
    def twoSum(self, nums: List[int], i: int, res: List[List[int]]):
        seen = set()
        j = i + 1
        while j < len(nums):
            complement = - nums[i] - nums[j]
            if complement in seen:
                res.append([nums[i],nunms[j], complement])
            while j + 1 < len(nums) and nums[j] == nums[j + 1]:
                j += 1
            seen.add(nums[j])
            j += 1



# 18 4-Sum 没想到吧
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
                # 这里的限制这么思考：最开始经过限制的k为3，不管k=4的时候是什么情况，k=3这一位答案，只能计算一遍；
                # 有点拗口，抽象理解成k=3这一位答案只能是不同的数字，其他怎么组合是其他地方的事情
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

"""
class Solution {
    public List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        return kSum(nums, target, 0, 4);
    }
	
    public List<List<Integer>> kSum(int[] nums, int target, int start, int k) {
        List<List<Integer>> res = new ArrayList<>();
        if (start == nums.length) {
            return res;
        }
        

        int average_value = target / k;
        if  (nums[start] > average_value || average_value > nums[nums.length - 1]) {
            return res;
        }
        
        if (k == 2) {
            return twoSum(nums, target, start);
        }
    
        for (int i = start; i < nums.length; ++i) {
            if (i == start || nums[i - 1] != nums[i]) {
                for (List<Integer> subset : kSum(nums, target - nums[i], i + 1, k - 1)) {
                    res.add(new ArrayList<>(Arrays.asList(nums[i])));                           //Arrays.asList就是将数组转化为list，在python中就是[nums[lo], nums[hi]]这样的操作。
                    res.get(res.size() - 1).addAll(subset);
                }
            }
        }
    
        return res;
    }
	
    public List<List<Integer>> twoSum(int[] nums, int target, int start) {
        List<List<Integer>> res = new ArrayList<>();
        int lo = start, hi = nums.length - 1;
    
        while (lo < hi) {
            int currSum = nums[lo] + nums[hi];
            if (currSum < target || (lo > start && nums[lo] == nums[lo - 1])) {
                ++lo;
            } else if (currSum > target || (hi < nums.length - 1 && nums[hi] == nums[hi + 1])) {
                --hi;
            } else {
                res.add(Arrays.asList(nums[lo++], nums[hi--]));
            }
        }
                                                          
        return res;
    }
}

"""

# 410 Split Array Largest Sum
class Solution:
    def splitArray(self, nums: List[int], m: int) -> int:
        n = len(nums)
        
        # Create a prefix sum array of nums.
        prefix_sum = [0] + list(itertools.accumulate(nums)) # prefix_sum就是0～len(nums)的位置都有数，比如i的data就是i的前缀和
        
        
        def get_min_largest_split_sum(curr_index: int, subarray_count: int):
            # 如果subarray_count为1，意味着还需要一个subarray，这个时候该subarray的sum直接用最开始的prefix_sum计算就好
            if subarray_count == 1:
                return prefix_sum[n] - prefix_sum[curr_index]
        
            minimum_largest_split_sum = prefix_sum[n]
            # 因为是连续的subarray，所以如果subarray_count还有多余的，那么我们不用把剩下的所有index全给tranverse掉。
            for i in range(curr_index, n - subarray_count + 1):
                
                # first_split_sum是暂定的、当前的sum综合
                first_split_sum = prefix_sum[i + 1] - prefix_sum[curr_index]


                # largest_split_sum存放的是目前的遍历过的所有片段的最大sum，也是我们要找的数据。
                largest_split_sum = max(first_split_sum, 
                                        get_min_largest_split_sum(i + 1, subarray_count - 1))

                # Find the minimum among all possible combinations.
                # 思考一个问题，所有的combinations是如何被纳入考虑，从而允许选出最小值的？-> 类似全局变量
                # 这种递归又类似树，类似递归栈中栈，所以各自继承各自的context，运行是单线程的，所有栈的结果最终会巧妙地集合执行在全局变量中。
                minimum_largest_split_sum = min(minimum_largest_split_sum, largest_split_sum)

                if first_split_sum >= minimum_largest_split_sum:
                    break
            
            return minimum_largest_split_sum
        
        return get_min_largest_split_sum(0, m)




# 560. Subarray Sum Equals K
# 这一题用两种写法吧（java），奇怪的是用python老师tle
"""
public class Solution {
    public int subarraySum(int[] nums, int k) {
        int count = 0;
        for (int start = 0; start < nums.length; start++) {
            int sum = 0;
            for (int end = start; end < nums.length; end++) {
                sum += nums[end]
                if (sum == k) {
                    count++;
                }
            }
        }
        return count;
    }
}
# 自己也用python写了这个方法，复杂度为n2。深深理解递归栈，已经for循环的应用文，这种机制其实帮助你排除了很多可能性。


// 第二种方法，利用hashmap，有点数学的意思了。复杂度为n，只用遍历一次就成了。
public class Solution { 
    public int subarraySum(int[] nums, int k) {
        int count = 0, sum = 0;
        // 这里的map相当于维护的是prefix sum，通过不同位的sum相差，可以得到连续几位数的和是否多少，是否为target。
        // value是值目前所有前缀和的组合中，和为key的组数。
        Hashmap<Integer, Integer> map = new Hashmap<>();
        map.put(0, 1);
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            // 刚开始sum<k的话，是不会进到这个循环的，如果之后发现有，那么就
            if (map.containsKey(sum - k)){
                count += map.getKey(sum - k);
            }
            map.put(sum, map.getOrDefault(sum, 0) + 1) // 如果没有sum这个key的话，就为0，那么map put的value就是0+1 = 1；
        }
        return count;0
    }
}

"""


# 2. Add Two Numbers
# 因为本题的特殊性，链表都是倒着的，所以也刚好方便我们进行顺序操作。
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        result = ListNode(0)
        result_tail = result
        carry = 0 #
        
        # 这个while很灵性，每次我们就针对l1,l2同时操作，如果发现两侧不齐的情况，也不用管了这下。
        while l1 or l2 or carry:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            carry, out = divmod(val1 + val2 + carry, 10)
            
            # 这里的result_tail.next就相当于是dummy node的操作。
            result_tail.next = ListNode(out)
            result_tail = result_tail.next
            
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
            
        return result.next

# 13. Roman into Integer
values = {
    "I": 1,
    "V": 5,
    "X": 10,
    "L": 50,
    "C": 100,
    "D": 500,
    "M": 1000,
}

class Solution:
    def romanToInt(self, s: str) -> int:
        total = 0
        i = 0
        while i < len(s):
            if i + 1 < len(s) and values[s[i]] < values[s[i + 1]]:
                total += values[s[i+1]] - values[s[i]]
                i += 2
            else:
                total += values[s[i]]
                i += 1
        return total
# 这一题唯一的难点在于不清楚罗马数字。现在记好了，罗马数字它本身的局限是只会有一个在前面。



# 134. Gas Station
# 这一题重要的是思想。题意是圆形轨道
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        n = len(gas)
        # 这两个变量很重要。
        # total_tank看整个路程是否可以走完。if > 0 一定意味着存在某条路可以走完。否则return -1
        # cur_tank看当前累计的，如果<0那么直接从下一个节点开始走，总之会找到一个可能的节点的。
        cur_tank, total_tank = 0, 0
        start_position = 0
        for i in range(n):
            total_tank += gas[i] - cost[i]
            cur_tank += gas[i] - cost[i]
            if cur_tank < 0:
                start_position = i + 1
                cur_tank = 0
        return start_position if total_tank >= 0 else -1
"""
class Solution {
    public int canCompleteCircuit(int[] gas, int[] cost) {
        int n = gas.length;
        int total_tank = 0, cur_tank = 0;
        int start_position = 0;
        
        for (int i = 0; i < n; i++) {
            total_tank += gas[i] - cost[i];
            cur_tank += gas[i] - cost[i];
            if(cur_tank < 0) {
                start_position = i + 1;
                cur_tank = 0;
            }
        }
        return total_tank >= 0 ? start_position : -1;
    }
}
"""

# 166. Fraction to Recurring Decimal
# 首先可以明确，两个数相除要么是「有限位小数」，要么是「无限循环小数」，而不可能是「无限不循环小数」。
# 这道题本质上来讲就是用来模拟人类计算余数的情况
class Solution:
    def fractionTodecimal(self, numerator: int, denominator: int) -> str:
        
        # 可以整除
        if numerator % denominator == 0: 
            return str(numerator // denominator)
        
        res = []
        # 判断两个数是否为同号，因为涉及到乘除。
        if (numerator < 0) != (denominator < 0):
            res.append('-')

        # 整数部分
        numerator = abs(numerator)
        denominator = abs(denominator)
        integerPart = (numerator // denominator)
        res.append(str(integerPart))
        res.append('.')

        # 小数部分
        indexMap = {} # dict()
        remain = numerator % denominator
        # remain不为0， 且remain没有出现过；我们每次都为添一位（模拟除法）
        while remain and remain not in indexMap:
            indexMap[remain] = len(res)
            remain *= 10
            res.append(str(remain//denominator))
            remain %= denominator
        # 跳出while且有remain意味着有循环
        if remain:
            # 这个好有灵性，通过push进hashtable的value为当前的len，即在这一步可以用作index，溯洄那个位置。
            insertIndex = indexMap[remain]
            res.insert(insertIndex,'(')
            res.append(')')
        return ''.join(res)



# 202. Happy Number
# 简简单单模拟一下计算过程
class Solution:
    def isHappy(self, n: int) -> bool:
        def get_next(n):
            total = 0
            while n > 0:
                n, digit = divmod(n, 10)
                total += digit ** 2
            return total
        seen = set()
        while n != 1 and n not in seen:
            seen.add(n)
            n = get_next(n)
        
        # 如果最后到这里是1，就意味着跳出循环了；如果不为1，意味着是因为n在seen里面出现过多次了。
        return n == 1


# 238. Product of Array Except Self
# 这道题的关键是，针对每一个index，用了两个for 2n都去计算了它的前缀和和后缀和，这个解题思路很OK，就和hashmap/set能记录是否出现过一样。
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        
        # The length of the input array 
        length = len(nums)
        
        # The answer array to be returned
        answer = [0]*length
        
        # answer[i] contains the product of all the elements to the left
        # Note: for the element at index '0', there are no elements to the left,
        # so the answer[0] would be 1
        answer[0] = 1
        for i in range(1, length):
            
            # answer[i - 1] already contains the product of elements to the left of 'i - 1'
            # Simply multiplying it with nums[i - 1] would give the product of all 
            # elements to the left of index 'i'
            answer[i] = nums[i - 1] * answer[i - 1]
        
        # R contains the product of all the elements to the right
        # Note: for the element at index 'length - 1', there are no elements to the right,
        # so the R would be 1
        R = 1
        for i in reversed(range(length)):
            
            # For the index 'i', R would contain the 
            # product of all elements to the right. We update R accordingly
            answer[i] = answer[i] * R
            R *= nums[i]
        
        return answer


# 273. Integer to English Words
# 垃圾题，不用看了
class Solution:
    def numberToWords(self, num):
        def one(num):
            switcher = {
                1: 'One',
                2: 'Two',
                3: 'Three',
                4: 'Four',
                5: 'Five',
                6: 'Six',
                7: 'Seven',
                8: 'Eight',
                9: 'Nine'
            }
            return switcher.get(num)

        def two_less_20(num):
            switcher = {
                10: 'Ten',
                11: 'Eleven',
                12: 'Twelve',
                13: 'Thirteen',
                14: 'Fourteen',
                15: 'Fifteen',
                16: 'Sixteen',
                17: 'Seventeen',
                18: 'Eighteen',
                19: 'Nineteen'
            }
            return switcher.get(num)
        
        def ten(num):
            switcher = {
                2: 'Twenty',
                3: 'Thirty',
                4: 'Forty',
                5: 'Fifty',
                6: 'Sixty',
                7: 'Seventy',
                8: 'Eighty',
                9: 'Ninety'
            }
            return switcher.get(num)
        

        def two(num):
            if not num:
                return ''
            elif num < 10:
                return one(num)
            elif num < 20:
                return two_less_20(num)
            else:
                tenner = num // 10
                rest = num - tenner * 10
                return ten(tenner) + ' ' + one(rest) if rest else ten(tenner)
        
        def three(num):
            hundred = num // 100
            rest = num - hundred * 100
            if hundred and rest:
                return one(hundred) + ' Hundred ' + two(rest) 
            elif not hundred and rest: 
                return two(rest)
            elif hundred and not rest:
                return one(hundred) + ' Hundred'
        
        billion = num // 1000000000
        million = (num - billion * 1000000000) // 1000000
        thousand = (num - billion * 1000000000 - million * 1000000) // 1000
        rest = num - billion * 1000000000 - million * 1000000 - thousand * 1000
        
        if not num:
            return 'Zero'
        
        result = ''
        if billion:        
            result = three(billion) + ' Billion'
        if million:
            result += ' ' if result else ''    
            result += three(million) + ' Million'
        if thousand:
            result += ' ' if result else ''
            result += three(thousand) + ' Thousand'
        if rest:
            result += ' ' if result else ''
            result += three(rest)
        return result