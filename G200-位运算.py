"""
二进制状况下跟1进行 位与& 运算，可以得到末尾是1还是0.然后两个数字之间就可以比较了。
位或 |   只要有1就为1
异或 ^   相异的时候为1
位与 &   都为1的时候才为1
取反 ~   1为0，0为1
"""

# 461 Haming Distance
# 方法一：逐位比较
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        return sum(1 for i in range(32) if x >> i & 1 != y >> i & 1)
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        res = 0
        for i in range(32):
            if x>>i & 1 != y>>i & 1:
                res += 1
        return res


# 方法二：右移统计
# 这里学到了bin()转化为二进制；len()一个二进制，可以看到它的1最高位；
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        return sum(1 for i in range(len(bin(max(x,y))) - 2) if (x >> i & 1) ^ (y >> i & 1))


# 方法三：lowbit
# 通过异或运算，找到相异的值，然后直接count出来。
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        return bin(x ^ y).count('1')

# 136 Single Number
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        return reduce(lambda x, y: x ^ y, nums)
"""
reduce() 函数会对参数序列中元素进行累积。
函数将一个数据集合（链表，元组等）中的所有数据进行下列操作：
用传给 reduce 中的函数 function（有两个参数）先对集合中的第 1、2 个元素进行操作，
得到的结果再与第三个数据用 function 函数运算，最后得到一个结果。


map() 会根据提供的函数对指定序列做映射。
第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。


filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
该接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，
然后返回 True 或 False，最后将返回 True 的元素放到新列表中。
"""

# 268 Missing Number
# 数学/位运算
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n = len(nums)
        total = n * (n + 1) // 2
        arrSum = sum(nums)
        return total - arrSum
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        xor = 0
        for i, num in enumerate(nums):
            xor ^= i ^ num
        return xor ^ len(nums)

# 这里又利用了index可以刚好和我们的一一对应上
# 复杂度都是n，空间是1


"""
非常巧妙但是可以之后再学
------------------------------------------------------------------------------------------------------
a & (-a) 可以获得a最低的非0位 ,比如a的二进制是 ??????10000，取反就是??????01111，
加1就是??????10000。前面?的部分是和原来a相反的，相与必然都是0，所以最后整体相与的结果就是00000010000。
------------------------------------------------------------------------------------------------------
-a在二进制中的表示是补码(2's complement code)形式，即先按位取反再加1

取反得 1111 0101(即1's complement code，反码)
加1得 1111 0110(即2's complement code，补码)
原码(0000 1010) 与 补码(1111 0110) 做与运算(&)，得 0000 0010，即原码 0000 1010的LSB

### 更详细的解释： 我们从右向左看发生了什么：

原码最低非0位右边所有的0，经由取反后全部变为1，反码+1会导致这些1逐位发生进位并变为0，最终进位记到最低非0位
原最低非0位是1，取反后是0，进位到这一位0变成1，不再向左进位
原最低非0位左边的每一位经由取反后 和 原码 进行与运算必为0
"""
# 260
# 数学上述的解释
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        bitmask = 0
        # 遍历求出最终的mask，mask是只出现了一次的两个数的异或
        for i in range(len(nums)):
            bitmask ^= nums[i]

        # 取异或值最后一个二进制位为 1 的数字作为 mask，如果是 1 则表示两个数字在这一位上不同。
        rightOne = bitmask & -bitmask
        res = 0

        # 通过与这个 mask 进行与操作，如果为 0 的分为一个数组，为 1 的分为另一个数组。
        # 这样就把问题降低成了：“有一个数组每个数字都出现两次，有一个数字只出现了一次，求出该数字”。
        # 对这两个子问题分别进行全异或就可以得到两个解。也就是最终的数组了。
        for i in range(len(nums)):
            if rightOne & nums[i] != 0:
                res ^= nums[i]
        return [res, res^bitmask]
###详细思路：
#1. 首先得到bitmask是两个只出现一次的数字x1和x2的异或和
#2. 利用bitmask & -bitmask得到从左往右第一次出现1的位置，并且利用这个值，rightOne，对nums里面的值再次进行分组！
#   而x1和x2将被分到两个组内，为什么呢？因为rightOne是x1x2的异或和，意味着这两个数在这个1的位置上的数字是不同的，因此能被分到两个组内
#3. 看最后一块代码。分完组res剩下的就是1个数字，然后再利用bitmask撤销异或和，就求到另一个数字。



#哈希
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        freq = collections.Counter(nums)
        return [num for num, occ in freq.items() if occ == 1]


#复杂度 n ; 1;



# 190 Reverse bits
# 题目中给到的是 a given 32 bits unsigned integer.
# Thought here: 每次把 res 左移，把 n 的二进制末尾数字，拼接到结果 res 的末尾。然后把 n 右移。
class Solution:
    def reverseBits(self, n):
        res = 0
        # traverse 32 次
        for i in range(32): 
            # res向左移动，将末位空出0来。
            # n & 1 位与运算会获得末位是0还是1；
            # 将上面的0/1放在res末位，通过位或运算
            res = (res << 1) | (n & 1)
            # 将n往右移动1位。
            n >>= 1
        return res

# 231 2的幂
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and n & (n - 1) == 0
# 下面来解释一下为什么满足这两个条件，就一定为2的幂
# 因为是2进制，所以n为2的幂，那么 n 二进制最高位为 1，其余所有位为 0
# 同理，n - 1 的二进制最高位为0，其余所有位为1。
# 根据题意知道n肯定大于0，才能满足。


# 342 4的幂
# 可以转化成2的幂solve，也可以用模版solve
class Solution:
    def isPowerOfFour(self, n: int) -> bool:
        if n <= 0: return False
        # 这里做int转化为舍弃掉小数部分。
        x = int(math.sqrt(n))
        return x * x == n and (x & -x) == x
# 这个x&-x与上文的n&n-1效果一样


class Solution:
    def isPowerOfFour(self, n: int) -> bool:
        if n<=0:return False
        if n==1:return True
        while n>1:
            if n%4==0:
                n=n/4
            else:
                return False
                break
        return True


# 693 Binary Number with Alternating Bits
class Solution:
    def hasAlternatingBits(self, n: int) -> bool:
        while n:
            # x用来判断最后一位是0是1
            x = n & 1 
            n>>=1
            # y是倒数第二位是0是1
            y = n & 1
            # 如果xy相同，意味着n位和n-1位相同。
            if x==y:
                return False
        # 遍历完就可以结束
        return True

class Solution:
    def hasAlternatingBits(self, n: int) -> bool:
        while n:
            prenum = (n//2) & 1   # n//2获得的是除去最后一位，不论
            nownum = n%2          # n % 2相当于二进制右移一位
            if prenum == nownum:
                return False
            n>>=1
        return True



# 476 Number Complement 补数
# 因为计算机里存储的是以32位形式，所以我们不能将所有的0都一口气用～取反成为1
class Solution:
    def findComplement(self, num: int) -> int:
        highbit = 0
        # 查询1～30位，找到最高位位1的地方
        for i in range(1, 30 + 1):
            # 1向左移动i位，就可以表示第i位就是1所在的地方
            if num >= (1 << i):
                highbit = i
            else:
                break
        
        # mask为掩码的运算 mask就是从highbit到0位都是1
        mask = (1 << (highbit + 1)) - 1 
        
        # 这个结果就是num的反码：异或运算
        return num ^ mask
# 巧妙！

# 371 sum of two integers
class Solution:
    def getSum(self, a: int, b: int) -> int:
        return sum([a,b])
# 318 Maximum product of word lengths
"""
class solution{
    public int maxProduct(String[] words) {
        int n = words.length();
        int[] nums = new int[n];

        for (int i = 0; i < n; i++) {
            // 将所有的word转化为我们想要的二进制数字
            // 重点：那么这个convert出来的究竟是什么？
            //      通过位与运算，将每个字母映射的数字嵌入到ans里面，好神奇！
            nums[i] = convert(words[i]);
        }
        int ans = 0;
        // 典型的双for loop traverse
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                // 因为是位与运算，如果=0，意味着所有位都不相同，即不存在相同的字母
                // 理解nums究竟是什么需要理解convert这个方法return的是什么内容。
                if (nums[i] & nums[j] == 0) { 
                    ans = Math.max(ans, words[i].length() 
                    * words[j].length());
                }
            }
        }
    }
    private int convert(String word) {
        int ans = 0;
        for (int i = 0; i < word.length(); i++) {
            // 这一步骤很关键，word.charAt(i) - 'a'是为了让每个字母与integer联系起来
            // 然后利用位或得到ans，ans就是表示word代表的一个数。
            ans |= 1 << (word.charAt(i) - 'a');
        }
        return ans;
    }
}
     
时间复杂度: O(n * m + n^2)O(n * m+n2)，n 为单词数量，m 为单词平均长度。
空间复杂度: O(n)O(n)。
"""

# 338 Counting Bits
"""
class Solution {
    public int[] countBits(int num) {
        int[] result = new int[num + 1];
        for(int i = 1; i <= num; i++){
                // result[i >> 1] : 相当于i//2；这是因为二进制中，一个偶数有1的个数与它的二分之一相同，只是这个1的位不同罢了。
                // i & 1: 位与运算，如果是奇数那么结果为1，否则为0；
                // 综上所属，result[i]，我们可以兼顾到奇数和偶数。
                result[i] = result[i >> 1] + (i & 1);
        }
        return result;
    }
}
"""