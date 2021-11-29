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


# 260
# 190
# 231



# 342
# 693
# 476
# 371
# 318
# 338