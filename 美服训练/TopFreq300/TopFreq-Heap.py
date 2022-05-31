
# 1046. Last Stone Weight
# 利用递归进行遍历的想法挺好的
import heapq


class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        if len(stones) == 0: return 0
        if len(stones) == 1: return stones[0]
    
        stones.sort()
        if stones[-1] == stones[-2]:
            stones.pop()
        else:
            stones[-2] = stones[-1] - stones[-2]
        stones.pop()
        return self.lastStoneWeight(stones)
        

# 703. Kth Largest Element in a Stream
class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.nums = nums
        self.k = k

    def add(self, val: int) -> int:
        self.nums.append(val)
        
        self.nums.sort()
        return self.nums[-self.k]
# 下面这种用heapq的方法还是挺好的。
class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.heap = nums
        heapq.heapify(self.heap)
        
        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val: int) -> int:
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]

# 215. Kth Largest Element in an Array
# 这一题有三种方案：sort、quicksort+divide&conquer，heap！return heapq.nlargest(k, nums)[-1]
# 如果利用sort，时空分别nlogn, 1； 如果用heap，时空分别为 nlogn, k
# 分治思路：
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        n = len(nums)
        self.divide(nums, 0, n-1, k)
        return nums[n-k]
    
    # divide作用就是利用position决定在哪区间内寻找position，如果找到就返回，没有的话继续divide/dive-deep去寻找(排序)
    def divide(self, nums, left, right, k):
        if left >= right: return 
        position = self.conquer(nums, left, right)
        
        if position == len(nums) - k: return    # position这个点已经求出，就是我们要得n-k，可以直接return
        elif position < len(nums) - k: self.divide(nums, position+1, right, k) # n-k在position右侧，右侧需要继续divide去寻找找到position；题外话，每排好一个position意味着，这个点就已经完全排好了。
        else: self.divide(nums, left, position - 1, k) # n-k我们找的点在左侧，因此我们需要在左侧找position。
    
    # conquer的作用就是将divide中指定的区间进行粗糙排序，并且返回pivot/position
    def conquer(self, nums, left, right):
        pivot, wall = nums[right], left
        for i in range(left, right):
            if nums[i] < pivot:
                # 碰到小的/应放左边的，就swap；碰到大的/应放右边的，continue，i继续增加，wall指向应该swap的大数；这样在下一轮循环的时候可以swap掉。
                # 而且在i与wall有间隔的情况，没碰到一个小的都要放在前面，这样更容易理解为什么每一次都需要swap
                nums[i], nums[wall] = nums[wall], nums[i]
                wall += 1
        # wall在loop结束后时+1的，因此肯定是较小区间的右侧，较大区间的第一位，因此将wall与pivot swap
        nums[wall], nums[right] = nums[right], nums[wall]
        # 我们返回的为什么是wall？因为返回的是分界点pi
        return wall
# 时间最好是n，最差是n2；空间为1


# 347. Top K Frequent Elements
# Heap的题目主要就是练习heapq的吧
from collections import Counter
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]: 
        # O(1) time 
        if k == len(nums):
            return nums
        
        # 1. build hash map : character and how often it appears
        # O(N) time
        count = Counter(nums)   
        # 2-3. build heap of top k frequent elements and
        # convert it into an output array
        # O(N log k) time
        return heapq.nlargest(k, count.keys(), key=count.get) 

# 692. Top K Frequent Words
# 这一题没有办法用nlargest的原因是因为nlargest的时候没有办法兼顾到字母 lexicographical order，而heapify -> heappop则可以。
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = Counter(nums)
        heap = [[-cnt, value] for value, cnt in count.items()]
        heapq.heapify(heap)
        
        result = []
        for _ in range(k):
            _, num = heapq.heappop(heap)
            print(num)
            result.append(num)
            
        return result
"""
为什么heapify的时间复杂度为O(n)? heapify的过程是bottom up 从最一个非子节点开始比较
考虑构造min-heap的过程 - 第k层有2^(k-1)个节点 / 一共有logN层 / 倒数第1层需要交换0次 倒数第二层需要交换1次 倒数第k层需要交换k次 
-> 这个交换次数怎么理解? 一次交换指的是1 parent & 2 sons的比较, 比如倒数第三层交换过之后, 可能会出现原先的root node被交换到倒数第二层, 而此时还要与倒数第一层进行交换, 因此倒数第k层的node最多要经历k-1次交换
因此一共的计算次数(k层):  
        S(k)  =  2^0 * (k-1) + 2^1 * (k-2) + ... + 2^(k-2)*1
利用错位相减法:
        2S(k) =                2^1 * (k-1) + ... + 2^(k-2)*2 + 2^(k-1)*1 

2S(k) - S(k)  =  -1 * (k-1)  + 2^1   + 2^2 + ... + 2^(k-2)   + 2^(k-1) 
        S(k)  =  -k + 1 + 2的等比数列求和 = 2^k - k - 1                             # k是层 n是个数 n = 2^k - 1 => k = logn
    =>  S(n)  =  n - logn - 1 = O(n)
"""

# 378. Kth Smallest Element in a Sorted Matrix
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        temp = []
        for m in matrix:
            temp.extend(m)
        return heapq.nsmallest(k, temp)[-1]
# heapq中的函数大多都是O(nlogn) or O(logn)



# 451. Sort Characters By Frequency
class Solution:
    def frequencySort(self, s: str) -> str:
        counts = collections.Counter(s)
        string_builder = []
        for letter, freq in counts.most_common():
            string_builder.append(letter * freq)
        return "".join(string_builder)
# 利用hashmap
# 利用merge sort
# 利用quick sort
# bucket sort也可以
        

# 973. K Closest Points to Origin
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        heap = []
        # 维持一个长度为k的heap
        for i in range(k):
            x, y = points[i][0], points[i][1]
            dis = x**2 + y**2
            heap.append((-dis, i))
        heapq.heapify(heap)
        # 或用下面一行代替，但是要写helper function
        # heap = [(-self.squared_distance(points[i]), i) for i in range(k)]
        
        # 开始遍历接下来的元素
        for i in range(k, len(points)):
            x, y = points[i][0], points[i][1]
            dis = -(x**2 + y**2)
            # 如果发现有比栈顶大的；因为是取负了，绝对值就小，就更近。就pushpop更新我们的heap，那么最后留在heap里的一定是k个最近的坐标。
            if dis > heap[0][0]:
                heapq.heappushpop(heap,(dis,i))
        return [points[i] for (_, i) in heap]        
# 时空复杂度为O(N*logK)\O(k)
        
    