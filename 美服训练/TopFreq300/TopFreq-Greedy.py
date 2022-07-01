# 264. Ugly Number II
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        ugly = [1]
        i2 = i3 = i5 = 0
        while n > 1:
            u2, u3, u5 = ugly[i2]*2, ugly[i3]*3, ugly[i5]*5
            u_min = min(u2, u3, u5)
            if u_min == u2:
                i2 += 1
            elif u_min == u3:
                i3 += 1
            else:
                i5 += 1
            if u_min == ugly[-1]: continue # 为了避免6即是2的倍数，又是3的倍数 ✨ 去看下面！！
            ugly.append(u_min)
            n -= 1
        return ugly[-1]
# 这一题值得学习的地方其实是如何依次找到公倍数！🌟
        """          
            if umin == u2:
                i2 += 1
            if umin == u3:
                i3 += 1
            if umin == u5:
                i5 += 1
            这么写的好处! umin有可能同时是2、3、5的倍数 因此我们用三个单独的if可以都进行一遍判断 而非用elif
            这样就不用判断是否是重复的了 因为重复的已经被跳过了 牛逼    
                
        """
    

# 946. Validate Stack Sequences
# 自己写的，逻辑这里卡了一段时间。本来想用double pointers
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        m, n = len(pushed), len(popped)
        if m != n: return False
        stack = []
        ptr = 0
        
        for i in range(m):
            stack.append(pushed[i])
            while stack and stack[-1] == popped[ptr]:
                stack.pop()
                ptr += 1
                 
        return ptr == n
            
            
"""
a = sorted(S, key=S.count)
a[1::2], a[::2] = a[:h], a[h:] 
#::2 每两个取一个 1::2 从1开始每两个取一个 :10:2前10个每两个取一个
"""

# 767. Reorganize String
# 这里有个非常聪明的处理，但是需要多费些心。
# 每次我们排的都是Most-Frequency的字母，但是如果本次参加排序的字母将不会在next loop中参加排序
# 因此我们利用两个变量p_a,p_b记录上一个loop的数据，并且update
# 本题好好盘一下heapq吧，heapq是小根堆
class Solution:
    def reorganizeString(self, S):
        result, cnt = [], collections.Counter(S)
        
        pq = [(-value, key) for key, value in cnt.items()]
        heapq.heapify(pq)
        pre_k, pre_v = '', 0 # 这里p_b其实不用初始化，因为可以在loop中，这里初始化只是因为第一遍要判断的时候避免undefine的缘故。
        while pq:
            v, k = heapq.heappop(pq)
            if pre_v < 0:
                heapq.heappush(pq, (pre_v, pre_k))
            result.append(k)
            v += 1
            pre_k, pre_v = k, v
        res_final = "".join(result)
        return res_final if len(res_final) == len(S) else ""
            
        
# 313. Super Ugly Number
# 这一题用heapq应该也可以
class Solution:
    def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
        if n < 1: return 0
        if n == 1: return 1
        
        m = len(primes)
        # 本题的精妙点就是利用index作为桥梁连接不同list，从而达到不同意义。
        # 也可以看作两个list share同一套的index系统，从而实现数据同步。
        u_num, u_num_list, index_list, dp = 1, [1] * m, [0] * m, [1]
        for i in range(1, n):
            for j in range(m):
                if u_num_list[j] == u_num:
                    u_num_list[j] = dp[index_list[j]] * primes[j]
                    index_list[j]+=1
            u_num = min(u_num_list)
            dp.append(u_num)
        return dp[-1]
# 利用heapq的话就是每次跳出heapq就成。N是构建，logK是pop
# After some thought:
# 1). Heap Solution:
# O(Nlogk) for runtime
# O(N) for space
# 2). DP Solution:
# O(NK) for runtime
# O(N) for space
       
# 373. Find K Pairs with Smallest Sums
class Solution:    
    # 一行代码能写就是对于memory的限制太高了。cannot pass all the test.
    # return sorted(product(nums1,nums2), key = lambda x:sum(x))[:k] # key = sum
    def kSmallestPairs(self, nums1, nums2, k):
        queue = []
        def push(i, j):
            if i < len(nums1) and j < len(nums2):
                heapq.heappush(queue, [nums1[i] + nums2[j], i, j])
        push(0, 0)
        pairs = []
        while queue and len(pairs) < k: # 这里用的也精妙。 我们要返回k对，如果没有k对，有多少返回多少。
            _, i, j = heapq.heappop(queue)
            pairs.append([nums1[i], nums2[j]])
            push(i, j + 1)
            if j == 0:      # 逻辑解释如下⬇️ # 这里充当的就是下一行的启动器。
                push(i + 1, 0)
        return pairs
# 🌟🌟🌟用heapq的思路太好了吧...I know shit about programming.
# 1. heapq我们用的2sum当作索引，这样每次pop出来的i和q应该是当下最小的。
# 2. 紧接着1，当下最小的i和j了，那么我们把紧接着的j+1入heap
# 3. 🌟这一点有意思：如果当前j为0，当前特定的i行所有最小的组合已经没有了。j可以进位了。这有点难理解。自己可以举例子尝试理解。
# 4. 利用一个helper而非if-clause决定是否入stack

# 解释下第三点的逻辑, 第一次入heap是(0,0)->(0,1)+(1,0)->(0,2)->(0,3)...一直到(0,n)不再pop出来而是(1,0)pop出来
# 然后往里面添加(1,1)+(2,0)->(1,2)...
# 关于遍历顺序，首先需要想明白的是2-dimension(i,j)中i和j轮流增加是完全可以遍历完所有的。如果其中一个过头了，就不会往heap中增加。
# 因为我们+1处理，如果某一行没有遍历完，那么之后heapq当满足条件的时候还是会继续遍历的。



            
