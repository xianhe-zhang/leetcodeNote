# 🌟🌟🌟这一题卡了我好久，值得学习🌟🌟🌟
# 46. Permutations
# 这一题的难点就是给你一个list如何找到所有的排序可能性
# 我自己思路是每一次递归找一个可能的数字进行排列。
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums, path):
            if not nums:
                res.append(path)
            for i in range(len(nums)): 
                # 如果这里只用path.append(nums[i])的话，要记的加入pop否则在该子树遍历完后，我们的path就会额外长，因为相当于每个recursion中选了多位数。
                dfs(nums[:i]+nums[i+1:], path+[nums[i]]) # 这里相当于把append+pop省略了
                """ 
                path.append(nums[i])
                dfs(nums[:i]+nums[i+1:], path)
                path.pop()
                但是上面res.append(path[:]) # 这个是关键的！
                """
        res = []
        dfs(nums, [])
        return res
# 本题开启了一个全新的知识点：参数传递/值传递/引用传递
# 本题两个key points(marked in the code): 
    # 1. 为什么不用path而一定要用path[:]？-python是对象/引用传递，因此如果只是用path，那么最后进res的都是同一对象，而且在其他递归中会被改变成一样的。
    # 2. 为什么path+[nums[i]]就可以呢？-如果在传参的过程中进了计算，其实相当于一个全新的变量了，因此可以实现变量隔离。

    # 浅拷贝有三种形式：切片操作，工厂函数，copy模块中的copy函数
    # 深拷贝只有deepcopy，完全隔离了与copy对象相当于
            
    # 切片操作：list_b = list_a[:]   或者 list_b = [each for each in list_a]
    # 工厂函数：list_b = list(list_a)
    # copy函数：list_b = copy.copy(list_a)

    # 浅拷贝产生的list_b不再是list_a了，使用is可以发现他们不是同一个对象，使用id查看，发现它们也不指向同一片内存。但是当我们使用 id(x) for x in list_a 和 id(x) for x in list_b 时，可以看到二者包含的元素的地址是相同的。
    # 在这种情况下，list_a和list_b是不同的对象，修改list_b理论上不会影响list_a。比如list_b.append([4,5])。
    # 但是要注意，浅拷贝之所以称为浅拷贝，是它仅仅只拷贝了一层，在list_a中有一个嵌套的list，如果我们修改了它，情况就不一样了。
    # list_a[4].append("C")。查看list_b，你将发现list_b也发生了变化。这是因为，你修改了嵌套的list。修改外层元素，会修改它的引用，让它们指向别的位置，修改嵌套列表中的元素，列表的地址并为发生变化，指向的都是同一个位置。

# 下面是官方答案，求permutation的方式不同而已，不过也很棒。
# 思路理解：first相当于一个位，针对first位，我们可以排什么数字，其实和我的思路是一样的，不过我用的path
# 这里是直接在原有num上变化了。
class Solution:
    def permute(self, nums):
        def backtrack(first = 0):
            if first == n:  
                output.append(nums[:])
            for i in range(first, n):
                nums[first], nums[i] = nums[i], nums[first]
                backtrack(first + 1)
                nums[first], nums[i] = nums[i], nums[first]
        
        n = len(nums)
        output = []
        backtrack()
        return output


# 22. Generate Parentheses
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        def backtrack(S=[], l = 0, r = 0):
            if len(S) == 2*n:
                res.append("".join(S))
                return 
            # 我们可以在每一层只进一层recursion，也可以平行进多层recursion，相当于Tree！
            if l < n:
                backtrack(S+["("], l+1, r) # 传参将append/pop省略了...
            if r < l:
                backtrack(S+[")"], l, r+1)
        
        res = []
        backtrack()
        return res

# 93. Restore IP Addresses 
# 第一种思路，还是寻找segment； 当然我们还有第二种解法，就是插入dot。
class Solution():
    def restoreIpAddresses(self, s):
        res = []
        self.dfs(s, 0, "", res)
        return res
    
    def dfs(self, s, idx, path, res):
        if idx > 4:
            return 
        if idx == 4 and not s:
            res.append(path[:-1]) # 这里处理直接是字符串，不是list，因此一点点小技巧吧...
            return 
        for i in range(1, len(s)+1): # 这里没有剪枝哦。剪枝的写法min(len(s)+1, 5)
            if s[:i]=='0' or (s[0]!='0' and 0 < int(s[:i]) < 256): 
                self.dfs(s[i:], idx+1, path+s[:i]+".", res)



# 78. Subsets
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def dfs(first = 0, cur = []):
            if first == k:
                output.append(cur[:])
                return 
            for i in range(first, n):
                dfs(i + 1, cur+[nums[i]]) 
        output = []
        n = len(nums)
        for k in range(n + 1):
            dfs()
        return output

# 17. Letter Combinations of a Phone Number
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if len(digits) == 0: return []
        letters = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", 
                   "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        
        def backtrack(index=0, path=[]):
            if len(path) == len(digits):
                result.append(''.join(path))
                return
                
            for letter in letters[digits[index]]:
                backtrack(index + 1, path+[letter]) # 这么写有点造成空间的浪费。
            
        result = []
        backtrack()
        return result
    

# 79. Word Search
# 利用res暂存变量的原因是因为我们要在进入递归后将原数据结构复原。
class Solution(object):
    def exist(self, board, word):   
        if not board: return 0
        
        def backtrack(i, j, word):
            if len(word) == 0: return True
            if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or word[0] != board[i][j]:
                return False
            
            temp = board[i][j]
            board[i][j] = '#'
            
            res = backtrack(i+1, j, word[1:]) or backtrack(i, j+1, word[1:]) or backtrack(i-1, j, word[1:]) or backtrack(i, j-1, word[1:])
            board[i][j] = temp
            return res
        

        for i in range(len(board)):
            for j in range(len(board[0])):
                if backtrack(i,j,word):
                    return True
        return False
        

# 90. Subsets II
# 如何去除duplicate，因为我们没办法确认重复元素的位置，因此要事先进行sort，确保重复元素的相对位置
# 本题的思路还是一样的，针对每一位仍在选择空间中选择一位数。但是这里需要注意：1. sort之后只能往后看，往前看会重复； 2.记得跳过重复的数，因为你不要duplicate
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums, path, res):
            res.append(path)           # 这很关键！每个path其实都是一种组合，哪怕它没有到尾。
            for i in range(len(nums)): # 🌟理解nums在不同层当中变化蛮重要的。1.每一次loop我们只选一位数 2. 每一次recursion我们都选的是其之后的数字
                if i > 0 and nums[i] == nums[i-1]: # 想清楚解题思路就不难了
                    continue
                dfs(nums[i+1:], path+[nums[i]], res)
                
        res = []
        # sort是没跑的+
        nums.sort()
        dfs(nums, [], res)
        return res
        

# 39. Combination Sum 跟上面这道题有点像哈。
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        
        def dfs(nums,path):
            tt = sum(path) 
            
            for i in range(len(nums)):
                n = nums[i]
                if tt + n == target: res.append(path+[n])
                elif tt + n > target: break
                else:
                    dfs(nums[i:],path+[n])  # 这里的nums[i:]处理需要注意下，我们可以重复选，但一但选了大的数，之后的不能选更小的，否则会重复。
        candidates.sort() # 为了保证nums[i:]的正常运行，要进行sort
        res = []     
        dfs(candidates, [])   
        return res
        
# 77. Combinations
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        
        def dfs(first, cur):
            if len(cur) == k:
                res.append(cur[:])
                return 
            for num in range(first, n + 1): # 这个first或者len(nums[1:])是个套路技巧。在指定空间，你要找是第first位数。
                cur.append(num)
                dfs(num + 1, cur)   # 不想用append/pop的话就直接cur+[num]
                cur.pop()
        res = []
        dfs(1, [])
        return res

# 47. Permutations II
# 这一题用Counter比较灵性，为什么呢？因为每一次选择数据的时候，我可以知道还有哪些数据可以选。
# 如果你要之间遍历nums，也是可以的！但是你的nums要每次手动更新nums[:i]+nums[i+1:]也可以起到同样的效果。
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        def dfs(com, counter):
            if len(com) == len(nums):
                res.append(list(com))
                return 
            
            for num in counter:
                if counter[num] > 0:
                    com.append(num)
                    counter[num] -= 1
                    
                    dfs(com, counter)
                    com.pop()
                    counter[num] += 1
                    
        dfs([], Counter(nums))
        return res


# 40. Combination Sum II
# 我自己写的。这一题的input有duplicate numbers！但是答案是不能重复的，如何跳过重复的情况是本题的亮点！
# 把判断条件放在下一次dfs也可以，或者放在for循环中也可以。
class Solution:
    def combinationSum2(self, nums: List[int], target: int) -> List[List[int]]:
        def dfs(nums, path):
            prev_val = 0 #跳过的关键！
            for i in range(len(nums)):
                # 🌟针对每一层recursion/loop. 我们的目标都是选1个数，如果这个数字选过了，我们就不选了！
                # 当然，我们的答案是可以包含duplicate num的，如果我们选了3，我们当前层就不能选3，但是我们可以在下一层继续选3. 理解并巩固思路最重要！
                if prev_val == nums[i]: 
                    continue
                prev_val = nums[i]
                new_path = path + [nums[i]]
                new_tt = sum(new_path)
                
                if new_tt > target:
                    break
                elif new_tt == target:
                    res.append(new_path)
                else:
                    dfs(nums[i+1:], new_path)

        nums.sort()
        res = []
        dfs(nums, [])
        return res
# 下面是官网解答，有几个点可以学习。
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        def backtrack(comb, remain, curr, results): # 首先remian参数，而不是利用了sum，可以减少计算！🌟
            if remain == 0:                         # 典型将判断/edge case放在开头
                results.append(list(comb))
                return

            # 🌟 我们传参的时候是有curr的！因为我们sort过，一般来说cur是可以取的。
            for next_curr in range(curr, len(candidates)):

                # 🌟利用next_curr和curr来判断是否是duplicate num，在同层中，如果碰到重复的事要跳过的！
                if next_curr > curr \
                  and candidates[next_curr] == candidates[next_curr-1]:
                    continue

                pick = candidates[next_curr]
                # optimization: skip the rest of elements starting from 'curr' index
                if remain - pick < 0:
                    break

                comb.append(pick)
                backtrack(comb, remain - pick, next_curr + 1, results)
                comb.pop()

        candidates.sort()

        comb, results = [], []
        backtrack(comb, target, 0, results)

        return results
        
        
# 216. Combination Sum III
class Solution:
    def combinationSum3(self, left: int, target: int) -> List[List[int]]:
        
        def backtrack(path, start, k, target):
            tt = sum(path)
            if k < 0 or tt > target: return 
            if k == 0 and sum(path) == target:
                res.append(path[:])
            for n in range(start, min(target+1,10)):
                path.append(n)
                backtrack(path,n+1, k-1, target)
                path.pop()
        
        res = []
        backtrack([], 1, left, target)
        return res
      
        
