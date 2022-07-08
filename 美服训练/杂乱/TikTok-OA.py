"""
1- circular printer // 
2- minimum size subarray sum //
3- Dominos Tiling 3D
4- preprocessed Dates
5- Shared Interest
6- # count Inversions
7 -Design a Stack With Increment Operation
8-  Ancestral Name
9- Reaching Points
10- roman to integer
11- stars and bars
12- jump to the flag
13- fizzbuzz
14-  Find the Distance Value Between Two Arrays
15- 最大depth of bst
16-  Count Binary Substrings
17- longest string chain
18- # Minumum moves to separate even and odd numbers in array | NextJump OA 2020
19- # aladin and his carpet
20- # Secret Array
21- # Prefix sum/ Matrix summation
22- # Minimum Number of Manipulations required to make two Strings Anagram Without Deletion of Character2
23- # is that a tree?





"""

########################################################################################################################
# circular printer
class Solution:
    def getTimes(self, string):
        res = 0
        pre_char = "A"
        for ch in string:
            gap = min((ord(ch) - ord(pre_char)) % 26, (ord(pre_char) - ord(ch)) % 26)
            pre_char = ch
            res += gap
        return res



# 209. Minimum Size Subarray Sum
class Solution:
    def minSubArrayLen(self, target:int, nums) -> int:
        if not nums or not target:
            return 0
        left = 0
        ans = float("inf")
        sum = 0
        for i in range(len(nums)):
            sum += nums[i]
            while sum >= target:
                ans = min(ans, i - left + 1)
                sum -= nums[left]
                left += 1
        return 0 if ans == float("inf") else ans

# 思路不难，你来我往

# 790. 多米诺和托米诺平铺
# class Solution {
#     private int mod = 1000000007;
#     public int numTilings(int N) {
#         int[] dp = new int[N+3];
#         dp[0] = 1;
#         dp[1] = 1;
#         dp[2] = 2;
#         dp[3] = 5;
#         for(int i = 4; i <= N; i++){
#             dp[i] = (2*(dp[i-1] % mod) % mod + dp[i-3] % mod) % mod;
#         }
#         return dp[N];
#     }
# }
# 最重要的里面的思路

# Dominos Tiling 3D
def numtil(n):
    dp = [0] * (n + 1)
    dpa = [0] * (n + 1)
    dp[0], dp[1], dpa[0], dpa[1] = 1, 2, 0, 1
    for i in range(2, n + 1):
        dpa[i] = dpa[i - 1] + dp[i - 1] % 1000000007
        dp[i] = (2*dpa[i]+2*dpa[i-1]+dp[i-2]) % 1000000007
    return dp[n]


# 1507 preprocessed Dates
class Solution:
    def reformatDate(self, date: str) -> str:
        month = {m: i+1 for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
        d, m, y = date.split()
        return "{}-{:02d}-{:02d}".format(y, month[m], int(d[:-2]))
        # :02是保留两位， d是十进制‘


# Shared Interest
from collections import defaultdict
import itertools
from re import S

def maxTokens(friends_nodes, friends_from, friends_to, friends_weight):
    # assigning set of friend_nodes to their shared weights
    #   Wieghts      nodes
    #     1         {1,2,3}  
    #     2         {1,2}
    #     3         {2,3,4}
    weights = defaultdict(set)
    for i in range(len(friends_from)):
        weights[friends_weight[i]].add(friends_from[i])
        weights[friends_weight[i]].add(friends_to[i])
    # print(weights)
    # make set of pairs for each weight 
    #    Wieghts      nodes
    #      1         (1,2),(2,3),(1,3)  
    #      2         (1,2)
    #      3         (2,3),(3,4),(2,4)
    # count no of pairs 
    # {(1,2):2, (2,3): 2, (1,3):1, (3,4):1, (2,4):1}
    count = defaultdict(int)
    for key, val in weights.items():
        for foo in list(itertools.combinations(val, 2)):
            count[foo]+=1 
    # print(count)
    for num in sorted(set(count.values()), reverse=True):
        # print(num, )
        pairs = [k for k,v in count.items() if v == num]
        if len(pairs) >= 2:
            return max([a*b for a, b in pairs])
friends_nodes=4
friends_from =  [1, 1, 2, 2, 2]
friends_to =   [2, 2, 3, 3, 4]
friends_weight =  [1, 2, 1, 3, 3 ]

print(maxTokens(friends_nodes, friends_from, friends_to, friends_weight))


# count Inversions
def getInvCount(arr, n):
 
    inv_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (arr[i] > arr[j]):
                inv_count += 1
 
    return inv_count
 
 
# Driver Code
arr = [1, 20, 6, 4, 5]
n = len(arr)
print("Number of inversions are",
      getInvCount(arr, n))
 
# Ancestral Name
def roman_to_int(s:str):
    rom_to_int_map = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    sub_map = {'IV': 4, 'IX':9, 'XL': 40, 'XC': 90, 'CD':400, 'CM': 900}
    summation = 0
    idx = 0

    while idx < len(s):
        if s[idx:idx+2] in sub_map:
            summation += sub_map.get(s[idx:idx+2])
            idx += 2
        else:
            summation += rom_to_int_map.get(s[idx])
            idx += 1
    return summation
    
def sort_roman(names):
    name_array = []
    for name in names:
        n, num = name.split()
        num = roman_to_int(num)
        name_array.append((n, num, name))
    name_array.sort(key=lambda x: [x[0], x[1]])
    return list(map(lambda x:x[2], name_array))

# 1381.Design a Stack With Increment Operation
class CustomStack:
    def __init__(self, maxSize):
        self.stack = []
        self.size = maxSize

    def push(self, x):
        if len(self.stack) < self.size:
            self.stack.append(x)

    def pop(self):
        return self.stack.pop() if self.stack else -1
    
    def increment(self, k, val):
        for i in range(min(k, len(self.stack))):
            self.stack[i] += val
    
# 780. Reaching Points
class Solution:
    def reachingPoints(self, sx, sy, tx, ty):
        if tx < sx or ty < sy: return False
        if sx == tx: return (ty-sy)%sx == 0
        if sy == ty: return (tx-sx)%sy == 0
        if tx>ty:
            return self.reachingPoints(sx, sy, tx%ty, ty)
        elif tx<ty:
            return self.reachingPoints(sx, sy, tx, ty%tx)
        else:
            return False

# 13. Roman to Integer
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

# stars and bars
res = S[0:6].strip('*').count('*')


# jump to the flag
def jumps(flag: int, big:int) -> int:
    if flag <= 1:
        return flag
    if big > flag:
        return flag
    
    dp = [float("inf") for i in range(flag)]
    for i in range(1, big):
        # 只操作到index=(big-2) big-1就是跳到big的步数
        dp[i - 1] = i
    dp[big - 1] = 1 #这一步操作big-1，init全部完成
    for i in range(big, flag):
        dp[i] = min(dp[i - 1] + 1, dp[i - big] + 1)
    return dp[-1]


# FizzBuzz
class Solution:
    def fizzBuzz(self, n):
        # ans list
        ans = []

        for num in range(1,n+1):

            divisible_by_3 = (num % 3 == 0)
            divisible_by_5 = (num % 5 == 0)

            if divisible_by_3 and divisible_by_5:
                # Divides by both 3 and 5, add FizzBuzz
                ans.append("FizzBuzz")
            elif divisible_by_3:
                # Divides by 3, add Fizz
                ans.append("Fizz")
            elif divisible_by_5:
                # Divides by 5, add Buzz
                ans.append("Buzz")
            else:
                # Not divisible by 3 or 5, add the number
                ans.append(str(num))

        return ans
        
# Find the Distance Value Between Two Arrays
class Solution(object):
    def findTheDistanceValue(self, arr1, arr2, d):
        arr2.sort()
        count=0
        for x in arr1:
            l, r = 0, len(arr2)
            while l < r:
                mid = (l + r) // 2
                if abs(arr2[mid] - x) <= d:
                    count-=1
                    break
                elif arr2[mid] > x:
                    r = mid
                else:
                    l = mid + 1
            count+=1
        return count


# 104. Maximum Depth of Binary Tree
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        def dfs(node, depth):
            if not node:
                return depth
            depth += 1
            left = dfs(node.left, depth)
            right = dfs(node.right, depth)
            return max(left, right)
        
        depth = 0
        return dfs(root, depth)
# 考虑好究竟返回的是什么。


# 696. Count Binary Substrings
class Solution:
    def countBinarySubstrings(self, s: str) -> int:
        groups = [1]
        for i in range(1, len(s)):
            if s[i-1] != s[i]:
                groups.append(1)
            else:
                groups[-1] += 1
            
        
        ans = 0
        for i in range(1, len(groups)):
            ans += min(groups[i-1], groups[i])
        return ans
# 节省了一下空间
class Solution:
    def countBinarySubstrings(self, s: str) -> int:
        ans, prev, cur = 0, 0, 1
        for i in range(1, len(s)):
            if s[i-1] != s[i]:
                ans += min(prev, cur)
                prev, cur = cur, 1
            else:
                cur += 1
        return ans + min(prev, cur)

# 1048. Longest String Chain
class Solution:
    def longestStrChain(self, words):
        dp = {}
        # 这里用了下排序，因为要有包含关系
        for w in sorted(words, key=len):
            # 首先针对单个word
            # 然后看它的所有组合，选择出最大的value，如果不用这样“内置iterator”的方法，可能要重新开一个list
            # 选出最大值，当dp[w]
            dp[w] = max(dp.get(w[:i] + w[i + 1:], 0) + 1 for i in range(len(w)))
        return max(dp.values())
        
# 1654. Minimum Jumps to Reach Home
# 涉及到minimum了，第一个思路是dp，第二个思路就可以是BFS
class Solution:
    def minimumJumps(self, forbidden: List[int], a: int, b: int, x: int) -> int:
    dq, seen, steps, furthest = deque([(True, 0)]), {(True, 0)}, 0, max(x, max(forbidden)) + a + b
    for pos in forbidden:
        seen.add((True, pos)) 
        seen.add((False, pos)) 
    while dq:
        for _ in range(len(dq)):
            dir, pos = dq.popleft()
            if pos == x:
                return steps
            forward, backward = (True, pos + a), (False, pos - b)
            if pos + a <= furthest and forward not in seen:
                seen.add(forward)
                dq.append(forward)
            if dir and pos - b > 0 and backward not in seen:
                seen.add(backward)
                dq.append(backward)    
        steps += 1         
    return -1


# 1654. Minimum Jumps to Reach Home
class Solution:
    def minimumJumps(self, forbidden: List[int], a: int, b: int, x: int) -> int:
        dq, seen, steps, furthest = deque([(True, 0)]), {(True, 0)}, 0, max(x, max(forbidden)) + a + b
        
        # 如果是不能去的pos，提前加入seen
        for pos in forbidden:
            seen.add((True, pos)) 
            seen.add((False, pos)) 
        # 当dq有值，意味着还能走
        while dq:
            # 这一步是模版，意味着下一步有多少种选择
            for _ in range(len(dq)):
                dir, pos = dq.popleft()
                if pos == x:
                    return steps
                forward, backward = (True, pos + a), (False, pos - b)
                
                #下一步后退的位置
                if pos + a <= furthest and forward not in seen:
                    seen.add(forward)
                    dq.append(forward)
                #下一步前进的位置
                if dir and pos - b > 0 and backward not in seen:
                    seen.add(backward)
                    dq.append(backward)    
            # 放在for外面意味着这一种情况完结后，才有下一步！
            steps += 1         
        return -1

# Minumum moves to separate even and odd numbers in array | NextJump OA 2020
# 类似双指针，复杂度应该为N
"""int minMovesToEvenFollowedByOdd(vector<int> arr) {
	int res = 0, left = 0, right = arr.size() - 1;
	// two-pointer approach
	while(left < right) {
		if(arr[left] % 2 != 0) {
			while(right > left && arr[right] % 2 != 0) {
				// Find the first occurrence on the righthand side that can be swapped
				right--;
			}

			if(right > left) {
				// if we're already in the midpoint and the left pointer is odd then there is no swap
				res++;
				// handled this rightmost occurrence that was even => adjust to account for this
				right--;
			}
		}

		left++;
	}

	return res;
}"""
class Solution:
    def minMovesToEvenFollowedByOdd(arr):
        res, left, right = 0, 0, len(arr) - 1
        while left < right:
            if arr[left] % 2 != 0:
                while(right > left and arr[right] %2 != 0):
                    right -= 1
                if right > left:
                    res += 1
                    right -= 1
            left += 1
        return res

# Minumum moves to separate even and odd numbers in array | NextJump OA 2020
# aladin and his carpet
def aladdin(magic, dist):
    # 首先看一下油够不够
    if sum(magic) < sum(dist): return -1
    n = len(magic)
    # total_val做个条件判断，就是为了防止跑不到地方
    total_val = 0
    start = 0
    for i in range(n):
        if total_val < 0:
            start = i
            total_val = 0
        total_val += (magic[i] - dist[i])
    return start

# Minumum moves to separate even and odd numbers in array | NextJump OA 2020
# aladin and his carpet
# Secret Array
def countAnalogousArrays(consecutiveDifference , lowerBound , upperBound):
    # 我的想法是你知道max和min的差距了
    maxdiff = float('-inf')
    mindiff = float('inf')
    runningsum = 0
    if len(consecutiveDifference) == 0 or upperBound < lowerBound:
        return 0
    # 遍历找到max与min
    for diff in consecutiveDifference:
        runningsum+=diff
        if runningsum > maxdiff:
            maxdiff = runningsum
        if runningsum < mindiff:
            mindiff = runningsum

    maxvalidupperbound = upperBound + mindiff if upperBound+mindiff < upperBound else upperBound
    minvalidlowerbound = lowerBound + maxdiff if lowerBound + maxdiff > lowerBound else lowerBound

    if maxvalidupperbound >= minvalidlowerbound:
        return maxvalidupperbound - minvalidlowerbound + 1
    else:
        return 0

# Minumum moves to separate even and odd numbers in array | NextJump OA 2020
# aladin and his carpet
# Secret Array
# Prefix sum/ Matrix summation
"prefix sum 还原 - Value[j] = Sum[j] - Sum[i-1][j] - Sum[j-1] + Sum[i-1][j-1]"
def print_matrix(mat):
    m = len(mat)
    n = len(mat[0])
    
    # 返回前缀和需要注意的点是？就是用nums[i] = nums[i] - nums[i - 1]

    # 第一个[1,2,3,4,5,6], 先是在一list中横向地返回
    for i in range(m-1,-1,-1):
        for j in range(n-1,0,-1):
            mat[i][j] -=mat[i][j-1]
    # 第二个双for循环是，纵向上依次返回。
    for i in range(m-1,0,-1):
        for j in range(n-1,-1,-1):
            mat[i][j] -=mat[i-1][j] 
    
    return mat
# Minumum moves to separate even and odd numbers in array | NextJump OA 2020
# aladin and his carpet
# Secret Array
# Prefix sum/ Matrix summation
# Minimum Number of Manipulations required to make two Strings Anagram Without Deletion of Character
def countManipulations(s1, s2):
     
    count = 0
 
    # store the count of character
    char_count = [0] * 26
     
    for i in range(26):
        char_count[i] = 0
 
    # iterate though the first String
    # and update count
    for i in range(len( s1)):
        char_count[ord(s1[i]) -
                   ord('a')] += 1
 
    # iterate through the second string
    # update char_count.
    # if character is not found in
    # char_count then increase count
    for i in range(len(s2)):
        char_count[ord(s2[i]) - ord('a')] -= 1
         
    for i in range(26):
        if char_count[i] != 0:
            count += abs(char_count[i])
         
 
    return count / 2
 
# Driver code
if __name__ == "__main__":
 
    s1 = "ddcf"
    s2 = "cedk"
     
    print(countManipulations(s1, s2))
 
# Minumum moves to separate even and odd numbers in array | NextJump OA 2020
# aladin and his carpet
# Secret Array
# Prefix sum/ Matrix summation
# Minimum Number of Manipulations required to make two Strings Anagram Without Deletion of Character
# is that a tree?
class Solution:
    def constructSExpression(s):
        E2 = False
        numofEdges = 0
        for i in range(1, len(s), 6):
            x = s.charAt(i)
"""
                boolean[][] graph = new boolean[26][26];
                Set<Character> set = new HashSet<Character>();
                boolean E2 = false;
                int numOfEdges = 0;
                for (int i = 1; i < s.length(); i += 6) {
                        int x = s.charAt(i) - 'A';
                        int y = s.charAt(i + 2) - 'A';
                        if (graph[x][y]) {
                                E2 = true;
                        }
                        graph[x][y] = true;
                        set.add(s.charAt(i));
                        set.add(s.charAt(i + 2));
                        numOfEdges++;
                }
                boolean E1 = false;
                for (int i = 0; i < 26; i++) {
                        int count = 0;
                        for (int j = 0; j < 26; j++) {
                                if (graph[i][j]) {
                                        count++;
                                }
                        }
                        if (count > 2) {
                                return "E1";
                        }
                }
                if (E2) return "E2";
                int numOfRoots = 0;
                char root = ' ';
                System.out.println(set);
                for (Character c : set) {
                        for (int i = 0; i < 26; i++) {
                                if (graph[i][c - 'A']) {
                                        break;
                                }
                                if (i == 25) {
                                        numOfRoots++;
                                        root = c;
                                        boolean[] visited = new boolean[26];
                                        if (detectCycle(c, graph, visited)) {
                                                return "E3";
                                        }
                                }
                        }
                }
                if (numOfRoots == 0) return "E3";
                if (numOfRoots > 1) return "E4";
                return getSexpression(root, graph);
        }
        private boolean detectCycle(char c, boolean[][] graph, boolean[] visited) {
                if (visited[c - 'A']) return true;
                visited[c - 'A'] = true;
                for (int i = 0; i < 26; i++) {
                        if (graph[c -'A'][i]) {
                                if (detectCycle((char)('A' + i), graph, visited)) {
                                        return true;
                                }
                        }
                }
                return false;
        }
        private String getSexpression(char root, boolean[][] graph) {
                String left = "";
                String right = "";
                for (int i = 0; i < 26; i++) {
                        if (graph[root - 'A'][i]) {
                                left = getSexpression((char)('A' + i), graph);
                                for (int j = i + 1; j < 26; j++) {
                                        if (graph[root - 'A'][j]) {
                                                right = getSexpression((char)('A' + j), graph);
                                                break;
                                        }
                                }
                                break;
                        }
                }
                return "(" + root + left + right + ")";
"""