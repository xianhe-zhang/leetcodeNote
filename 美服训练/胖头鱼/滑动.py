# 3. Longest Substring Without Repeating Characters
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        i = 0
        res = 0
        seen = {}    # 不能用set，要用dict
        for j in range(len(s)):     
            if s[j] in seen:
                i = max(seen[s[j]], i)  # 更新left pointer的方式不错。
            res = max(res, j - i + 1)
            seen[s[j]] = j + 1
        return res

# 76. Minimum Window Substring
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not s or not t: return ""
        dict_t = collections.Counter(t)
        required = len(dict_t)
        l = r = 0
        formed = 0  # 这个formed用来表示已经有多少个字母满足了，倒是我没有想过的。
        window_counts = dict()  
        ans = (float("inf"), None, None)
        
        while r < len(s):
            character = s[r]
            window_counts[character] = window_counts.get(character, 0) + 1 # 可以用defaultdict避免
            
            # 用两个dict分别保存窗口和t的情况，如果发现窗口的元素满足了，用一个变量保存。
            if character in dict_t and window_counts[character] == dict_t[character]:
                formed += 1
            
            # 满足condition的情况下，我们开始缩小左边届
            while l <= r and formed == required:
                character = s[l]
                if r - l + 1 < ans[0]:
                    ans = (r - l + 1, l, r) # 用个tuple组合起来，不仅仅存了data，也存了metadata
                
                window_counts[character] -= 1
                if character in dict_t and window_counts[character] < dict_t[character]:
                    formed -= 1
                l += 1
            r += 1
        return "" if ans[0] == float("inf") else s[ans[1] : ans[2] + 1]

