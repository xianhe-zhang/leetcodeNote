
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import javax.swing.tree.TreeNode;

class Solution1 {
        
// 扫描线 - 252. Meeting room
    public boolean canAttendMeetings(int[][] intervals) {
        // intervals.sort();
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        
        for (int i = 0; i < intervals.length - 1; i++) {
            if (intervals[i][1] > intervals[i+1][0]) return false;
        }
        return true;
    }
}


// DP - 55. Jump Game
// 第一种写法-DP
// WTH,Greedy的想法好棒，从最后一位开始走，往前遍历，如果前面的index能走到last_position，就把lastPos更新到最前面的一位，最后看看能不能到达0
class Solution2 { 
    public boolean canJump(int[] nums) {
        int lastPos = nums.length - 1;
        for (int i = nums.length - 1; i >= 0; i--) {
            if (i + nums[i] >= lastPos) lastPos = i; 
        }
        return lastPos == 0;
    }
}

// 第二种写法-top down
enum Index {
    GOOD, BAD, UNKNOWN
}
class Solution3 {
    Index[] memo;

    public boolean canJumpFromPosition(int position, int[] nums) {
        if (memo[position] != Index.UNKNOWN) {
            return memo[position] == Index.GOOD ? true : false;
        }

        // 这里是针对每一个可以到达的index，再去看他们能不能继续往下走
        // 如果可以继续往下走，那么继续走，只要能继续往下走，都会可以一直返回True，
        // 否则针对某一个index，一旦它走不了，for循环完毕，就会返回False
        // 那么如何才能停止？当走到终点，Base case起到拦截/剪枝的作用。只要每一层又一个index可以走，那么最终返回的都是true
        int furthestJump = Math.min(position + nums[position], nums.length - 1);
        for (int nextPosition = position + 1; nextPosition <= furthestJump; nextPosition++) {
            if (canJumpFromPosition(nextPosition, nums)) {
                memo[position] = Index.GOOD;
                return true;
            }
        }
        memo[position] = Index.BAD;
        return false;
    }
 
    // 初始化并进入我们的helper function
    public boolean canJump(int[] nums) {
        memo = new Index[nums.length];
        for (int i = 0; i < memo.length; i++) {
            memo[i] = Index.UNKNOWN;
        }
        memo[memo.length - 1] = Index.GOOD;
        return canJumpFromPosition(0, nums);
    }
}

// 第三种写法 - bottom up
class Solution444 {
    public boolean canJump(int[] nums) {
        Index[] memo = new Index[nums.length];
        for (int i = 0; i < memo.length; i++) {
            memo[i] = Index.UNKNOWN;
        }
        memo[memo.length - 1] = Index.GOOD;

        for (int i = nums.length - 2; i >= 0; i--) {
            int furthestJump = Math.min(i + nums[i], nums.length - 1);
            for (int j = i + 1; j <= furthestJump; j++) {
                if (memo[j] == Index.GOOD) {
                    memo[i] = Index.GOOD;
                    break;
                }
            }
        }

        return memo[0] == Index.GOOD;
    }
}


// 扫描线 - 57. Insert Interval
class Solution5 {
    public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
        List<Interval> result = new LinkedList<>();
        int i = 0;
        // add all the intervals ending before newInterval starts
        // add属于linked list里面的方法
        while (i < intervals.size() && intervals.get(i).end < newInterval.start)
            // i++这里也太有趣了。
            result.add(intervals.get(i++));
        
        
        // 把从为overlap的添加进result里，那么剩下的就是有overlap的
        // merge all overlapping intervals to one considering newInterval
        while (i < intervals.size() && intervals.get(i).start <= newInterval.end) {
            // .get(i)是获取list里index为i的那一项，通过iteration，直到newInterval的end在intervals的前面。
            newInterval = new Interval( // we could mutate newInterval here also
                    Math.min(newInterval.start, intervals.get(i).start),
                    Math.max(newInterval.end, intervals.get(i).end));
            i++;
        }
        result.add(newInterval); // add the union of intervals we got
        
        
        // add all the rest
        while (i < intervals.size()) result.add(intervals.get(i++)); 
        return result;
    }
}


class Solution6 {
    public int[][] insert(int[][] intervals, int[] newInterval) {
      // init data
      int newStart = newInterval[0], newEnd = newInterval[1];
      int idx = 0, n = intervals.length;
      LinkedList<int[]> output = new LinkedList<int[]>();
  
      // add all intervals starting before newInterval
      while (idx < n && newStart > intervals[idx][0])
        output.add(intervals[idx++]);
  
      // add newInterval
      int[] interval = new int[2];
      // if there is no overlap, just add the interval
      if (output.isEmpty() || output.getLast()[1] < newStart)
        output.add(newInterval);
      // if there is an overlap, merge with the last interval
      else {
        interval = output.removeLast();
        interval[1] = Math.max(interval[1], newEnd);
        output.add(interval);
      }
  
      // add next intervals, merge with newInterval if needed
      while (idx < n) {
        interval = intervals[idx++];
        int start = interval[0], end = interval[1];
        // if there is no overlap, just add an interval
        if (output.getLast()[1] < start) output.add(interval);
        // if there is an overlap, merge with the last interval
        else {
          interval = output.removeLast();
          interval[1] = Math.max(interval[1], end);
          output.add(interval);
        }
      }
      return output.toArray(new int[output.size()][2]);
    }
  }

//   191. Number of 1 Bits
class Solution7 {
    // you need to treat n as an unsigned value
     public int hammingWeight(int n) {
        // return Integer.bitCount(n);
        // 初始化
        int bits = 0;
        int mask = 1;
         
        for (int i = 0; i < 32; i++) {
            // 得知最后一位是1是0
            if ((n & mask) != 0) {
                bits++;
            }
            mask <<= 1;
        }
        return bits;
    }
}

// 153. Find Minimum in Rotated Sorted Array
class Solution {
    public int findMin(int[] nums) {
        int lo = 0, hi = nums.length - 1;
        // 最后lo == hi
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if(nums[mid] < nums[hi]) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return nums[lo];
    }
}

// 543. Diameter of Binary Tree
class Solution8 {
    private int diameter;
    
    public int diameterOfBinaryTree(TreeNode root) {
        diameter = 0;
        helper(root);
        return diameter;
    }
    
    // helper的写法很有意思，全局变量很有意思，recursion的顺序很有意思。
    private int helper(TreeNode root) {
        if (root == null) return 0;
        int left = helper(root.left);
        int right = helper(root.right);
        diameter = Math.max(diameter, left+right);
        return Math.max(left, right) + 1;
    }
}

// 257
class Solution10 {
    public List<String> binaryTreePaths(TreeNode root) {
        LinkedList<String> paths = new LinkedList<>();
        helper(root, "", paths);
        return paths;
    }
    
    private void helper(TreeNode root, String path, LinkedList<String> paths) {
        if (root != null) {
            path += Integer.toString(root.val);
            if((root.left == null) && (root.right == null))
                paths.add(path);
            else {
                path += "->";
                helper(root.left, path, paths);
                helper(root.right, path, paths);
            }
        }
    }
}
// 110. Balanced Binary Tree
class Solution9 {
    public boolean isBalanced(TreeNode root) {
        if (root == null) return true;
        return Math.abs(helper(root.left) - helper(root.right)) < 2 &&
            isBalanced(root.left) && isBalanced(root.right);
    }
    
    private int helper(TreeNode root) {
        if (root == null) return 0;
        return 1 + Math.max(helper(root.left), helper(root.right));
        
    }
}


// 207 207. Course Schedule
// bfs/dfs/拓扑排序
"a = b++; // ++写在后面，说明前面那个东西前用了，也就是b先赋值给a了，然后b再+1

a = ++b; // ++写在前面，说明++先有效，即b要+1，然后赋值给a"
class Solution11 {
    public boolean canFinish(int n, int[][] prerequisites) {
        
        ArrayList<Integer>[] G = new ArrayList[n];
        int[] degree = new int[n];
        ArrayList<Integer> bfs = new ArrayList();
        
        // 前++是先自加再使用而后++是先使用再自加
        // 这里相当于双维列表了
        for (int i = 0; i < n; ++i) G[i] = new ArrayList<Integer>();
        
        // 相当于注册课程；e[0]是课程，e[1]是先修；那么这里G中就是如果g[e] == 0 就可以选修了
        // 这里的课程数量用另一个degree表示
        for (int[] e : prerequisites) {
            G[e[1]].add(e[0]);
            degree[e[0]]++;
        }
        
        // 初始化我们的bfs，相当于那个q
        for (int i = 0; i < n; ++i) if (degree[i] == 0) bfs.add(i);
        
        for (int i = 0; i < bfs.size(); ++i)
            for (int j: G[bfs.get(i)])
                if (--degree[j] == 0) bfs.add(j);
        
        return bfs.size() == n;
    }
}
// 33 
// 21 merge two sorted linkedlist
class Solution12 {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(-1);
        ListNode prev = dummy;
        
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                prev.next = l1;
                l1 = l1.next;
            } else {
                prev.next = l2;
                l2 = l2.next;
            }
            prev = prev.next;
        }
        prev.next = l1 == null? l2: l1;
            
        return dummy.next;
    }
}

class Solution13 {
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if (l1 == null) return l2;
        else if (l2 == null) return l1;
        else if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        }
        else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }
}

// 617 
class Solution14 {
    public TreeNode mergeTrees(TreeNode t1, TreeNode t2) {
        if (t1 == null) return t2;
        Stack<TreeNode[]> stack = new Stack<>();
        stack.push(new TreeNode[] {t1, t2});
        
        while (!stack.isEmpty()) {
            TreeNode[] t = stack.pop();
            if (t[0] == null || t[1] == null) continue;
            
            t[0].val += t[1].val;
            
            
            // 这里左右的顺序不重要，只要能够遍历就成了
                   
            if(t[0].right == null) {
                t[0].right = t[1].right;
            } else {
                stack.push(new TreeNode[] {t[0].right, t[1].right});
            }
            if (t[0].left == null) {
                t[0].left = t[1].left;
            } else {
                stack.push(new TreeNode[] {t[0].left, t[1].left});
            }
            
     
        }
        return t1;
    }
}
// 226. Invert Binary Tree
class Solution14 {
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return null;
        TreeNode right = invertTree(root.right);
        TreeNode left = invertTree(root.left);
        root.left = right;
        root.right = left;
        return root;
    }
}       


// 3. Longest Substring Without Repeating Characters
// 用了hashmap
// HashMap<Character, Integer> map = new HashMap<Chracter, Integer>();
// containsKey,length, charAt, Math.max, .get(), .put()
class Solution15 {
    public int lengthOfLongestSubstring(String s) {
        if (s == null) return 0;
        if (s.length() == 1) return 1;
        // map存的是目前遍历过的char的最右侧index
        HashMap<Character, Integer> map = new HashMap<Character, Integer>();
        int max = 0;
        for (int i = 0, j = 0; i < s.length(); i++) {
            // 如果map包含s[i]，j直接跳转到index + 1； j是左边界。
            if (map.containsKey(s.charAt(i))){
                j = Math.max(j, map.get(s.charAt(i)) + 1);
            }
            map.put(s.charAt(i), i);
            max = Math.max(max, i-j+1);
        }
        return max;
    }
}

// 23. Merge K sorted LinkedLists
class Solution16 {
    public ListNode mergeKLists(ListNode[] lists) {
        
    }
}