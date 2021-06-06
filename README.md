# LeetCode每日早刷

## **每天起床刷一题**(每天刷一题,强壮程序员)

``` go
注意: 这里面算法思想重要,可以用API,而不是手写多余没用东西
```

``` go
注意: 这里面算法思想重要,可以用API,而不是手写多余没用东西
```

``一直在忙着java学习和面试,少了刷题时间! 每天早起半小时/晚上1h刷题``

### 打乱数组 (2021-03-26 10:17:57)

![image-20210326101625676]( .\README.assets\image-20210326101625676.png)

#### 洗牌算法 Fisher-Yates

![image-20210326101658606]( .\README.assets\image-20210326101658606.png)

![image-20210326101730525]( .\README.assets\image-20210326101730525.png)



```java
class Solution {
    private int[] array;
    private int[] original;

    Random rand = new Random();

    private int randRange(int min, int max) {
        return rand.nextInt(max - min) + min;
    }

    private void swapAt(int i, int j) {
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }

    public Solution(int[] nums) {
        array = nums;
        original = nums.clone();
    }
    
    public int[] reset() {
        array = original;
        original = original.clone();
        return original;
    }
    
    public int[] shuffle() {
        for (int i = 0; i < array.length; i++) {
            swapAt(i, randRange(i, array.length));
        }
        return array;
    }
}

```

![image-20210326101744011]( .\README.assets\image-20210326101744011.png)



###  双指针

![image-20210325164324007]( .\README.assets\image-20210325164324007.png)

```java
class Solution {
    public void moveZeroes(int[] nums) {
        int n = nums.length, left = 0, right = 0;
        while (right < n) {
            if (nums[right] != 0) {
                swap(nums, left, right);
                left++;
            }
            right++;
        }
    }

    public void swap(int[] nums, int left, int right) {
        int temp = nums[left];
        nums[left] = nums[right];
        nums[right] = temp;
    }
}

```

![image-20210325164410737]( .\README.assets\image-20210325164410737.png)



### 数组中是否存在重复元素

**2021-03-24 14:15:15**

![image-20210324141519082]( .\README.assets\image-20210324141519082.png)

![image-20210324141544299]( .\README.assets\image-20210324141544299.png)

```java
class Solution {
    public boolean containsDuplicate(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        for (int i = 0; i < n - 1; i++) {
            if (nums[i] == nums[i + 1]) {
                return true;
            }
        }
        return false;
    }
}
```

![image-20210324141616105]( .\README.assets\image-20210324141616105.png)

方法二

### 哈希表

对于数组中每个元素，我们将它插入到哈希表中。如果插入一个元素时发现该元素已经存在于哈希表中，则说明存在重复的元素。

```java
class Solution {
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<Integer>();
        for (int x : nums) {
            if (!set.add(x)) {
                return true;
            }
        }
        return false;
    }
}
```

![image-20210324141650618]( .\README.assets\image-20210324141650618.png)



### 旋转数组

#### 2021-02-25 11:02:22 

![image-20210225104528957]( .\README.assets\image-20210225104528957.png)

```java
我的想法就是,从后向前找到k个元素,依次头插到最前面, 下面方法1就类似于我说的那样
```

![image-20210225104728957]( .\README.assets\image-20210225104728957.png)

![image-20210225104910000]( .\README.assets\image-20210225104910000.png)

### 多数元素

![image-20210222110611834]( .\README.assets\image-20210222110611834.png)



![image-20210222110652647]( .\README.assets\image-20210222110652647.png)



```java

class Solution {
    public int majorityElement(int[] nums) {
        Arrays.sort(nums);
        return nums[nums.length / 2];
    }
}

```



### 乘积最大子数组

#### 2021-02-20 10:33:44  

看解析都没看懂的题!  直接放上连接,  [乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/solution/cheng-ji-zui-da-zi-shu-zu-by-leetcode-solution/)

![image-20210220103327968]( .\README.assets\image-20210220103327968.png)

```java

class Solution {
    public int maxProduct(int[] nums) {
        int maxF = nums[0], minF = nums[0], ans = nums[0];
        int length = nums.length;
        for (int i = 1; i < length; ++i) {
            int mx = maxF, mn = minF;
            maxF = Math.max(mx * nums[i], Math.max(nums[i], mn * nums[i]));
            minF = Math.min(mn * nums[i], Math.min(nums[i], mx * nums[i]));
            ans = Math.max(maxF, ans);
        }
        return ans;
    }
}
```



### 反转字符串

**2021-02-18 09:29:10** **简单题重拳出击!**

![image-20210218092941061]( .\README.assets\image-20210218092941061.png)

![image-20210218093004944]( .\README.assets\image-20210218093004944.png)

**代码**

```java
class Solution {
    public void reverseString(char[] s) {
        int len = s.length;
        for (int i = 0; i < len/2; i++) {
            char temp = s[i];
            s[i] = s[len-i-1];
            s[len-i-1] = temp;
        }
    }
}
```



### 字符串中的第一个唯一字符

**2021-02-17 11:00:41**  **简单题重拳出击!**

![image-20210217110017055]( .\README.assets\image-20210217110017055.png)

```java
class Solution {

public int firstUniqChar(String s) {
       int[] arr = new int[26];
        int n = s.length();
        for (int i = 0; i < n; i++) {
            // chaAt() 是返回的AIIC码,-'a'是为了得到下标
            // 先遍历字符串, 由于已知为小写字母,故做一个标记
            // 标记:  字符串每个字母到a的距离,然后再做一次遍历
            // 如果有重复的字母, +1 
            arr[s.charAt(i)-'a']++ ;
            
        }
        for (int i = 0; i < n; i++) {
            if (arr[s.charAt(i)-'a'] == 1) {
                // 如果数组=1 ,返回下标
                return i;
            }
        }
        return -1;
    }
} 
```



### 有效的字母异位词

#### 2021-02-15 10:19:17(多想想API,而不是手写底层,拜托)

![image-20210215101923699]( .\README.assets\image-20210215101923699.png)



![image-20210215101956199]( .\README.assets\image-20210215101956199.png)

```java

class Solution {
    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length()) {
            return false;
        }
        char[] str1 = s.toCharArray();
        char[] str2 = t.toCharArray();
        Arrays.sort(str1);
        Arrays.sort(str2);
        return Arrays.equals(str1, str2);
    }
}


```



### 单词搜索 Ⅱ (又是很难得题)

![image-20210213160647593]( .\README.assets\image-20210213160647593.png)

![image-20210213160700717]( .\README.assets\image-20210213160700717.png)

[看参考答案吧,太难了](https://leetcode-cn.com/problems/word-search-ii/solution/dan-ci-sou-suo-ii-by-leetcode/)

#### 给个评论区优化后的代码

```go
//利用前缀树剪枝，优化DFS回溯  

// 前缀树（字典树）结点
    private static class TrieNode {
        TrieNode[] paths; // 因为题目规定只有小写字母，创建长度26的数组即可
        int end; // 前缀树【以当前字符结尾的单词的数量】
        int pass; // 前缀树【通过当前字符的单词的数量】

        public TrieNode() {
            paths = new TrieNode[26];
            end = 0;
            pass = 0;
        }
    }

    public static List<String> findWords(char[][] board, String[] words) {
        ArrayList<String> ans = new ArrayList<>();
        if (board == null || words == null || board.length == 0) return ans;

        int N = board.length;
        int M = board[0].length;
        HashSet<String> foundWords = new HashSet<>(); // 收集到的单词

        TrieNode root = buildTrieTree(words); // 根据单词表，构建前缀树

        // 尝试从每个位置出发，玩深度优先遍历（回溯），沿途收集单词放入foundWords
        // 通过前缀树【剪枝】优化
        for (int row = 0; row < N; row++) {
            for (int col = 0; col < M; col++) {
                LinkedList<Character> path = new LinkedList<>();
                boolean[][] access = new boolean[N][M]; // 标记是否访问过
                findWords(board, root, foundWords, row, col, path, access); // 从(row,col)出发，走出的单词放入foundWords
            }
        }

        return new ArrayList<>(foundWords);
    }

    // 当前来到board[row][col]位置，前面沿途走过的路径放在path中，走出的单词放入foundWords中
    // 当前在前缀树上走到pre上
    // 【返回本条DFS路径深度遍历总共走出了多少单词】
    private static int findWords(char[][] board, TrieNode pre, HashSet<String> foundWords,
                                  int row, int col, LinkedList<Character> path, boolean[][] access) {
        if (row < 0 || row >= board.length || col < 0 || col >= board[0].length) return 0;

        if (access[row][col]) return 0; // 已经走过这个位置 【不走回头路】

        char curChar = board[row][col]; // 当前迈上的字符
        if (pre.paths[curChar - 'a'] == null) return 0; // 【剪枝】没有这条路，当前尝试的路径走不下去了 
        if (pre.paths[curChar - 'a'].pass == 0) return 0; // 【剪枝】【有这条路，但是这条路上的单词之前都已经加入过了，无需再走下去】 

        TrieNode cur = pre.paths[curChar - 'a']; // 有这条路，走上去
        access[row][col] = true; // 访问了这个位置
        path.add(curChar); // 当前字符加入路径中
        int count = 0; // 搜出的单词数量

        if (cur.end > 0) { // 发现一个单词，收集
            foundWords.add(toWord(path));
            cur.end--; // 【收集完，end减1】
            count++; // 【搜索出1个单词】
        }

        int p1 = findWords(board, cur, foundWords, row-1, col, path, access); // 往上走
        int p2 = findWords(board, cur, foundWords, row+1, col, path, access); // 往下走
        int p3 = findWords(board, cur, foundWords, row, col-1, path, access); // 往左走
        int p4 = findWords(board, cur, foundWords, row, col+1, path, access); // 往右走

        path.removeLast(); // 轨迹擦除
        access[row][col] = false; // 走过一条DFS路径后 【恢复现场】
        count += (p1 + p2 + p3 + p4);
        cur.pass -= count; // 【前缀树上pass值更新】
        return count;
    }

    // 根据单词表，构建前缀树
    private static TrieNode buildTrieTree(String[] words) {
        TrieNode root = new TrieNode();

        for (String word : words) {
            TrieNode cur = root;
            for (char c : word.toCharArray()) {
                int path = c - 'a';
                if (cur.paths[path] == null) { // 当前cur结点没有这条路
                    cur.paths[path] = new TrieNode(); // 新建这条路
                }
                cur = cur.paths[path]; // cur走到这条路上
                cur.pass++;
            }
            cur.end++; // 有一个单词以cur结尾
        }

        return root;
    }

    private static String toWord(LinkedList<Character> path) {
        char[] chars = new char[path.size()];
        int i = 0;
        for (char c : path) {
            chars[i++] = c;
        }
        return new String(chars);
    }
```



###  实现Trie(前缀树)

![image-20210211123816216]( .\README.assets\image-20210211123816216.png)

[Trie树的官方解析!!!](https://leetcode-cn.com/problems/implement-trie-prefix-tree/solution/shi-xian-trie-qian-zhui-shu-by-leetcode/)

```java
// 递归版本
class Trie {
    private Node root;

    private class Node {
        private TreeMap<Character, Node> next;
        private boolean isWord;

        public Node(boolean isWord) {
            this.next = new TreeMap<>();
            this.isWord = isWord;
        }

        public Node() {
            this(false);
        }
    }

    /**
     * Initialize your data structure here.
     */
    public Trie() {
        root = new Node();
    }

    /**
     * Inserts a word into the trie.
     */
    public void insert(String word) {
        //获取根节点
        insertHelper(word, 0, root);
    }

    private void insertHelper(String word, int index, Node node) {
        if (index == word.length()) {
            node.isWord = true;
            return;
        }
        node.next.putIfAbsent(word.charAt(index), new Node());
        insertHelper(word, index + 1, node.next.get(word.charAt(index)));
    }

    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        Node cur = root;
        return searchTrie(word, 0, cur);
    }

    private boolean searchTrie(String word, int index, Node cur) {
        if (index == word.length()) {
            return cur.isWord;
        }
        if (cur.next.containsKey(word.charAt(index))) {
            return searchTrie(word, index + 1, cur.next.get(word.charAt(index)));
        } else {
            return false;
        }
    }
    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        return searchPre(prefix, 0, root);
    }

    private boolean searchPre(String prefix, int index, Node cur) {
        if (index == prefix.length()) {
            return true;
        }
        if (cur.next.containsKey(prefix.charAt(index))) {
            return searchPre(prefix, index + 1, cur.next.get(prefix.charAt(index)));
        } else {
            return false;
        }
    }
}
```



```java
// 非递归版本
class Trie {
    private Node root;

    private class Node {
        private TreeMap<Character, Node> next;
        private boolean isWord;

        public Node(boolean isWord) {
            this.next = new TreeMap<>();
            this.isWord = isWord;
        }

        public Node() {
            this(false);
        }
    }

    /**
     * Initialize your data structure here.
     */
    public Trie() {
        root = new Node();
    }

    /**
     * Inserts a word into the trie.
     */
    public void insert(String word) {
        //获取根节点
        Node cur = root;
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            cur.next.putIfAbsent(c,new Node());
            cur = cur.next.get(c);
        }
        if (!cur.isWord) {
            cur.isWord = true;
        }
    }
    /**
     * Returns if the word is in the trie.
     */
    public boolean search(String word) {
        Node cur = root;
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            if (!cur.next.containsKey(c)) {
                return false;
            } else {
                cur = cur.next.get(c);
            }
        }
        return cur.isWord;
    }

    /**
     * Returns if there is any word in the trie that starts with the given prefix.
     */
    public boolean startsWith(String prefix) {
        Node cur = root;
        for (int i = 0; i < prefix.length(); i++) {
            if (!cur.next.containsKey(prefix.charAt(i))) {
                return false;
            } else {
                cur = cur.next.get(prefix.charAt(i));
            }
        }
        return true;
    }
}
```



### 拆分单词Ⅱ

**2021-02-08 09:57:51**

![image-20210208095757131]( .\README.assets\image-20210208095757131.png)

![image-20210208095830506]( .\README.assets\image-20210208095830506.png)

```java
func wordBreak(s string, wordDict []string) (sentences []string) {
    wordSet := map[string]struct{}{}
    for _, w := range wordDict {
        wordSet[w] = struct{}{}
    }

    n := len(s)
    dp := make([][][]string, n)
    var backtrack func(index int) [][]string
    backtrack = func(index int) [][]string {
        if dp[index] != nil {
            return dp[index]
        }
        wordsList := [][]string{}
        for i := index + 1; i < n; i++ {
            word := s[index:i]
            if _, has := wordSet[word]; has {
                for _, nextWords := range backtrack(i) {
                    wordsList = append(wordsList, append([]string{word}, nextWords...))
                }
            }
        }
        word := s[index:]
        if _, has := wordSet[word]; has {
            wordsList = append(wordsList, []string{word})
        }
        dp[index] = wordsList
        return wordsList
    }
    for _, words := range backtrack(0) {
        sentences = append(sentences, strings.Join(words, " "))
    }
    return
}

```

![image-20210208095846707]( .\README.assets\image-20210208095846707.png)



### 拆分单词Ⅰ

#### 2021-02-07 10:46:09

![image-20210207104614655]( .\README.assets\image-20210207104614655.png)

#### 解析

![image-20210207104641737]( .\README.assets\image-20210207104641737.png)

``` java
public class Solution {
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> wordDictSet = new HashSet(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                if (dp[j] && wordDictSet.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[s.length()];
    }
}

```

``` go
func wordBreak(s string, wordDict []string) bool {
    wordDictSet := make(map[string]bool)
    for _, w := range wordDict {
        wordDictSet[w] = true
    }
    dp := make([]bool, len(s) + 1)
    dp[0] = true
    for i := 1; i <= len(s); i++ {
        for j := 0; j < i; j++ {
            if dp[j] && wordDictSet[s[j:i]] {
                dp[i] = true
                break
            }
        }
    }
    return dp[len(s)]
}
```

![image-20210207104735766]( .\README.assets\image-20210207104735766.png)

### 分割回文串

#### 2021-02-02 10:52:00

![image-20210202105213955]( .\README.assets\image-20210202105213955.png)

### 竟然用到了深度优先遍历

这个 算法 很秀!

![image-20210202105256943]( .\README.assets\image-20210202105256943.png)

![image-20210202105312377]( .\README.assets\image-20210202105312377.png)

### 方法一：回溯

```java
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

public class Solution {

    public List<List<String>> partition(String s) {
        int len = s.length();
        List<List<String>> res = new ArrayList<>();
        if (len == 0) {
            return res;
        }

        // Stack 这个类 Java 的文档里推荐写成 Deque<Integer> stack = new ArrayDeque<Integer>();
        // 注意：只使用 stack 相关的接口
        Deque<String> stack = new ArrayDeque<>();
        backtracking(s, 0, len, stack, res);
        return res;
    }

    /**
     * @param s
     * @param start 起始字符的索引
     * @param len   字符串 s 的长度，可以设置为全局变量
     * @param path  记录从根结点到叶子结点的路径
     * @param res   记录所有的结果
     */
    private void backtracking(String s, int start, int len, Deque<String> path, List<List<String>> res) {
        if (start == len) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = start; i < len; i++) {

            // 因为截取字符串是消耗性能的，因此，采用传子串索引的方式判断一个子串是否是回文子串
            // 不是的话，剪枝
            if (!checkPalindrome(s, start, i)) {
                continue;
            }

            path.addLast(s.substring(start, i + 1));
            backtracking(s, i + 1, len, path, res);
            path.removeLast();
        }
    }

    /**
     * 这一步的时间复杂度是 O(N)，因此，可以采用动态规划先把回文子串的结果记录在一个表格里
     *
     * @param str
     * @param left  子串的左边界，可以取到
     * @param right 子串的右边界，可以取到
     * @return
     */
    private boolean checkPalindrome(String str, int left, int right) {
        // 严格小于即可
        while (left < right) {
            if (str.charAt(left) != str.charAt(right)) {
                return false;
            }
            left++;
            right--;
        }
        return true;
    }
}

```

### 方法二：回溯的优化（加了动态规划）

在上一步，验证回文串那里，每一次都得使用“两边夹”的方式验证子串是否是回文子串。于是“用空间换时间”，利用「力扣」第 5 题：最长回文子串 的思路，利用动态规划把结果先算出来，这样就可以以 O(1)O(1) 的时间复杂度直接得到一个子串是否是回文。

```java
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Stack;

public class Solution {

    public List<List<String>> partition(String s) {
        int len = s.length();
        List<List<String>> res = new ArrayList<>();
        if (len == 0) {
            return res;
        }

        // 预处理
        // 状态：dp[i][j] 表示 s[i][j] 是否是回文
        boolean[][] dp = new boolean[len][len];
        // 状态转移方程：在 s[i] == s[j] 的时候，dp[i][j] 参考 dp[i + 1][j - 1]
        for (int right = 0; right < len; right++) {
            // 注意：left <= right 取等号表示 1 个字符的时候也需要判断
            for (int left = 0; left <= right; left++) {
                if (s.charAt(left) == s.charAt(right) && (right - left <= 2 || dp[left + 1][right - 1])) {
                    dp[left][right] = true;
                }
            }
        }

        Deque<String> stack = new ArrayDeque<>();
        backtracking(s, 0, len, dp, stack, res);
        return res;
    }

    private void backtracking(String s,
                              int start,
                              int len,
                              boolean[][] dp,
                              Deque<String> path,
                              List<List<String>> res) {
        if (start == len) {
            res.add(new ArrayList<>(path));
            return;
        }

        for (int i = start; i < len; i++) {
            // 剪枝
            if (!dp[start][i]) {
                continue;
            }
            path.addLast(s.substring(start, i + 1));
            backtracking(s, i + 1, len, dp, path, res);
            path.removeLast();
        }
    }
}

```





### 回文字符串遍历

**2021-02-01 10:12:15**

![image-20210201101249997]( .\README.assets\image-20210201101249997.png)

```java
class Solution {
    public boolean isPalindrome(String s) {
        StringBuffer sgood = new StringBuffer();
        int length = s.length();
        for (int i = 0; i < length; i++) {
            char ch = s.charAt(i);
            if (Character.isLetterOrDigit(ch)) {
                sgood.append(Character.toLowerCase(ch));
            }
        }
        StringBuffer sgood_rev = new StringBuffer(sgood).reverse();
        return sgood.toString().equals(sgood_rev.toString());
    }
}

```

![image-20210201101345397]( .\README.assets\image-20210201101345397.png)

![image-20210201101402843]( .\README.assets\image-20210201101402843.png)

```java
class Solution {
    public boolean isPalindrome(String s) {
        int n = s.length();
        int left = 0, right = n - 1;
        while (left < right) {
            while (left < right && !Character.isLetterOrDigit(s.charAt(left))) {
                ++left;
            }
            while (left < right && !Character.isLetterOrDigit(s.charAt(right))) {
                --right;
            }
            if (left < right) {
                if (Character.toLowerCase(s.charAt(left)) != Character.toLowerCase(s.charAt(right))) {
                    return false;
                }
                ++left;
                --right;
            }
        }
        return true;
    }
}

```



### 面试知识储备方向!

**2021-01-30 10:22:40**

![image-20210130102123090]( .\README.assets\image-20210130102123090.png)

```go
可以明确的一点是，面试算法题目在难度上（尤其是代码难度上）会略低一些，倾向于考察一些基础数据结构与算法，对于高级算法和奇技淫巧一般不作考察。

代码题主要考察编程语言的应用是否熟练，基础是否扎实，一般来会让面试者写出代码完成一些简单的需求或者使用递归实现某些功能，而数学题倾向于考察概率相关的问题。

以上这两类问题，出现的频率不会很高，即使出现了也应该是面试中的简单部分，相信一定难不倒在座的各位。
```

![image-20210130102226464]( .\README.assets\image-20210130102226464.png)

#### 越努力,越幸运!

### 鸡蛋掉落(5)

2021-01-27 11:40:02

这尼玛是谷歌经典面试题,超难那种,,直接放上解析了

![image-20210127114016387]( .\README.assets\image-20210127114016387.png)

**解析:** 

![image-20210127114107157]( .\README.assets\image-20210127114107157.png)

​	![image-20210127114148989]( .\README.assets\image-20210127114148989.png)

代码

```java

class Solution {
    public int superEggDrop(int K, int N) {
        return dp(K, N);
    }

    Map<Integer, Integer> memo = new HashMap();
    public int dp(int K, int N) {
        if (!memo.containsKey(N * 100 + K)) {
            int ans;
            if (N == 0) {
                ans = 0;
            } else if (K == 1) {
                ans = N;
            } else {
                int lo = 1, hi = N;
                while (lo + 1 < hi) {
                    int x = (lo + hi) / 2;
                    int t1 = dp(K-1, x-1);
                    int t2 = dp(K, N-x);

                    if (t1 < t2) {
                        lo = x;
                    } else if (t1 > t2) {
                        hi = x;
                    } else {
                        lo = hi = x;
                    }
                }

                ans = 1 + Math.min(Math.max(dp(K - 1, lo - 1), dp(K, N - lo)), Math.max(dp(K - 1, hi - 1), dp(K, N - hi)));
            }
                        memo.put(N * 100 + K, ans);
        }

        return memo.get(N * 100 + K);
    }
}


```

```python

class Solution:
    def superEggDrop(self, K: int, N: int) -> int:
        memo = {}
        def dp(k, n):
            if (k, n) not in memo:
                if n == 0:
                    ans = 0
                elif k == 1:
                    ans = n
                else:
                    lo, hi = 1, n
                    # keep a gap of 2 X values to manually check later
                    while lo + 1 < hi:
                        x = (lo + hi) // 2
                        t1 = dp(k-1, x-1)
                        t2 = dp(k, n-x)

                        if t1 < t2:
                            lo = x
                        elif t1 > t2:
                            hi = x
                        else:
                            lo = hi = x

                    ans = 1 + min(max(dp(k-1, x-1), dp(k, n-x))
                                  for x in (lo, hi))

                memo[k, n] = ans
            return memo[k, n]

        return dp(K, N)

```

![image-20210127114256995]( .\README.assets\image-20210127114256995.png)

### 合并两个有序数组(4)

**2021-01-26 10:32:58**



![image-20210126103457313]( .\README.assets\image-20210126103457313.png)

0. 贴上自己的思路代码(思路对,代码错)

   ```python
   class Solution(object):
     #思路: 先循环遍历nums1,nums1每一个元素与nums2所有元素比较,如果大于,添加在nums2的后面
     # 思路很对,就是实现错误,因为里面指针是指 下标 
       def merge(self, nums1, m, nums2, n):
           for i in nums1
               for  j in nums2 
                   if nums1[i]> nums2[j]
                       nums2[j+1] = nums1[i]
   ```

1. ####  双指针 / 从前往后

   ![image-20210126103601874]( .\README.assets\image-20210126103601874.png)

   **代码:**

   ```python
   
   class Solution(object):
       def merge(self, nums1, m, nums2, n):
           """
           :type nums1: List[int]
           :type m: int
           :type nums2: List[int]
           :type n: int
           :rtype: void Do not return anything, modify nums1 in-place instead.
           """
           # Make a copy of nums1.
           nums1_copy = nums1[:m] 
           nums1[:] = []
   
           # Two get pointers for nums1_copy and nums2.
           p1 = 0 
           p2 = 0
           
           # Compare elements from nums1_copy and nums2
           # and add the smallest one into nums1.
           while p1 < m and p2 < n: 
               if nums1_copy[p1] < nums2[p2]: 
                   nums1.append(nums1_copy[p1])
                   p1 += 1
               else:
                   nums1.append(nums2[p2])
                   p2 += 1
   
           # if there are still elements to add
           if p1 < m: 
               nums1[p1 + p2:] = nums1_copy[p1:]
           if p2 < n:
   
   ```

   **复杂度分析**

   - 时间复杂度 : O(n + m)*O*(*n*+*m*)。
   - 空间复杂度 : O(m)*O*(*m*)。

   **2.双指针/从后向前**

   

   ![image-20210126103744353]( .\README.assets\image-20210126103744353.png)

   ```java
   class Solution {
     public void merge(int[] nums1, int m, int[] nums2, int n) {
       // two get pointers for nums1 and nums2
       int p1 = m - 1;
       int p2 = n - 1;
       // set pointer for nums1
       int p = m + n - 1;
   
       // while there are still elements to compare
       while ((p1 >= 0) && (p2 >= 0))
         // compare two elements from nums1 and nums2 
         // and add the largest one in nums1 
         nums1[p--] = (nums1[p1] < nums2[p2]) ? nums2[p2--] : nums1[p1--];
   
       // add missing elements from nums2
       System.arraycopy(nums2, 0, nums1, 0, p2 + 1);
     }
   }
   ```

   

### 搜索二维数组(3)

2021-01-25 11:24:44

每日一题,开阔视野

![image-20210125112352722]( .\README.assets\image-20210125112352722.png)

![image-20210125112407035]( .\README.assets\image-20210125112407035.png)

**解析**:

1. 方法一：暴力法
   对于每一行我们可以像搜索未排序的一维数组——通过检查每个元素来判断是否有目标值。

   算法：
   这个算法并没有做到聪明的事情。我们循环数组，依次检查每个元素。如果，我们找到了，我们返回 true。否则，对于搜索到末尾都没有返回的循环，我们返回 false。此算法在所有情况下都是正确的答案，因为我们耗尽了整个搜索空间。
 ```python
   class Solution:
       def searchMatrix(self, matrix, target):
           for row in matrix:
               if target in row:
                   return True
           
           return False
 ```

2. 方法2
   因为矩阵的行和列是排序的（分别从左到右和从上到下），所以在查看任何特定值时，我们可以修剪O(m)O(m)或O(n)O(n)元素。

   算法：
   首先，我们初始化一个指向矩阵左下角的 (row，col)(row，col) 指针。然后，直到找到目标并返回 true（或者指针指向矩阵维度之外的 (row，col)(row，col) 为止，我们执行以下操作：如果当前指向的值大于目标值，则可以 “向上” 移动一行。 否则，如果当前指向的值小于目标值，则可以移动一列。不难理解为什么这样做永远不会删减正确的答案；因为行是从左到右排序的，所以我们知道当前值右侧的每个值都较大。 因此，如果当前值已经大于目标值，我们知道它右边的每个值会比较大。也可以对列进行非常类似的论证，因此这种搜索方式将始终在矩阵中找到目标（如果存在）。

   (大于tag row--(向上 减一)  小于tag  col++ (向右加1))

   ```jave
   
   class Solution {
       public boolean searchMatrix(int[][] matrix, int target) {
           // start our "pointer" in the bottom-left
           int row = matrix.length-1;
           int col = 0;
   
           while (row >= 0 && col < matrix[0].length) {
               if (matrix[row][col] > target) {
                   row--;
               } else if (matrix[row][col] < target) {
                   col++;
               } else { // found it
                   return true;
               }
           }
   
           return false;
       }
   }
   ```

   

### 多数元素(2)

2021-01-24 11:25:00

![image-20210124102026067]( .\README.assets\image-20210124102026067.png)

```go
func majorityElement(nums []int) int {
    n  := nums.length
    map := map[...]int
    // 1.先定义一个map 存放元素与相应的次数
    // 2. for循环遍历,把每个元素的出现次数存到map中
    //3. 遍历map集合,如果下次数大于 n/2 则打印出 
    // 这种方法会出现O(n^2) 不符合题
}
```

********

**解析**

![image-20210124102118587]( .\README.assets\image-20210124102118587.png)

```python
class Solution:
    def majorityElement(self, nums):
        counts = collections.Counter(nums)
        return max(counts.keys(), key=counts.get)

```

**解析2:摩尔投票法**

![image-20210124102320836]( .\README.assets\image-20210124102320836.png)

```python
class Solution:
    def majorityElement(self, nums):
        res = nums[0]
        times = 1
        for i in range(1, len(nums)):
            if times == 0:
                res = nums[i]
                times = 1
            elif nums[i] == res:
                times += 1
            else:
                times -= 1
        return res

```

### 只出现一次的数(1)

2021-01-23 21:06:19 

解析:![image-20210123210620105]( .\README.assets\image-20210123210620105.png)

***********

法一：位运算
如果没有时间复杂度和空间复杂度的限制，这道题有很多种解法，可能的解法有如下几种。

**使用集合存储数字**。遍历数组中的每个数字，如果集合中没有该数字，则将该数字加入集合，如果集合中已经有该数字，则将该数字从集合中删除，最后剩下的数字就是只出现一次的数字。

**使用哈希表**存储每个数字和该数字出现的次数。遍历数组即可得到每个数字出现的次数，并更新哈希表，最后遍历哈希表，得到只出现一次的数字。

**使用集合存**储数组中出现的所有数字，并计算数组中的元素之和。由于集合保证元素无重复，因此计算集合中的所有元素之和的两倍，即为每个元素出现两次的情况下的元素之和。由于数组中只有一个元素出现一次，其余元素都出现两次，因此用集合中的元素之和的两倍减去数组中的元素之和，剩下的数就是数组中只出现一次的数字。

上述三种解法都需要额外使用 O(n) 的空间，其中 n 是数组长度。如果要求使用线性时间复杂度和常数空间复杂度，上述三种解法显然都不满足要求。那么，如何才能做到线性时间复杂度和常数空间复杂度呢？

*****************

![image-20210123211157712]( .\README.assets\image-20210123211157712.png)

```go
func singleNumber(nums []int) int {
    single := 0
    for _, num := range nums {
        single ^= num
    }
    return single
}

```

