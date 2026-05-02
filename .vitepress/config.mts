import { defineConfig } from 'vitepress'

export default defineConfig({
  base: '/leetcode-notes/',
  srcDir: 'docs',
  title: "LeetCode Notes",
  description: "Personal algorithm journey",
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '算法分类', link: '/search/BFS' },
      { text: '其他题目', link: '/misc/leetcode' }
    ],
    sidebar: [
      {
        text: '🔍 搜索与图论',
        items: [
          { text: 'BFS 广度优先', link: '/search/BFS' },
          { text: 'BFS 二叉树', link: '/search/BFS Binary Tree' },
          { text: 'BFS 拓扑排序', link: '/search/BFS Topological Sort' },
          { text: 'DFS 深度优先', link: '/search/DFS' },
          { text: 'DFS 组合排列', link: '/search/DFS Combinations & Permutations' },
          { text: 'DFS N皇后', link: '/search/DFS N-Queens' }
        ]
      },
      {
        text: '🧩 动态规划 (DP)',
        items: [
          { text: 'DP 基础', link: '/DP/DP' },
          { text: 'DP 背包问题', link: '/DP/DP knapsack' },
          { text: 'DP LIS', link: '/DP/DP LIS' },
          { text: '72. 编辑距离', link: '/DP/72. Edit Distance' },
          { text: '688. 骑士在棋盘上的概率', link: '/DP/688.Knight Probability in Chessboard' },
          { text: '1143. 最长公共子序列', link: '/DP/1143. Longest Common Subsequence' },
          { text: '1269. 停在原地的方案数', link: '/DP/1269.Number of Ways to Stay in the Same Place After Some Steps' }
        ]
      },
      {
        text: '🏗️ 数据结构与设计',
        items: [
          { text: '设计类题目 (Design)', link: '/data-structures/Design' },
          { text: '链表 (Linked List)', link: '/data-structures/Linked List' },
          { text: '栈 (Stack)', link: '/data-structures/Stack' },
          { text: '单调栈 (Monotonic Stack)', link: '/data-structures/Monotonic Stack' },
          { text: '堆 (Heap)', link: '/data-structures/Heap' },
          { text: '字典树 (Trie)', link: '/data-structures/Trie' },
          { text: '并查集 (Union Find)', link: '/data-structures/Union Find' },
          { text: '哈希表 (Hashmap)', link: '/data-structures/Hashmap' }
        ]
      },
      {
        text: '⚡ 常用算法',
        items: [
          { text: '二分查找 (Binary Search)', link: '/algorithms/Binary Search' },
          { text: '滑动窗口 (Sliding Window)', link: '/algorithms/Sliding Window' },
          { text: '双指针 (Two Pointers)', link: '/algorithms/Two Pointers' },
          { text: '前缀和 (Prefix)', link: '/algorithms/Prefix' },
          { text: '贪心 (Greedy)', link: '/algorithms/Greedy' },
          { text: '位运算 (Bit Manipulation)', link: '/algorithms/Bit Manuputation' },
          { text: '递归 (Recursion)', link: '/algorithms/Recursion' },
          { text: '二叉树递归', link: '/algorithms/Binary Tree Recursion' },
          { text: '桶排序 (Bucket Sort)', link: '/algorithms/Bucket Sort' },
          { text: '归并排序 (Merge Sort)', link: '/algorithms/Merge Sort' },
          { text: '快速排序 (Quick Sort)', link: '/algorithms/Quick Sort' },
          { text: '数学 (Math)', link: '/algorithms/Math' }
        ]
      },
      {
        text: '📦 其他题型',
        items: [
          { text: '未归类题目', link: '/misc/leetcode' }
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/DropGuard/leetcode-notes' }
    ],
    search: {
      provider: 'local'
    }
  },
  markdown: {
    math: true
  }
})
