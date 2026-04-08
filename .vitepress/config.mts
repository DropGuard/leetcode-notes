import { defineConfig } from 'vitepress'

export default defineConfig({
  base: '/leetcode-notes/',
  srcDir: 'docs',
  title: "LeetCode Notes",
  description: "Personal algorithm journey",
  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Algorithms', link: '/search/BFS' },
      { text: 'Misc', link: '/misc/leetcode' }
    ],
    sidebar: [
      {
        text: '🔍 Search & Graph',
        items: [
          { text: 'BFS', link: '/search/BFS' },
          { text: 'BFS Binary Tree', link: '/search/BFS Binary Tree' },
          { text: 'BFS Topological Sort', link: '/search/BFS Topological Sort' },
          { text: 'DFS', link: '/search/DFS' },
          { text: 'DFS Combinations & Permutations', link: '/search/DFS Combinations & Permutations' },
          { text: 'DFS N-Queens', link: '/search/DFS N-Queens' }
        ]
      },
      {
        text: '🧩 Dynamic Programming (DP)',
        items: [
          { text: 'DP Basics', link: '/DP/DP' },
          { text: 'DP Knapsack', link: '/DP/DP knapsack' },
          { text: 'DP LIS', link: '/DP/DP LIS' },
          { text: '72. Edit Distance', link: '/DP/72. Edit Distance' },
          { text: '688. Knight Probability', link: '/DP/688.Knight Probability in Chessboard' },
          { text: '1143. Longest Common Subsequence', link: '/DP/1143. Longest Common Subsequence' },
          { text: '1269. Number of Ways to Stay', link: '/DP/1269.Number of Ways to Stay in the Same Place After Some Steps' }
        ]
      },
      {
        text: '🏗️ Data Structures & Design',
        items: [
          { text: 'Design', link: '/data-structures/Design' },
          { text: 'Linked List', link: '/data-structures/Linked List' },
          { text: 'Stack', link: '/data-structures/Stack' },
          { text: 'Monotonic Stack', link: '/data-structures/Monotonic Stack' },
          { text: 'Heap', link: '/data-structures/Heap' },
          { text: 'Trie', link: '/data-structures/Trie' },
          { text: 'Union Find', link: '/data-structures/Union Find' },
          { text: 'Hashmap', link: '/data-structures/Hashmap' }
        ]
      },
      {
        text: '⚡ Common Algorithms',
        items: [
          { text: 'Binary Search', link: '/algorithms/Binary Search' },
          { text: 'Sliding Window', link: '/algorithms/Sliding Window' },
          { text: 'Two Pointers', link: '/algorithms/Two Pointers' },
          { text: 'Prefix Sum', link: '/algorithms/Prefix' },
          { text: 'Greedy', link: '/algorithms/Greedy' },
          { text: 'Bit Manipulation', link: '/algorithms/Bit Manuputation' },
          { text: 'Recursion', link: '/algorithms/Recursion' },
          { text: 'Binary Tree Recursion', link: '/algorithms/Binary Tree Recursion' },
          { text: 'Bucket Sort', link: '/algorithms/Bucket Sort' },
          { text: 'Merge Sort', link: '/algorithms/Merge Sort' },
          { text: 'Quick Sort', link: '/algorithms/Quick Sort' },
          { text: 'Math', link: '/algorithms/Math' }
        ]
      },
      {
        text: '📦 Others',
        items: [
          { text: 'Uncategorized', link: '/misc/leetcode' }
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
