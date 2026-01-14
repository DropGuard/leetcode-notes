#### [1219. Path with Maximum Gold](https://leetcode.com/problems/path-with-maximum-gold/)

java

```java
class Solution {
    public int getMaximumGold(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int total = fullGrid(grid, m, n);
        if (total != -1)
            return total;
        boolean[][] visited = new boolean[m][n];
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) continue;
                res = Math.max(res, dfs(grid, m, n, i, j, visited));
            }
        }
        return res;
    }

    private int dfs(int[][] grid, int m, int n, int x, int y, boolean[][] visited) {
        visited[x][y] = true;
        int res = 0;
        int[][] dirs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        for (int[] dir : dirs) {
            int nx = x + dir[0];
            int ny = y + dir[1];
            if (nx < 0 || nx >= m || ny < 0 || ny >= n || grid[nx][ny] == 0 || visited[nx][ny]) {
                continue;
            }
            res = Math.max(res, dfs(grid, m, n, nx, ny, visited));
        }
        visited[x][y] = false;
        return grid[x][y] + res;
    }

    private int fullGrid(int[][] grid, int m, int n) {
        int total = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) return -1;
                else total += grid[i][j];
            }
        }
        return total;
    }
}
```

#### [1034. Coloring A Border](https://leetcode.com/problems/coloring-a-border/)

```java
class Solution {
    public int[][] colorBorder(int[][] grid, int row, int col, int color) {
        int m = grid.length, n = grid[0].length;
        boolean[][] visited = new boolean[m][n], isBoarder = new boolean[m][n];
        dfs(grid, row, col, m, n, visited, isBoarder);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (isBoarder[i][j]) grid[i][j] = color;
            }
        }
        return grid;
    }

    private void dfs(int[][] grid, int x, int y, int m, int n, boolean[][] visited, boolean[][] isBoarder) {
        // mark as visited
        visited[x][y] = true;
        // the border is on the boundary of the grid or adjacent to squares of a different color.
        isBoarder[x][y] = x == 0 || x == m - 1 || y == 0 || y == n - 1
                || grid[x][y] != grid[x - 1][y]
                || grid[x][y] != grid[x + 1][y]
                || grid[x][y] != grid[x][y - 1]
                || grid[x][y] != grid[x][y + 1];
        // a recursive call for each of 4 directions
        int[][] dirs = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
        for (int[] dir : dirs) {
            int nx = dir[0] + x;
            int ny = dir[1] + y;
            if (nx < 0 || nx >= m || ny < 0 || ny >= n || grid[nx][ny] != grid[x][y] || visited[nx][ny]) {
                continue;
            }
            dfs(grid, nx, ny, m, n, visited, isBoarder);
        }
    }
}
```

#### [797. All Paths From Source to Target](https://leetcode.com/problems/all-paths-from-source-to-target/)

java

```java
class Solution {
    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        List<List<Integer>> res = new ArrayList<>();
        int n = graph.length;
        dfs(graph, res, new ArrayList<>(), 0, n);
        return res;
    }

    private void dfs(int[][] graph, List<List<Integer>> res, ArrayList<Integer> path, int curr, int n) {
        path.add(curr);
        if (curr == n - 1) {
            res.add(new ArrayList<>(path));
        } else {
            for (int next : graph[curr]) {
                dfs(graph, res, path, next, n);
            }
        }
        path.remove(path.size() - 1);
    }
}
```

#### 934. Shortest Bridge

java

```java
class Solution {
    public int shortestBridge(int[][] grid) {
        int[][] dup = Arrays.stream(grid).map(int[]::clone).toArray(int[][]::new);
        int n = grid.length;
        Queue<int[]> queue = new ArrayDeque<>();
        int[][] dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (dup[i][j] == 0) continue;
                dfs(dup, i, j, queue, n, dirs);
                return bfs(dup, queue, n, dirs);
            }
        }
        throw new Error("unreachable");
    }

    private int bfs(int[][] grid, Queue<int[]> queue, int n, int[][] dirs) {
        int step = 0;
        while (!queue.isEmpty()) {
            for (int k = queue.size(); k > 0; k--) {
                int[] poll = queue.poll();
                for (int[] dir : dirs) {
                    int x = poll[0] + dir[0];
                    int y = poll[1] + dir[1];
                    if (x < 0 || y < 0 || x >= n || y >= n || grid[x][y] == 2) continue;
                    if (grid[x][y] == 1) return step;
                    grid[x][y] = 2;
                    queue.offer(new int[]{x, y});
                }
            }
            step++;
        }
        return -1;
    }

    // change connected 1 to 2, collect them to the queue
    private void dfs(int[][] grid, int i, int j, Queue<int[]> queue, int n, int[][] dirs) {
        grid[i][j] = 2;
        queue.offer(new int[]{i, j});
        for (int[] dir : dirs) {
            int x = i + dir[0];
            int y = j + dir[1];
            if (x < 0 || y < 0 || x >= n || y >= n) continue;
            if (grid[x][y] == 1) {
                dfs(grid, x, y, queue, n, dirs);
            }
        }
    }
}
```

#### 399. Evaluate Division

java

```java
class Solution {
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        var graph = graphBuild(equations, values);
        int n = queries.size();
        double[] res = new double[n];
        for (int i = 0; i < n; i++) {
            res[i] = dfs(graph, queries.get(i).get(0), queries.get(i).get(1), new HashSet<>());
        }
        return res;
    }

    private double dfs(HashMap<String, HashMap<String, Double>> graph, String start, String end, HashSet<String> visited) {
        if (!graph.containsKey(start)) {
            return -1;
        }
        if (graph.get(start).containsKey(end)) {
            return graph.get(start).get(end);
        }
        visited.add(start);
        for (String neighbor : graph.get(start).keySet()) {
            if (visited.contains(neighbor)) continue;
            double weight = dfs(graph, neighbor, end, visited);
            if (weight == -1) continue;
            return weight * graph.get(start).get(neighbor);
        }
        return -1;
    }

    private HashMap<String, HashMap<String, Double>> graphBuild(List<List<String>> equations, double[] values) {
        HashMap<String, HashMap<String, Double>> graph = new HashMap<>();
        for (int i = 0; i < equations.size(); i++) {
            String u = equations.get(i).get(0);
            String v = equations.get(i).get(1);
            graph.putIfAbsent(u, new HashMap<>());
            graph.get(u).put(v, values[i]);
            graph.putIfAbsent(v, new HashMap<>());
            graph.get(v).put(u, 1 / values[i]);
        }
        return graph;
    }
}
```

#### 79. Word Search

go

```go
func exist(board [][]byte, word string) bool {
	m, n := len(board), len(board[0])
	visited := make([][]bool, m)
	for i := range visited {
		visited[i] = make([]bool, n)
	}
	for i := range board {
		for j := range board[i] {
			if dfs(board, word, i, j, m, n, visited) {
				return true
			}
		}
	}
	return false
}

func dfs(board [][]byte, word string, i, j, m, n int, visited [][]bool) bool {
	if len(word) == 1 {
		return word[0] == board[i][j]
	}
	if board[i][j] != word[0] {
		return false
	}
	visited[i][j] = true
	dirs := [4][2]int{{-1, 0}, {0, 1}, {1, 0}, {0, -1}}
	for _, dir := range dirs {
		x := i + dir[0]
		y := j + dir[1]
		if x < 0 || x >= m || y < 0 || y >= n || visited[x][y] {
			continue
		}
		if dfs(board, word[1:], x, y, m, n, visited) {
			return true
		}
	}
	visited[i][j] = false
	return false
}
```

rust

```rust
impl Solution {
    pub fn exist(board: Vec<Vec<char>>, word: String) -> bool {
        let m = board.len();
        let n = board[0].len();
        let word: Vec<char> = word.chars().collect();
        let mut visited = vec![vec![false; n]; m];
        for i in 0..m {
            for j in 0..n {
                if Solution::dfs(&board, &word, i, j, m, n, &mut visited) {
                    return true;
                }
            }
        }
        false
    }
    fn dfs(board: &Vec<Vec<char>>, word: &[char], x: usize, y: usize, m: usize, n: usize, visited: &mut Vec<Vec<bool>>) -> bool {
        if word.len() == 1 {
            return board[x][y] == word[0];
        }
        if board[x][y] != word[0] {
            return false;
        }
        visited[x][y] = true;
        let dirs = [[-1, 0], [0, 1], [1, 0], [0, -1]];
        for dir in dirs {
            let x = x as i32 + dir[0];
            let y = y as i32 + dir[1];
            if x < 0 || y < 0 {
                continue;
            }
            let x = x as usize;
            let y = y as usize;
            if x >= m || y >= n || visited[x][y] {
                continue;
            }
            if Solution::dfs(board, &word[1..], x, y, m, n, visited) {
                visited[x][y] = false;
                return true;
            }
        }
        visited[x][y] = false;
        false
    }
}
```

#### 130. Surrounded Regions

go

```go
func solve(board [][]byte) {
	m, n := len(board), len(board[0])
	for i := range board {
		for j := range board[0] {
			if board[i][j] == 'O' && (i == 0 || j == 0 || i == m-1 || j == n-1) {
				dfs(board, i, j, m, n)
			}
		}
	}
	for i := range board {
		for j := range board[0] {
			if board[i][j] == 'O' {
				board[i][j] = 'X'
			}
			if board[i][j] == '#' {
				board[i][j] = 'O'
			}
		}
	}
}

func dfs(board [][]byte, x, y, m, n int) {
	board[x][y] = '#'
	dirs := [4][2]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
	for _, dir := range dirs {
		nx := x + dir[0]
		ny := y + dir[1]
		if nx < 0 || nx >= m || ny < 0 || ny >= n || board[nx][ny] != 'O' {
			continue
		}
		dfs(board, nx, ny, m, n)
	}
}
```

#### 133. Clone Graph

go

```go
func cloneGraph(node *Node) *Node {
	if node == nil {
		return nil
	}
	copies := make(map[*Node]*Node)
	return dfs(node, copies)
}
func dfs(node *Node, copies map[*Node]*Node) *Node {
	if _, ok := copies[node]; !ok {
		copies[node] = &Node{Val: node.Val}
		for _, neighbor := range node.Neighbors {
			copies[node].Neighbors = append(copies[node].Neighbors, dfs(neighbor, copies))
		}
	}
	return copies[node]
}
```

python

```python
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return None
        copies = {}
        return self.dfs(node, copies)

    def dfs(self, node: 'Node', copies: dict) -> 'Node':
        if node not in copies:
            copies[node] = Node(node.val, [])
            for neighbor in node.neighbors:
                copies[node].neighbors += self.dfs(neighbor, copies),
        return copies[node]

```

java

```java
class Solution {  
    public Node cloneGraph(Node node) {  
        if (node == null) {  
            return null;  
        }  
        HashMap<Node, Node> copies = new HashMap<>();  
        return dfs(node, copies);  
    }  
  
    Node dfs(Node node, HashMap<Node, Node> copies) {  
        if (!copies.containsKey(node)) {  
            copies.put(node, new Node(node.val));  
            for (Node neighbor : node.neighbors) {  
                copies.get(node).neighbors.add(dfs(neighbor, copies));  
            }  
        }  
        return copies.get(node);  
    }  
}
```

#### 200. Number of Islands

go

```go
func numIslands(grid [][]byte) int {
	m, n := len(grid), len(grid[0])
	visited := make([][]bool, m)
	for i := range visited {
		visited[i] = make([]bool, n)
	}
	res := 0
	for i := range grid {
		for j := range grid[0] {
			if grid[i][j] == '0' || visited[i][j] {
				continue
			}
			res++
			dfs(i, j, m, n, grid, visited)
		}
	}
	return res
}
func dfs(i, j, m, n int, grid [][]byte, visited [][]bool) {
	visited[i][j] = true
	dirs := [4][2]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
	for _, dir := range dirs {
		x := i + dir[0]
		y := j + dir[1]
		if x < 0 || y < 0 || x >= m || y >= n || grid[x][y] == '0' || visited[x][y] {
			continue
		}
		dfs(x, y, m, n, grid, visited)
	}
}
```

rust

```rust
impl Solution {
    pub fn num_islands(grid: Vec<Vec<char>>) -> i32 {
        let m = grid.len();
        let n = grid[0].len();
        let mut res = 0;
        let mut visited = vec![vec![false; n]; m];
        for i in 0..m {
            for j in 0..n {
                if grid[i][j] == '0' || visited[i][j] {
                    continue;
                }
                res += 1;
                Solution::dfs(&grid, &mut visited, i, j, m, n)
            }
        }
        res
    }
    fn dfs(grid: &[Vec<char>], visited: &mut [Vec<bool>], i: usize, j: usize, m: usize, n: usize) {
        visited[i][j] = true;
        let dirs = [[-1, 0], [0, -1], [0, 1], [1, 0]];
        for dir in dirs {
            let x = i as i32 + dir[0];
            let y = j as i32 + dir[1];
            if x < 0 || y < 0 {
                continue;
            }
            let x = x as usize;
            let y = y as usize;
            if x >= m || y >= n || grid[x][y] == '0' || visited[x][y] {
                continue;
            }
            Solution::dfs(grid, visited, x, y, m, n)
        }
    }
}
```

#### 207. Course Schedule

有向图中dfs找环

判断DAG

每次当前层将该节点涂成unsafe然后递归查找邻居

java

```java
class Solution {
    private static final int safe = 1;
    private static final int unsafe = 2;

    public boolean canFinish(int n, int[][] prerequisites) {
        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++)
            graph.add(new ArrayList<>());
        for (int[] pre : prerequisites)
            graph.get(pre[0]).add(pre[1]);
        int[] dp = new int[n];
        for (int i = 0; i < n; i++) {
            if (cyclic(graph, dp, i)) return false;
        }
        return true;
    }

    boolean cyclic(ArrayList<ArrayList<Integer>> graph, int[] dp, int i) {
        if (dp[i] > 0) return dp[i] == unsafe;
        dp[i] = unsafe;
        for (Integer v : graph.get(i)) {
            if (cyclic(graph, dp, v)) return true;
        }
        dp[i] = safe;
        return false;
    }
}
```

#### 210. Course Schedule II

java

```java
public class Solution {
    private static final int safe = 1;
    private static final int unsafe = 2;

    public int[] findOrder(int n, int[][] prerequisites) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++)
            graph.add(new ArrayList<>());
        for (int[] pre : prerequisites) {
            graph.get(pre[0]).add(pre[1]);
        }
        int[] dp = new int[n];

        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < n; i++)
            if (cyclic(res, graph, dp, i)) return new int[0];
        return res.stream().mapToInt(i -> i).toArray();
    }

    private boolean cyclic(List<Integer> path, List<List<Integer>> graph, int[] dp, int i) {
        if (dp[i] > 0) return dp[i] == unsafe;
        dp[i] = unsafe;
        for (int j : graph.get(i))
            if (cyclic(path, graph, dp, j)) return true;
        dp[i] = safe;
        path.add(i);
        return false;
    }
}
```

#### 310. Minimum Height Trees

在无向无环图中找到图的中点

java

```java
class Solution {  
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {  
        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();  
        for (int i = 0; i < n; i++) {  
            graph.add(new ArrayList<>());  
        }  
        for (int[] edge : edges) {  
            graph.get(edge[0]).add(edge[1]);  
            graph.get(edge[1]).add(edge[0]);  
        }  
        int[] parent = new int[n];  
        int farthest = findFarthest(0, n, graph, parent);  
        int end = findFarthest(farthest, n, graph, parent);  
  
        ArrayList<Integer> path = new ArrayList<>();  
        while (end != -1) {  
            path.add(end);  
            end = parent[end];  
        }  
        Integer a = path.get(path.size() / 2);  
        Integer b = path.get((path.size() - 1) / 2);  
        if (a.equals(b)) {  
            return List.of(a);  
        }  
        return List.of(a, b);  
    }  
  
    int findFarthest(int start, int n, ArrayList<ArrayList<Integer>> graph, int[] parent) {  
        int[] distance = new int[n];  
        Arrays.fill(distance, -1);  
        Arrays.fill(parent, -1);  
        distance[start] = 0;  
        dfs(start, distance, parent, graph);  
        int res = 0, maxDis = 0;  
        for (int i = 0; i < distance.length; i++) {  
            if (distance[i] > maxDis) {  
                maxDis = distance[i];  
                res = i;  
            }  
        }  
        return res;  
    }  
  
    void dfs(int start, int[] distance, int[] parent, ArrayList<ArrayList<Integer>> graph) {  
        for (Integer neighbor : graph.get(start)) {  
            if (distance[neighbor] >= 0) continue;  
            distance[neighbor] = distance[start] + 1;  
            parent[neighbor] = start;  
            dfs(neighbor, distance, parent, graph);  
        }  
    }  
}
```

#### 329. Longest Increasing Path in a Matrix

记忆化搜索，命中缓存就直接返回，长度默认填1，DFS寻找四周的高点累加1

java

```java
class Solution {
    public int longestIncreasingPath(int[][] matrix) {
        int m = matrix.length, n = matrix[0].length;
        int[][] dp = new int[m][n];
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                res = Math.max(res, dfs(i, j, m, n, dp, matrix));
            }
        }
        return res;
    }

    int dfs(int i, int j, int m, int n, int[][] dp, int[][] matrix) {
        if (dp[i][j] > 0) {
            return dp[i][j];
        }
        int[][] dirs = {{0, 1}, {1, 0}, {-1, 0}, {0, -1}};
        dp[i][j] = 1;
        for (int[] dir : dirs) {
            int x = i + dir[0];
            int y = j + dir[1];
            if (x < 0 || y < 0 || x >= m || y >= n || matrix[x][y] <= matrix[i][j]) {
                continue;
            }
            dp[i][j] = Math.max(dp[i][j], dfs(x, y, m, n, dp, matrix) + 1);
        }
        return dp[i][j];
    }
}
```

go

```go
func longestIncreasingPath(matrix [][]int) int {  
   m, n := len(matrix), len(matrix[0])  
   res := 0  
   dp := make([][]int, m)  
   for i := range dp {  
      dp[i] = make([]int, n)  
   }  
   for i := 0; i < m; i++ {  
      for j := 0; j < n; j++ {  
         res = max(res, dfs(i, j, m, n, matrix, dp))  
      }  
   }  
   return res  
}  
func dfs(i, j, m, n int, matrix, dp [][]int) int {  
   if dp[i][j] > 0 {  
      return dp[i][j]  
   }  
   dirs := [4][2]int{{0, 1}, {1, 0}, {0, -1}, {-1, 0}}  
   dp[i][j] = 1  
   for _, dir := range dirs {  
      x := i + dir[0]  
      y := j + dir[1]  
      if x < 0 || y < 0 || x >= m || y >= n || matrix[x][y] <= matrix[i][j] {  
         continue  
      }  
      dp[i][j] = max(dp[i][j], dfs(x, y, m, n, matrix, dp)+1)  
   }  
   return dp[i][j]  
}  
func max(a, b int) int {  
   if a > b {  
      return a  
   }  
   return b  
}
```

rust

```rust
use std::cmp::max;  
  
impl Solution {  
    pub fn longest_increasing_path(matrix: Vec<Vec<i32>>) -> i32 {  
        let m = matrix.len();  
        let n = matrix[0].len();  
        let mut dp = vec![vec![0; n]; m];  
        let mut res = 0;  
        for i in 0..m {  
            for j in 0..n {  
                res = max(res, dfs(i, j, &mut dp, &matrix, m, n))  
            }  
        }  
  
        fn dfs(i: usize, j: usize, dp: &mut Vec<Vec<i32>>, matrix: &Vec<Vec<i32>>, m: usize, n: usize) -> i32 {  
            if dp[i][j] > 0 {  
                return dp[i][j];  
            }  
            let dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]];  
            dp[i][j] = 1;  
            for dir in dirs {  
                let x = i as i32 + dir[0];  
                let y = j as i32 + dir[1];  
                if x < 0 || y < 0 {  
                    continue;  
                }  
                let x = x as usize;  
                let y = y as usize;  
                if x >= m || y >= n || matrix[x][y] <= matrix[i][j] {  
                    continue;  
                }  
                dp[i][j] = max(dp[i][j], dfs(x, y, dp, matrix, m, n) + 1);  
            }  
            dp[i][j]  
        }  
        res  
    }  
}
```

#### 694.Number of Distinct Islands

testing [https://www.lintcode.com/problem/860/description](https://www.lintcode.com/problem/860/description)

java

```java
public class Solution {
    public int numberofDistinctIslands(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        boolean[][] visited = new boolean[m][n];
        Set<ArrayList<Integer>> res = new HashSet<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 0 || visited[i][j]) continue;
                ArrayList<Integer> path = new ArrayList<>();
                dfs(i, j, grid, path, visited, i, j, m, n);
                res.add(path);
            }
        }
        return res.size();
    }

    void dfs(int i, int j, int[][] grid, ArrayList<Integer> path, boolean[][] visited, int startI, int startJ, int m, int n) {
        visited[i][j] = true;
        path.add(i - startI);
        path.add(j - startJ);
        int[][] dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int[] dir : dirs) {
            int x = i + dir[0], y = j + dir[1];
            if (x < 0 || x >= m || y < 0 || y >= n || visited[x][y] || grid[x][y] == 0) continue;
            dfs(x, y, grid, path, visited, startI, startJ, m, n);
        }
    }
}
```

#### 695. Max Area of Island

java

```java
class Solution {
    public int maxAreaOfIsland(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        boolean[][] visited = new boolean[m][n];
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0 || visited[i][j]) continue;
                res = Math.max(res, dfs(grid, visited, m, n, i, j));
            }
        }
        return res;
    }

    int dfs(int[][] grid, boolean[][] visited, int m, int n, int i, int j) {
        visited[i][j] = true;
        int[][] dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        int res = 1;
        for (int[] dir : dirs) {
            int x = i + dir[0];
            int y = j + dir[1];
            if (x < 0 || y < 0 || x >= m || y >= n || visited[x][y] || grid[x][y] == 0) continue;
            res += dfs(grid, visited, m, n, x, y);
        }
        return res;
    }
}
```

#### 721. Accounts Merge

java

```java
class Solution {
    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        int n = accounts.size();
        HashMap<String, Integer> mailToIndex = new HashMap<>();
        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            graph.add(new ArrayList<>());
        }
        for (int i = 0; i < n; i++) {
            for (int j = 1; j < accounts.get(i).size(); j++) {
                String curr = accounts.get(i).get(j);
                if (mailToIndex.containsKey(curr)) {
                    Integer prev = mailToIndex.get(curr);
                    graph.get(prev).add(i);
                    graph.get(i).add(prev);
                } else mailToIndex.put(curr, i);
            }
        }
        HashMap<Integer, HashSet<String>> indexToMail = new HashMap<>();
        boolean[] visited = new boolean[n];
        for (int i = 0; i < n; i++) {
            if (visited[i]) continue;
            HashSet<String> path = new HashSet<>();
            DFS(i, graph, path, visited, accounts);
            indexToMail.put(i, path);
        }
        List<List<String>> res = new ArrayList<>();
        for (int i : indexToMail.keySet()) {
            ArrayList<String> path = new ArrayList<>();
            path.add(accounts.get(i).get(0));
            path.addAll(indexToMail.get(i).stream().sorted().toList());
            res.add(path);
        }
        return res;
    }

    void DFS(Integer i, ArrayList<ArrayList<Integer>> graph, HashSet<String> path, boolean[] visited, List<List<String>> accounts) {
        visited[i] = true;
        path.addAll(accounts.get(i).stream().skip(1).toList());
        for (int neighbor : graph.get(i)) {
            if (visited[neighbor]) continue;
            DFS(neighbor, graph, path, visited, accounts);
        }
    }
}
```

#### 785. Is Graph Bipartite?

验证二分图

java

```java
class Solution {
    public boolean isBipartite(int[][] graph) {
        int n = graph.length;
        int[] colors = new int[n];
        for (int i = 0; i < n; i++) {
            if (colors[i] != 0) continue;
            if (!dfs(graph, i, 1, colors)) return false;
        }
        return true;
    }

    boolean dfs(int[][] graph, int i, int color, int[] colors) {
        if (colors[i] != 0) return colors[i] == color;
        colors[i] = color;
        for (int j : graph[i]) {
            if (!dfs(graph, j, -color, colors)) return false;
        }
        return true;
    }
}
```

python

```python
class Solution:  
    def isBipartite(self, graph: List[List[int]]) -> bool:  
        n = len(graph)  
        dp = [0] * n  
        for i in range(n):  
            if not dp[i] and not self.dfs(1, i, dp, graph):  
                return False  
        return True  
    def dfs(self, color: int, i: int, dp: List[int], graph: List[List[int]]) -> bool:  
        if dp[i]:  
            return dp[i] == color  
        dp[i] = color  
        return all(self.dfs(-color, j, dp, graph) for j in graph[i])
```

#### 802. Find Eventual Safe States

标记为unsafe之后递归查找邻居，如果查找到unsafe，就说明成环了，自身也是unsafe，直接return，否则说明自己safe

java

```java
class Solution {
    private static final int safe = 1;
    private static final int unsafe = 2;

    public List<Integer> eventualSafeNodes(int[][] graph) {
        int n = graph.length;
        int[] dp = new int[n];

        for (int i = 0; i < n; i++) {
            dfs(i, dp, graph);
        }
        ArrayList<Integer> res = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (dp[i] == safe) res.add(i);
        }
        return res;
    }

    boolean dfs(int i, int[] color, int[][] graph) {
        if (color[i] > 0) return color[i] == safe;
        color[i] = unsafe;
        for (int j : graph[i]) {
            if (!dfs(j, color, graph)) return false;
        }
        color[i] = safe;
        return true;
    }
}
```

go

```go
const safe = 1
const unsafe = 2

func eventualSafeNodes(graph [][]int) []int {
	n := len(graph)
	color := make([]int, n)
	for i := range graph {
		dfs(graph, i, color)
	}
	var res []int
	for i, v := range color {
		if v == safe {
			res = append(res, i)
		}
	}
	return res
}
func dfs(graph [][]int, i int, color []int) bool {
	if color[i] > 0 {
		return color[i] == safe
	}
	color[i] = unsafe
	for _, j := range graph[i] {
		if !dfs(graph, j, color) {
			return false
		}
	}
	color[i] = safe
	return true
}
```

rust

```rust
impl Solution {
    const SAFE: i32 = 1;
    const UNSAFE: i32 = 2;
    pub fn eventual_safe_nodes(graph: Vec<Vec<i32>>) -> Vec<i32> {
        let n = graph.len();
        let mut color = vec![0; n];

        fn dfs(graph: &Vec<Vec<i32>>, color: &mut Vec<i32>, i: usize) -> bool {
            if color[i] > 0 {
                return color[i] == Solution::SAFE;
            }
            color[i] = Solution::UNSAFE;
            for j in &graph[i] {
                if !dfs(graph, color, *j as usize) {
                    return false;
                }
            }
            color[i] = Solution::SAFE;
            true
        }

        for i in 0..n {
            dfs(&graph, &mut color, i);
        }
        color.iter().enumerate().filter(|(_, &v)| v == Solution::SAFE).map(|(i, _)| i as i32).collect::<Vec<i32>>()
    }
}
```

#### 827. Making A Large Island

java

```java
class Solution {
    public int largestIsland(int[][] grid) {
        HashMap<Integer, Integer> area = new HashMap<>();
        int mark = 2, res = 0, n = grid.length;
        int[][] cp = Arrays.copyOf(grid, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (cp[i][j] == 1) {
                    area.put(mark, dfs(cp, i, j, n, mark));
                    res = Math.max(res, area.get(mark++));
                }
            }
        }
        int[][] dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (cp[i][j] == 0) {
                    ArrayList<Integer> seen = new ArrayList<>();
                    int curr = 1;
                    for (int[] dir : dirs) {
                        int x = i + dir[0], y = j + dir[1];
                        if (out(x, y, n) || cp[x][y] == 0 || seen.contains(cp[x][y])) continue;
                        seen.add(cp[x][y]);
                        curr += area.get(cp[x][y]);
                    }
                    res = Math.max(res, curr);
                }
            }
        }
        return res;
    }

    int dfs(int[][] grid, int i, int j, int n, int mark) {
        grid[i][j] = mark;
        int[][] dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        int res = 1;
        for (int[] dir : dirs) {
            int x = i + dir[0];
            int y = j + dir[1];
            if (out(x, y, n) || grid[x][y] != 1) continue;
            res += dfs(grid, x, y, n, mark);
        }
        return res;
    }

    boolean out(int x, int y, int n) {
        return 0 > x || x >= n || 0 > y || y >= n;
    }
}
```

#### 886. Possible Bipartition

java

```java
class Solution {

    public boolean possibleBipartition(int n, int[][] dislikes) {
        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < n + 1; i++) graph.add(new ArrayList<>());
        for (int[] dislike : dislikes) {
            int u = dislike[0], v = dislike[1];
            graph.get(u).add(v);
            graph.get(v).add(u);
        }
        int[] colors = new int[n + 1];
        for (int i = 1; i < n + 1; i++) {
            if (colors[i] != 0) {
                continue;
            }
            if (!dfs(i, 1, colors, graph)) {
                return false;
            }
        }
        return true;
    }

    boolean dfs(int u, int target, int[] colors, ArrayList<ArrayList<Integer>> graph) {
        if (colors[u] != 0) {
            return colors[u] == target;
        }
        colors[u] = target;
        for (Integer v : graph.get(u)) {
            if (!dfs(v, -target, colors, graph)) {
                return false;
            }
        }
        return true;
    }
}
```

#### 980. Unique Paths III

java

```java
class Solution {
    int empty;

    public int uniquePathsIII(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int startI = 0, startJ = 0;
        empty = 1; // starting point
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) {
                    empty++;
                    continue;
                }
                if (grid[i][j] == 1) {
                    startI = i;
                    startJ = j;
                }
            }
        }
        return dfs(grid, startI, startJ, m, n, new boolean[m][n]);
    }

    int dfs(int[][] grid, int i, int j, int m, int n, boolean[][] visited) {
        if (grid[i][j] == 2) {
            return empty == 0 ? 1 : 0;
        }
        visited[i][j] = true;
        empty--;
        int[][] dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        int res = 0;
        for (int[] dir : dirs) {
            int x = i + dir[0];
            int y = j + dir[1];
            if (x < 0 || y < 0 || x >= m || y >= n || grid[x][y] == -1 || visited[x][y]) continue;
            res += dfs(grid, x, y, m, n, visited);
        }
        visited[i][j] = false;
        empty++;
        return res;
    }
}
```

compress

```java
class Solution {
    int empty;

    public int uniquePathsIII(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        int startI = 0, startJ = 0;
        empty = 1; // starting point
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) {
                    empty++;
                    continue;
                }
                if (grid[i][j] == 1) {
                    startI = i;
                    startJ = j;
                }
            }
        }
        return dfs(grid, startI, startJ, m, n, 0);
    }

    int dfs(int[][] grid, int i, int j, int m, int n, int visited) {
        if (grid[i][j] == 2) {
            return empty == 0 ? 1 : 0;
        }
        visited |= 1 << i * n + j;
        empty--;
        int[][] dirs = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
        int res = 0;
        for (int[] dir : dirs) {
            int x = i + dir[0];
            int y = j + dir[1];
            if (x < 0 || y < 0 || x >= m || y >= n || grid[x][y] == -1 || (visited & 1 << x * n + y) != 0) continue;
            res += dfs(grid, x, y, m, n, visited);
        }
        empty++;
        return res;
    }
}
```

#### 1020. Number of Enclaves

java

```java
class Solution {
    public int numEnclaves(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        boolean[][] visited = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if ((i == 0 || j == 0 || i == m - 1 || j == n - 1) && grid[i][j] == 1) {
                    dfs(i, j, m, n, visited, grid);
                }
            }
        }
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0 || visited[i][j]) continue;
                res++;
            }
        }
        return res;
    }

    void dfs(int i, int j, int m, int n, boolean[][] visited, int[][] grid) {
        visited[i][j] = true;
        int[][] directions = new int[][]{{0, 1}, {0, -1}, {-1, 0}, {1, 0}};
        for (int[] dir : directions) {
            int x = dir[0] + i;
            int y = dir[1] + j;
            if (x < 0 || y < 0 || x >= m || y >= n || grid[x][y] == 0 || visited[x][y]) continue;
            dfs(x, y, m, n, visited, grid);
            visited[x][y] = true;
        }
    }
}
```

#### 1254. Number of Closed Islands

java

```java
class Solution {
    public int closedIsland(int[][] grid) {
        int m = grid.length, n = grid[0].length;
        boolean[][] visited = new boolean[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1) continue;
                if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
                    dfs(i, j, m, n, visited, grid);
                }
            }
        }
        int res = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 1 || visited[i][j]) continue;
                res++;
                dfs(i, j, m, n, visited, grid);
            }
        }
        return res;
    }

    void dfs(int i, int j, int m, int n, boolean[][] visited, int[][] grid) {
        visited[i][j] = true;
        int[][] dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
        for (int[] dir : dirs) {
            int x = i + dir[0];
            int y = j + dir[1];
            if (x < 0 || y < 0 || x >= m || y >= n || grid[x][y] == 1 || visited[x][y]) continue;
            dfs(x, y, m, n, visited, grid);
        }
    }
}
```

#### 36. Valid Sudoku

```go
func isValidSudoku(board [][]byte) bool {
	var (
		row [9]int
		col [9]int
		box [9]int
	)
	for i := range board {
		for j := range board[0] {
			if board[i][j] == '.' {
				continue
			}
			idx := 1 << (board[i][j] - '0')
			if row[i]&idx > 0 ||
				col[j]&idx > 0 ||
				box[i/3*3+j/3]&idx > 0 {
				return false
			}
			row[i] |= idx
			col[j] |= idx
			box[(i/3)*3+j/3] |= idx
		}
	}
	return true
}
```

#### 37. Sudoku Solver

 / 3  X   3 可以将坐标映射到3 * 3九宫格的左上角

```go
func solveSudoku(board [][]byte) {
	dfs(board)
}
func dfs(board [][]byte) bool {
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			if board[i][j] != '.' {
				continue
			}
			for k := '1'; k <= '9'; k++ {
				if valid(i, j, byte(k), board) {
					board[i][j] = byte(k)
					if dfs(board) {
						return true
					}
					board[i][j] = '.'
				}
			}
			return false
		}
	}
	return true
}
func valid(row, col int, k byte, board [][]byte) bool {
	for i := 0; i < 9; i++ {
		if board[row][i] == k {
			return false
		}
	}
	for i := 0; i < 9; i++ {
		if board[i][col] == k {
			return false
		}
	}
	nr := row / 3 * 3
	nc := col / 3 * 3
	for i := nr; i < nr+3; i++ {
		for j := nc; j < nc+3; j++ {
			if board[i][j] == k {
				return false
			}
		}
	}
	return true
}
```

