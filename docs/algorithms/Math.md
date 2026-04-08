Refer to Lesson 26 of the 'Zuo Shen' system learning class on matrices and the Fibonacci sequence.

#### 509. Fibonacci Number

The recurrence relation of the Fibonacci sequence is F(n) = F(n - 1) + F(n - 2).

F(0) = 0, F(1) = 1, F(2) = 1, F(3) = 2, f(4) = 3.

Calculate the n-th Fibonacci number F(n).

The recurrence relation can be represented by a matrix:

$$
\begin{bmatrix} 
 F(n)\\ 
 F(n-1)\\
 \end{bmatrix}
 =
 \begin{bmatrix} 
 a & b \\ 
 c & d \\
 \end{bmatrix}
 \times
 \begin{bmatrix} 
 F(n-1)\\ 
 F(n-2)\\
 \end{bmatrix}
$$

The recurrence relation is strictly invariant. Multiplying F(0) and F(1) by the matrix N times yields F(n-1) and F(n).

Multiplying n times, the time complexity is of order O(log(N)).

Substituting F(2) = 1, F(3) = 2, f(4) = 3, solve the equations:

$$
\begin{cases}
2a+b=3\\
2c+d=2\\
\end {cases}
$$
Substituting F(5) = 5, f(4) = 3, F(3) = 2, solve the equations:
$$
\begin{cases}
3a+2b=5\\
3c+2d=3\\
\end {cases}
$$
Therefore, the base matrix is:
$$
\begin{bmatrix}
1&1\\
1&0
\end{bmatrix}
$$


Go

```go
func fib(n int) int {
	if n == 0 {
		return 0
	}
	matrix := [][]int{
		{1, 1},
		{1, 0},
	}
	pow := matrixPow(matrix, n)
	return matrixMulti(pow, [][]int{{1}, {0}})[0][0]
}
func matrixPow(matrix [][]int, n int) [][]int {
	res := [][]int{
		{1, 0},
		{0, 1},
	}
	for n--; n > 0; n >>= 1 {
		if n&1 == 1 {
			res = matrixMulti(res, matrix)
		}
		matrix = matrixMulti(matrix, matrix)
	}
	return res
}
func matrixMulti(m1, m2 [][]int) [][]int {
	res := make([][]int, len(m2))
	for i := range res {
		res[i] = make([]int, len(m2[0]))
	}
	for i := range m1 {
		for j := range m2[0] {
			for k := range m1 {
				res[i][j] += m1[i][k] * m2[k][j]
			}
		}
	}
	return res
}
```

java

```java
class Solution {
    public int fib(int n) {
        if (n == 0) {
            return 0;
        }
        int[][] start = {{1}, {0}};
        int[][] matrix = {{1, 1}, {1, 0}};
        int[][] pow = pow(matrix, n);
        return matrixMulti(pow, start)[0][0];
    }

    int[][] pow(int[][] matrix, int n) {
        int[][] res = {{1, 0}, {0, 1}};
        for (n--; n > 0; n >>= 1) {
            if ((n & 1) == 1) {
                res = matrixMulti(res, matrix);
            }
            matrix = matrixMulti(matrix, matrix);
        }
        return res;
    }

    int[][] matrixMulti(int[][] m1, int[][] m2) {
        int m = m2.length, n = m2[0].length;
        int[][] res = new int[m][n];
        for (int i = 0; i < m1.length; i++) {
            for (int j = 0; j < m2[0].length; j++) {
                for (int k = 0; k < m1.length; k++) {
                    res[i][j] += m1[i][k] * m2[k][j];
                }
            }
        }
        return res;
    }
}
```

#### 935. Knight Dialer

Represent the undirected graph with an adjacency matrix.

https://alexgolec.medium.com/google-interview-questions-deconstructed-the-knights-dialer-impossibly-fast-edition-c288da1685b8

https://leetcode.com/problems/knight-dialer/solutions/189252/o-logn/

Go log(n) Solution

Since it is an undirected graph, the matrix is symmetric along the diagonal from top-left to bottom-right.

```go
func knightDialer(n int) int {
	graph := [10][]int{
		{4, 6},
		{6, 8},
		{7, 9},
		{4, 8},
		{3, 9, 0},
		{},
		{1, 7, 0},
		{2, 6},
		{1, 3},
		{4, 2},
	}
	matrix := [10][10]int{}
	for i := range matrix {
		for _, j := range graph[i] {
			matrix[i][j] = 1
		}
	}
	var res = [10][10]int{}
	for i := range res {
		res[i][i] = 1
	}
	mod := int(1e9 + 7)
	for n--; n > 0; n >>= 1 {
		if n&1 == 1 {
			res = matrixMulti(res, matrix, mod)
		}
		matrix = matrixMulti(matrix, matrix, mod)
	}
	sum := 0
	for i := range res {
		for j := range res[0] {
			sum = (sum + res[i][j]) % mod
		}
	}
	return sum
}
func matrixMulti(m1, m2 [10][10]int, mod int) [10][10]int {
	res := [10][10]int{}
	for i := 0; i < 10; i++ {
		for j := 0; j < 10; j++ {
			for k := 0; k < 10; k++ {
				res[i][j] = (res[i][j] + m1[i][k]*m2[k][j]) % mod
			}
		}
	}
	return res
}
```

python

```python
import numpy


class Solution:
    def knightDialer(self, N):
        if N == 1:
            return 10
        m = numpy.matrix([[0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                          [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                          [1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                          [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 1, 0, 0, 0, 0, 0]])
        res = 1
        mod = 10 ** 9 + 7
        N -= 1
        while N:
            if N % 2:
                res = res * m % mod
            m = m * m % mod
            N >>= 1
        return int(numpy.sum(res)) % mod
```
