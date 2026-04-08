## Two-way Quick Sort (Two-way Partitioning)

**Core Idea: Uniform Distribution**
Two-way quick sort distributes the partitioning elements evenly between the two intervals `<= pivot` and `>= pivot`. This not only handles duplicate elements but also prevents degradation to $O(N^2)$ in cases containing many duplicate elements (e.g., `[2, 2, ..., 2]`).

**Implementation Details: Move-to-Head Strategy**
A strategy of swapping the `pivot` to the head of the interval `l` is uniformly adopted:
1. **Random Selection**: Randomly select an index within the current interval `[l, r)` as the `pivot`.
2. **Swap to Head**: Swap the `pivot` to index `l` and temporarily store `pivotVal`.
3. **Two-way Scan**: The left pointer `i` starts from `l+1` and the right pointer `j` starts from `r-1`. They move towards the center and swap elements that violate the partitioning rules.
4. **Positioning**: After the scan, swap the `pivot` at head index `l` with the element at `j`. Now `j` is the final position of the `pivot`.
5. **Recursion**: The next recursive intervals are `[l, j)` and `[j+1, r)`, **completely skipping the already positioned index `j`**.

Elements equal to the `pivot` will be evenly distributed on both sides of the array.

Swap the `pivot value` to the head/tail and exclude the `pivot` element from recursive calls. This effectively reduces the problem size for recursion and avoids infinite loops.

**Optimization**
Use Insertion Sort for small arrays, as it has a smaller constant factor and is cache-friendly.

---

#### 215. Kth Largest Element in an Array

Recursive binary search with a time complexity of $O(N)$. Note: The partition function using the Move-to-Head strategy returns an index `j` that guarantees **`nums[j]` is in its final sorted position**. Therefore, recursion can be pruned or narrowed based on the relationship between `j` and `k`.

Go

```go
import "math/rand"

func findKthLargest(nums []int, k int) int {
    n := len(nums)
    // The k-th largest is the (n - k)-th smallest (0-based index)
    return quickSelect(nums, 0, n, n-k)
}

func quickSelect(nums []int, l, r, k int) int {
    if r-l <= 1 {
        return nums[l]
    }
    
    // partition returns the final position j of the pivot
    j := partition(nums, l, r)
    
    if k == j {
        return nums[j]
    }
    if k < j {
        return quickSelect(nums, l, j, k)
    }
    return quickSelect(nums, j+1, r, k)
}

func partition(nums []int, l, r int) int {
    // Randomly select pivot and swap to the head
    pivotIdx := l + rand.Intn(r-l)
    pivotVal := nums[pivotIdx]
    // Move to Head
    nums[l], nums[pivotIdx] = nums[pivotIdx], nums[l]
    
    i, j := l+1, r-1
    for {
        // Find the first element >= pivotVal to the right
        for i <= j && nums[i] < pivotVal { i++ }
        // Find the first element <= pivotVal to the left
        for i <= j && nums[j] > pivotVal { j-- }
        if i >= j { break }
        nums[i], nums[j] = nums[j], nums[i]
        i++
        j--
    }
    // Place pivot in its final position j
    nums[l], nums[j] = nums[j], nums[l]
    return j
}
```

Python

```python
import random
from typing import List

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # The k-th largest is the (n - k)-th smallest (0-based index)
        target = n - k
        return self.quick_select(nums, 0, n, target)

    def quick_select(self, nums: List[int], l: int, r: int, target: int) -> int:
        if r - l <= 1:
            return nums[l]
        
        # partition returns the final position j of the pivot
        j = self.partition(nums, l, r)
        
        if target == j:
            return nums[j]
        if target < j:
            return self.quick_select(nums, l, j, target)
        return self.quick_select(nums, j + 1, r, target)

    def partition(self, nums: List[int], l: int, r: int) -> int:
        # Randomly select pivot and swap to the head
        pivot_idx = random.randint(l, r - 1)
        pivot_val = nums[pivot_idx]
        # Move to Head
        nums[l], nums[pivot_idx] = nums[pivot_idx], nums[l]
        
        i, j = l + 1, r - 1
        while True:
            # Find the first element >= pivotVal to the right
            while i <= j and nums[i] < pivot_val:
                i += 1
            # Find the first element <= pivotVal to the left
            while i <= j and nums[j] > pivot_val:
                j -= 1
            if i >= j:
                break
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1
            
        # Place pivot in its final position j
        nums[l], nums[j] = nums[j], nums[l]
        return j
```

Rust

```rust
impl Solution {
    pub fn find_kth_largest(mut nums: Vec<i32>, k: i32) -> i32 {
        let n = nums.len();
        // The k-th largest is the (n - k)-th smallest (0-based index)
        let target = n - k as usize;
        Self::quick_select(&mut nums, 0, n, target)
    }

    fn quick_select(arr: &mut [i32], l: usize, r: usize, k: usize) -> i32 {
        if r - l <= 1 {
            return arr[l];
        }
        let j = Self::partition(arr, l, r);
        
        if k == j {
            return arr[j];
        } 
        if k < j {
            return Self::quick_select(arr, l, j, k);
        } 
        Self::quick_select(arr, j + 1, r, k)
    }

    fn partition(arr: &mut [i32], l: usize, r: usize) -> usize {
        // Randomly select pivot and swap to the head
        let pivot_index = rand::random_range(l..r); 
        let pivot_value = arr[pivot_index];
        // Move to Head
        arr.swap(l, pivot_index);
        
        let mut i = l + 1;
        let mut j = r - 1;
        
        loop {
            // Find the first element >= pivotVal to the right
            while i <= j && arr[i] < pivot_value { i += 1; }
            // Find the first element <= pivotVal to the left
            while i <= j && arr[j] > pivot_value { j -= 1; }
            if i >= j { break; }
            arr.swap(i, j);
            i += 1;
            j -= 1;
        }
        // Place pivot in its final position j
        arr.swap(l, j);
        j
    }
}
```

Java

```java
import java.util.concurrent.ThreadLocalRandom;

public class Solution {
    public int findKthLargest(int[] nums, int k) {
        int n = nums.length;
        // The k-th largest is the element at index n-k (after sorting)
        return quickSelect(nums, 0, n, n - k);
    }

    private int quickSelect(int[] nums, int l, int r, int k) {
        if (r - l <= 1) return nums[l];
        
        int j = partition(nums, l, r);
        
        if (k == j) {
            return nums[j];
        } 
        if (k < j) {
            return quickSelect(nums, l, j, k);
        } 
        return quickSelect(nums, j + 1, r, k);
    }    private int partition(int[] nums, int l, int r) {
        // Randomly select pivot and swap to the head
        int pivotIdx = l + ThreadLocalRandom.current().nextInt(r - l);
        int pivotVal = nums[pivotIdx];
        // Move to Head
        swap(nums, l, pivotIdx);
        
        int i = l + 1, j = r - 1;
        while (true) {
            // Find the first element >= pivotVal to the right
            while (i <= j && nums[i] < pivotVal) i++;
            // Find the first element <= pivotVal to the left
            while (i <= j && nums[j] > pivotVal) j--;
            if (i >= j) {
                break;
            }
            swap(nums, i, j);
            i++;
            j--;
        }
        // Place pivot in its final position j
        swap(nums, l, j);
        return j;
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

#### 912. Sort an Array

Go

```go
import (
	"math/rand"
)

func sortArray(nums []int) []int {
	n := len(nums)
	if n < 2 {
		return nums
	}
	quickSort(nums, 0, n)
	return nums
}

const threshold = 47

func quickSort(nums []int, l, r int) {
	if r-l < threshold {
		insertionSort(nums, l, r)
		return
	}
	// j is the final position of the pivot
	j := partition(nums, l, r)
	quickSort(nums, l, j)
	quickSort(nums, j+1, r)
}

func insertionSort(nums []int, l, r int) {
	for i := l + 1; i < r; i++ {
		key := nums[i]
		j := i
		for j > l && nums[j-1] > key {
			nums[j] = nums[j-1]
			j--
		}
		nums[j] = key
	}
}

func partition(nums []int, l, r int) int {
	// Randomly select pivot and swap to the head
	pivotIdx := l + rand.Intn(r-l)
	pivotVal := nums[pivotIdx]
	// Move to Head
	nums[l], nums[pivotIdx] = nums[pivotIdx], nums[l]

	i, j := l+1, r-1
	for {
		for i <= j && nums[i] < pivotVal {
			i++
		}
		for i <= j && nums[j] > pivotVal {
			j--
		}
		if i >= j {
			break
		}
		nums[i], nums[j] = nums[j], nums[i]
		i++
		j--
	}
	// Place pivot in its final position j
	nums[l], nums[j] = nums[j], nums[l]
	return j
}
```

Python

```python
import random
from typing import List

class Solution:
    THRESHOLD = 47

    def sortArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        if n < 2:
            return nums
        self.quick_sort(nums, 0, n)
        return nums

    def quick_sort(self, nums: List[int], l: int, r: int):
        if r - l < self.THRESHOLD:
            self.insertion_sort(nums, l, r)
            return
        
        # j is the final position of the pivot
        j = self.partition(nums, l, r)
        self.quick_sort(nums, l, j)
        self.quick_sort(nums, j + 1, r)

    def insertion_sort(self, nums: List[int], l: int, r: int):
        for i in range(l + 1, r):
            key = nums[i]
            j = i
            while j > l and nums[j - 1] > key:
                nums[j] = nums[j - 1]
                j -= 1
            nums[j] = key

    def partition(self, nums: List[int], l: int, r: int) -> int:
        # Randomly select pivot and swap to the head
        pivot_idx = random.randint(l, r - 1)
        pivot_val = nums[pivot_idx]
        # Move to Head
        nums[l], nums[pivot_idx] = nums[pivot_idx], nums[l]
        
        i, j = l + 1, r - 1
        while True:
            while i <= j and nums[i] < pivot_val:
                i += 1
            while i <= j and nums[j] > pivot_val:
                j -= 1
            if i >= j:
                break
            
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1
        
        # Place pivot in its final position j
        nums[l], nums[j] = nums[j], nums[l]
        return j
```

Rust

```rust
impl Solution {
    const THRESHOLD: usize = 47;

    pub fn sort_array(mut nums: Vec<i32>) -> Vec<i32> {
        let n = nums.len();
        if n < 2 {
            return nums;
        }
        Self::quick_sort_recursion(&mut nums);
        nums
    }

    fn quick_sort_recursion(arr: &mut [i32]) {
        let len = arr.len();
        if len < Self::THRESHOLD {
            Self::insertion_sort(arr);
            return;
        }
        let j = Self::partition(arr);
        // j is the final position of the pivot (relative index)
        // arr[0..j] is the left side, arr[j] is the pivot, arr[j+1..] is the right side
        let (left, right) = arr.split_at_mut(j);
        Self::quick_sort_recursion(left);
        Self::quick_sort_recursion(&mut right[1..]);
    }

    fn insertion_sort(arr: &mut [i32]) {
        for i in 1..arr.len() {
            let key = arr[i];
            let mut j = i;
            while j > 0 && arr[j - 1] > key {
                arr[j] = arr[j - 1];
                j -= 1;
            }
            arr[j] = key;
        }
    }

    fn partition(arr: &mut [i32]) -> usize {
        let len = arr.len();
        // Randomly select pivot and swap to the head
        let pivot_index = rand::random_range(0..len); 
        let pivot_value = arr[pivot_index];
        // Move to Head
        arr.swap(0, pivot_index);
        
        let mut i = 1;
        let mut j = len - 1;
        
        loop {
            while i <= j && arr[i] < pivot_value {
                i += 1;
            }
            while i <= j && arr[j] > pivot_value {
                j -= 1;
            }
            if i >= j {
                break;
            }
            arr.swap(i, j); 
            i += 1;
            j -= 1;
        }
        // Place pivot in its final position j
        arr.swap(0, j);
        j
    }
}
```

Java

```java
import java.util.concurrent.ThreadLocalRandom;

public class Solution {
    private static final int THRESHOLD = 47;

    public int[] sortArray(int[] nums) {
        if (nums == null) return null;
        int n = nums.length;
        if (n < 2) return nums;
        quickSort(nums, 0, n);
        return nums;
    }

    private void insertionSort(int[] nums, int l, int r) {
        for (int i = l + 1; i < r; i++) {
            int key = nums[i], j = i;
            while (j > l && nums[j - 1] > key) {
                nums[j] = nums[j - 1];
                j--;
            }
            nums[j] = key;
        }
    }

    private void quickSort(int[] nums, int l, int r) {
        if (r - l < THRESHOLD) {
            insertionSort(nums, l, r);
            return;
        }
        int j = partition(nums, l, r);
        quickSort(nums, l, j);
        quickSort(nums, j + 1, r);
    }

    private int partition(int[] nums, int l, int r) {
        // Randomly select pivot and swap to the head
        int pivotIdx = l + ThreadLocalRandom.current().nextInt(r - l);
        int pivotVal = nums[pivotIdx];
        // Move to Head
        swap(nums, l, pivotIdx);
        
        int i = l + 1, j = r - 1;
        while (true) {
            while (i <= j && nums[i] < pivotVal) i++;
            while (i <= j && nums[j] > pivotVal) j--;
            if (i >= j) {
                break;
            }
            swap(nums, i, j);
            i++;
            j--;
        }
        // Place pivot in its final position j
        swap(nums, l, j);
        return j;
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```
