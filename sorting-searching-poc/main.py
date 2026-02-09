import numpy as np

print("=" * 40)
print("     NumPy Sorting & Searching")
print("=" * 40, "\n")


# 1. np.sort (copy) vs ndarray.sort (in-place)
arr = np.array([42, 7, 18, 3, 25, 11, 33, 1])
print("Original:", arr)
print("np.sort (copy):", np.sort(arr))
print("Descending:", np.sort(arr)[::-1])

arr.sort()
print("After in-place .sort():", arr)
print()


# 2. np.argsort — indirect sorting
values = np.array([50, 10, 40, 20, 30])
indices = np.argsort(values)
print("Values:", values)
print("Argsort indices:", indices)
print("Sorted via indices:", values[indices])

# Ranking elements
ranks = np.empty_like(indices)
ranks[indices] = np.arange(len(values))
print("Ranks (0-based):", ranks)
print()


# 3. Sorting along axes
matrix = np.array([[9, 3, 7], [1, 8, 2], [5, 6, 4]])
print("Matrix:\n", matrix)
print("Sorted axis=0 (columns):\n", np.sort(matrix, axis=0))
print("Sorted axis=1 (rows):\n", np.sort(matrix, axis=1))
print()


# 4. Sort algorithms and stability
data = np.array([3, 1, 2, 1, 3, 2])
print("Data:", data)
print("Mergesort argsort (stable):", np.argsort(data, kind="mergesort"))
print("Quicksort argsort:         ", np.argsort(data, kind="quicksort"))
print()


# 5. np.searchsorted — binary search in sorted arrays
sorted_arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
print("Sorted array:", sorted_arr)
print("Insert index for 35:", np.searchsorted(sorted_arr, 35))

arr_dupes = np.array([10, 20, 20, 20, 30, 40])
print("With duplicates:", arr_dupes)
print("  searchsorted(20, 'left'):", np.searchsorted(arr_dupes, 20, side="left"))
print("  searchsorted(20, 'right'):", np.searchsorted(arr_dupes, 20, side="right"))
print()


# 6. np.where — conditional selection
arr = np.array([15, -3, 42, -8, 7, -1, 23, 0])
print("Array:", arr)
print("Indices where > 0:", np.where(arr > 0)[0])
print("Replace negatives with 0:", np.where(arr > 0, arr, 0))
print()


# 7. np.argmax, np.argmin, np.argwhere
grid = np.array([[10, 50, 30], [70, 20, 60], [40, 80, 15]])
print("Grid:\n", grid)
print("argmax axis=0 (per col):", np.argmax(grid, axis=0))
print("argmin axis=1 (per row):", np.argmin(grid, axis=1))

flat_idx = np.argmax(grid)
print(f"Global max: index={np.unravel_index(flat_idx, grid.shape)}, "
      f"value={grid.flat[flat_idx]}")
print("argwhere(grid > 50):\n", np.argwhere(grid > 50))
print()


# 8. np.nonzero and np.extract
arr = np.array([0, 5, 0, 12, 0, 3, 8, 0, 7])
print("Array:", arr)
print("Nonzero indices:", np.nonzero(arr)[0])
print("extract(> 5):", np.extract(arr > 5, arr))
print()


# 9. np.partition and np.argpartition
arr = np.array([38, 12, 75, 4, 91, 23, 56, 8, 67, 42])
partitioned = np.partition(arr, 3)
print("Original:", arr)
print("Partition k=3:", partitioned)
print(f"  arr[3]={partitioned[3]} is the 4th smallest")
print()


# 10. Practical: top-k with argpartition
np.random.seed(42)
scores = np.random.uniform(0, 100, size=1_000_000)
k = 5

top_idx = np.argpartition(scores, -k)[-k:]
top_scores = scores[top_idx]
top_scores_sorted = top_scores[np.argsort(top_scores)[::-1]]

print(f"Top {k} from 1M scores: {np.round(top_scores_sorted, 2)}")
print()

