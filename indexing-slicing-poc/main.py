import numpy as np

print("====================================")
print(" NumPy Array Indexing & Slicing Demo ")
print("====================================\n")

# 1. Indexing 1D arrays
arr_1d = np.array([10, 20, 30, 40, 50])

print("1D array:", arr_1d)
print("First element:", arr_1d[0])
print("Last element:", arr_1d[-1])
print("Third element:", arr_1d[2])
print()

# 2. Indexing 2D arrays
arr_2d = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("2D array:\n", arr_2d)
print("Element at row 0, col 1:", arr_2d[0, 1])
print("Second row:", arr_2d[1])
print("Third column:", arr_2d[:, 2])
print()

# 3. Indexing n-dimensional arrays
arr_3d = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

print("3D array:\n", arr_3d)
print("Element [1, 0, 1]:", arr_3d[1, 0, 1])
print("First block:\n", arr_3d[0])
print("All elements at [:, 1, :]:\n", arr_3d[:, 1, :])
print()

# 4. Slicing arrays
arr = np.array([0, 1, 2, 3, 4, 5, 6])

print("Original array:", arr)
print("arr[2:5]:", arr[2:5])
print("arr[:4]:", arr[:4])
print("arr[::2]:", arr[::2])
print("arr[::-1]:", arr[::-1])
print()

# 5. Views vs Copies
original = np.array([10, 20, 30, 40])

view = original[1:3]
copy = original[1:3].copy()

view[0] = 999
copy[1] = 777

print("Original after modifying view:", original)
print("View:", view)
print("Copy:", copy)
print()

# 6. Boolean indexing
data = np.array([5, 10, 15, 20, 25, 30])

mask = data > 15

print("Data:", data)
print("Mask:", mask)
print("Filtered data:", data[mask])
print("Divisible by 10:", data[data % 10 == 0])
print()

# 7. Fancy indexing
arr = np.array([100, 200, 300, 400, 500])
indices = [0, 2, 4]

print("Array:", arr)
print("Fancy indexing result:", arr[indices])

matrix = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

print("Matrix:\n", matrix)
print("matrix[[0, 2], [1, 2]]:", matrix[[0, 2], [1, 2]])
print()

# 8. Modifying arrays via indexing
arr = np.array([1, 2, 3, 4, 5])

print("Original:", arr)

arr[arr > 3] = 99
print("After boolean assignment:", arr)

arr[[0, 2]] = -1
print("After fancy assignment:", arr)

arr[1:4] = 0
print("After slice assignment:", arr)
