import numpy as np
import time

print("========================================")
print(" NumPy Vector Operations & Broadcasting ")
print("========================================\n")


# 1. Element-wise operations

a = np.array([1, 2, 3])
b = np.array([10, 20, 30])

print("a:", a)
print("b:", b)

print("a + b:", a + b)
print("a - b:", a - b)
print("a * b:", a * b)
print("a / b:", a / b)
print()


# 2. Broadcasting rules

# Rule summary:
# 1. Compare shapes from right to left
# 2. Dimensions must be equal OR one of them must be 1

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

vector = np.array([10, 20, 30])

print("Matrix:\n", matrix)
print("Vector:", vector)
print("Matrix + Vector:\n", matrix + vector)
print()

# Broadcasting with column vector
column = np.array([[1], [2]])

print("Column vector:\n", column)
print("Matrix + Column:\n", matrix + column)
print()


# 3. Real-world broadcasting examples

# Normalize data (subtract mean, divide by std)

data = np.array([
    [50, 60, 70],
    [55, 65, 75],
    [60, 70, 80]
])

mean = data.mean(axis=0)
std = data.std(axis=0)

normalized = (data - mean) / std

print("Data:\n", data)
print("Mean:", mean)
print("Std:", std)
print("Normalized data:\n", normalized)
print()

# Apply weights to features
weights = np.array([0.2, 0.3, 0.5])
weighted = data * weights

print("Weights:", weights)
print("Weighted data:\n", weighted)
print()


# 4. Loops vs vectorized code

size = 1_000_000
x = np.random.rand(size)
y = np.random.rand(size)

# Loop version
start = time.time()
result_loop = np.empty(size)
for i in range(size):
    result_loop[i] = x[i] + y[i]
loop_time = time.time() - start

# Vectorized version
start = time.time()
result_vectorized = x + y
vector_time = time.time() - start

print("Loop time:", loop_time)
print("Vectorized time:", vector_time)
print("Results equal:", np.allclose(result_loop, result_vectorized))
print()


# 5. Avoid broadcasting mistakes

# Common mistake: wrong axis alignment

data = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

wrong_vector = np.array([10, 20])  # shape (2,)

print("Data shape:", data.shape)
print("Wrong vector shape:", wrong_vector.shape)

try:
    print(data + wrong_vector)
except ValueError as e:
    print("Broadcasting error:", e)

# Fix by reshaping
correct_vector = wrong_vector.reshape(-1, 1)
print("\nCorrect vector shape:", correct_vector.shape)
print("Fixed broadcasting:\n", data + correct_vector)

