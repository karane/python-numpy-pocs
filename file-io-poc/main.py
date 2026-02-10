import numpy as np
import tempfile
import os
import shutil

print("=" * 40)
print("        NumPy File I/O")
print("=" * 40, "\n")

tmpdir = tempfile.mkdtemp()

# 1. np.save / np.load (.npy - single array, binary)

print("--- 1. np.save / np.load (.npy) ---\n")

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
npy_path = os.path.join(tmpdir, "array.npy")
np.save(npy_path, arr)

loaded = np.load(npy_path)
print("Original:", arr.shape, arr.dtype)
print("Match:", np.array_equal(arr, loaded), "| Dtype preserved:", loaded.dtype)
print()

# 2. np.savez / np.savez_compressed (.npz - multiple arrays)

print("--- 2. np.savez / np.savez_compressed (.npz) ---\n")

x = np.random.rand(100, 50)
y = np.random.randint(0, 10, size=100)

npz_path = os.path.join(tmpdir, "arrays.npz")
npzc_path = os.path.join(tmpdir, "arrays_compressed.npz")
np.savez(npz_path, features=x, labels=y)
np.savez_compressed(npzc_path, features=x, labels=y)

data = np.load(npz_path)
print("Keys:", list(data.keys()))
print("features shape:", data["features"].shape, "| labels shape:", data["labels"].shape)
data.close()

ratio = os.path.getsize(npzc_path) / os.path.getsize(npz_path) * 100
print(f"Compressed is {ratio:.1f}% of uncompressed size")
print()

# 3. np.savetxt / np.loadtxt (text/CSV)

print("--- 3. np.savetxt / np.loadtxt (text/CSV) ---\n")

table = np.array([[1.0, 2.5, 3.7], [4.1, 5.0, 6.3], [7.8, 8.2, 9.9]])

csv_path = os.path.join(tmpdir, "data.csv")
np.savetxt(csv_path, table, fmt="%.2f", delimiter=",", header="a,b,c", comments="")

loaded_csv = np.loadtxt(csv_path, delimiter=",", skiprows=1)
print("Match:", np.allclose(table, loaded_csv))

with open(csv_path) as f:
    print(f.read())

# 4. np.genfromtxt (missing values)

print("--- 4. np.genfromtxt (missing values) ---\n")

messy_path = os.path.join(tmpdir, "messy.csv")
with open(messy_path, "w") as f:
    f.write("a,b,c\n85,90,78\n,88,92\n91,,85\n76,82,\n")

result = np.genfromtxt(messy_path, delimiter=",", skip_header=1)
print("With NaN defaults:\n", result)

col_means = np.nanmean(result, axis=0)
filled = result.copy()
for col in range(filled.shape[1]):
    mask = np.isnan(filled[:, col])
    filled[mask, col] = col_means[col]
print("Filled with column means:\n", filled)
print()

# 5. Memory-mapped files (np.memmap)

print("--- 5. np.memmap ---\n")

mmap_path = os.path.join(tmpdir, "mapped.dat")
shape = (1000, 200)

fp = np.memmap(mmap_path, dtype="float64", mode="w+", shape=shape)
fp[:] = np.random.rand(*shape)
fp[0, :3] = [1.1, 2.2, 3.3]
fp.flush()
del fp

fp = np.memmap(mmap_path, dtype="float64", mode="r", shape=shape)
print(f"Shape: {shape} | File size: {os.path.getsize(mmap_path):,} bytes")
print("First 3 elements:", fp[0, :3])
del fp
print()

# 6. tofile / np.fromfile (raw binary)

print("--- 6. tofile / np.fromfile (raw binary) ---\n")

original = np.arange(12, dtype=np.int16).reshape(3, 4)
bin_path = os.path.join(tmpdir, "raw.bin")
original.tofile(bin_path)

# Must know dtype and shape to reload
restored = np.fromfile(bin_path, dtype=np.int16).reshape(3, 4)
print("Match:", np.array_equal(original, restored))
print("Wrong dtype:", np.fromfile(bin_path, dtype=np.float64).shape, "(expected 12 elements)")
print()

# Cleanup
shutil.rmtree(tmpdir)
print("Cleaned up temp directory.")
