import numpy as np

def main():
    print("=== NumPy Masked Arrays Demo ===\n")

    
    # 1. Create a masked array
    
    print("=== Creating masked arrays ===")

    data = np.array([1, 2, -999, 4, 5, -999, 7])
    masked = np.ma.masked_equal(data, -999)

    print("Original data:", data)
    print("Masked array:", masked)
    print("Mask:", masked.mask)
    print("Fill value:", masked.fill_value)
    print()

    
    # 2. Mask by condition
    
    print("=== Masking by condition ===")

    values = np.array([10, -1, 30, -5, 50, 0, 70])
    masked_neg = np.ma.masked_less(values, 0)

    print("Original:", values)
    print("Masked (negative values hidden):", masked_neg)
    print()

    
    # 3. Manual mask creation
    
    print("=== Manual mask ===")

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = [False, False, True, False, True]
    manual_masked = np.ma.array(data, mask=mask)

    print("Data:", data)
    print("Mask:", mask)
    print("Masked array:", manual_masked)
    print()

    
    # 4. Operations with masked arrays
    
    print("=== Operations ignore masked values ===")

    temperatures = np.ma.array(
        [22.5, 23.1, -999.0, 24.0, -999.0, 25.5, 21.0],
        mask=[False, False, True, False, True, False, False]
    )

    print("Temperatures:", temperatures)
    print("Mean (ignoring masked):", temperatures.mean())
    print("Max:", temperatures.max())
    print("Min:", temperatures.min())
    print("Sum:", temperatures.sum())
    print("Count (valid):", temperatures.count())
    print()

    
    # 5. Filling masked values
    
    print("=== Filling masked values ===")

    filled_zero = temperatures.filled(0)
    filled_mean = temperatures.filled(temperatures.mean())

    print("Filled with 0:", filled_zero)
    print("Filled with mean:", filled_mean)
    print()

    
    # 6. Masking invalid values (NaN, Inf)
    
    print("=== Masking invalid values ===")

    data_with_nan = np.array([1.0, np.nan, 3.0, np.inf, 5.0])
    masked_invalid = np.ma.masked_invalid(data_with_nan)

    print("Original:", data_with_nan)
    print("Masked invalid:", masked_invalid)
    print("Mean (ignoring invalid):", masked_invalid.mean())
    print()

    
    # 7. Combining masks
    
    print("=== Combining masks ===")

    data = np.array([10, 20, 30, 40, 50, 60, 70, 80])

    mask_low = np.ma.masked_less(data, 25)
    mask_high = np.ma.masked_greater(data, 65)

    # Combine: mask values outside [25, 65]
    combined = np.ma.masked_outside(data, 25, 65)

    print("Data:", data)
    print("Masked < 25:", mask_low)
    print("Masked > 65:", mask_high)
    print("Masked outside [25, 65]:", combined)
    print()

    
    # 8. 2D masked arrays
    # --------------------------------------------------
    print("=== 2D masked arrays ===")

    grid = np.array([
        [1, 2, -1],
        [4, -1, 6],
        [7, 8, 9]
    ])

    masked_grid = np.ma.masked_equal(grid, -1)

    print("Grid:\n", grid)
    print("Masked grid:\n", masked_grid)
    print("Column means:", masked_grid.mean(axis=0))
    print("Row means:", masked_grid.mean(axis=1))


if __name__ == "__main__":
    main()
