import numpy as np

def main():
    print("=== NumPy Linear Algebra Demo ===\n")

    
    # 1. Create vectors and matrices
    
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])

    A = np.array([
        [1, 2],
        [3, 4]
    ])

    B = np.array([
        [5, 6],
        [7, 8]
    ])

    print("Vector v1:", v1)
    print("Vector v2:", v2)
    print("\nMatrix A:\n", A)
    print("Matrix B:\n", B)
    print()

    
    # 2. Dot products
    
    print("=== Dot products ===")

    dot_vectors = np.dot(v1, v2)
    print("v1 · v2 =", dot_vectors)

    dot_matrix_vector = np.dot(A, v1[:2])
    print("A · [1, 2] =", dot_matrix_vector)
    print()

    
    # 3. Matrix multiplication
    
    print("=== Matrix multiplication ===")

    matmul_1 = np.matmul(A, B)
    matmul_2 = A @ B  # preferred modern syntax

    print("A @ B:\n", matmul_1)
    print("Same result using @ operator:\n", matmul_2)
    print()

    
    # 4. Determinant
    
    print("=== Determinant ===")

    det_A = np.linalg.det(A)
    print("det(A) =", det_A)
    print()

    
    # 5. Solve linear systems
    
    print("=== Solve linear system ===")
    print("Solving: A · x = b")

    b = np.array([5, 11])

    x = np.linalg.solve(A, b)
    print("b:", b)
    print("Solution x:", x)

    # Verify solution
    print("A @ x =", A @ x)
    print()

    
    # 6. Eigenvalues and eigenvectors
    
    print("=== Eigenvalues & Eigenvectors ===")

    eigenvalues, eigenvectors = np.linalg.eig(A)

    print("Eigenvalues:", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)

    # Verify: A · v = λ · v
    i = 0
    v = eigenvectors[:, i]
    lam = eigenvalues[i]

    print("\nVerification for first eigenpair:")
    print("A @ v =", A @ v)
    print("λ * v =", lam * v)
    print()


if __name__ == "__main__":
    main()
