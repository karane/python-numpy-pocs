import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.chebyshev import Chebyshev

SEP1 = 30 * "="
SEP2 = 30 * "-"

print("       NumPy Polynomials                 ")
print(SEP1)

# 1. Creating and evaluating polynomials
# Coefficients are in ASCENDING order: [1, -3, 2] means 1 - 3x + 2x^2

print("1. Creating and Evaluating Polynomials")
print(SEP2)

p = Polynomial([1, -3, 2])  # 1 - 3x + 2x^2
print("Polynomial p:", p)
print("Coefficients:", p.coef)
print("Degree:", p.degree())
print("p(0) =", p(0), " p(1) =", p(1), " p(2) =", p(2))
print()

# Create from roots
p_roots = Polynomial.fromroots([1, 2, 3])
print("From roots [1, 2, 3]:", p_roots)
print("Verify p(1) =", round(p_roots(1), 10), " p(2) =", round(p_roots(2), 10))
print()


# 2. Polynomial arithmetic

print("2. Polynomial Arithmetic")
print(SEP2)

a = Polynomial([1, 2])       # 1 + 2x
b = Polynomial([3, 0, -1])   # 3 - x^2

print("a =", a)
print("b =", b)
print("a + b =", a + b)
print("a * b =", a * b)
print("a ** 2 =", a ** 2)
print()

# Division with remainder
c = Polynomial([2, 3, 1])  # (2 + x)(1 + x)
d = Polynomial([1, 1])     # 1 + x
quot, rem = divmod(c, d)
print(f"({c}) / ({d})")
print("Quotient:", quot, " Remainder:", rem)
print()


# 3. Derivative and integral

print("3. Derivative and Integral")
print(SEP2)

p3 = Polynomial([2, 0, -4, 1])  # 2 - 4x^2 + x^3
print("p3:", p3)
print("p3' (deriv):", p3.deriv())
print("p3'' (2nd deriv):", p3.deriv(2))
print("Integral:", p3.integ())
print()


# 4. Finding roots

print("4. Finding Roots")
print(SEP2)

# Real roots
q = Polynomial([-6, 11, -6, 1])  # x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
print("Polynomial:", q)
print("Roots:", np.sort(np.real(q.roots())))
print()

# Complex roots: x^2 + 1 = 0
q_complex = Polynomial([1, 0, 1])
print("x^2 + 1 roots:", q_complex.roots())
print()

# Round-trip: roots -> polynomial -> roots
q_high = Polynomial.fromroots([-2, -1, 0, 1, 2])
print("From roots [-2,-1,0,1,2], recovered:", np.sort(np.real(q_high.roots())))
print()


# 5. Polynomial fitting

print("5. Polynomial Fitting")
print(SEP2)

np.random.seed(42)
x_data = np.linspace(-3, 3, 30)
y_data = 0.5 * x_data**2 - 2 * x_data + 1 + np.random.normal(0, 0.5, len(x_data))

fit = Polynomial.fit(x_data, y_data, 2)
print("Fit (deg 2):", fit.convert())
print("Expected:    ~1.0 - 2.0x + 0.5x^2")
print()

# Compare degrees
for deg in [1, 2, 3, 5]:
    f = Polynomial.fit(x_data, y_data, deg)
    rss = np.sum((y_data - f(x_data))**2)
    print(f"  Degree {deg}: RSS = {rss:.4f}")
print()


# 6. Chebyshev polynomials

print("6. Chebyshev Polynomials")
print(SEP2)

# T0=1, T1=x, T2=2x^2-1
T2 = Chebyshev([0, 0, 1])
x_cheb = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
print("T2(x):", T2(x_cheb))
print("Manual 2x^2-1:", 2 * x_cheb**2 - 1)
print("Match:", np.allclose(T2(x_cheb), 2 * x_cheb**2 - 1))
print()

# 6. Overfitting demonstration

print("6. Overfitting Demonstration")
print(SEP2)

np.random.seed(123)
x_train = np.linspace(-3, 3, 40)
y_exact = np.sin(x_train) + 0.5 * x_train
y_train = y_exact + np.random.normal(0, 0.3, len(x_train))

x_test = np.linspace(-3, 3, 100)
y_test = np.sin(x_test) + 0.5 * x_test

print(f"{'Degree':>6}  {'Train RSS':>10}  {'Test RSS':>10}")
print("-" * 32)

for deg in [1, 3, 5, 10, 15]:
    f = Polynomial.fit(x_train, y_train, deg)
    train_rss = np.sum((y_train - f(x_train))**2)
    test_rss = np.sum((y_test - f(x_test))**2)
    print(f"{deg:>6}  {train_rss:>10.4f}  {test_rss:>10.4f}")

print()
print("Higher degree fits training better but generalizes worse = overfitting")
print()

