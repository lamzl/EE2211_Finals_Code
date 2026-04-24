"""
run_me.py - Example usage for all EE2211 functions

Paste your ee2211_functions.py in the same directory, then run:
    python run_me.py
"""

import numpy as np
from EE2211_exam_functions import (
    # Ch 3-5
    bayes, check_rank, is_invertible,
    # Ch 6
    mean_squared_error, add_bias, make_poly,
    least_squares_prime, binary_classif,
    multi_category_classif, ridge_reg,
    # Ch 7
    pearsons, bias_squared, variance,
    # Ch 8
    grad_descent, grad_descent_2,
    # Ch 9
    gini_impurity, entropy, misclassification_rate,
    overall_metric, split_data, mse_node, mse_depth_1,
)

# ===========================================================
# Chapter 3-5
# ===========================================================
print("=" * 50)
print("CHAPTER 3-5")
print("=" * 50)

# --- bayes() ---
# P(A|B) where A=0.3, B=0.5, P(B|A)=0.8
print("\n-- bayes() --")
bayes(A=0.3, B=0.5, B_given_A=0.8)

# --- check_rank() ---
print("\n-- check_rank() --")
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
check_rank(X)

# --- is_invertible() ---
print("\n-- is_invertible() --")
A = np.array([[1, 2],
              [3, 4]])
is_invertible(A)

singular = np.array([[1, 2],
                     [2, 4]])   # det = 0
is_invertible(singular)


# ===========================================================
# Chapter 6
# ===========================================================
print("\n" + "=" * 50)
print("CHAPTER 6")
print("=" * 50)

# Shared training data
X_train = np.array([[1, 0, 1],
                    [2, -1, 1],
                    [1, 1, 5]])
y_train = np.array([1, 2, 3])
X_test  = np.array([[-1, 2, 8],
                    [1, 5, -1]])

# --- mean_squared_error() ---
print("\n-- mean_squared_error() --")
mean_squared_error(X_train, y_train)

# --- add_bias() ---
print("\n-- add_bias() --")
X_biased = add_bias(X_train)
print("X with bias column:\n", X_biased)

# --- make_poly() ---
print("\n-- make_poly() --")
X_1d = np.array([[1], [2], [3], [4]])
X_poly = make_poly(X_1d, order=2)
print("Polynomial features (order 2):\n", X_poly)

# --- least_squares_prime() ---
print("\n-- least_squares_prime() WITH BIAS --")
least_squares_prime(X_biased, y_train, add_bias(X_test))
print("\n-- least_squares_prime() WITHOUT BIAS --")
least_squares_prime(X_train, y_train, X_test)

# --- binary_classif() ---
print("\n-- binary_classif() --")
y_binary = np.array([-1, -1, 1, 1])
binary_classif(X_biased, y_binary, add_bias(X_test))

# --- multi_category_classif() ---
print("\n-- multi_category_classif() --")
X_mc = np.array([[1, 0], [0, 1], [1, 1], [0, 0],
                 [2, 1], [1, 2]])
y_mc = np.array([[0], [1], [2], [0], [1], [2]])
X_mc_b = add_bias(X_mc)
X_test_mc = add_bias(np.array([[1, 0], [0, 1]]))
multi_category_classif(X_mc_b, y_mc, X_test_mc)

# --- ridge_reg() ---
print("\n-- ridge_reg() --")
ridge_reg(X_biased, y_train, reg_factor=0.1, Xt=add_bias(X_test))


# ===========================================================
# Chapter 7
# ===========================================================
print("\n" + "=" * 50)
print("CHAPTER 7")
print("=" * 50)

# --- pearsons() ---
print("\n-- pearsons() --")
x_vals = [1, 2, 3, 4, 5]
y_vals = [2, 4, 5, 4, 5]
r = pearsons(x_vals, y_vals)
print(f"Pearson's r: {r:.4f}")

# --- bias_squared() ---
print("\n-- bias_squared() --")
y_true = 5.0
predictions = np.array([4.5, 5.2, 4.8, 5.1])
bs = bias_squared(y_true, predictions)
print(f"Bias^2: {bs:.4f}")

# --- variance() ---
print("\n-- variance() --")
var = variance(predictions)
print(f"Variance: {var:.4f}")


# ===========================================================
# Chapter 8
# ===========================================================
print("\n" + "=" * 50)
print("CHAPTER 8")
print("=" * 50)

# --- grad_descent() ---
print("\n-- grad_descent() (f(x) = x^2) --")
f       = lambda x: x ** 2
grad_f  = lambda x: 2 * x
grad_descent(x=5.0, eta=0.1, iterations=10, function=f, grad=grad_f)

# --- grad_descent_2() ---
print("\n-- grad_descent_2() (f(x,y) = x^2 + y^2) --")
f2      = lambda x, y: x**2 + y**2
grad_x2 = lambda x: 2 * x
grad_y2 = lambda y: 2 * y
grad_descent_2(x=4.0, y=3.0, eta=0.1, iterations=10,
               function=f2, grad_x=grad_x2, grad_y=grad_y2)


# ===========================================================
# Chapter 9
# ===========================================================
print("\n" + "=" * 50)
print("CHAPTER 9")
print("=" * 50)

# --- gini_impurity() ---
print("\n-- gini_impurity() --")
gini_impurity([0.8, 0.2])

# --- entropy() ---
print("\n-- entropy() --")
entropy([0.5, 0.5])

# --- misclassification_rate() ---
print("\n-- misclassification_rate() --")
misclassification_rate([0.7, 0.3])

# --- overall_metric() ---
print("\n-- overall_metric() (using entropy) --")
child_nodes = [
    (40, [0.8, 0.2]),   # Node 1: 40 samples
    (60, [0.5, 0.5]),   # Node 2: 60 samples
]
overall_metric(child_nodes, metric_function=entropy)

# --- split_data() ---
print("\n-- split_data() --")
data = np.array([
    [0.2, 2.1], [0.7, 1.5], [1.8, 5.8], [2.2, 6.1], [3.7, 9.1],
    [4.1, 9.5], [4.5, 9.8], [5.1, 12.7], [6.3, 13.8], [7.4, 15.9]
])
x_left, y_left, x_right, y_right = split_data(data, threshold=3.0)
print(f"Left  x: {x_left},  y: {y_left}")
print(f"Right x: {x_right}, y: {y_right}")

# --- mse_node() ---
print("\n-- mse_node() --")
mse_node(y_left)

# --- mse_depth_1() ---
print("\n-- mse_depth_1() --")
mse_depth_1(y_left, y_right)