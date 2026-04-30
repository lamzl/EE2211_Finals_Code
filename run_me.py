"""
run_me.py - Example usage for all EE2211 functions

Paste your ee2211_functions.py in the same directory, then run:
    python run_me.py
"""

import numpy as np
import math as math

from EE2211_exam_functions import (
    bayes, check_rank, is_invertible,
    mean_squared_error, add_bias, make_poly,
    least_squares_prime, binary_classif,
    multi_category_classif, ridge_reg,
    pearsons, bias_squared, variance,
    grad_descent, grad_descent_2,
    gini_impurity, entropy, misclassification_rate,
    overall_metric, split_data, mse_depth_1, mse_node
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
bayes(A=0.3, B=0.78, B_given_A=0.6)

# --- check_rank() ---
print("\n-- check_rank() --")
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
check_rank(X)

# --- is_invertible() ---
print("\n-- is_invertible() --")
A = np.array([[1, 1],
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


# X_train_raw = np.array([[1, 3, -2], [-4, 0, -1], [3, 1, 8], [2, 1, 6], [8, 4, 6]])
X_train_raw = np.array([[4], [7], [10], [2], [3], [9]])
y_train = np.array([-1, -1, -1, 1, 1, 1])
X_test_raw = np.array([[4], [7], [10], [2], [3], [9]])
# X_test_raw = np.array([[4], [7], [10], [2], [3], [9]])

# y_test = np.array([1, 1, 2, 3, 3])

# --- add_bias() ---
print("\n-- add_bias() --")
X_biased = add_bias(X_train_raw)
print("X with bias column:\n", X_biased)


# --- make_poly() ---
print("\n-- make_poly() --")
X_poly_train = make_poly(X_train_raw, order=4)
X_poly_test  = make_poly(X_test_raw, order=4)
print("Polynomial features (order 4):\n", X_poly_train)


# --- mean_squared_error() ---
print("\n-- mean_squared_error() --")
mean_squared_error(add_bias(X_train_raw), y_train)

# --- least_squares_prime() WITH BIAS ---
print("\n-- least_squares_prime() WITH BIAS --")
least_squares_prime(X_biased, y_train, add_bias(X_test_raw))

# --- least_squares_prime() WITHOUT BIAS ---
print("\n-- least_squares_prime() WITHOUT BIAS --")
least_squares_prime(X_train_raw, y_train, X_test_raw)

# --- least_squares_prime() POLYNOMIAL  ---
print("\n-- least_squares_prime() POLYNOMIAL--")
least_squares_prime(X_poly_train, y_train, X_poly_test)

# --- ridge_reg() ---
print("\n-- ridge_reg() --")
ridge_reg(X_biased, y_train, reg_factor=0.1, Xt=add_bias(X_test_raw))

# --- binary_classif() ---
print("\n-- binary_classif() --")
y_binary = np.array([-1, -1, -1, 1, 1, 1])
# binary_classif(X_biased, y_binary, add_bias(X_test_raw))
binary_classif(X_poly_train, y_binary, X_poly_test) # for polynomial features

# --- multi_category_classif() ---
print("\n-- multi_category_classif() --")
X_mc = np.array([[1, 3, -2], [-4, 0, -1], [3, 1, 8], [2, 1, 6], [8, 4, 6]])
y_mc = np.array([[1], [1], [2], [3], [3]])
X_mc_b = add_bias(X_mc)
X_test_mc = add_bias(np.array([[1, -2, 4]]))
multi_category_classif(X_mc_b, y_mc, X_test_mc)



# ===========================================================
# Chapter 7
# ===========================================================
print("\n" + "=" * 50)
print("CHAPTER 7")
print("=" * 50)

# --- pearsons() ---
print("\n-- pearsons() --")
x_vals = [[3.3459, 2.7435, -1.7253], [1.0893, 2.9113, -0.7804], [3.2103,1.4706,-0.9944], [1.744,1.2895,0.5307], [1.6762,2.1366,-1.0502]]
y_vals = [[2.9972], [1.1399], [2.228], [0.3387], [2.5042]]
r = pearsons(x_vals, y_vals)
for i, val in enumerate(r):
    print(f"Pearson’s correlation of Feature {i+1} and Target y = {val:.6f}")

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
print("\n-- grad_descent()--")
x_val = 3.0
f       = lambda x: math.sin(x)**2
grad_f  = lambda x: 2 * math.sin(x) * math.cos(x)
gradient = grad_f(x_val)
print(f"Calculated gradient at x={x_val:.4f} is: {gradient:.4f}")
grad_descent(x=x_val, eta=0.1, iterations=1, function=f, grad=grad_f)

# --- grad_descent_2() ---
print("\n-- grad_descent_2() --")
f2      = lambda x, y: x**2 + x*y**2
grad_x2 = lambda x, y: 2 * x + y**2
grad_y2 = lambda x, y: 2 * x * y
grad_descent_2(x=3.0, y=2.0, eta=0.2, iterations=1,
               function=f2, grad_x=grad_x2, grad_y=grad_y2)


# ===========================================================
# Chapter 9
# ===========================================================
print("\n" + "=" * 50)
print("CHAPTER 9")
print("=" * 50)

# --- gini_impurity() ---
print("\n-- gini_impurity() --")
gini_impurity([7/15, 8/15])

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
y_root = data[:, 1] 
print("\n-- MSE at Root --")
mse_node(y_root)

# --- mse_depth_1() ---
print("\n-- mse_depth_1() --")
mse_depth_1(y_left, y_right)



# # --- mse_node() ---
# print("\n-- mse_node() --")
# mse_tree([y_train])

# # --- mse_depth_1() ---
# print("\n-- mse_depth_1() --")
# x_left, y_left, x_right, y_right = split_data(data, threshold=3.0)
# mse_tree([y_left, y_right])

# # --- mse_depth_2() ---
# print("\n-- mse_depth_2() --")
# # 1. First split (Depth 1) -> Creates 2 nodes
# x_L, y_L, x_R, y_R = split_data(data, threshold=3.0)

# # 2. Second splits (Depth 2) -> Creates 4 nodes
# # (Combine x and y temporarily so split_data can process them)
# data_L = np.column_stack((x_L, y_L))
# data_R = np.column_stack((x_R, y_R))

# # Split the left side
# x_LL, y_LL, x_LR, y_LR = split_data(data_L, threshold=1.5) 
# # Split the right side
# x_RL, y_RL, x_RR, y_RR = split_data(data_R, threshold=6.0) 

# # 3. Call the function with all 4 final leaf nodes!
# mse_tree([y_LL, y_LR, y_RL, y_RR])