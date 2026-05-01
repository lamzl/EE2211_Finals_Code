"""
run_me.py - Example usage for all EE2211 functions

Paste your ee2211_functions.py in the same directory, then run:
    python run_me.py
"""

import numpy as np
import torch
from torch import sigmoid

from EE2211_exam_functions import (
    bayes, check_rank, is_invertible, k_means,
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

# Training Data
# X_train_raw = np.array([[1, 3, -2], [-4, 0, -1], [3, 1, 8], [2, 1, 6], [8, 4, 6]])
X_train_raw = np.array([[4], [7], [10], [2], [3], [9]])

# y_train = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
y_train = np.array([-1, -1, -1, 1, 1, 1])

# Testing Data
X_test_raw = np.array([[-9], [-7], [-5], [-4], [-2], [1], [4], [5], [6], [9]])
y_test = np.array([3, 1.81, 0.80, 0.25, -0.19, 0.4, 1.24, 1.68, 2.32, 5.05])

# --- add_bias() ---
print("\n-- add_bias() --")
X_biased = add_bias(X_train_raw)
print("X with bias column:\n", X_biased)


# --- make_poly() ---
print("\n-- make_poly() --")
X_poly_train = make_poly(X_train_raw, order=2)
X_poly_test  = make_poly(X_test_raw, order=2)
print("Polynomial features (order 3):\n", X_poly_train)


# --- mean_squared_error() ---
print("\n-- mean_squared_error() --")
mean_squared_error(add_bias(X_train_raw), y_train)
# mean_squared_error(add_bias(X_poly_train), y_train) # for polynomial function

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
ridge_reg(X_biased, y_train, reg_factor=1, Xt=add_bias(X_test_raw))

# --- binary_classif() ---
print("\n-- binary_classif() --")
y_binary = np.array([4.18, 2.42, 0.22, 0.12, 0.25, 3.09])  # Binary labels for the training data
binary_classif(X_biased, y_binary, add_bias(X_test_raw))
# binary_classif(X_poly_train, y_binary, X_poly_test) # for polynomial features

# --- multi_category_classif() ---
print("\n-- multi_category_classif()--> linear --")
X_mc = np.array([[-1], [0], [0.5], [0.3], [0.8]])
y_mc = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
X_mc_b = add_bias(X_mc) 
X_test_mc = add_bias(np.array([[-0.1], [0.4]]))
multi_category_classif(X_mc_b, y_mc, X_test_mc)

#==========================================
print("\n-- multi_category_classif 2() --> polynomial --")
X_mc = np.array([[1, 2, 3], [4, 0, 6], [1, 1, 0], [0, 1, 2], [5, 7, -2], [-1, 4, 0]])
y_mc = np.array([[1], [1], [2], [3], [2], [3]]) 

X_mc_poly = make_poly(X_mc, order=3)
X_test_raw = np.array([[1, -2, 3]])
X_test_poly = make_poly(X_test_raw, order=3)
X_mc_final = add_bias(X_mc_poly)
X_test_final = add_bias(X_test_poly)
multi_category_classif(X_mc_final, y_mc, X_test_final)



# ===========================================================
# Chapter 7
# ===========================================================
print("\n" + "=" * 50)
print("CHAPTER 7")
print("=" * 50)

# --- pearsons() ---
print("\n-- pearsons() --")
x_vals = [[0.3510, 1.1796, -0.9852], [2.1812, 2.1068, 1.3766], [0.2415, 1.7753,-1.3244], [-0.1096, 1.2747, -0.6316], [0.1544, 2.0815, -0.8320]]
y_vals = [[0.2758], [1.4392], [-0.4611], [0.6154], [1.0006]]
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

# --- grad_descent() 1 variable in C(w) ---
print("\n-- grad_descent()--")
x_val = 2.0
f       = lambda x: x**4
grad_f  = lambda x: 4*x**3
gradient = grad_f(x_val)
print(f"Calculated gradient at x={x_val:.4f} is: {gradient:.4f}")
# eta is learning rate, iterations is number of steps to take
grad_descent(x=x_val, eta=0.1, iterations=1, function=f, grad=grad_f)

# --- grad_descent_2() --> 2 variables in C(w) ---
print("\n-- grad_descent_2() --")
# f2      = lambda x, y: np.e**(x-y)+ x**2*y
# grad_x2 = lambda x, y: np.e**(x-y) + 2*x*y # partial derivative w.r.t x
# grad_y2 = lambda x, y: -1*np.e**(x-y) + x**2 # partial derivative w.r.t y
f2      = lambda x, y: x**2+y**4
grad_x2 = lambda x, y: 2*x
grad_y2 = lambda x, y: 4*y**3
grad_descent_2(x=1.0, y=1.0, eta=0.2, iterations=1,
               function=f2, grad_x=grad_x2, grad_y=grad_y2)


# ===========================================================
# Chapter 9
# ===========================================================
print("\n" + "=" * 50)
print("CHAPTER 9")
print("=" * 50)

# --- gini_impurity() ---
print("\n-- gini_impurity() --")
gini_impurity([8/18, 5/18, 5/18])

# --- entropy() ---
print("\n-- entropy() --")
entropy([8/18, 5/18, 5/18])

# --- misclassification_rate() ---
print("\n-- misclassification_rate() --")
misclassification_rate([8/18, 5/18, 5/18])

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
    [0.1, 1.9], [0.7, 1.5], [1.6, 5.4], [2.2, 6.1], [3.6, 8.9],
    [4.1, 9.5], [4.5, 9.6], [5.2, 12.9], [6.2, 13.6], [7.3, 15.7]
])
x_left, y_left, x_right, y_right = split_data(data, threshold=4.0)
print(f"Left  x: {x_left},  y: {y_left}")
print(f"Right x: {x_right}, y: {y_right}")

# --- mse_node() ---
y_root = data[:, 1] 
print("\n-- MSE at Root --")
mse_node(y_root)

# --- mse_depth_1() ---
print("\n-- mse_depth_1() --")
mse_depth_1(y_left, y_right)

# ===========================================================
# Chapter 11
# ===========================================================
print("\n" + "=" * 50)
print("CHAPTER 11")
print("=" * 50)

# --- K-Means---------------------
print("\n-----implementing k-means algorithm-----")
# Data points
x1 = np. array ([0 , 0])
x2 = np. array ([0 , 1])
x3 = np. array ([1 , 1])
x4 = np. array ([1 , 0])
x5 = np. array ([3 , 0])
x6 = np. array ([3 , 1])
x7 = np. array ([4 , 0])
x8 = np. array ([4 , 1])

data_points = np. array ([x1 , x2 , x3 , x4 , x5 , x6 , x7 , x8 ])

# Initial centers
c1_init = np. array ([0 , 0])
c2_init = np. array ([3 , 0])

centers = np. array ([ c1_init , c2_init ])
centers , labels = k_means ( data_points , centers , n_clusters =2)
print (" Converged centers :", centers )


# ===========================================================
# Chapter 12
# ===========================================================
print("\n" + "=" * 50)
print("CHAPTER 12")
print("=" * 50)

print("\n-----implementing  sigmoid algorithm-----")

X = torch.tensor([[1 , 2 , 1],[1 , 5, 1]], dtype=torch.float32)

W1 = torch.tensor([[-1 , 0 , 1],[0 , -1, 0],[1, 0, -1]], dtype=torch.float32)
W2 = torch.tensor([[-1 , 0 , 1],[0 , -1, 0],[1, 0, 1],[1, -1, 1]], dtype=torch.float32)
W3 = W2
XW1 = X@W1
#F_1 = 𝑓(𝑿𝑾_1 )
F_1 = sigmoid(XW1)
print(F_1)
print('')

#F_2 = 𝑓([𝟏,𝑓(𝑿𝑾_1 )] 𝑾_2 )
bias = torch.ones((2, 1), dtype=torch.float32)

F_2 = torch.cat([bias, F_1], dim=1)
F_2 = F_2@W2
F_2 = sigmoid(F_2)
print(F_2)
print('')

#F_3 = 𝑭_𝒘 (𝑿) = 𝑓([𝟏,𝑓([𝟏,𝑓(𝑿𝑾_1 )] 𝑾_2 )] 𝑾_3 )
F_3 = torch.cat([bias, F_2], dim=1)
F_3 = F_3@W3
F_3 = sigmoid(F_3)
print(F_3)





