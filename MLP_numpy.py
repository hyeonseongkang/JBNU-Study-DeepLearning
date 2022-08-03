import numpy as np


# ReLU
def relu(x):
  return np.maximum(0, x)

# relu test
print(relu(1))

print(relu(0.2))

print(relu(-2))

print(relu(-0.5))

# softMax
def softmax(a):
  exp_a = np.exp(a)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a

  return y


a = np.array([0.3, 2.9, 4.0])
print(a)
exp_a = np.exp(a)
print(exp_a)
sum_exp_a = np.sum(exp_a)
print(sum_exp_a)
y = exp_a / sum_exp_a
print(y)
print(softmax(a))