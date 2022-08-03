import numpy as np
import matplotlib.pyplot as plt

# ReLU
def relu(x):
  return np.maximum(0, x)

# relu test
"""
print(relu(1))
print(relu(0.2))
print(relu(-2))
print(relu(-0.5))
"""


# softMax
def softmax(a):
  exp_a = np.exp(a)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a

  return y

# softmax test
"""
a = np.array([0.3, 2.9, 4.0])
print(a)
exp_a = np.exp(a)
print(exp_a)
sum_exp_a = np.sum(exp_a)
print(sum_exp_a)
y = exp_a / sum_exp_a
print(y)
print(softmax(a))
"""

X = np.random.rand(100)
Y = 0.2 * X * 0.5
print(X)
def plot_prediction(pred, y):
  plt.figure(figsize=(8, 6))
  plt.scatter(X, Y)
  plt.scatter(X, pred)
  plt.show()

## Gradient Descent

W = np.random.uniform(-1, 1)
# W1 = np.random.uniform(-1, 1)
# W2 = np.random.uniform(-1, 1)
b = np.random.uniform(-1, 1)

learning_rate = 0.7

for epoch in range(100):
  # Y_pred = W1 * x1 + W2 * x2 + .... + Wn * xn + b
  Y_pred = W * X + b # Y: 실제값, Y_pred: 예측값

  error = np.abs(Y_pred - Y).mean() # 오차값
  if error < 0.001:
    break


  # gradient descent
  # 업데이트할 W: Learning Rate * ((Y예측 - Y실제) * X)평균
  # 업데이트할 b: Learning Rate * ((Y예측 - Y실제) * 1)평균
  w_grad = learning_rate * ((Y_pred - Y) * X).mean()
  b_grad = learning_rate * (Y_pred - Y).mean()

  # W, b 갱신
  W = W - w_grad
  b = b - b_grad

  if epoch % 5 == 0:
    Y_pred = W * X + b
    plot_prediction(Y_pred, Y)