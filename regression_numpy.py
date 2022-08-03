import numpy as np

x_data = np.array([1, 2, 3, 4, 5]).reshape(5, 1) # 입력 데이터
t_data = np.array([3, 6, 9, 12, 15]).reshape(5, 1) # 정답 데이터

W = np.random.rand(1,1) # 0 ~ 1 사이의 random 값 - 1차원
b = np.random.rand(1)

# 손실 함수
def loss_func(x, t):
  y = np.dot(x, W) + b
  return np.sum((t - y) ** 2) / len(x)

# 미분 함수
def numerical_derivative(f, x):
  delta_x = 1e-4

  grad = np.zeros_like(x)

  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

  while not it.finished:
    idx = it.multi_index
    tmp_val = x[idx]

    x[idx] = tmp_val + delta_x
    fx1 = f(x)

    x[idx] = tmp_val - delta_x
    fx2 = f(x)

    grad[idx] = (fx1 - fx2) / (2 * delta_x)

    x[idx] = tmp_val


    it.iternext()
  return grad


# 예측 함수
def predict(x):
  y = np.dot(x, W) + b
  return y


# 학습
learning_rate = 1e-2

f = lambda x : loss_func(x_data, t_data)

for step in range(6001):
  W -= learning_rate * numerical_derivative(f, W)
  b -= learning_rate * numerical_derivative(f, b)

  if step % 1000 == 0:
    print("W = ", W, "b = ", b)


input_data = np.array([12, 8, 11, 15, 1000, 120]).reshape(6, 1)
print(predict(np.array([25])))
print(predict(input_data))