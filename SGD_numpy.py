import numpy as np

"""
logistic regression: 회귀를 사용하여 데이터가 어떤 범주에 속할 확률을 0 ~ 1 사이의 값으로 예측, 가능성이 더 높은 범주에 속하는 것으로 분류해주는 지도학습
ex) 스팸 메일 분류기
스팸일 확률 0.5 이상 -> 스팸 O
         0.5 이하 -> 스팸 X
hidden layer: 1
dense: 3
minibatch size: 10(default)
epoch size: 100(default)
"""

# 은닉층
# hidden_layer = int(input())
hidden_layer = 1

# 차원
# dimension = int(input())
dimension = 3

# batch size (default: 10)
# batch_size = int(input())
batch_size = 10

# epoch size (default: 100)
# epoch_size = int(input())
epoch_size = 100

learning_rate = 0.1

np.random.seed(1)

# np.random.rand(m, n): 0 ~ 1 난수를 matrix array(m, n) 생성
x1 = np.random.rand(100)
x2 = np.random.rand(100)
x3 = np.random.rand(100)

# 정답
y1 = np.random.rand(1).round(1)
y2 = np.random.rand(1).round(1)
y3 = np.random.rand(1).round(1)
y_bias = np.random.rand(1).round(1)

print(y1, y2, y3, y_bias)

y = y1 * x1 + y2 * x2 + y3 * x3 + y_bias

w_list = []
for i in range(3):
    w_list.append(np.random.uniform(-1, 1))

print(w_list)
for i in range(len(w_list)):
    print(w_list[i])

w1 = np.random.uniform(-1, 1)
w2 = np.random.uniform(-1, 1)
w3 = np.random.uniform(-1, 1)

bias = np.random.uniform(-1, 1)

print(f"구하고자하는 다항식: Y={y1}X1 + {y2}X2 + {y3}X3 + {y_bias}")
print(f"SGD 시작 다항식: Y={w1}X1 + {w2}X2 + {w3}X3 + {bias}")

for epoch in range(epoch_size):
    print("epoch: ", epoch)
    print("-"*50)

    start = 0
    end = 10

    for i in range(batch_size):
        # 매 iteration마다 batch_size=10에 해당하는 데이터 셋을 가져옴
        x1_batch = x1[start: end]
        x2_batch = x2[start: end]
        x3_batch = x3[start: end]
        y_batch = y1 * x1_batch + y2 * x2_batch + y3 * x3_batch + y_bias

        start += 10
        end += 10

        # 선택한 batch의 예측값
        predict_batch = w1 * x1_batch + w2 * x2_batch + w3 * x3_batch + bias

        # 가중치 업데이트
        w1 = w1 - 2 * learning_rate * ((predict_batch - y_batch) * x1_batch).mean()
        w2 = w2 - 2 * learning_rate * ((predict_batch - y_batch) * x2_batch).mean()
        w3 = w3 - 2 * learning_rate * ((predict_batch - y_batch) * x3_batch).mean()
        bias = bias - 2 * learning_rate * (predict_batch - y_batch).mean()

        # error값은 전체 데이터셋의 오류값을 계산해야한다.
        predict = w1 * x1 + w2 * x2 + w3 * x3 + bias
        error = ((y - predict) ** 2).mean()

        print("iteration ", i, "w1= ", w1.round(2), "w2= ", w2.round(2), "w3= ", w3.round(2), "bias= ", bias.round(2), "error= ", error)

    if error < 0.000001:
            break
print("최종: ", "w1= ", w1.round(2), "w2= ", w2.round(2), "w3= ", w3.round(2), "bias= ", bias.round(2), "error= ", error)

print(f"구하고자하는 다항식: Y={str(y1)}X1 + {str(y2)}X2 + {str(y3)}X3 + {str(y_bias)}")
print(f"SGD 최종 다항식: Y= {w1}X1 + {w2}X2 + {w3}X3 + {bias}")
