# model load
from tensorflow.keras.datasets.mnist import load_data

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

(x_train_full, y_train_full), (x_test, y_test) = load_data(path='mnist.npz')

x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full,
                                                  test_size=0.3,
                                                  random_state=111)


x_train_full = x_train_full.reshape(60000, 784)
x_train = x_train.reshape(42000, 784)
x_val = x_val.reshape(18000, 784)
x_test = x_test.reshape(10000, 784)

# 0 ~ 9 사이의 값을 예측하기 위해 one_hot_encoding 사용
def one_hot_encoding(labels, dimension=10):
  one_hot_labels = labels[..., None] == np.arange(dimension)[None]
  return one_hot_labels.astype(np.float64)

training_labels = one_hot_encoding(y_train[:42000])
test_labels = one_hot_encoding(y_test[:10000])

# relu
def relu(x):
    return (x >= 0) * x

def relu2deriv(output):
    return output >= 0


rng = np.random.default_rng(12345231)

learning_rate = 0.005
epochs = 100


weights_1 = 0.2 * rng.random((784, 100)) - 0.1
weights_2 = 0.2 * rng.random((100, 10)) - 0.1

save_training_loss = []
save_training_accurate_pred = []
save_test_loss = []
save_test_accurate_pred = []

for epoch in range(epochs):
    training_loss = 0.0
    training_accurate_predictions = 0

    for i in range(len(x_train)):
        # 학습할 데이터 가져오기
        layer_0 = x_train[i]

        # layer_0째 데이터와 가중치1 곱하기
        layer_1 = np.dot(layer_0, weights_1)

        # relu 함수 적용하기
        layer_1 = relu(layer_1)

        """
        드랍아웃은 인공신경망 훈련 과정을 최적화하기 위한 방법 중 하나이며, 오버피팅(overfitting)을 방지함으로써 모델의 정확도를 높여준다.
    
        Note
        오버피팅(overfitting)이란?
        신경망을 학습하는 과정에서 훈련데이터에만 적합한 형태로 학습되는 현상을 오버피팅이라고 한다. 훈련데이터의 정확도는 거의 100%를 달성하는데 실제데이터에서는 일정 이상의 정확도에서 상승하지 않는 것이다. 이런 현상은 보통 훈련데이터를 너무 적게 사용한 경우 또는 모델 파라미터가 너무 많은 경우에 발생한다.
    
        드랍아웃은 각각의 훈련데이터들이 결과값으로 연결되는 신호(엣지, Edge)를 일정한 퍼센트로 삭제함으로써 훈련데이터의 일부만 파라미터에 영향을 줄 수 있도록 조정하는 역할을 한다. 드랍아웃을 함수로 만들면 다음과 같다.
        """
        dropout_mask = rng.integers(low=0, high=2, size=layer_1.shape)
        layer_1 *= dropout_mask * 2

        # layer_1과 가중치2 곱하기
        layer_2 = np.dot(layer_1, weights_2)

        # loss값 계산
        training_loss += np.sum((y_train[i] - layer_2) ** 2)

        # error 구한 뒤 가중치 업데이트
        layer_2_delta = y_train[i] - layer_2

        layer_1_delta = np.dot(weights_2, layer_2_delta) * relu2deriv(layer_1)

        layer_1_delta *= dropout_mask

        weights_1 += learning_rate * np.outer(layer_0, layer_1_delta)
        weights_2 += learning_rate * np.outer(layer_1, layer_2_delta)
        if i % 10000 == 0:
            print("weights_1: ", weights_1)
            print("weights_2: ", weights_2)

    # training_loss 저장
    save_training_loss.append(training_loss)
    save_training_accurate_pred.append(training_accurate_predictions)

    # @ 행렬곱 연산
    results = relu(x_test @ weights_1) @ weights_2

    # 오차 값 계산
    test_loss = np.sum((test_labels - results) ** 2)

    # 정확도 측정
    test_accurate_predictions = np.sum(
        np.argmax(results, axis=1) == np.argmax(test_labels, axis=1)
    )

    # test_loss 저장
    save_test_loss.append(test_loss)
    save_test_accurate_pred.append(test_accurate_predictions)

    # print result
