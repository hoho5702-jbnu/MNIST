# 3. Linear regression: Numpy를 이용하여 구현하기
## 3.1 Logistic regression학습을 위한 식 정리

### 3.1.1 기본 Gradient식 정리

### 3.1.2 L2정규화를 반영한 parameter update식 정리

## 3.2 Early stopping을 반영한 SGD Method

## 3.3 [Main Project]  Logistic regression학습을 위한 Stochastic Gradient Descent Method 구현
```python

import numpy as np
from tensorflow import keras
import math
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

np.random.seed(42)
# mnist 데이터를 받아옴
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 검증 세트 생성
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, stratify=y_train, test_size=0.2)

# 28 x 28 배열을 1 x 784 배열로 바꿈
x_train = x_train.reshape(x_train.shape[0], 28 * 28)
x_val = x_val.reshape(x_val.shape[0], 28 * 28)
x_test = x_test.reshape(x_test.shape[0], 28 * 28)

# 0~255 값 분포를 0~1 사이에 분포하도록 바꿈
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# one-hot encoding
y_train = keras.utils.to_categorical(y_train)
y_val = keras.utils.to_categorical(y_val)
y_test = keras.utils.to_categorical(y_test)
print(math.ceil(len(x_train) / 256))

np.random.seed(0)
learning_rate = 0.01
epochs = 10
batch_size = 32


# 1. true 파라미터 값의 랜덤 할당
r = 10
# w = d x 1
w = np.random.uniform(-r, r, (784, 10))
print("w: ", w.shape)

# b = 1 x 1
b = np.random.uniform(-r, r, (1, 1))
print("b: ", b.shape)

num = x_train.shape[0]
for epoch in range(epochs):
    for i in range(0, x_train.shape[0], batch_size):
        batch = i + batch_size # 0 ~ 60000까지 10씩 증가
        # print(batch)
        x_batch = x_train[i : batch]  # 미니배치 생성 batch_size x 784
        # print(x_batch.shape)
        # print("x_batch",x_batch)
        y_batch = y_train[i : batch]    # batch_size x 10
        # print("t_batch",t_batch)

        f = np.dot(x_batch, w) + b # batch_size x 10
        # print("f: ", f.shape)
        # 소프트맥스 함수 y
        c = np.max(f, axis=1).reshape(-1, 1)
        exp_f = np.exp(f-c)
        sum_exp_f = np.sum(exp_f, axis=1).reshape(-1, 1)
        softmax = exp_f / sum_exp_f # batch_size x 10
        # print("softmax: ", softmax.shape)

        # cost
        loss = np.sum(-np.sum(y_batch * np.log(softmax)))
        cost = loss / batch_size
        # ("cost: ", cost) # 1 x 1

        # softmax cost func의 미분 결과는 pi - yi
        gradient = softmax - y_batch # batch_size x 10
        dw = np.dot(x_batch.T, gradient) # 784 x 10
        # print(dw.shape)
        dw /= batch_size
        # print("dw: ", dw.shape)
        # db = gradient / batch_size
        # print("db: ", db.shape)

        w -= dw * learning_rate
        # print("w :",w)
        # b -= db * learning_rate
        # print("d :",b)
    # if epoch % 10 == 0:
    f = np.dot(x_train, w) + b  # 60000 x 10
    # print("f: ", f.shape)
    # 소프트맥스 함수 y
    exp_f = np.exp(f)
    sum_exp_f = np.sum(exp_f)
    softmax = exp_f / sum_exp_f  # 60000 x 10
    # print("softmax: ", softmax.shape)

    result = 0
    for i in range (x_train.shape[0]):
        if np.argmax(softmax[i]) == np.argmax(y_train[i]):
            result += 1

    print('Epoch (', epoch + 1, '/', epochs, ') result: ', result/x_train.shape[0])

```
![image](https://user-images.githubusercontent.com/91112750/162623612-02bb29e5-6d80-445b-97e4-70fb4da2108e.png)
