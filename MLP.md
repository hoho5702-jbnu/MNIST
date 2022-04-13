# 4. Multi-layer perceptron (MLP)

## 4.1 backpropagation유도를 위한 기본 규칙

### 4.1.1 layer l의 post-activation delta vector로부터 pre-activation delta vector를 구하는 backprop 과정

### 4.1.2 layer l delta vector로부터 layer l − 1 delta vector를 구하는 backprop 과정

### 4.1.3 layer l의 delta vector가 주어질때 Parameter matrix update

## 4.2 L개의 layer로 구성된 MLP의 backpropagation알고리즘

## 4.3 (250 points) [Main project] MLP학습을 위한 Stochastic Gradient Descent Method 구현
```python
import numpy as np
from tensorflow import keras
import math
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class MLP:
    def __init__(self):
        self.hidden_list = None
        self.w_list = None
        self.b_list = None
        self.num_layer = 3  # 층 수
        print('nl: ', self.num_layer)
        self.learning_rate = 0.0009  # 학습률
        print('lr: ', self.learning_rate)
        self.batch_size = 32  # 배치 사이즈
        self.epochs = 40
        self.lambd = 0.7
        self.losses = []  # 손실(훈련세트)
        self.val_losses = []  # 손실(검증세트)
        self.accuracy = []  # 정확도(훈련세트)
        self.val_accuracy = []  # 정확도(검증세트)

    def init_weights(self):
        self.hidden_list = [200, 100]  # 은닉층에 들어갈 히든수 1~10 랜덤 생성
        '''
        for i in range(self.num_layer - 1):
            self.hidden_list.append(np.random.randint(10) + 1)
        '''
        print("hidden: ", self.hidden_list)

        # 1. true 파라미터 값의 랜덤 할당, w_list 와 b_list 의 크기는 은닉층 수 + 1 로 동일하다.
        self.w_list = [np.random.normal(0, 1, (784, self.hidden_list[0]))]
        for hidden in range(1, self.num_layer - 1):  # 첫번째 은닉층을 제외한 은닉층들의 w 생성 후 저장
            # w = ( 이전 은닉층 hidden x 다음 은닉층 hidden )
            self.w_list.append(np.random.normal(0, 1, (self.hidden_list[hidden - 1], self.hidden_list[hidden])))
        # 마지막 w = ( 이전 은닉층 hidden x 10 )
        self.w_list.append(np.random.normal(0, 1, (self.hidden_list[self.num_layer - 2], 10)))

        # b = 1 x hidden
        self.b_list = [np.zeros((self.hidden_list[0]))]
        for hidden in range(1, self.num_layer - 1):
            self.b_list.append(np.zeros((self.hidden_list[hidden])))
        self.b_list.append(np.zeros((10)))

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, z1):  # sigmoid 함수
        z1 = np.clip(z1, -100, None)
        a1 = 1 / (1 + np.exp(-z1))
        return a1

    def softmax(self, x):  # softmax 함수
        x = np.clip(x, -100, None)  # -100 보다 작으면 -100 으로
        c = np.max(x, axis=1).reshape(-1, 1)
        exp_x = np.exp(x-c)
        out = exp_x / np.sum(exp_x, axis=1).reshape(-1, 1)
        return out

    def forward(self, x, num_layer):
        out = np.dot(x, self.w_list[num_layer]) + self.b_list[num_layer]
        #out = np.clip(out, -100, None)  # NaN 방지
        return out

    def loss(self, x_data, y_data):  # 손실 계산
        y = np.zeros((self.batch_size, 10))
        z = x_data.copy()
        for layer in range(self.num_layer):  # 0 ~ num_layer-1
            z = self.forward(z, layer)
            if layer < self.num_layer - 1:  # 0 ~ num_layer-2(은닉층)
                active = self.relu(z)  # relu 적용
                #active = self.sigmoid(z)
                z = active
            else:  # num_layer-1(출력층)
                y = self.softmax(z)  # softmax 적용
                y = np.clip(y, 1e-10, 1 - 1e-10)  # 로그 안에 0 이 들어가 nan 이 나오는 것을 방지

        l2_cost = 0
        for index in range(len(self.w_list)):
            l2_cost += np.sum(np.square(self.w_list[index]))
        l2_cost *= (self.lambd / (2 * self.batch_size))

        return -np.sum(y_data * np.log(y)) + l2_cost

    def backpropagation(self, x, y):
        z_list = []
        h_list = []
        z = x.copy()
        for layer in range(self.num_layer):  # 0 ~ num_layer-1
            z = self.forward(z, layer)
            z_list.append(z)
            if layer < self.num_layer - 1:  # 0 ~ num_layer-2(은닉층)
                active = self.relu(z)  # relu 적용
                #active = self.sigmoid(z)
                h_list.append(active)
                z = active
            else:  # num_layer-1(출력층)
                active = self.softmax(z)  # softmax 적용
                h_list.append(active)  # 순전파 종료

        # 역전파
        # 출력층 w 업데이트
        # softmax cost func의 미분 결과는 pi - yi
        do = h_list[self.num_layer - 1] - y
        # print("do: ", do.shape)
        du = np.dot(h_list[self.num_layer - 2].T, do) + self.lambd * self.w_list[self.num_layer-1] # ( 마지막 은닉층의 hidden x 10 ).T
        # print("du: ", du.shape)
        db = np.sum(do)
        du /= self.batch_size
        db /= self.batch_size
        self.w_list[self.num_layer - 1] -= self.learning_rate * du
        self.b_list[self.num_layer - 1] -= self.learning_rate * db

        dz = do.copy()
        # j 는 현재 은닉층
        for layer in range(self.num_layer - 2, -1, -1):  # num_layer-1 ~ 0
            # print("역전파 ", j+1, " 번째 은닉층")
            # print("dh: ", dh.shape)

            d_relu = h_list[layer].copy()
            d_relu[d_relu < 0] = 0
            d_relu[d_relu > 0] = 1
            dz = np.dot(dz, self.w_list[layer + 1].T) * d_relu

            # print("dz: ", dz.shape)
            if layer > 0:  # 첫 은닉층이 아니라면
                dw = np.dot(h_list[layer - 1].T, dz) + self.lambd * self.w_list[layer]
                # print("dw: ", dw.shape)
                # print("해당 w: ", w[j].shape)
                db = np.sum(dz, axis=0)
                dw /= self.batch_size
                db /= self.batch_size
                self.w_list[layer] -= self.learning_rate * dw
                self.b_list[layer] -= self.learning_rate * db
            else:  # layer = 0  첫번째 은닉층이면
                dw = np.dot(x.T, dz) + self.lambd * self.w_list[layer]
                db = np.sum(dz, axis=0)
                dw /= x.shape[0]
                db /= x.shape[0]
                self.w_list[layer] -= self.learning_rate * dw
                self.b_list[layer] -= self.learning_rate * db

    def minibatch(self, x, y):
        iter = math.ceil(len(x) / self.batch_size)

        x, y = shuffle(x, y)

        for i in range(iter):
            start = self.batch_size * i
            end = self.batch_size * (i + 1)
            yield x[start:end], y[start:end]

    def fit(self, x_tran, y_train, x_val=None, y_val=None):
        self.init_weights()
        for epoch in range(self.epochs):
            loss = 0
            for i in range(0, x_train.shape[0], self.batch_size):  # 미니배치 크기만큼 증가하면서
                batch_index = i + self.batch_size  # 0 ~ 59999 까지 batch_size 씩 증가
                # print(batch)
                x_batch = x_train[i: batch_index]  # 미니배치 생성 ( batch_size x 784 ).T
                # print(np.argmax(x_batch[0]))
                y_batch = y_train[i: batch_index]  # 미니배치에 대한 target 값 ( batch_size x 10 ).T
                # print("t_batch",np.argmax(t_batch[0]))

                loss += self.loss(x_batch, y_batch)
                self.backpropagation(x_batch, y_batch)

            val_loss = self.val_loss(x_val, y_val)

            self.losses.append(loss / len(y_train))
            self.val_losses.append(val_loss)
            self.accuracy.append(self.score(x_train, y_train))
            self.val_accuracy.append(self.score(x_val, y_val))

            print(f'epoch({epoch + 1}) ===> loss : {loss / len(y_train):.5f} | val_loss : {val_loss:.5f}',
                  f' | accuracy : {self.score(x_train, y_train):.5f} | val_accuracy : {self.score(x_val, y_val):.5f}')

    def predict(self, x_data):
        z = x_data.copy()
        for layer in range(self.num_layer):  # 0 ~ num_layer-1
            z = self.forward(z, layer)
            if layer < self.num_layer - 1:  # 0 ~ num_layer-2(은닉층)
                active = self.relu(z)  # relu 적용
                # active = self.sigmoid(z)
                z = active
        return np.argmax(z, axis=1)  # 가장 큰 인덱스 반환

    def score(self, x_data, y_data):
        return np.mean(self.predict(x_data) == np.argmax(y_data, axis=1))

    def val_loss(self, x_val, y_val):  # 검증 손실 계산
        val_loss = self.loss(x_val, y_val)
        return val_loss / len(y_val)


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

model = MLP()
model.fit(x_train, y_train, x_val=x_val, y_val=y_val)
print(model.score(x_test, y_test))

```
![image](https://user-images.githubusercontent.com/91112750/162624016-cfe069e2-cc59-4528-959c-e37fab281f50.png)
