# 2. Linear regression: Numpy를 이용하여 구현하기
## 2.1 Linear regression을 위한 Parameter의 Gradient 식 정리


## 2.2 Linear regression학습을 위한 Stochastic Gradient Descent (SGD)Method

### 2.2.1 SGD를 이용한 parameter update식 정리

### 2.2.2 L2 regularization추가

### 2.2.3 Early stopping추가

## 2.3 [Main Project] Linear regression을 위한 SGD 구현

### 2.3.1 랜덤 데이터 생성기 구현: Gaussian분포에 기반
```python
import numpy as np
import matplotlib.pyplot as plt

# 1. true 파라미터 값의 랜덤 할당
r = 10
d = 1  # int(input())
n = 1000  # int(input())
alpha = 0.1
# w = d x 1
w = np.random.uniform(-r, r, (d, 1))
# print("w:\n",w)

# b = 1 x 1
b = np.random.uniform(-r, r, (1, 1))
# print("b:\n",b)

# 2. 데이터 셋 생성
# x = n x d
x = np.random.uniform(-r, r, (n, d))
'''
# x x(i)를 품고 있는 리스트
x = []
# xi d x 1
for i in range (n):
    xi = np.zeros((d, 1))
    for j in range (d):
        p = np.random.uniform(-r, r)
        xi[j] = p
    x.append(xi)
'''
# print("x:\n",x)
# t
t = []
for i in range(n):
    sd = (alpha * r)
    ti = np.random.normal((x[i] * w + b)[0, 0], sd, d)  # 평균, 표준편차, 개수
    t.append(ti)
t = np.array(t)
# print("t:\n",t)

y = w * x + b
# print(y)

plt.scatter(x, y)
plt.scatter(t, y)
plt.show()

# 3. 데이터셋 분리
n_of_train = int(len(x) * 0.85)
n_of_test = int(len(x) * 0.1)
n_of_dev = int(len(x) - n_of_train - n_of_test)

x_test = x[: n_of_test]
t_test = t[: n_of_test]

x_train = x[n_of_test: n_of_test + n_of_train]
t_train = t[n_of_test: n_of_test + n_of_train]

x_dev = x[n_of_test + n_of_train:]
t_dev = t[n_of_test + n_of_train:]
```

### 2.3.2 scikit 샘플 예제: diabets
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# 1. true 파라미터 값의 랜덤 할당
r = 10
d = 10  # int(input())
n = 1000  # int(input())
alpha = 0.1
# w = d x 1
w = np.random.uniform(-r, r, (d, 1))
# print("w:\n",w)

# b = 1 x 1
b = np.random.uniform(-r, r, (1, 1))
# print("b:\n",b)

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Split the data into training/testing sets
x_train = diabetes_X[:-20]
x_test = diabetes_X[-20:]

# Split the targets into training/testing sets
t_train = diabetes_y[:-20]
t_test = diabetes_y[-20:]
```


### 2.3.3 Linear regression학습기 테스트
1) 랜덤 데이터
```python
learning_rate = 0.01
epochs = 10
batch_size = 10
l2_lam = 0.7  # l2_regularization의 lambda값 임의로 설정
for epoch in range(epochs):
    batch = np.random.choice(x_train.shape[0], batch_size)  # 배치 크기에 맞게 임의의 인덱스 번호 생성
    # print(batch)
    x_batch = x_train[batch]  # 미니배치 생성
    # print("x_batch",x_batch)
    t_batch = t_train[batch]
    # print("t_batch",t_batch)

    f = np.dot(x_batch, w) + b
    # print("f", f)
    dw = ((f - t_batch) * x_batch * 2 + 2 * l2_lam * w).mean(0)
    # print("l2_reg: ", (f - t_batch) * x_batch * 2 + 2 * l2_lam * w)
    # print("dw :", dw)
    dw = dw.reshape(dw.shape[0], 1)
    # print("new dw :", dw)
    db = (f - t_batch).mean()

    w -= dw * learning_rate
    # print("w :",w)
    b -= db * learning_rate
    # print("d :",b)

    # if epoch % 10 == 0:
    f = np.dot(x_train, w) + b
    cost = np.mean(np.square(f - t_train))
    print('Epoch (', epoch + 1, '/', epochs, ') cost: ', cost, 'W: ', w, 'b:', b)

'''
# sgd 구현
n_data = len(x_train)
epochs = 5000
learning_rate = 0.01
l2_lam = 0.01

for i in range(epochs):
    hypothesis = w * x_train + b
    cost = np.sum((hypothesis - t_train) ** 2 ) / n_data
    l2_cost = np.sum((hypothesis - t_train) ** 2 ) + l2_lam * (w ** 2)
    gradient_w = np.sum((w * x_train - t_train + b) * 2 * x_train + 2 * l2_lam * w) / n_data
    gradient_b = np.sum((w * x_train - t_train + b) * 2) / n_data

    w -= learning_rate * gradient_w
    b -= learning_rate * gradient_b

    if i % 100 == 0:
        print('Epoch ({', i, '}/{', epochs, '}) cost: {', cost, '}, W: {', w, '}, b:{', b, '}')

print('result : ')
print(x_train * w + b)
'''
```
2) sklearn 데이터
```python
learning_rate = 0.01
epochs = 10
batch_size = 10
l2_lam = 0.7  # l2_regularization의 lambda값 임의로 설정

for epoch in range(epochs):
    batch = np.random.choice(x_train.shape[0], batch_size)  # 배치 크기에 맞게 임의의 인덱스 번호 생성
    # print(batch)
    x_batch = x_train[batch]  # 미니배치 생성
    # print("x_batch",x_batch)
    t_batch = t_train[batch]
    # print("t_batch",t_batch)

    f = np.dot(x_batch, w) + b
    # print("f", f)
    dw = ((f - t_batch) * x_batch * 2 + 2 * l2_lam * w).mean(0)
    # print("l2_reg: ", (f - t_batch) * x_batch * 2 + 2 * l2_lam * w)
    # print("dw :", dw)
    dw = dw.reshape(dw.shape[0], 1)
    # print("new dw :", dw)
    db = (f - t_batch).mean()

    w -= dw * learning_rate
    # print("w :",w)
    b -= db * learning_rate
    # print("d :",b)

    # if epoch % 10 == 0:
    f = np.dot(x_train, w) + b
    cost = np.mean(np.square(f - t_train))
    print('Epoch (', epoch + 1, '/', epochs, ') cost: ', cost, 'W: ', w, 'b:', b)
```
