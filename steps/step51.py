if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
import math
from dezero import Variable, as_variable
from dezero import Function
import dezero.functions as F
import matplotlib.pyplot as plt
import dezero.layers as L
import dezero
from dezero.utils import plot_dot_graph
from dezero.models import Model
from dezero import optimizers
from dezero.models import MLP
from dezero.dataloaders import DataLoader
import matplotlib.pyplot as plt


def f(x):
    x = x.flatten()
    x = x.astype(np.float32)
    x /= 255.0
    return x


train_set = dezero.datasets.MNIST(train=True, transform=f)
test_set = dezero.datasets.MNIST(train=False, transform=f)

# ハイパラの指定
max_epoch = 10
batch_size = 100
hidden_size = 1000


train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
optimizer = optimizers.Adam().setup(model)

train_loss = []
test_loss = []
for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        # ミニバッチの生成
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    train_loss.append(sum_loss)
    print('epoch {}' .format(epoch+1))
    print('train loss {:.4f}, accuracy: {:.4f}' .format(
        sum_loss / len(train_set), sum_acc / len(train_set)))

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)

            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    test_loss.append(sum_loss)
    print('test loss: {:.4f}, accuracy: {:.4f}'.format(
        sum_loss / len(test_set), sum_acc / len(test_set)))

# 作図
plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(train_loss)), train_loss, label='train')  # 損失
plt.plot(np.arange(len(test_loss)), test_loss, label='test')  # 損失
plt.xlabel('iterations (epoch)')  # x軸ラベル
plt.ylabel('loss')  # y軸ラベル
plt.title('Cross Entropy Loss', fontsize=20)  # タイトル
plt.grid()  # グリッド線
# plt.ylim(0, 0.2) # y軸の表示範囲
plt.show()
