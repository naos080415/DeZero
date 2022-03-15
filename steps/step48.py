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
import matplotlib.pyplot as plt


# ハイパラの指定
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0


x, t = dezero.datasets.get_spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

trace_loss = []
for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        # ミニバッチの生成
        batch_index = index[i * batch_size:(i+1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        y = model(batch_x)
        loss = F.softmax_cross_entropy_simple(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    ave_loss = sum_loss / data_size
    trace_loss.append(ave_loss)

    print('epoch %d, loss %.2f' % (epoch+1, ave_loss))


# 作図
plt.figure(figsize=(8, 6))
plt.plot(np.arange(len(trace_loss)), trace_loss, label='train') # 損失
plt.xlabel('iterations (epoch)') # x軸ラベル
plt.ylabel('loss') # y軸ラベル
plt.title('Cross Entropy Loss', fontsize=20) # タイトル
plt.grid() # グリッド線
#plt.ylim(0, 0.2) # y軸の表示範囲
plt.show()
