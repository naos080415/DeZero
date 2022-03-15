if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
import math
import time
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


# ハイパラの指定
max_epoch = 5
batch_size = 100
hidden_size = 1000

train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)

model = MLP((1000, 10))
optimizer = optimizers.SGD().setup(model)


# GPU mode
if dezero.cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()

for epoch in range(max_epoch):
    start = time.time()
    sum_loss = 0

    for x, t in train_loader:
        # ミニバッチの生成
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)

    elapsed_time = time.time() - start
    print('epoch {}, loss: {:.4f}, time: {:.4f}[sec]' .format(
        epoch+1, sum_loss / len(train_set), elapsed_time))

