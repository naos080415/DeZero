if '__file__' in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
import math
from dezero import Variable
from dezero import Function
import dezero.functions as F
import matplotlib.pyplot as plt
import dezero.layers as L
from dezero.utils import plot_dot_graph
from dezero.models import Model
from dezero import optimizers
from dezero.models import MLP


# データセット
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
max_iters = 10000
hidden_size = 10

model = MLP((hidden_size, 1))
optimizer = optimizers.SGD(lr)
optimizer.setup(model)

for i in range(max_iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()  # 以下の2行と同様にパラメータの更新
    # for p in model.params():
    #     p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(loss)
