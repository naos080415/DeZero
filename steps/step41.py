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
from dezero.utils import plot_dot_graph


x = Variable(np.random.randn(2, 3))
W = Variable(np.random.randn(3, 4))
y = F.matmul(x, W)
print(y, y.shape)
y.backward()

print(x.grad.shape)
print(W.grad.shape)
