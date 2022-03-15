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


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
print(x.transpose())
# y = F.reshape(x, (6, ))
y = F.transpose(x)
print(y)
y.backward(retain_grad=True)
print(x.grad)
