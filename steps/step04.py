import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


# 各種関数の基底クラスとして、共通する機能の実装
class Function:
    def __call__(self, input):
        x = input.data        # Variableインスタンスから入力
        y = self.forward(x)
        output = Variable(y)  # Variableインスタンスで出力
        return output

    def forward(self, x):
        raise NotImplementedError()  # 継承先のクラスで実装する


class Square(Function):
    def forward(self, x):
        return x ** 2


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


# 中心差分近似による数値微分
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)

print(dy)
