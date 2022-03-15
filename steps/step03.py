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


A = Square()
B = Exp()
C = Square()

data = np.array(0.5)
x = Variable(data)

a = A(x)
b = B(a)
y = C(b)

print(type(y))
print(y.data)
