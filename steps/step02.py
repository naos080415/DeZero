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


data = np.array(10)
x = Variable(data)
f = Square()
y = f(x)

print(type(y))
print(y.data)
