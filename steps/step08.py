import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None    # 微分値
        self.creator = None  # 生成元の関数

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()     # 1. 関数を取得
            x, y = f.input, f.output    # 2. 関数の入出力を取得
            x.grad = f.backward(y.grad)  # backwardメソッドを呼ぶ

            if x.creator is not None:
                funcs.append(x.creator)     # 1つ前の関数をリストに追加


# 各種関数の基底クラスとして、共通する機能の実装
class Function:
    def __call__(self, input):
        x = input.data        # Variableインスタンスから入力
        y = self.forward(x)
        output = Variable(y)    # Variableインスタンスで出力
        output.set_creator(self)    # 出力変数に生みの親を覚えさせる
        self.input = input    # 入力された変数を記憶
        self.output = output  # 出力を記憶
        return output

    # 順伝搬
    def forward(self, x):
        raise NotImplementedError()  # 継承先のクラスで実装する

    # 逆伝搬
    def backward(self, gy):
        raise NotImplementedError()  # 継承先のクラスで実装する


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


# 中心差分近似による数値微分
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
y.backward()

print(x.grad)
