import numpy as np
import unittest


class Variable:
    def __init__(self, data):
        if data is not None:    # ndarrayだけを扱う
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is notsupported'.format(type(data)))

        self.data = data
        self.grad = None    # 微分値
        self.creator = None  # 生成元の関数

    def set_creator(self, func):
        self.creator = func

    def cleargrad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()     # 1. 関数を取得
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    funcs.append(x.creator)     # 1つ前の関数をリストに追加


# 各種関数の基底クラスとして、共通する機能の実装
class Function:
    def __call__(self, *inputs):    # * をつけることで可変長引数となる
        xs = [x.data for x in inputs]   # Variableインスタンスから入力
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]  # Variableインスタンスで出力

        for output in outputs:
            output.set_creator(self)    # 出力変数に生みの親を覚えさせる
        self.inputs = inputs    # 入力された変数を記憶
        self.outputs = outputs  # 出力を記憶

        return outputs if len(outputs) > 1 else outputs[0]  # 要素が1つのときは最初の要素を返す

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
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def square(x):
    return Square()(x)  # 1行でまとめて書く


def exp(x):
    return Exp()(x)


def add(x0, x1):
    return Add()(x0, x1)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


# 中心差分近似による数値微分
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def numerical_diff(self):
        x = Variable(np.random.rand(1))     # ランダムな入力値を生成
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


x = Variable(np.array(2.0))
z = add(x, x)
z.backward()
print(z.data)
print(x.grad)

x.cleargrad()
z = add(add(x, x), x)
z.backward()
print(z.data)
print(x.grad)
