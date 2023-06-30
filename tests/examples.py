import math

from autograd import grad, jacobian
import numpy as np


def quadratic1_func(x, need_hessian_eval=False):
    Q = np.array([[1, 0],
                  [0, 1]])
    func = lambda t: (np.transpose(t) @ Q) @ t
    f = func(x)
    df = grad(func)
    g = df(x)
    if need_hessian_eval:
        hf = jacobian(df)
        h = hf(x)
        return f, g, h
    else:
        return f, g


def quadratic2_func(x, need_hessian_eval=False):
    Q = np.array([[1, 0],
                  [0, 100]])
    func = lambda t: np.transpose(t) @ Q @ t
    f = func(x)
    df = grad(func)
    g = df(x)
    if need_hessian_eval:
        hf = jacobian(df)
        h = hf(x)
        return f, g, h
    else:
        return f, g


def quadratic3_func(x, need_hessian_eval=False):
    q1 = np.array([[math.sqrt(3)/2, -0.5],
                   [0.5, math.sqrt(3)/2]])
    q2 = np.array([[100, 0],
                   [0, 1]])
    q3 = np.array([[math.sqrt(3)/2, -0.5],
                   [0.5, math.sqrt(3)/2]])
    Q = np.transpose(q1) @ q2 @ q3
    func = lambda t: np.transpose(t) @ Q @ t
    f = func(x)
    df = grad(func)
    g = df(x)
    if need_hessian_eval:
        hf = jacobian(df)
        h = hf(x)
        return f, g, h
    else:
        return f, g


def rosenbrock_func(x, need_hessian_eval=False):
    func = lambda t: 100 * (t[1] - t[0] ** 2) ** 2 + (1 - t[0]) ** 2
    f = func(x)
    df = grad(func)
    g = df(x)
    if need_hessian_eval:
        hf = jacobian(df)
        h = hf(x)
        return f, g, h
    else:
        return f, g


def linear_func(x, need_hessian_eval=False):
    a = np.array([2, 3])
    func = lambda t: np.transpose(a) @ t
    f = func(x)
    df = grad(func)
    g = df(x)
    if need_hessian_eval:
        hf = jacobian(df)
        h = hf(x)
        return f, g, h
    else:
        return f, g


def g_func(x, need_hessian_eval=False):
    func = lambda t: math.e ** (t[0] + 3 * t[1] - 0.1) + math.e ** (t[0] - 3 * t[1] - 0.1) + math.e ** (-1 * t[0] - 0.1)
    f = func(x)
    df = grad(func)
    g = df(x)
    if need_hessian_eval:
        hf = jacobian(df)
        h = hf(x)
        return f, g, h
    else:
        return f, g

