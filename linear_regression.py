# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_data(a=0.5, b=2, noise=0.3):
    x = np.arange(10, dtype=np.float32) - 5
    y = a * x + b + noise * np.random.randn(10)

    return x, y


if __name__ == "__main__":
    a, b = 0.5, 2
    x, y = generate_data(a, b)
    plt.plot(x, y, "x")

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    X = np.hstack((x, np.ones(x.shape)))
    A = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)  # 最小二乗法
    print "actual:", a, b
    print "estimated", A[0, 0], A[1, 0]

    plt.plot(x, A[0, 0]*x+A[1, 0])
    plt.show()
