# -*- coding: utf-8 -*-
"""
Created on Mon Apr 04 12:29:27 2016

@author: kodai
"""

from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
import time


if __name__ == "__main__":
    digits = load_digits(2)
    img_data, img_label = (digits.data, digits.target)  # データとラベルに分ける
    alpha = 0.5  # 学習率の設定

    w = np.zeros(len(img_data[0]))  # wの初期値の設定
    for x in range(100):
        for data, label in zip(img_data, img_label):
            g_x = np.dot(w, data.reshape(-1, 1))  # 識別を行う
            # 0が-1で1が1
            if label == 0:
                if g_x >= 0:
                    modify_w = w - (alpha * data)
            else:
                if g_x < 0:
                    modify_w = w + (alpha * data)
            w = modify_w

        # alpha = alpha - 0.001  # 学習率をだんだん減らす

    plt.matshow(w.reshape(8, 8), cmap=plt.cm.gray)
    plt.show()
