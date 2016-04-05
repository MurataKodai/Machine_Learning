# -*- coding: utf-8 -*-
"""
Created on Mon Apr 04 12:29:27 2016

@author: kodai
"""

from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    digits = load_digits(2)
    img_data, img_label = (digits.data, digits.target)  # データとラベルに分ける
    alpha = 0.5  # 学習率の設定

    w = np.zeros(len(img_data[0]))  # wの初期値の設定

    for x in range(50):
        n_data = len(img_data)
        for data, label in zip(img_data, img_label):
            g_x = np.dot(w, data.reshape(-1, 1))  # 識別を行う
            # 0が-1で1が1
            if label == 0:
                if g_x >= 0:  # labelが0のときに識別関数g(x)が正の場合wを修正
                    modify_w = w - (alpha * data)
                    n_data = n_data - 1
            else:
                if g_x < 0:  # labelが1のときに識別関数g(x)が負の場合wを修正
                    modify_w = w + (alpha * data)
                    n_data = n_data - 1
            w = modify_w
        if n_data == 360:  # 一度も修正がなければ終わり
            break
        # alpha = alpha - 0.001  # 学習率をだんだん減らす

    plt.matshow(w.reshape(8, 8), cmap=plt.cm.gray)
    plt.show()
