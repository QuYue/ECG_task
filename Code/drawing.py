# -*- coding: utf-8 -*-
"""
@Time    : 2019/4/3 18:41
@Author  : QuYue
@File    : drawing.py
@Software: PyCharm
Introduction: Drawing the result.
"""
#%% Import Packages
import matplotlib.pyplot as plt
import numpy as np
#%% Functions
def draw_result(result, fig, title=[], show=False):
    #  actionly draw the result
    num = len(result)
    # check
    if len(title) < num:
        for i in range(len(title), num):
            title.append(str(i))
    xaxis = [list(range(len(i))) for i in result] # axis -x
    subplot = []
    fig.clf()
    for i in range(num):
        subplot.append(fig.add_subplot(num, 1, i+1))
        subplot[i].plot(xaxis[i], result[i], marker='o')
        subplot[i].grid()
        subplot[i].set_title(title[i])
        subplot[i].set_ylim(0, 1)
        if show:
            subplot[i].annotate(s=title[i] + ': %.3f' % result[i][-1], xy=(xaxis[i][-1], result[i][-1]),
                                xytext=(-20, 10), textcoords='offset points')
            r = np.array(result[i])
            subplot[i].annotate(s='Max: %.3f' % r.max(), xy=(r.argmax(), r.max()), xytext=(-20, -10),
                             textcoords='offset points')
    plt.pause(0.01)
#%% Main Function
if __name__ == '__main__':
    fig = plt.figure(1)
    plt.ion()
    b = []
    c = []
    for i in range(100):
        a = np.random.randn(2)
        b.append(a[0])
        c.append(a[1])
        draw_result([b, c], fig, ['a', 'b'], True)
    plt.ioff()
    plt.show()
