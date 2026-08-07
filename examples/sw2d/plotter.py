import numpy as np
import matplotlib.pyplot as plt
import os

plt.ion()

npys = ['uums.npy', 'hb.npy', 'vvms.npy', 'hhms.npy']

def plot_field(n, t=0):

    field = np.load(npys[n])
    if field.ndim == 3:
        field = field[t]

    plt.close('all')
    plt.figure(figsize=(6, 5))
    plt.imshow(field)
    plt.show()