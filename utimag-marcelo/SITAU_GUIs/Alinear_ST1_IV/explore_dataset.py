import sys
sys.path.append(r'C:\Marcelo\utimag-marcelo')

import pickle

import numpy as np
from matplotlib import pyplot as plt

import utils
from imag2D.pintar import arcoiris_cmap
from imag3D.CNN_superficie.dataset.merge_datasets import merge
import imag3D.CNN_superficie.cnnsurf_funcs as fu
import imag3D.CNN_superficie.cnnsurf_plot as cnnplot

import sys
sys.path.append(r'C:\Marcelo\utimag-marcelo')

folder = r'C:\Marcelo\utimag-marcelo\SITAU_GUIs\Alinear_ST1_IV\pickles\\'
filename = ['plano_base_9tx_rf.pickle',
            'plano_base_9tx_rf_cercano.pickle',
            'esfera_19mm_9tx_rf.pickle',
            'cilindroU_inv_25mm_9tx_rf_1.pickle',
            'cilindroU_40mm_9tx_rf.pickle',
            'plano_fibra_9tx_rf.pickle',
            'cilindro_12mm_9tx_rf_3.pickle',
            'cilindro_12mm_9tx_rf_cercano.pickle',
            'cilindro_35mm_9tx_rf.pickle',
            'cilindroU_25mm_9tx_rf.pickle']


def load_data(filename):
    subdata = {}
    with open(folder + filename, 'rb') as f:
    # with open(filename, 'rb') as f:
        print(f)
        subdata = pickle.load(f)
        poses_c = pickle.load(f)
    subdata['filename'] = filename
    return subdata, poses_c


class ImageBrowser:
    def __init__(self, subdata, u, peco_interval, poses_c, set='test', figsize=(16, 10), gamma=10):

        self.current_index = 0
        self.subdata = subdata
        self.poses_c = poses_c[np.int16(subdata[set + '_idx'] / 9)]  # OJO!!!!!!!!!!!!
        self.fmc = subdata[set + '_fmc']
        self.t = subdata[set + '_idx']
        self.u = u
        self.set = set
        self.peco_interval = peco_interval
        self.num_images = self.fmc.shape[0]

        self.fig, self.ax = plt.subplots(figsize=figsize)

        n = np.arange(121)
        img = self.fmc[self.current_index, :, :, :].reshape(121, -1)
        idx = utils.first_thr_cross(img, peco_interval, u, window_max=10)
        vmax, vmin = img.max(), img.min()
        self.im1 = self.ax.imshow(img.T, cmap='seismic')
        self.im1.set_norm(cnnplot.SigmoidNorm(gamma, vmin, vmax))
        self.ax.set_aspect(0.1)
        self.l1 = self.ax.plot(idx[:, 0], '.', label='threshold', color='k')
        self.l2 = self.ax.plot(self.t[self.current_index, ...].flatten(), label='Ground Truth', color='g')
        self.ax.legend()
        plt.colorbar(self.im1)
        self.ax.set_title(self.subdata['filename'] + 2*'\n' + str(self.poses_c[self.current_index, :]))

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.show()

    def on_key_press(self, event):
        if event.key == 'right':
            self.next_image()
        elif event.key == 'left':
            self.prev_image()

    def next_image(self):
        self.current_index = (self.current_index + 1) % self.num_images
        self.update_image()

    def prev_image(self):
        self.current_index = (self.current_index - 1) % self.num_images
        self.update_image()

    def update_image(self):
        img = self.fmc[self.current_index, :, :, :].reshape(121, -1)
        idx = utils.first_thr_cross(img, self.peco_interval, self.u, window_max=10)

        self.im1.set_array(img.T)
        self.l1[0].set_ydata(idx[:, 0])
        self.l2[0].set_ydata(self.t[self.current_index, ...].flatten())
        self.ax.set_title(self.subdata['filename'] + 2*'\n' + str(self.poses_c[self.current_index, :]))
        self.fig.canvas.draw()

