import sys
sys.path.append(r'C:\Marcelo\utimag-marcelo')

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider
from imag2D.pintar import arcoiris_cmap
import imag3D.CNN_superficie.cnnsurf_funcs as fu
from scipy.ndimage import correlate1d
from obspy.signal.trigger import recursive_sta_lta, plot_trigger, classic_sta_lta
import matplotlib.gridspec as gridspec


class SigmoidNorm(Normalize):
    def __init__(self, gamma, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        self.gamma = gamma
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        q = np.interp(value, x, y)
        return 1/(1 + np.exp(-self.gamma*(q - 0.5)))


def plot_input(data, i, t_idx, offset=0):
    fig, ax = plt.subplots()
    n_samples = data.shape[-2]
    aux = data[i, :11, :11, :, 0].reshape((121, n_samples))
    ax.imshow(aux.T, cmap=arcoiris_cmap)
    ax.set_aspect(0.1)
    ax.plot(t_idx[i, :, :].reshape(121, 1) + offset, 'k')


def plot_input_and_labels(data, labels, i):
    fig, ax = plt.subplots(1, 2)
    n_samples = data.shape[-2]
    aux1 = data[i, :11, :11, :].reshape((121, n_samples))
    ax[0].imshow(aux1.T, cmap=arcoiris_cmap)
    ax[0].set_aspect(0.1)
    aux2 = labels[i, :11, :11, :].reshape((121, n_samples))
    ax[1].imshow(aux2.T, cmap=arcoiris_cmap)
    ax[1].set_aspect(0.1)


def plot_model_inference(data, i, t_idx, model):
    fig, ax = plt.subplots(1, 2)
    n_samples = data.shape[-2]
    aux1 = data[i, :11, :11, :].reshape((121, n_samples))
    ax[0].imshow(aux1.T, cmap=arcoiris_cmap)
    ax[0].set_aspect(0.1)
    x = np.expand_dims(data[i, :, :, :, :], axis=0)
    y = model(x).numpy()  # trasnformar a array numpy
    y = y[0, :11, :11, :].reshape((121, n_samples))
    ax[1].imshow(y.T, cmap=arcoiris_cmap)
    ax[1].set_aspect(0.1)
    ax[1].plot(t_idx[i, :, :].reshape(121, 1), 'k')


class ImageBrowser:
    def __init__(self, model, fmc, u1, u2, peco_interval, t, sta_lta1, sta_lta2, methods_to_show, leg_y=1.1, figsize=(16, 10), vnet_thr_inverso=False, gamma=10, gt=True, ci=0):
        self.current_index = ci
        self.model = model
        self.fmc = fmc
        self.u1 = u1
        self.u2 = u2
        self.peco_interval = peco_interval
        self.t = t
        self.num_images = fmc.shape[0]
        self.vnet_thr_inverso = vnet_thr_inverso
        self.sta_lta1 = sta_lta1
        self.sta_lta2 = sta_lta2
        self.methos_to_show = methods_to_show
        self.leg_y = leg_y
        self.gt = gt

        self.fig, self.ax = plt.subplots(1, 2, figsize=figsize)
        q = fu.return_example(self.model, self.current_index, self.fmc, self.u1, self.u2, self.peco_interval,
                               self.sta_lta1, self.sta_lta2, vnet_thr_inverso=self.vnet_thr_inverso)

        self.m1, self.m2, self.m3, self.m4, self.m5 = self.methos_to_show

        n = np.arange(121)
        vmax, vmin = q[0].max(), q[0].min()
        self.im1 = self.ax[0].imshow(q[0].T, cmap='seismic')
        self.im1.set_norm(SigmoidNorm(gamma, vmin, vmax))
        self.ax[0].set_aspect(0.1)
        if self.m1 == 1:
            self.l1 = self.ax[0].plot(n[q[2]], q[1][q[2]], '.', label='threshold', color='k')
        if self.m2 == 1:
            self.l2 = self.ax[0].plot(n[q[5]], q[4][q[5]], 'v', label='V-net', color='darkorange')
        if self.m3 == 1:
            self.l3 = self.ax[0].plot(n[q[8]], q[7][q[8]], '.', label='Matched-Filter', color='k')
        if self.m4 == 1:
            self.l4 = self.ax[0].plot(n[q[11]], q[10][q[11]], '.', label='STA/LTA (1) = {}'.format(str(self.sta_lta1)), color='k')
        if self.m5 == 1:
            self.l14 = self.ax[0].plot(n[q[13]], q[12][q[13]], '.', label='STA/LTA (2) = {}'.format(str(self.sta_lta2)), color='k')
        if self.gt:
            self.l5 = self.ax[0].plot(self.t[self.current_index, ...].flatten(), label='Ground Truth', color='g')
        self.ax[0].set_title('Índice actual: {}'.format(str(self.current_index)), fontsize=16)
        self.ax[0].legend(fontsize=13, bbox_to_anchor=(1.05, self.leg_y))
        self.im2 = self.ax[1].imshow(q[3].T, cmap=arcoiris_cmap)
        self.ax[1].set_aspect(0.1)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.ax[0].set_xlabel("Índice del elemento del arreglo", fontsize=16)
        self.ax[0].set_ylabel("Número de muestra", fontsize=16)
        self.ax[1].set_xlabel("Índice del elemento del arreglo", fontsize=16)
        self.ax[1].set_ylabel("Número de muestra", fontsize=16)

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
        q = fu.return_example(self.model, self.current_index, self.fmc, self.u1, self.u2, self.peco_interval,
                              self.sta_lta1, self.sta_lta2, vnet_thr_inverso=self.vnet_thr_inverso)
        n = np.arange(121)
        self.im1.set_array(q[0].T)
        self.im2.set_array(q[3].T)
        if self.m1 == 1:
            self.l1[0].set_xdata(n[q[2]])
            self.l1[0].set_ydata(q[1][q[2]])
        if self.m2 == 1:
            self.l2[0].set_xdata(n[q[5]])
            self.l2[0].set_ydata(q[4][q[5]])
        if self.m3 == 1:
            self.l3[0].set_xdata(n[q[8]])
            self.l3[0].set_ydata(q[7][q[8]])
        if self.m4 == 1:
            self.l4[0].set_xdata(n[q[11]])
            self.l4[0].set_ydata(q[10][q[11]])
        if self.m5 == 1:
            self.l14[0].set_xdata(n[q[13]])
            self.l14[0].set_ydata(q[12][q[13]])
        self.ax[0].set_title('Índice actual: {}'.format(str(self.current_index)))
        if self.gt:
            self.l5[0].set_ydata(self.t[self.current_index, ...].flatten())
        self.fig.canvas.draw()


class ImageBrowserAscan:
    def __init__(self, model, fmc, u1, u2, peco_interval, t, sta_lta1, sta_lta2,  el, methods_to_show, leg_y=1.2, figsize=(16, 10),vnet_thr_inverso=False, gamma=10, ci=0):

        self.current_index = ci
        self.model = model
        self.fmc = fmc
        self.u1 = u1
        self.u2 = u2
        self.peco_interval = peco_interval
        self.t = t
        self.num_images = fmc.shape[0]
        self.sta_lta1 = sta_lta1
        self.sta_lta2 = sta_lta2
        self.el = el
        self.vnet_thr_inverso = vnet_thr_inverso
        self.methos_to_show = methods_to_show
        self.leg_y = leg_y

        self.m1, self.m2, self.m3, self.m4, self.m5 = self.methos_to_show

        # self.fig, self.ax = plt.subplots(3, 1, figsize=figsize)
        self.fig = plt.figure(figsize=figsize)

        self.gs = gridspec.GridSpec(6, 2, height_ratios=[1,1,1,1,1,1], wspace=0.1, hspace=1.0)

        q0 = fu.return_example(self.model, self.current_index, self.fmc, self.u1, self.u2, self.peco_interval,
                              self.sta_lta1, self.sta_lta2, vnet_thr_inverso=self.vnet_thr_inverso)

        q = fu.return_example_ascan(self.current_index, self.fmc, self.peco_interval, self.sta_lta1, self.sta_lta2, self.u1)

        self.ax1 = self.fig.add_subplot(self.gs[0,0])
        self.l1 = self.ax1.plot(q[0][self.el, :], label='A-scan', color='k')  # A-scan
        self.ax1.set_ylim(np.min(q[0][self.el, :]) - 20, np.max(q[0][self.el, :])+ 20)
        self.ax1.legend()
        self.ax1.set_title('Elemento: {} / Índice actual: {}'.format(str(self.el), str(self.current_index)))

        self.ax2 = self.fig.add_subplot(self.gs[1, 0])
        self.l2 = self.ax2.plot(np.abs(q[1][self.el, :]), label='Matched Filter', color='red')  # matched filter
        self.ax2.set_ylim(np.min(np.abs(q[1][self.el, :])) -100 , np.max(q[1][self.el, :]) + 50)
        self.ax2.legend()

        self.ax3 = self.fig.add_subplot(self.gs[2, 0])
        self.ax4 = self.fig.add_subplot(self.gs[3, 0])
        if self.m4 == 1:
            self.l3 = self.ax3.plot(np.abs(q[2][self.el, :]), label='STA/LTA (1) = {}'.format(str(self.sta_lta1)),  color='k')  # sta/lta 1
            self.l5 = self.ax4.plot(np.abs(q[4][self.el, :]),  label='STA/LTA derivative (1) = {}'.format(str(self.sta_lta1)), color='k')  # sta/lta 1

        self.ax3.legend(fontsize=7)


        if self.m5 == 1:
            self.l4 = self.ax3.plot(np.abs(q[3][self.el, :]), label='STA/LTA (2) = {}'.format(str(self.sta_lta2)),  color='k')  # sta/lta 2
            self.l6 = self.ax4.plot(np.abs(q[5][self.el, :]), label='STA/LTA derivative (2) = {}'.format(str(self.sta_lta2)), color='k')  # sta/lta 2
        self.ax4.legend(fontsize=7)

        self.ax5 = self.fig.add_subplot(self.gs[4, 0])
        self.l7 = self.ax5.plot(np.abs(q0[3][self.el, :]), label='V-net', color='darkorange')  # sta/lta 1
        self.ax5.legend()

        self.ax6 = self.fig.add_subplot(self.gs[5, 0])
        self.l8 = self.ax6.plot(np.abs(q[6][self.el, :]), label='Threshold Cross', color='k')  # threshold cross
        self.l9 = self.ax6.axvline(q[7][self.el], color='g', linestyle='--')  # threshold cross
        self.ax6.legend()

        ################################################################################################################
        n = np.arange(121)
        vmax, vmin = q0[0].max(), q0[0].min()
        self.ax7 = self.fig.add_subplot(self.gs[0:3, 1])
        self.im1 = self.ax7.imshow(q0[0].T, cmap='seismic')
        self.im1.set_norm(SigmoidNorm(gamma, vmin, vmax))
        self.ax7.set_aspect(0.1)


        if self.m1 == 1:
            self.l10 = self.ax7.plot(n[q0[2]], q0[1][q0[2]], '.', label='threshold', color='k')
        if self.m2 == 1:
            self.l11 = self.ax7.plot(n[q0[5]], q0[4][q0[5]], 'v', label='V-net', color='darkorange')
        if self.m3 == 1:
            self.l12 = self.ax7.plot(n[q0[8]], q0[7][q0[8]], '.', label='Matched-Filter', color='k')
        if self.m4 == 1:
            self.l13 = self.ax7.plot(n[q0[11]], q0[10][q0[11]], '.', label='STA/LTA (1) = {}'.format(str(self.sta_lta1)), color='k')
        if self.m5 == 1:
            self.l14 = self.ax7.plot(n[q0[13]], q0[12][q0[13]], '.', label='STA/LTA (2) = {}'.format(str(self.sta_lta2)), color='k')
        self.l15 = self.ax7.plot(self.t[self.current_index, ...].flatten(), label='Ground Truth', color='g')
        plt.colorbar(self.im1)

        self.ax7.legend(fontsize=13, bbox_to_anchor=(1.04, self.leg_y))
        plt.tight_layout()

        self.ax8 = self.fig.add_subplot(self.gs[3:, 1])
        self.im2 = self.ax8.imshow(q0[3].T, cmap=arcoiris_cmap)
        self.ax8.set_aspect(0.1)
        ################################################################################################################

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
        q0 = fu.return_example(self.model, self.current_index, self.fmc, self.u1, self.u2, self.peco_interval,
                               self.sta_lta1, self.sta_lta2, vnet_thr_inverso=self.vnet_thr_inverso)
        q = fu.return_example_ascan(self.current_index, self.fmc, self.peco_interval, self.sta_lta1, self.sta_lta2, self.u1)

        self.l1[0].set_ydata(q[0][self.el, :])
        self.ax1.set_ylim(np.min(q[0][self.el, :]) - 20, np.max(q[0][self.el, :]) + 20)

        self.l2[0].set_ydata(np.abs(q[1][self.el, :]))
        self.ax2.set_ylim(np.min(np.abs(q[1][self.el, :])) -100, np.max(q[1][self.el, :]) + 50)

        if self.m4 == 1:
            self.l3[0].set_ydata(np.abs(q[2][self.el, :]))
            self.l5[0].set_ydata(np.abs(q[4][self.el, :]))

        if self.m5 == 1:
            self.l4[0].set_ydata(np.abs(q[3][self.el, :]))
            self.l6[0].set_ydata(np.abs(q[5][self.el, :]))

        self.l7[0].set_ydata(np.abs(q0[3][self.el, :]))

        self.l8[0].set_ydata(np.abs(q[6][self.el, :]))
        self.l9.set_xdata(np.abs(q[7][self.el]))
        self.ax6.set_ylim(-0.2, 1.2)

        # ---------------------------------------------------------------------------------------------------------------
        n = np.arange(121)
        self.im1.set_array(q0[0].T)
        self.im2.set_array(q0[3].T)
        if self.m1 == 1:
            self.l10[0].set_xdata(n[q0[2]])
            self.l10[0].set_ydata(q0[1][q0[2]])
        if self.m2 == 1:
            self.l11[0].set_xdata(n[q0[5]])
            self.l11[0].set_ydata(q0[4][q0[5]])
        if self.m3 == 1:
            self.l12[0].set_xdata(n[q0[8]])
            self.l12[0].set_ydata(q0[7][q0[8]])
        if self.m4 == 1:
            self.l13[0].set_xdata(n[q0[11]])
            self.l13[0].set_ydata(q0[10][q0[11]])
        if self.m5 == 1:
            self.l14[0].set_xdata(n[q0[13]])
            self.l14[0].set_ydata(q0[12][q0[13]])
        self.l15[0].set_ydata(self.t[self.current_index, ...].flatten())
        # ---------------------------------------------------------------------------------------------------------------

        self.ax1.set_title('Elemento: {} / Índice actual: {}'.format(str(self.el), str(self.current_index)))
        self.fig.canvas.draw()

