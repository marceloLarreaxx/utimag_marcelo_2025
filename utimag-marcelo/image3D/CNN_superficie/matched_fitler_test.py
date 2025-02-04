import sys
sys.path.append(r'C:\Marcelo\utimag-marcelo')

import numpy as np
from scipy.ndimage import correlate1d
import pickle
import utils
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

import obspy

plt.ion()

filename = \
    r'C:\Marcelo\utimag-marcelo\imag3D\CNN_superficie\pickles_train\plano_base_9tx_rf.pickle'

with open(filename, 'rb') as f:
    subdata = pickle.load(f)

# parametros para general el pulso gaussiano /  kernel para la correlacion
freq = 3
bw = 0.8 # bandwith fraccional
fs = 40
t0 = 0
t_max = 4/freq # define el ancho/ nro de muestras del kernel
kernel = utils.gaussian_pulse(freq, bw, t0, -t_max, t_max, fs)

fig, ax = plt.subplots()
ax.plot(kernel.real)

q = correlate1d(subdata['train_fmc'], kernel, axis=-1) # hace la correlacion

# busca el máximo en un intervalo (t1, t2)
t1, t2 = 0, 700
t_idx = np.argmax(np.abs(q[:, :, :, t1:t2]), axis=-1) + t1

# pinta cosas
i, j, k = 150, 0, 0

fig2, ax2 = plt.subplots(1, 2)
ax2[0].imshow(subdata['train_fmc'][i, :, :, :].reshape((121, -1)).T)
ax2[0].set_aspect(0.1)
ax2[1].imshow(np.abs(q[i, :, :, :]).reshape((121, -1)).T)
ax2[1].set_aspect(0.1)
ax2[1].plot(t_idx[i, :, :].reshape((121,)))
ax2[0].plot(t_idx[i, :, :].reshape((121,)))

fig3, ax3 = plt.subplots()
ax3.plot(subdata['train_fmc'][i, j, k, :])
ax3.plot(np.abs(q[i, j, k, :]))