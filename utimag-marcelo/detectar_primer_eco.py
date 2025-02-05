import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import utils
from imag2D import ifaz
plt.style.use(r'C:\Users\ggc\PROYECTOS\Paper_array_virtual_3D\figuras\estilo.txt')

# cargar config file
cfg = utils.load_cfg(os.path.dirname(os.path.realpath(__file__)) + '/')

# Primer eco
nel_x, nel_y = cfg['nel_x'], cfg['nel_y']
diag = np.arange(0, cfg['n_elementos'])
umbral = None if cfg['umbral'] == 0 else cfg['umbral']

mat_vars = loadmat(cfg['matfile'])
temp = mat_vars['FMC_crudo'].astype(np.float32)
cfg['t_start'] = mat_vars['ret_ini'][0, 0]

# swapear y simetrizar
temp = temp.swapaxes(0, 2)
temp += temp.swapaxes(0, 1)
matrix = temp / 2
peco = matrix[diag, diag, :]  # matriz saft, pulso eco
cfg['n_samples'] = matrix.shape[-1]

# peco_rms = utils.rolling_rms(peco, 5)
q = utils.detect_first_echo(peco, cfg['peco_interval'], cfg['umbral'], cfg['n_flanco'])

# colores del default color cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

labels = ['flanco -', 'flanco +', 'umbral absoluto', 'umbral relativo', 'máximo']
plt.figure()
plt.plot(q[:, 0:5], 'o', label=labels)
plt.legend()


def pecoplot(i):
    plt.figure()
    plt.plot(peco[i, :], 'k.-', markersize=3)
    for v in range(5):
        plt.vlines(q[i, v], -50, 50, colors=colors[v], label=labels[v])
    plt.ylim(-100, 100)
    plt.legend()