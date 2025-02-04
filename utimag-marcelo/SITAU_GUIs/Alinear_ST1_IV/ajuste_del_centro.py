import sys
sys.path.append(r'/\\')
#import os
#import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#from scipy.io import loadmat
#from importlib import reload
import utils
#from imag3D import utils_3d
from scipy.optimize import least_squares
from scipy.spatial import transform as tra
from SITAU_GUIs.Alinear_ST1_IV import alinear_funcs as ali
from scipy.interpolate import griddata

data_A_scan = np.load(r"C:\Marcelo\utimag-marcelo\SITAU_GUIs\Alinear_ST1_IV\A_scan_xy_I.npy")
euler_angs = np.load(r"C:\Marcelo\utimag-marcelo\SITAU_GUIs\Alinear_ST1_IV\A_scan_xy_I_degrees.npy")

i_angs0 = np.where((euler_angs == [0, 0]).all(axis=1))[0][0]  # indice tal que euler_angs[i0, :] = [0, 0]

rot_list = [tra.Rotation.from_euler('xy', euler_angs[i, :], degrees=True) for i in range(euler_angs.shape[0])]

idx_thr = utils.first_thr_cross(data_A_scan, [300, 1200], 50, 20, axis=-1)

idx_thr_angs0 = np.expand_dims(idx_thr[i_angs0,:,0], 0)

fs = 40  # frecuencia de muestreo

delta_tof = (-1)*(idx_thr[:, :,0] - idx_thr_angs0) / (2*fs)

weight = idx_thr[:, :,1]
weight = weight[:,2] # solamente del elemento central

x = [30, 0, 262, 180, 0, 0]; # es el TCP_offset inicial (sin correcciones)
x = x[0:3]  # solamente nos interesa las coordenadas de traslación

def resid_fun_2(u):
    # solo elemento centrals
     return (weight * (ali.calcula_delta_tof_un_elem(u, i_angs0, rot_list) - delta_tof[:,2])).flatten()


result_2 = least_squares(resid_fun_2, [0, 0, 0], max_nfev=1000)
print(np.around(result_2.x, 2))

idx_std = np.std(idx_thr[:, :, 0], axis=0)
i_min = np.argmin(idx_std)
plt.figure()
plt.plot(idx_std)

x_adjusted = list(x + result_2.x)
delta_tof_corr = ali.calcula_delta_tof_un_elem(result_2.x, i_angs0, rot_list)  # tof_calculados a partir de coordenadas corregidas


def plot_delta_tof_un_elem(i_center=2, array_center=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sca = ax.scatter(euler_angs[:, 0], euler_angs[:, 1], delta_tof[:, i_center], c = idx_thr[:, i_center, 1])
    if array_center is not None:
        delta_tof_model = ali.calcula_delta_tof_un_elem(array_center, i_angs0, rot_list)
        sca_model = ax.scatter(euler_angs[:, 0], euler_angs[:, 1], delta_tof_model)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sca = ax.scatter(euler_angs[:, 0], euler_angs[:, 1], idx_thr[:, i_center, 1], c=idx_thr[:, i_center, 1])



def plot_delta_tof_model(delta_tof_exp, modelo=None, alph=0.02, i_center=2):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # delta_tof_model_1 = ali.calcula_delta_tof_un_elem(mov, i_angs0, rot_list)
    # delta_tof_model_2 = ali.calcula_delta_tof_un_elem(mov_adjusted, i_angs0, rot_list)

    if modelo is not None:
        xi, yi = np.meshgrid(euler_angs[:, 0], euler_angs[:, 1])
        zi = griddata((euler_angs[:, 0], euler_angs[:, 1]), modelo , (xi, yi), method='linear')
        ax.plot_surface(xi, yi, zi, color='gray', alpha=alph)


   # ax.scatter(euler_angs[:, 0], euler_angs[:, 1], delta_tof_exp, c=idx_thr[:, i_center, 1])
    ax.scatter(euler_angs[:, 0], euler_angs[:, 1], delta_tof_exp)
    #ax.scatter(euler_angs[:, 0], euler_angs[:, 1], delta_tof_adjusted, c=idx_thr[:, i_center, 1])

    ax.set_ylabel(r'$\theta$y')
    ax.set_xlabel(r'$\theta$x')
    ax.set_zlabel(r'$\Delta{TDV}$ vs $\Delta{TDV}^{*}$')


def plot_delta_tof_model_2(delta_tof_exp, modelo, alph=0.02, i_center=2):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # delta_tof_model_1 = ali.calcula_delta_tof_un_elem(mov, i_angs0, rot_list)
    # delta_tof_model_2 = ali.calcula_delta_tof_un_elem(mov_adjusted, i_angs0, rot_list)


   # ax.scatter(euler_angs[:, 0], euler_angs[:, 1], delta_tof_exp, c=idx_thr[:, i_center, 1])
    ax.scatter(euler_angs[:, 0], euler_angs[:, 1], delta_tof_exp)
    ax.scatter(euler_angs[:, 0], euler_angs[:, 1], modelo, c='red')
    ax.set_ylabel(r'$\theta$y')
    ax.set_xlabel(r'$\theta$x')
    ax.set_zlabel(r'$\Delta{TDV}$ vs $\Delta{TDV}^{*}$')


def plot_delta_tof_un_elem(i_center=2, array_center=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sca = ax.scatter(euler_angs[:, 0], euler_angs[:, 1], delta_tof[i_center, :], c=idx_thr[1, i_center, :])
    if array_center is not None:
        delta_tof_model = ali.calcula_delta_tof_un_elem(array_center, i_angs0, rot_list)
        sca_model = ax.scatter(euler_angs[:, 0], euler_angs[:, 1], delta_tof_model)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sca = ax.scatter(euler_angs[:, 0], euler_angs[:, 1], idx_thr[1, i_center, :], c=idx_thr[1, i_center, :])



