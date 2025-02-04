import sys

sys.path.append(r'/\\')
import os
import time
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from importlib import reload
import utils
from imag3D import utils_3d
from scipy.optimize import least_squares
from scipy.spatial import transform as tra
import SITAU_GUIs.Alinear_array_robot.alinear_funcs as ali
import pickle

TCP_OFFSET_0 = [31, 0.1, 262, 180, 0, 0]

elems_idx = [0, 10, 60, 110, 120]  # tienen que estar en el orden en que están los ascans adquiridos
# coordenadas de los elementos en el sistema del array
e_array = utils_3d.array_coordinates_list(11, 11, 1, 1)
# seleccionar solo los elementos utilizados
e_array = np.array(e_array)[elems_idx]

# ángulos de euler de las rotaciones, son extrínsecas 'xy'

data_A_scan = np.load(r"C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\A_scan_xy_II.npy")
data_degrees = np.load(r"C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\A_scan_xy_II_degrees.npy")

i_angs0 = int(data_degrees.shape[0] / 2)  # indice tal que data_degrees[i0, :] = [0, 0]
rot_list = [tra.Rotation.from_euler('xy', data_degrees[i, :], degrees=True) for i in range(data_degrees.shape[0])]

# ascan_rot = mat_vars['ascan_rot']
# if cfg['no_usar_central']:
#     ascan_rot = ascan_rot[:, [0, 1, 3, 4], :]
#     e_array = e_array[[0, 1, 3, 4]]

idx_thr = utils.first_thr_cross(data_A_scan, [500, 1200], 30, 20, axis=-1)
# idx_thr [3, nro de elemento, nro de pose], poner los ejes en este orden
idx_thr = np.swapaxes(idx_thr, 0, 2)

# idx_thr = mat_vars['idx_thr'].astype('float')
idx_thr_angs0 = np.expand_dims(idx_thr[0, :, i_angs0], -1)

# OJO EL SIGNO !!!
fs = 40
delta_tof = (idx_thr[0, :, :] - idx_thr_angs0) / (2 * fs)  # shape(indice de elemento, indice de pose)
weight = idx_thr[1, :, :]  # > cfg['umbral_weight']


#
# def array_coords_to_TCP_0_coords(e, mov):
#     """ Pasar de coordenadas en el sistema propio del array a cordenadas en TCP_0"""
#     xyz_array_tcp_0 = np.array(mov[0:3])
#     rot_array_tcp_0 = tra.Rotation.from_euler('xyz', mov[3:], degrees=True)
#     e_tcp_0 = rot_array_tcp_0.apply(e) + xyz_array_tcp_0
#     return e_tcp_0
#
#
# def rotate_element(e, mov, rot):
#     e_tcp_0 = array_coords_to_TCP_0_coords(e, mov)
#     return rot.apply(e_tcp_0)
#
#
# def calcula_delta_tof(mov):
#     """
#     mov: [x0, y0, z0, rx, ry, rz]
#     Dado un vector en sistema del array, se debe rotar  y luego trasladar segun "mov" para obtener
#     sus coordenadas en el TCP_0. Una vez calculadas estas, se les aplican las rotaciones definidas
#     por los data_degrees que se usaron en el experimento.
#     """
#     e_tcp_0 = array_coords_to_TCP_0_coords(e_array, mov)
#     e_tcp_0_rot = np.array([q.apply(e_tcp_0) for q in rot_list])
#     delta_z = e_tcp_0_rot[:, :, 2] - e_tcp_0_rot[i_angs0, :, 2]
#     delta_tof_calc = delta_z.T / cfg['c1']  # transpongo pa que quede igual que delta_tof medido
#     return delta_tof_calc


def resid_fun_1(mov):
    return (weight * (ali.calcula_delta_tof_mov(mov, i_angs0, rot_list, e_array) - delta_tof)).flatten()


def resid_fun_2(x):
    # solo elemento central
    return (weight * (ali.calcula_delta_tof_un_elem(x, i_angs0, rot_list) - delta_tof[2, :])).flatten()


result_1 = least_squares(resid_fun_1, [0, 0, 0, 180, 0, 0], max_nfev=1000)
result_2 = least_squares(resid_fun_2, [0, 0, 0], max_nfev=1000)
# tcp_1_pose = np.concatenate([result_1.x[0:3], tra.Rotation.from_euler('xyz', result_1.x[3:], degrees=True).as_rotvec()], 0)
print(np.around(result_1.x, 2))
print(np.around(result_2.x, 2))

idx_std = np.std(idx_thr[0, :, :], axis=0)
i_min = np.argmin(idx_std)
plt.figure()
plt.plot(idx_std)

rot0 = tra.Rotation.from_euler('xyz', TCP_OFFSET_0[3:], degrees=True)
rot_mov = tra.Rotation.from_euler('xyz', result_1.x[3:], degrees=True)
rot1 = rot_mov * rot0

nuevos_angs = rot1.as_euler('xyz', degrees=True)

TCP_OFFSET_1 = TCP_OFFSET_0.copy()
TCP_OFFSET_1[0:3] += result_2.x


def plot_delta_tof_mov(i, mov=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sca = ax.scatter(data_degrees[:, 0], data_degrees[:, 1], delta_tof[i, :], c=idx_thr[1, i, :])
    if mov is not None:
        delta_tof_model = ali.calcula_delta_tof_mov(mov, i_angs0, rot_list, e_array)
        sca_model = ax.scatter(data_degrees[:, 0], data_degrees[:, 1], delta_tof_model[i, :])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sca = ax.scatter(data_degrees[:, 0], data_degrees[:, 1], idx_thr[1, i, :], c=idx_thr[1, i, :])


def plot_delta_tof_un_elem(i_center=2, array_center=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sca = ax.scatter(data_degrees[:, 0], data_degrees[:, 1], delta_tof[i_center, :], c=idx_thr[1, i_center, :])
    if array_center is not None:
        delta_tof_model = ali.calcula_delta_tof_un_elem(array_center, i_angs0, rot_list)
        sca_model = ax.scatter(data_degrees[:, 0], data_degrees[:, 1], delta_tof_model)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sca = ax.scatter(data_degrees[:, 0], data_degrees[:, 1], idx_thr[1, i_center, :], c=idx_thr[1, i_center, :])


def plot_delta_tof_model(i, mov):
    delta_tof_model = ali.calcula_delta_tof_mov(mov, i_angs0, rot_list, e_array)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    sca = ax.scatter(data_degrees[:, 0], data_degrees[:, 1], delta_tof_model[i, :])


def plot_ascan(elem_idx, i):
    fig, ax = plt.subplots()
    ax.plot(np.abs(ascan_rot[:, elem_idx, i]))
    ax.hlines(cfg['umbral'], xmin=0, xmax=cfg['n_samples'])
    ax.vlines(idx_thr[0, elem_idx, i], ymin=0, ymax=idx_thr[1, elem_idx, i], colors='lime')
    # ax.vlines(results[i]['idx'][j, 0] + window_num, ymin=0, ymax=results[i]['idx'][j, 1], colors='lime')
