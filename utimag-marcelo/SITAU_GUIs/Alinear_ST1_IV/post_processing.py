import numpy as np
import matplotlib.pyplot as plt
import utils
from mpl_toolkits.mplot3d import Axes3D
import utils
from imag3D import utils_3d
from scipy.spatial import transform as tra

# %% section 1
plt.ion()

# data_A_scan = np.load(r"C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_III\data_A_scan_2deg_step.npy")
# data_degrees = np.load(r"C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_III\data_A_scan_2deg_step_degrees.npy")
# data_degrees = np.degrees(data_degrees)
# idx = utils.first_thr_cross(data_A_scan, [400, 1200], 50, 10)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x, y = np.meshgrid(np.degrees(data_degrees[:, 0]), np.degrees(data_degrees[:, 1]))
# z = idx[:, 0, 0]
# #
# ax.scatter(data_degrees[:, 0], data_degrees[:, 1], z)
# ax.scatter(x, y, z)


data_A_scan = np.load(r"C:\Marcelo\utimag-marcelo\SITAU_GUIs\Alinear_ST1_IV\A_scan_xy_I.npy")
data_degrees = np.load(r"C:\Marcelo\utimag-marcelo\SITAU_GUIs\Alinear_ST1_IV\A_scan_xy_I_degrees.npy")

# data_A_scan = np.load(r"C:\Marcelo\utimag-marcelo\SITAU_GUIs\Alinear_ST1_IV\A_scan_xy_II.npy")
# data_degrees = np.load(r"C:\Marcelo\utimag-marcelo\SITAU_GUIs\Alinear_ST1_IV\A_scan_xy_II_degrees.npy")

i_angs0 = np.where((data_degrees == [0,0]).all(axis=1))[0][0]


# data_A_scan = np.load(r"C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_III\A_scan_20deg_step_2deg_II.npy")
# data_degrees = np.load(r"C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_III\A_scan_20deg_step_2deg_II_degrees.npy")

# DEFINO UN RANGO DE UMBRALES
thr_min = 20;
thr_max = 80;
steps = 5

thresholds = np.arange(thr_min, thr_max + steps, steps)
print('BARRIDO XY:')
print(' ')
for thr in thresholds:
    idx_thr = utils.first_thr_cross(data_A_scan, [300, 1200], thr, 20, axis=-1)

    t_idx = idx_thr[:, :, 0]  # obtengo los índices de cada combinanciopn de angulos para cada punto

    mx_idx = np.max(t_idx, axis=1)
    mn_idx = np.min(t_idx, axis=1)

    dif_idx = np.abs(mx_idx - mn_idx)

    # aux_var = np.logical_and(np.abs(data_degrees[:, 0]) < 10, np.abs(data_degrees[:,1] < 10))
    #
    # idx_min = np.argmin(dif_idx[aux_var], axis=0)

    idx_min = np.argmin(dif_idx, axis=0)

    ang_min = data_degrees[idx_min, :]
    ang_min = np.append(ang_min, [0.0])

    rot0 = tra.Rotation.from_euler('x', 180, degrees=True)
    rot1 = tra.Rotation.from_euler('xyz', ang_min, degrees=True)
    rot1_inv = rot1.inv()
    rot2 = rot1_inv * rot0
    angs = np.around(rot2.as_euler('xyz', degrees=True), 1)

    print('threshold: ' + str(thr) + ' with angle: ' + str(ang_min))
    print(angs)
    print('------------------------')


def pintar():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_degrees[:, 0], data_degrees[:, 1], dif_idx)
