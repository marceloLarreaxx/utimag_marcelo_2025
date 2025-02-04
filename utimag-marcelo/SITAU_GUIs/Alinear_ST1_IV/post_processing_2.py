import numpy as np
import matplotlib.pyplot as plt
# import utils
# from mpl_toolkits.mplot3d import Axes3D
# import utils
# from imag3D import utils_3d
from scipy.spatial import transform as tra
import methods

plt.ion()

# Cargo los archivos de A scan de barridos en XY & XZ

# DATOS 1:
# data_A_scan_xy = np.load(r"C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\Registros_extra\Alineacion 1\A_scan_xy_I.npy")
# data_degrees_xy = np.load(r"C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\Registros_extra\Alineacion 1\A_scan_xy_I_degrees.npy")
#
# data_A_scan_xz = np.load(r"C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\Registros_extra\Alineacion 1\A_scan_xz_I.npy")
# data_degrees_xz = np.load(r"C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\Registros_extra\Alineacion 1\A_scan_xz_I_degrees.npy")


# DATOS 3:
data_A_scan_xy = np.load(r"C:\Marcelo\utimag-marcelo\SITAU_GUIs\Alinear_ST1_IV\A_scan_xy_II.npy")
data_degrees_xy = np.load(r"C:\Marcelo\utimag-marcelo\SITAU_GUIs\Alinear_ST1_IV\A_scan_xy_II_degrees.npy")

data_A_scan_xz = np.load(r"C:\Marcelo\utimag-marcelo\SITAU_GUIs\Alinear_ST1_IV\A_scan_xz_III.npy")
data_degrees_xz = np.load(r"C:\Marcelo\utimag-marcelo\SITAU_GUIs\Alinear_ST1_IV\A_scan_xz_III_degrees.npy")

threshold = 50
rango_xy = (500, 800)
rango_xz = (300, 800)

ang_min_xy, dif_idx_xy = methods.ang_min(data_A_scan_xy, data_degrees_xy, threshold, rango_xy)
ang_min_xz, dif_idx_xz = methods.ang_min(data_A_scan_xz, data_degrees_xz, threshold, rango_xz)

rot0 = tra.Rotation.from_euler('x', 180,  degrees=True)  # primera rotación (TCP_offset0) rotación en eje x de 180 # grados
rot1 = tra.Rotation.from_euler('xy', ang_min_xy * np.array([1, -1]), degrees=True)  # segunda rotación (TCP_offset1) rotación en eje x & en eje y
rot2 = tra.Rotation.from_euler('z', (-1) * ang_min_xz[1], degrees=True)  # tercera rotación (TCP_offset2) rotación en eje z
rot1_inv = rot1.inv()  # Inverting the rotation
rot2_inv = rot2.inv()  # Inverting the rotation

rot_1_xy = rot1_inv * rot0 # PRIMERA ROTACIÓN (CORRECCIÓN INICIAL SOLO EN XY)
rotf2 = rot2_inv * rot1_inv * rot0  # SEGUNDA ROTACIÓN (CORRECCIÓN FINAL POSTERIOR AL BARRIDO EN XZ)


angs_xy_correcion = np.around(rot_1_xy.as_euler('xyz', degrees=True), 2)
angs2 = np.around(rotf2.as_euler('xyz', degrees=True), 2)

print('Euler angles xy(1): ' + str(angs_xy_correcion))
print('Euler angles (2): ' + str(angs2))


def pintar(data_degrees, dif_idx):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_degrees[:, 0], data_degrees[:, 1], dif_idx)
    ax.set_ylabel(r'$\theta$z')
    ax.set_xlabel(r'$\theta$x')
    ax.set_zlabel(r'$\Delta{TDV}_{i,j}$')


def pintar2D(data_degrees, dif_idx):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_degrees[:, 1], dif_idx)
    ax.set_xlabel(r'$\theta$z')
    ax.set_ylabel(r'$\Delta{TDV}_{i,j}$')
    ax.grid(True)