import numpy as np
import matplotlib.pyplot as plt
# import utils
# from mpl_toolkits.mplot3d import Axes3D
import utils
# from imag3D import utils_3d
# from scipy.spatial import transform as tra


###################### FUNCIONES ######################
def pintar(data_degrees, dif_idx):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_degrees[:, 0], data_degrees[:, 1], dif_idx)


def ang_min(data_a_scan, data_degrees, threshold, rango):
    idx_thr = utils.first_thr_cross(data_a_scan, rango, threshold, 20, axis=-1)
    t_idx = idx_thr[:, :, 0]  # obtengo los índices de corte de umbral

    mx_idx = np.max(t_idx, axis=1)
    mn_idx = np.min(t_idx, axis=1)

    dif_idx = np.abs(mx_idx - mn_idx)  # estimando error
    idx_min = np.argmin(dif_idx, axis=0)
    angmin = data_degrees[idx_min, :]

    return angmin, dif_idx


def get_list_of_positions(n, dz, rx, ry):
    dz_range = np.random.uniform(dz[0], dz[1], n)
    rx_range = np.random.uniform(rx[0], rx[1], n)
    ry_range = np.random.uniform(ry[0], ry[1], n)
    pose_combinations = list(zip(dz_range, rx_range, ry_range))
    return pose_combinations


def get_list_of_positions2(n, dz, dy, rz, ry):
    dz_range = np.random.uniform(dz[0], dz[1], n)
    dy_range = np.random.uniform(dy[0], dy[1], n)
    ry_range = np.random.uniform(ry[0], ry[1], n)
    rz_range = np.random.uniform(rz[0], rz[1], n)
    pose_combinations = list(zip(dz_range, dy_range, rz_range, ry_range))
    return pose_combinations


def get_list_of_positions3(n, dx, dy, dz):
    dx_range = np.random.uniform(dx[0], dx[1], n)
    dy_range = np.random.uniform(dy[0], dy[1], n)
    dz_range = np.random.uniform(dz[0], dz[1], n)
    pose_combinations = list(zip(dx_range, dy_range, dz_range))
    return pose_combinations


def filter_list(pose_combinations, z_min, ang_max):
    pose_combinations2 = []
    for pose_i in pose_combinations:
        if pose_i[0] <= z_min:
            if pose_i[1] <= ang_max and pose_i[2] <= ang_max:
                pose_combinations2.append(pose_i)
        else:
            pose_combinations2.append(pose_i)
    return pose_combinations2


def filter_list2(pose_combinations, z_min, ang_max, anglim1, anglim2):
    pose_combinations2 = []
    for pose_i in pose_combinations:
        if pose_i[0] <= z_min:
            if pose_i[3] <= ang_max:
                pose_combinations2.append(pose_i)
        else:
            pose_combinations2.append(pose_i)
    pose_combinations3 = []
    for pose_i in pose_combinations2:
        if pose_i[2] >= anglim1:
            if pose_i[3] <= anglim2:
                pose_combinations3.append(pose_i)
        else:
            pose_combinations3.append(pose_i)
    return pose_combinations3


def sweep_z_rx_ry(pose, t=10):
    pass

###################### ---------- ######################
