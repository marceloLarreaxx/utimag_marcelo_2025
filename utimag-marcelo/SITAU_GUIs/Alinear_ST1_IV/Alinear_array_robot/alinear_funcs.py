import numpy as np
from scipy.spatial import transform as tra


def array_coords_to_TCP_0_coords(e, mov):
    """ Pasar de coordenadas en el sistema propio del array a cordenadas en TCP_0"""
    xyz_array_tcp_0 = np.array(mov[0:3])
    rot_array_tcp_0 = tra.Rotation.from_euler('xyz', mov[3:], degrees=True)
    e_tcp_0 = rot_array_tcp_0.apply(e) + xyz_array_tcp_0
    return e_tcp_0


def rotate_element(e, mov, rot):
    e_tcp_0 = array_coords_to_TCP_0_coords(e, mov)
    return rot.apply(e_tcp_0)


def calcula_delta_tof_un_elem(array_center, i_angs0, rot_list, c=1.48):
    """solo elemento central del array"""
    array_center_rot = np.array([q.apply(array_center) for q in rot_list])
    delta_z = array_center_rot[:, 2] - array_center_rot[i_angs0, 2]
    delta_tof_calc = delta_z / c
    return delta_tof_calc


def calcula_delta_tof_mov(mov, i_angs0, rot_list, e_array, c=1.48):
    """
    mov: [x0, y0, z0, rx, ry, rz]
    Dado un vector en sistema del array, se debe rotar  y luego trasladar segun "mov" para obtener
    sus coordenadas en el TCP_0. Una vez calculadas estas, se les aplican las rotaciones definidas
    por los euler_angs que se usaron en el experimento.
    """
    e_tcp_0 = array_coords_to_TCP_0_coords(e_array, mov)
    e_tcp_0_rot = np.array([q.apply(e_tcp_0) for q in rot_list])
    delta_z = e_tcp_0_rot[:, :, 2] - e_tcp_0_rot[i_angs0, :, 2]
    delta_tof_calc = delta_z.T / c  # transpongo pa que quede igual que delta_tof medido
    return delta_tof_calc
