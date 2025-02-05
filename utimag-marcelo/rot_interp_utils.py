import os
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import utils
import imag3D.utils_3d as u3d


def roi_array2lab_2d(roi_array, array_center, theta, pintar=False):

    # calcular coordenadas x,z de los vértices de roi_array
    # roi_array: [u1, u2, w2, w1]
    u1, u2, w2, w1 = roi_array

    # orden de los vertices
    # 0------------1
    # |            |
    # |            |
    # |            |
    # 3------------2

    rotmat = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
    
    # conozco las coordenadas de los vertices en sistema array.
    vex_sa = np.array([[u1, w2], [u2, w2], [u2, w1], [u1, w1]]).T  # traspongo para que sean vectores columna
    
    # coordeanadas x,z de los vertices vx
    # rotar
    vex_sl = np.matmul(rotmat, vex_sa)
    # trasladar los vertices
    vex_sl += np.expand_dims(np.array(array_center), 1)
    
    # calcular "bounding rectangle"
    roi_lab = [vex_sl[0, :].min(), vex_sl[0, :].max(),
               vex_sl[1, :].max(), vex_sl[1, :].min()]

    if pintar:
        fig, ax = plt.subplots()
        for v in vex_sl.T:
            ax.plot(v[0], v[1], 'o')
            ax.set_aspect('equal')
        rect_roi_lab = Rectangle((roi_lab[0], roi_lab[3]),
                                 roi_lab[1] - roi_lab[0], roi_lab[2] - roi_lab[3], fill=0)
        ax.add_artist(rect_roi_lab)

    return roi_lab


def roi_array2lab_3d(roi_array, array_center, rot, pintar=False):
    # calcular coordenadas x,y,z de los vértices de roi_array
    # roi_array: [u1, u2, v1, v2, w2, w1]
    u1, u2, v1, v2, w2, w1 = roi_array

    # conozco las coordenadas de los vertices en sistema array.
    vex_sa = np.array([[u1, v1, w2], [u2, v1, w2], [u2, v2, w2], [u1, v2, w2],
                       [u1, v1, w1], [u2, v1, w1], [u2, v2, w1], [u1, v2, w1]])

    # coordanadas x,y,z de los vertices vex
    # rotar
    vex_sl = rot.apply(vex_sa)
    # trasladar los vertices
    vex_sl += np.array(array_center)

    # calcular prisma paralelo a ejes X Y Z que contiene al roi
    roi_lab = [vex_sl[:, 0].min(), vex_sl[:, 0].max(),
               vex_sl[:, 1].min(), vex_sl[:, 1].max(),
               vex_sl[:, 2].max(), vex_sl[:, 2].min()]

    return roi_lab, vex_sl


def roi_list2roi_all(roi_list, x_step, y_step, z_step):

    """Calcular una ROI grande que contiene a todas las rois de roi_list. Y calcula el índice de el ´vertice
    "origen" de cada roi en la ROI grande"""
    roi_list = np.array(roi_list)
    x_min = roi_list[:, 0].min(axis=0)
    y_min = roi_list[:, 2].min(axis=0)
    z_min = roi_list[:, 5].min(axis=0)

    x_max = roi_list[:, 1].max(axis=0)
    y_max = roi_list[:, 3].max(axis=0)
    z_max = roi_list[:, 4].max(axis=0)

    roi_grande = [x_min, x_max, y_min, y_max, z_max, z_min]

    orig = [[r[0], r[2], r[4]] for r in roi_list]
    idx = [u3d.xyz2index(v, roi_grande, x_step, y_step, z_step) for v in orig]

    return roi_grande, idx
