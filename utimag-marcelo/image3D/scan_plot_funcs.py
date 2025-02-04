import numpy as np
from matplotlib.colors import LogNorm

import imag3D.utils_3d as utils_3d
import imag3D.ifaz_3d as ifaz_3d
from imag2D.pintar import arcoiris_cmap
import utils
import matplotlib.pyplot as plt


def compute_cscan(img, z1, z2, roi, xyz_step, surf_fun, db_mm=0, db_0=0):
    dx, dy, dz = xyz_step
    tgc = ifaz_3d.surf_tgc(surf_fun, roi, dx, dy, dz, db_mm, db_0)
    wmask = ifaz_3d.surf_water_mask(surf_fun, roi, dx, dy, dz)
    img = img * wmask * tgc
    i1 = int(np.round((roi[4] - z1) / dz))
    i2 = int(np.round((roi[4] - z2) / dz))
    cscan = img[i1:i2, :, :].max(axis=0)
    return cscan.T


def compute_dscan_xz(img, y1, y2, roi, xyz_step, surf_fun, db_mm=0, db_0=0):
    dx, dy, dz = xyz_step
    tgc = ifaz_3d.surf_tgc(surf_fun, roi, dx, dy, dz, db_mm, db_0)
    wmask = ifaz_3d.surf_water_mask(surf_fun, roi, dx, dy, dz)
    img = img * wmask * tgc
    i1 = int(np.round((y1 - roi[2]) / dy))
    i2 = int(np.round((y2 - roi[2]) / dy))
    dscan = img[:, :, i1:i2].max(axis=2)
    return dscan


def plot_cscan(img, z1, z2, roi, xyz_step, surf_fun, db_mm=0, db_0=0, db_color=False, db_min=-20, vmax=70,
               interpolation='bilinear', color_db=False):

    cscan = compute_cscan(img, z1, z2, roi, xyz_step, surf_fun, db_mm, db_0)
    fig, ax = plt.subplots()
    if db_color:
        cscan_db = utils.db(cscan / cscan.max())
        im = ax.imshow(cscan_db, extent=roi[0:4], cmap=arcoiris_cmap, vmin=db_min, vmax=0,
                       interpolation=interpolation)
    else:
        im = ax.imshow(cscan, extent=roi[0:4], cmap=arcoiris_cmap, vmax=vmax, interpolation=interpolation)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')

    if color_db:
        im.set_norm(LogNorm(vmax=vmax))
    plt.colorbar(im)

    # if png_filename is not None:
    #     fig.savefig(cfg['data_path'] + png_filename)
    # if pickle_name is not None:
    #     with open(cfg['data_path'] + pickle_name, 'wb') as f:
    #         pickle.dump(total_cscan, f)

    return ax, cscan


def plot_dscan_xz(img, y1, y2, roi, xyz_step, surf_fun, z1=None, z2=None, db_mm=0, db_0=0, db_color=False, vmin=0,
                  vmax=70, interpolation='bilinear'):
    dx, dy, dz = xyz_step
    dscan = compute_dscan_xz(img, y1, y2, roi, xyz_step, surf_fun, db_mm, db_0)
    if z1 is not None:
        i1 = int(np.round((roi[4] - z1) / dz))
        i2 = int(np.round((roi[4] - z2) / dz))
        dscan = dscan[i1:i2, :]
        aux = (roi[0], roi[1], z2, z1)
    else:
        aux = (roi[0], roi[1], roi[5], roi[4])

    fig, ax = plt.subplots()
    # ax.imshow(dscan, extent=aux, cmap=arcoiris_cmap)
    # if db_color:
    #     dscan_db = utils.db(dscan / dscan.max())
    #     im = ax.imshow(dscan_db, extent=aux,
    #                    cmap=arcoiris_cmap, vmin=db_min, vmax=0, interpolation=interpolation)
    # else:
    im = ax.imshow(dscan, extent=aux, cmap=arcoiris_cmap, vmax=vmax, interpolation=interpolation)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('z [mm]')
    if db_color:
        im.set_norm(LogNorm(vmin=vmin, vmax=vmax))
    plt.colorbar(im)


    # if png_filename is not None:
    #     fig.savefig(cfg['data_path'] + png_filename)
    # if pickle_name is not None:
    #     with open(cfg['data_path'] + pickle_name, 'wb') as f:
    #         pickle.dump(total_cscan, f)

    return ax, dscan
