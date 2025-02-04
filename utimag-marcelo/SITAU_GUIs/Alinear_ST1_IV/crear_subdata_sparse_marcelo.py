import sys
sys.path.append(r'C:\Marcelo\utimag-marcelo')

from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
import utils
import os
import numpy as np
import pickle
import imag3D.CNN_superficie.cnnsurf_funcs as fu
from imag3D import utils_3d, ifaz_3d
import imag3D.CNN_superficie.cnnsurf_plot as cnnplot
from imag2D.pintar import arcoiris_cmap
import imag3D.utils_3d as u3d
from scipy.optimize import minimize_scalar
from importlib import reload

"""Carga las adquisiciones de una carpeta, indicada en el json. Pasa los filtros y genera el array
(batch, 11, 11, samples). Se recortan las taps/2 muestras al principio y al final,  que valen 0 por efecto
 de los filtros. En "mat_vars" estan las poses del robot. Se selecciona un subconjunto random y se divide en 
 train y test"""

# cargar config file
cfg = utils.load_cfg(os.path.dirname(os.path.realpath(__file__)) + '/')
dict_params = np.load(cfg['data_path'] + 'measure_parameters.npz')
dict_params = {ky: dict_params[ky] for ky in dict_params}
cfg.update(dict_params)  # fusionar los diccionarios
cfg['t_start'] = 0
cfg['idx_ret_ini'] = 0

codepath = utils.utimag_path + r'\imag2D\OpenCL_code\filtro_fir_conv.cl'
input_data = fu.crear_input_data(cfg, codepath, envelope=cfg['envelope'], format='npy')  # n_adq, 11, 11, n_samples

# --------------------------------------------------------------------------------------------------
# Pose combinations contiene (en orden de índices):
# desplazamiento en z [0]; desplazamiento e y [1]; rotación en z [2];  y rotación e y [3]
pose_comb = cfg['pose_combinations']
poses = np.zeros((cfg['n_adq'], 6))

delta_z = np.load(cfg['delta_z_path'])

# -------------------------------------------------------------------------------------------------
# Trasnformar del formato marceliano de pose_comb a poses en formato robotiano
if cfg['forma'] == 'plane':
    poses[:, 2] = pose_comb[0:cfg['n_adq'], 0] + delta_z - cfg['radio_externo_nominal']
    poses[:, 3:5] = pose_comb[:, 1:3]
elif cfg['forma'] == 'cylinder':
    poses[:, 2] = pose_comb[0:cfg['n_adq'], 0] + delta_z - cfg['radio_externo_nominal']  # desplazamiento en z
    poses[:, 1] = pose_comb[0:cfg['n_adq'], 1]  # desplazamiento en y
    poses[:, 4] = pose_comb[0:cfg['n_adq'], 3]  # rotación en y
    poses[:, 5] = pose_comb[0:cfg['n_adq'], 2]  # rotación en z
elif cfg['forma'] == 'sphere':
    poses[:, 0] = pose_comb[0:cfg['n_adq'], 0]  # desplazamiento en x
    poses[:, 1] = pose_comb[0:cfg['n_adq'], 1]  # desplazamiento en y
    poses[:, 2] = pose_comb[0:cfg['n_adq'], 2] + delta_z - cfg['radio_externo_nominal']   # desplazamiento en z

# --------------------------------------------------------------------------------------------------------------------
# coordenadas en el sistema del array
array_coords = np.array(utils_3d.array_coordinates_list(cfg['nel_x'], cfg['nel_y'], cfg['pitch'], cfg['pitch']))
# pasar las coordenadas de los elementos al sistema del componente/home
# (al cambiar el sentido del eje z se invierte el signo de x)
rot_array = Rotation.from_euler(*cfg['rot_array'], degrees=True)
array_coords_c = rot_array.apply(array_coords)

ascan_elements = []
for tx in cfg['tx_list']:
    linear_index, _ = u3d.array_ij2element(tx, 11, 11, 1, 1)
    ascan_elements.append(linear_index)

tx_coords = array_coords_c[ascan_elements, :]
rx_coords = array_coords_c

# -------------------------------------------------------------------------------------------------------------------
# definir el modelo de cálculo de tiempos de vuelo
match cfg['forma']:
    case 'cylinder':
        if cfg['ray_model'] == 'interpolation':
            angs = 30 * (np.linspace(0, 1, 200)) ** 0.7
            distfun = ifaz_3d.return_cyl_pitch_catch_interporays_fun(tx_coords, rx_coords, angs, cfg['curv_sign'])
        else:
            distfun = ifaz_3d.return_pitch_catch_cylfun_circmirror(tx_coords, rx_coords, cfg['curv_sign'])
    case 'plane':
        distfun = ifaz_3d.return_pitch_catch_plane_fun(tx_coords, rx_coords)
    case 'sphere':
        if cfg['ray_model'] == 'interpolation':
            angs = 30 * (np.linspace(0, 1, 200)) ** 0.7
            distfun = ifaz_3d.return_sphere_pitch_catch_interporays_fun(tx_coords, rx_coords, angs, cfg['curv_sign'])
        else:
            # model aproximado espejo circular
            distfun = ifaz_3d.return_pitch_catch_sphere_fun_circmirror(tx_coords, rx_coords, cfg['curv_sign'])

# # ---------------------------------------------------------------------------------------------------------------
# calcular tiempos de vuelo teóricos

poses_c = poses.copy()
n_disp = len(cfg['tx_list'])  # OJO: si es FMC completa esto tiene que ser 121
dist = []
nan_idx = []
rot_z_grande_idx = [] # para identificar los casos en que z muy grande, y el calculo de tof falla

for i in range(cfg['n_adq']):
    if cfg['forma'] == 'cylinder':
        dist.append(
            distfun(cfg['radio_nominal'], poses_c[i, :3] + np.array([0, 0, cfg['z_offset']]),
                    ['xyz', poses_c[i, 3:]]).reshape((n_disp, 11, 11)))
    elif cfg['forma'] == 'plane':
        dist.append(
            distfun(poses_c[i, :3] + np.array([0, 0, cfg['z_offset']]),
                    ['xyz', poses_c[i, 3:]]).reshape((n_disp, 11, 11)))
    elif cfg['forma'] == 'sphere':
        dist.append(distfun(cfg['radio_nominal'], poses_c[i, :3] + np.array([0, 0, cfg['z_offset']]),
                    ['xyz', poses_c[i, 3:]]).reshape((n_disp, 11, 11)))

    # chequear si hay algún "nan". Puede haber poses en las cuales al calcularse la intersecion de linea con cilindro
    # en el modelo, no haya interseccion, y salgan nanes.
    if np.any(np.isnan(dist[i])):
        nan_idx.append(i)
    if poses_c[i, -1] > cfg['rot_z_max']:
        rot_z_grande_idx.append(i)

dist = np.concatenate(dist)
t_idx = np.int16(cfg['fs'] * dist / cfg['c1'] - cfg['idx_ret_ini'] - cfg['taps'] / 2)

# # ------------------------------------------------------------------------------------
# # eliminar las poses que dan "nanes" o que tienen z_grande
idx_poses_to_delete = nan_idx + rot_z_grande_idx # indice de adquisicion

if len(idx_poses_to_delete) > 0:
    idx_to_delete = []  # indice de imagen (9 por cada adquisicion, si 9 disparos)
    for i in idx_poses_to_delete:
        idx_to_delete.append(np.arange(n_disp * i, n_disp * (i + 1)))
    idx_to_delete = np.concatenate(idx_to_delete)
    t_idx = np.delete(t_idx, idx_to_delete, 0)
    input_data = np.delete(input_data, idx_to_delete, 0)
    poses_c = np.delete(poses_c, idx_poses_to_delete, 0)

#----------- eliminando índices mayores a 2000 --------
big_idx = []
for i in range(t_idx.shape[0]):
    if np.any(t_idx[i, :, :] > 2000):
        # print(i)
        big_idx.append(i)
input_data = np.delete(input_data, big_idx, 0)
t_idx = np.delete(t_idx, big_idx, 0)

# # ------------------------------------------------------------------------------------

# ahora hay que hacer una seleccion de un subconjunto random y dividirlo en train, validation y test
subdata = {}
n_fulldata = input_data.shape[0]
np.random.seed(0)
idx = np.random.permutation(n_fulldata)
# ahora hay que dividir entre train y test
fra = cfg['set_fractions']
n_train, n_val = int(n_fulldata * fra[0]), int(n_fulldata * fra[1])
subdata['train_idx'], subdata['val_idx'], subdata['test_idx'] = np.split(idx, [n_train, n_train + n_val])
ofs = {}
for x in ['train', 'val', 'test']:
    subdata[x + '_fmc'] = input_data[subdata[x + '_idx'], :, :, :]
    subdata[x + '_t'] = t_idx[subdata[x + '_idx'], :, :]
    ofs[x] = np.zeros(subdata[x + '_fmc'].shape[0])
#
# # --------------------------------------------------------------------------------------------------------------

print('calculando offset óptimos')
# ajustar offsets
for x in ['train', 'val', 'test']:
    match cfg['offset_optim']:
        case -1:
            # no pone ofsset
            pass
        case 0:
            for i in range(ofs[x].size):
                ofs[x][i] = fu.compute_t_offset(subdata[x + '_fmc'][i, :, :, :], subdata[x + '_t'][i, :, :],
                                                cfg['offset_window_width'], max_offset=cfg['max_offset'])
        case 1:
            hw = (cfg['kernel_size'] - 1) / 2
            t_max = hw / cfg['fs']
            ofs_knl = utils.gaussian_pulse(cfg['freq'], 0.8, 0, -t_max, t_max, cfg['fs'])
            for i in range(ofs[x].size):
                ofs_fun = fu.return_offset_fun(subdata[x + '_fmc'][i, :, :, :], subdata[x + '_t'][i, :, :], ofs_knl)
                opt = minimize_scalar(lambda u: -ofs_fun(u), bounds=[-50, 50])
                ofs[x][i] = np.round(opt.x) - hw  # ALERTA: restar o no media ancho de kernel??????

    subdata[x + '_t'] += ofs[x].reshape((-1, 1, 1)).astype(np.int16)
    subdata[x + '_ofs'] = ofs[x]  # guardar los offset en subdata

with open(cfg['save_path'] + cfg['save_name'] + '.pickle', 'wb') as f:
    pickle.dump(subdata, f)
    pickle.dump(poses_c, f)
    pickle.dump(cfg, f)

# guardar cfg utilizado en un txt con el mismo nombre del dataset -------------------------------------------------
utils.dict2txt(cfg, cfg['save_path'] + cfg['save_name'] + '.txt')


def plot_algo(i, j):
    fig, ax = plt.subplots()
    ax.imshow(input_data[i, j, :, :].T, cmap=arcoiris_cmap)
    ax.set_aspect(0.02)
    ax.plot(t_idx[i, j, :], 'k')
    ax.set_aspect(0.02)


def plot_algo_2(i, vmax=1000):
    fig, ax = plt.subplots()
    ax.imshow(np.abs(input_data[i, :, :, :]).reshape((121, -1)).T, cmap=arcoiris_cmap, vmax=vmax)
    ax.set_aspect(0.02)
    ax.plot(t_idx[i, :, :].reshape((121, -1)), 'k')
    ax.set_aspect(0.02)


def plot_random():
    fig, ax = plt.subplots()
    i, j = np.random.randint(input_data.shape[0]), np.random.randint(11)
    ax.imshow(input_data[i, j, :, :].T, cmap='seismic', norm=cnnplot.SigmoidNorm(gamma=10))
    ax.set_aspect(0.02)
    ax.plot(t_idx[i, j, :], 'k')
    ax.set_aspect(0.02)


def plot2(i, j):
    fig, ax = plt.subplots()
    # i, j = np.random.randint(subdata['train_fmc'].shape[0]), np.random.randint(11)
    ax.imshow(subdata['train_fmc'][i, j, :, :].T, cmap=arcoiris_cmap)
    ax.set_aspect(0.02)
    ax.plot(subdata['train_t'][i, j, :], 'k')
    ax.set_aspect(0.02)
