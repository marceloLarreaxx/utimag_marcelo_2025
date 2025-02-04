from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize_scalar
from imag2D.pintar import arcoiris_cmap
import imag3D.Adquisiciones.procesar_adq_functions as aq
import imag3D.Adquisiciones.geom_robot as gr
import utils
import os
import numpy as np
import pickle
import imag3D.CNN_superficie.cnnsurf_funcs as fu
import imag3D.CNN_superficie.cnnsurf_plot as cnnplot
from imag3D import utils_3d, ifaz_3d
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
input_data = fu.crear_input_data(cfg, codepath, envelope=cfg['envelope'], sparse_tx=True)  # n_adq, 11, 11, n_samples

# Pose combinations contiene (en orden de índices): desplazamiento en z [0]; desplazamiento e y [1]; rotación en z [2];  y rotación e y [3]
pose_comb = cfg['pose_combinations']
poses = np.zeros((cfg['n_adq'], 6))

delta_z = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\delta_z.npy') # Para utilizarlo en el caso del plano

# depedne la foooormamaaaaaaa
if cfg['forma'] == 'plane':
    poses[:, 2] = pose_comb[:, 0] + delta_z - cfg['radio_externo_nominal']
    poses[:, 3:5] = pose_comb[:, 1:3]
elif cfg['forma'] == 'cylinder':
    poses[:, 2] = pose_comb[:, 0] + delta_z - cfg['radio_externo_nominal']  # desplazamiento en z
    poses[:, 1] = pose_comb[:, 1]  # desplazamiento en y
    poses[:, 4] = pose_comb[:, 3]  # rotación en y
    poses[:, 5] = pose_comb[:, 2]  # rotación en z

# # cargar config file
# cfg = utils.load_cfg(os.path.dirname(os.path.realpath(__file__)) + '/')
# matrix, mat_vars = aq.load_adq(cfg, '1', format='matlab')
#
# cfg['n_samples'] = matrix.shape[2]
# cfg['t_start'] = mat_vars['ret_ini'][0, 0]
# cfg['idx_ret_ini'] = cfg['t_start'] * cfg['fs']
#
# codepath = utils.utimag_path + r'\imag2D\OpenCL_code\\filtro_fir_conv.cl'
# # cargar adquisiones y filtrar
# input_data = fu.crear_input_data(cfg, codepath, format='matlab', envelope=cfg['envelope'])  # n_adq, 11, 11, n_samples
#
# poses = mat_vars['poses']
# # pasar de metros a mm
# poses[:, 0:3] *= 1000
# z_toca = mat_vars['z_0'][0][0] * 1000
# # cuando no esté definida la pose central, se usa la primera, dado que se trata de un caso de
# # solo rotaciones????
# pose_central = 1000 * mat_vars['pose_central'][0] if 'pose_central' in mat_vars.keys() else poses[0, :]
# r_ext = cfg['radio_nominal'] if cfg['curv_sign'] > 0 else cfg['radio_externo_nominal']
#
# # centro del cilindro o esfera
# ori = gr.return_component_center(z_toca, cfg['array_gap'], r_ext, pose_central)

# coordenadas en el sistema del array
array_coords = np.array(utils_3d.array_coordinates_list(cfg['nel_x'], cfg['nel_y'], cfg['pitch'], cfg['pitch']))
# pasaar las coordenadas de los elementos al sistema del componente
rot_array = Rotation.from_euler(*cfg['rot_array'], degrees=True)
array_coords_c = rot_array.apply(array_coords)

# definir el modelo
match cfg['forma']:
    case 'cylinder':
        if cfg['ray_model'] == 'interpolation':
            angs = 30*(np.linspace(0, 1, 100))**0.7
            distfun = ifaz_3d.return_cyl_pitch_catch_interporays_fun(array_coords_c, array_coords_c, angs, 1)
        else:
            distfun = ifaz_3d.return_pitch_catch_cylfun_circmirror(array_coords_c, array_coords_c, cfg['curv_sign'])
    case 'plane':
        distfun = ifaz_3d.return_pitch_catch_plane_fun(array_coords_c, array_coords_c)
    case 'sphere':
        if cfg['ray_model'] == 'interpolation':
            angs = 30*(np.linspace(0, 1, 100))**0.7
            distfun = ifaz_3d.return_sphere_pitch_catch_interporays_fun(array_coords_c, array_coords_c, angs, 1)
        else:
            # model aproximado espejo circular
            distfun = ifaz_3d.return_pitch_catch_sphere_fun_circmirror(array_coords_c, array_coords_c, cfg['curv_sign'])


# --------------------------------------------------------------------------------------------------------------------
# pasar las poses al sistema del componente, y calcular los TOF teoricos
print('calculado tof model')
poses_c = np.zeros((cfg['n_adq'], 6))
trafo, rot0_inv = gr.return_pose_transform(ori, cfg['rot_component'])
dist = []
nan_idx = []

for i in range(cfg['n_adq']):
    poses_c[i, :] = trafo(poses[i, :])
    if cfg['forma'] == 'cylinder':
        dist.append(
            distfun(cfg['radio_nominal'], poses_c[i, :3] + np.array([0, 0, cfg['delta_z']]),
                    ['xyz', poses_c[i, 3:]]).reshape((121, 11, 11)))
    elif cfg['forma'] == 'plane':
        dist.append(
            distfun(poses_c[i, :3] + np.array([0, 0, cfg['delta_z']]),
                    ['xyz', poses_c[i, 3:]]).reshape((121, 11, 11)))
    elif cfg['forma'] == 'sphere':
        dist.append(
            distfun(cfg['radio_nominal'], poses_c[i, :3], ['xyz', poses_c[i, 3:]]).reshape((121, 11, 11)))

    # chequear si hay algún "nan". Puede haber poses en las cuales al calcularse la intersecion de linea con cilindro
    # en el modelo, no haya interseccion, y salgan nanes.
    if np.any(np.isnan(dist[i])):
        nan_idx.append(i)

dist = np.concatenate(dist)
t_idx = np.int16(cfg['fs'] * dist / cfg['c1'] - cfg['idx_ret_ini'] - cfg['taps'] / 2)

# -------------------------------------------------------------------------------------------------------------------
# eliminar las poses que dan "nanes"
if len(nan_idx) > 0:
    idx_to_delete = []
    for i in nan_idx:
        idx_to_delete.append(np.arange(121 * i, 121 * (i + 1)))
    idx_to_delete = np.concatenate(idx_to_delete)
    t_idx = np.delete(t_idx, idx_to_delete, 0)
    input_data = np.delete(input_data, idx_to_delete, 0)
    poses_c = np.delete(poses_c, nan_idx, 0)

# --------------------------------------------------------------------------------------------------------------------
# seleccionar las submatrices correspondientes a los emisores seleccionados
n_fmc = int(input_data.shape[0] / 121)
selec = fu.return_sparse_fmc_selec(n_fmc, cfg['tx_list'])
input_data = input_data[selec, :, :, :].copy()
t_idx = t_idx[selec, :, :].copy()

# --------------------------------------------------------------------------------------------------------------------
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

# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
print('calculando offset óptimos')
# ajustar offsets
hw = (cfg['kernel_size'] - 1) / 2
t_max = hw / cfg['fs']
ofs_knl = utils.gaussian_pulse(cfg['freq'], 0.8, 0, -t_max, t_max, cfg['fs'])

for x in ['train', 'val', 'test']:
    for i in range(ofs[x].size):
        if cfg['offset_optim'] == 0: # max energy
            ofs[x][i] = fu.compute_t_offset(subdata[x + '_fmc'][i, :, :, :], subdata[x + '_t'][i, :, :],
                                            cfg['offset_window_width'], max_offset=cfg['max_offset'])
        elif cfg['offset_optim'] == 1: # matched filter
            ofs_fun = fu.return_offset_fun(subdata[x + '_fmc'][i, :, :, :], subdata[x + '_t'][i, :, :], ofs_knl)
            opt = minimize_scalar(lambda u: -ofs_fun(u), bounds=[-40, 40])
            ofs[x][i] = np.round(opt.x) - hw  # ALERTA: restar o no media ancho de kernel??????
            pass

    subdata[x + '_t'] += ofs[x] .reshape((-1, 1, 1)).astype(np.int16)

# --------------------------------------------------------------------------------------------------------------
# guardar en pickles_train
if cfg['save_data']:
    with open(cfg['save_path'] + cfg['save_name'] + '.pickle', 'wb') as f:
        pickle.dump(subdata, f)
        pickle.dump(poses_c, f)
        pickle.dump(mat_vars, f)
        pickle.dump(cfg, f)

    # guardar cfg utilizado en un txt con el mismo nombre del dataset -------------------------------------------------
    utils.dict2txt(cfg, cfg['save_path'] + cfg['save_name'] + '.txt')


def plot_algo(i, j):
    fig, ax = plt.subplots()
    ax.imshow(input_data[i, j, :, :].T, cmap=arcoiris_cmap)
    ax.set_aspect(0.02)
    ax.plot(t_idx[i, j, :], 'k')
    ax.set_aspect(0.02)

def plot_algo_2(i):
    fig, ax = plt.subplots()
    ax.imshow(np.abs(subdata['train_fmc'][i, :, :, :].reshape((121,-1))).T, cmap=arcoiris_cmap)
    ax.set_aspect(0.02)
    ax.plot(subdata['train_t'][i, :, :].reshape((121,-1)), 'k')
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
    ax.imshow(np.abs(subdata['train_fmc'][i, j, :, :]).T, cmap=arcoiris_cmap)
    ax.set_aspect(0.02)
    ax.plot(subdata['train_t'][i, j, :], 'k')
    ax.set_aspect(0.02)

def plot_ascan(i, j, k, amp=1000):
    fig, ax = plt.subplots()
    ax.plot(subdata['train_fmc'][i, j, k, :])
    ax.axvline(subdata['train_t'][i, j, k])
    aux = cfg['fs']*np.arange(-t_max, t_max + 1/cfg['fs'], 1/cfg['fs']) + subdata['train_t'][i, j, k] + ofs['train'][i] + hw
    ax.plot(aux, ofs_knl*amp)
