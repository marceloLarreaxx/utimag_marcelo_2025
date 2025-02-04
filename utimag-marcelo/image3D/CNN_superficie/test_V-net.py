import sys
sys.path.append(r'C:\Marcelo\utimag-marcelo')

import numpy as np
import os
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
import utils
import pickle
from imag2D.pintar import arcoiris_cmap
import imag3D.CNN_superficie.cnnsurf_funcs as fu
from keras.optimizers import Adam
import imag3D.CNN_superficie.keras_unet_mod.models.custom_vnet as cu
from imag3D.CNN_superficie.dataset.merge_datasets import merge
import imag3D.CNN_superficie.cnnsurf_plot as cnnplot
from imag3D.CNN_superficie.loss_funcs import load_segmentation_losses
import tensorflow as tf
from importlib import reload
import time
plt.ion()

reload(cu)

# cargar config file
cfg = utils.load_cfg(os.path.dirname(os.path.realpath(__file__)) + '/')
fmc, tof, mask = merge(cfg, decimar=cfg['decimar'])  # crear los datasets

model = tf.keras.models.load_model(cfg['model_path'],
            custom_objects={'tversky_loss': tf.keras.utils.get_custom_objects()['Custom>tversky loss'],
            'idx_error': tf.keras.utils.get_custom_objects()['Custom>idx_error'],
              'n_out_1': tf.keras.utils.get_custom_objects()['Custom>n_out_1'],
              'n_out_2': tf.keras.utils.get_custom_objects()['Custom>n_out_2']})

          # custom_objects={'weighted binary crossentropy': tf.keras.utils.get_custom_objects()['Custom>weighted binary crossentropy'],
          #           'idx_error': tf.keras.utils.get_custom_objects()['Custom>idx_error'],
          #                 'n_out_1': tf.keras.utils.get_custom_objects()['Custom>n_out_1'],
          #                 'n_out_2': tf.keras.utils.get_custom_objects()['Custom>n_out_2']})
        #custom_objects={'tversky_loss': tf.keras.utils.get_custom_objects()['Custom>tversky loss'],
        #             'idx_error': tf.keras.utils.get_custom_objects()['Custom>idx_error']})


def example(i, umbral1, umbral2, peco_interval, w_min=0, zoom=None, co=('magenta', 'cyan', 'red'), window_num=10,
            figsize=(16, 8), cmap=arcoiris_cmap):
    """Pinta un ejemplo del test data, con el indice i"""
    x = fmc['test'][i, :, :, :, 0].numpy()
    x = x.reshape((121, -1))
    idx1 = utils.first_thr_cross(x, peco_interval, umbral1, window_num)
    valid1 = idx1[:, 1] > w_min

    y = fmc['test'][i, :, :, :, 0]
    y = model(np.expand_dims(y, axis=0)).numpy()
    y = y[0, :, :, :, 0].reshape((121, -1))
    idx2 = utils.first_thr_cross(y, peco_interval, umbral2, window_num)
    valid2 = idx2[:, 1] > w_min

    n = np.arange(121)

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(x.T, cmap=cmap)
    ax[0].set_aspect(0.1)
    ax[0].plot(n[valid1], idx1[valid1, 0], '.', label='threshold', color=co[0])
    ax[0].plot(n[valid2], idx2[valid2, 0], '.-', label='V-net', color=co[1])
    ax[0].plot(tof['test'][i, :, :].flatten(), label='Ground Truth', color=co[2])
    ax[0].legend()
    ax[0].set_xlabel('Element linear index')
    ax[0].set_ylabel('time sample index')

    ax[1].imshow(y.T, cmap=arcoiris_cmap)
    ax[1].set_aspect(0.1)
    ax[1].plot(n[valid2], idx2[valid2, 0], color=co[1])
    ax[1].set_xlabel('Element linear index')

    if zoom is not None:
        for q in ax:
            q.set_xlim(zoom[0:2])
            q.set_ylim(zoom[2:])

    return idx1, idx2, fig


def timing_fun(n=20):
    t = []
    for i in range(n):
        t0 = time.perf_counter()
        y = model.predict_on_batch(np.expand_dims(fmc['test'][0, :, :, :, :], axis=0))
        t.append((time.perf_counter() - t0) * 1000)
    return t


def guardar_figs_varios_ejemplos(n, umbral1, umbral2, peco_interval, w_min=0, zoom=None, co=('magenta', 'cyan', 'red'),
                                 window_num=10, figsize=(16, 10)):
    for i in range(n):
        _, _, fig = example(i, umbral1, umbral2, peco_interval, w_min=w_min, zoom=zoom, co=co, window_num=window_num,
                            figsize=figsize)
        fig.savefig(cfg['fig_path'] + str(i))
        plt.close(fig)


def umbralizar(u, peco_interval, caso, sta_lta=[10, 100], window_num=10):
    """Calcula el indice de primer cruce de umbral sobre el test dataset"""
    n_test = fmc['test'].shape[0]
    idx = []

    for i in range(n_test):
        if caso == 'thr':
            # aplicar umbral a imagenes originales
            x = fmc['test'][i, :11, :11, :, 0].numpy()
            x = x.reshape((121, -1))
            idx.append(utils.first_thr_cross(x, peco_interval, u, window_num))
        elif caso == 'matched_filter':
            x = fmc['test'][i, :11, :11, :, 0].numpy()
            x = x.reshape((121, -1))
            freq = 3; fs = 40; t0 = 0; bw = 0.8; t_max = 4 / freq  # Parámetros para pulso gaussiano
            kernel = utils.gaussian_pulse(freq, bw, t0, -t_max, t_max, fs)
            t_idx, _ = fu.matched_filter(x, kernel, peco_interval)
            idx.append(t_idx)
        elif caso == 'sta_lta':
            x = fmc['test'][i, :11, :11, :, 0].numpy()
            x = x.reshape((121, -1))
            sta, lta = sta_lta
            t_idx, _, _ = fu.sta_lta_fun(x, sta, lta, peco_interval)
            idx.append(t_idx)
        elif caso == 'vnet':
            # aplicar umbral a resultados del modelo
            x = fmc['test'][i, :, :, :, 0]
            x = model(np.expand_dims(x, axis=0)).numpy()
            x = x[0, :11, :11, :, 0].reshape((121, -1))
            idx.append(utils.first_thr_cross(x, peco_interval, u, window_num))

    return np.array(idx),


def histograma(idx_thr, idx_vnet, caso, sta_lta=[10, 100], err_min=40, bins=20, xlim=(-20, 20), ymax=90000):
    """Calcula y pinta histogramas de error de los 2 métodos:
    threshold común y V-net + threshold.
    Calcula el nro de outliers en ambos casos, considerando
    que son outliers los que tienen error mayor a err_min"""

    if caso == 'Threshold':
        err_thr = idx_thr[:, :, 0].flatten() - tof['test'].flatten()
    else:
        if 'STA/LTA' in caso:
            caso = caso + ' - ' + str(sta_lta)
        err_thr = idx_thr[:, :].flatten() - tof['test'].flatten()
    err_vnet = idx_vnet[:, :, 0].flatten() - tof['test'].flatten()
    fig, ax = plt.subplots(1, 2)
    outli_thr = np.abs(err_thr) > err_min
    outli_vnet = np.abs(err_vnet) > err_min
    ax[0].hist(err_thr[np.logical_not(outli_thr)], bins=bins)
    ax[0].set_title(caso)
    ax[0].set_xlim(*xlim)
    ax[0].set_ylim((0, ymax))
    ax[0].grid()
    ax[0].set_xlabel('error (number of samples)')
    ax[0].set_ylabel('count')
    ax[1].hist(err_vnet[np.logical_not(outli_vnet)], bins=bins)
    ax[1].set_title('V-net')
    ax[1].set_xlim(*xlim)
    ax[1].set_ylim((0, ymax))
    ax[1].grid()
    ax[1].set_xlabel('error (number of samples)')
    res = {'thr': [err_thr.mean(), err_thr.std(), outli_thr.sum()],
           'vnet': [err_vnet.mean(), err_vnet.std(), outli_vnet.sum()]}
    # usando los pesos
    if caso == 'Threshold':
        w_thr = idx_thr[:, :, 1].flatten()
        w_vnet = idx_vnet[:, :, 1].flatten()
        res_w = {'thr': [np.average(err_thr, weights=w_thr), np.sqrt(np.cov(err_thr, aweights=w_thr)), outli_thr.sum()],
             'vnet': [np.average(err_vnet, weights=w_vnet), np.sqrt(np.cov(err_vnet, aweights=w_vnet)),
                      outli_vnet.sum()]}
    else:
        w_vnet = idx_vnet[:, :, 1].flatten()
        res_w = {'thr': [np.average(err_thr), np.sqrt(np.cov(err_thr)), outli_thr.sum()],
                 'vnet': [np.average(err_vnet, weights=w_vnet), np.sqrt(np.cov(err_vnet, aweights=w_vnet)),
                          outli_vnet.sum()]}

    return res, res_w


def error_stats(idx_thr, idx_vnet):
    err_thr = idx_thr[:, :, 0] - tof['test'].reshape(-1, 121)
    err_vnet = idx_vnet[:, :, 0] - tof['test'].reshape(-1, 121)
    valid_thr = idx_thr[:, :, 1] > 0
    valid_vnet = idx_vnet[:, :, 1] > 0
    n_valid = [valid_thr.sum(axis=1), valid_vnet.sum(axis=1)]
    # err_thr_valid = err_thr[valid]
    # err_vnet_valid = err_vnet[valid]
    return n_valid


def plot_error_histogram_multi(errors_dict, bin_width=5):
    """
    Función para generar un histograma de los errores de índice para varias soluciones, con el número de errores en el eje y y bins fijos en el eje x.

    Args:
    - errors_dict: Un diccionario donde las claves son los nombres de las soluciones y los valores son listas de errores de índice correspondientes a cada solución.
    """
    num_solutions = len(errors_dict)
    # bin_width = 5  # Ancho del bin en el eje x

    # Crear subplots para cada solución
    fig, axs = plt.subplots(num_solutions, figsize=(8, 2*num_solutions), sharex=True)


    num_bins = int((40 - (-40)) / bin_width) + 1

    # Generar bins para el histograma
    bins = np.linspace(-40, 40, num_bins + 1)  # Bins para valores dentro del rango [-40, 40]


    for i, (solution, errors) in enumerate(errors_dict.items()):
            # Calcular los valores dentro y fuera del rango
            in_range = [error for error in errors if -40 <= error <= 40]
            out_of_range_left = [error for error in errors if error < -40]
            out_of_range_right = [error for error in errors if error > 40]

            # Generar el histograma en el subplot correspondiente
            counts, bins, _ = axs[i].hist(in_range, bins=bins, color='blue', edgecolor='black', alpha=0.7)
            axs[i].set_title(f'- {solution}')
            axs[i].set_ylabel('Errores')
            axs[i].grid(True)

            # Añadir barras para valores fuera del rango
            if out_of_range_left:
                axs[i].bar(-45, len(out_of_range_left), width=5, color='red', label='Outliers left')
            if out_of_range_right:
                axs[i].bar(45, len(out_of_range_right), width=5, color='red', label='Outliers right')

            # Anotar el número total de outliers en el gráfico
            total_outliers = len(out_of_range_left) + len(out_of_range_right)
            if total_outliers > 0:
                axs[i].text(0.85, 0.85, f'Total de Outliers: {total_outliers}', transform=axs[i].transAxes,
                            fontsize=10, ha='center', va='center', color='red')
            axs[i].set_ylim((0, 85000))

    # Ajustar los ejes x
    plt.xticks(np.arange(-40, 41, bin_width))

    # Añadir etiqueta y título común
    axs[-1].set_xlabel('Error de índice')
    #plt.title('Index error per solution')

    plt.tight_layout()
    plt.show()


def plot_error_boxplot(error_data, outlier_counts):
    """
    Función para generar un diagrama de caja (box plot) de los errores de índice de distintas soluciones.

    Args:
    - error_data: Un diccionario donde las claves son los nombres de las soluciones y los valores son listas de errores de índice.
    - outlier_counts: Un diccionario donde las claves son los nombres de las soluciones y los valores son tuplas que contienen el número de outliers de cada tipo.
    """
    # Crear una lista de errores para cada solución
    errors = list(error_data.values())

    # Crear una lista de nombres de soluciones
    solution_names = list(error_data.keys())
    plt.ion()
    # Generar el box plot principal

    fig, axs = plt.subplots(2,figsize=(10, 8), sharex=True, sharey=False)

    axs[0].boxplot(errors, labels=solution_names,vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     showfliers=False)
    #axs[0].set_title('Box Plot de Errores de Índice por Solución')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Errores de Índice')
    axs[0].grid(True)

    # Obtener el número total de outliers de cada tipo para cada solución
    total_outliers_type_1 = [outlier_counts[solution][0] for solution in solution_names]
    total_outliers_type_2 = [outlier_counts[solution][1] for solution in solution_names]

    # Añadir subplot encima del box plot principal para mostrar outliers
    # Añadir los marcadores 'X' en el segundo subplot para outliers de tipo 1
    axs[1].scatter(range(1,len(solution_names)+1), total_outliers_type_1, marker='X', color='red', label='Outliers Tipo 1')
    axs[1].set_ylabel('Número de Outliers')
    axs[1].grid(True)
    axs[1].legend()

    # Añadir los marcadores '0' en el segundo subplot para outliers de tipo 2
    axs[1].scatter(range(1,len(solution_names)+1), total_outliers_type_2, marker='o', color='blue', label='Outliers Tipo 2')
    axs[1].set_xlabel('Métodos de detección')
    axs[1].legend()

    #plt.tight_layout()
    plt.show()

    # # Añadir los marcadores 'X' y '0' en el gráfico en la misma posición del eje x
    # for i, solution in enumerate(solution_names):
    #     x_position = i + 1  # Posición en el eje x
    #     y_position_x = total_outliers_type_1[i]  # Altura para el marcador 'X'
    #     y_position_0 = total_outliers_type_2[i]  # Altura para el marcador '0'
    #     plt.scatter([x_position], [y_position_x], marker='X', color='red', label='Outliers Tipo 1')
    #     plt.scatter([x_position], [y_position_0], marker='o', color='blue', label='Outliers Tipo 2')
    #
    # plt.tight_layout()
    # plt.show()
    #
    # plt.tight_layout()
    # plt.show()

def get_outlier(idx_list, e_max=40):
    print(idx_list.shape)
    n_out1 = 0
    n_out2 = 0
    errors = idx_list[:, :].flatten() - tof['test'].flatten()
    for i in range(len(errors)):
        if idx_list[:, :].flatten()[i] == 0:
            n_out1+=1
        elif abs(errors[i])>e_max:
            n_out2+=1
        # n_out1 = np.sum(idx_list[:, :, 0].flatten() == 0)
        # n_out2 = np.sum(np.abs(err)>e_max)

    #idx_cleaned=idx_list[(idx_list[:, :, 0].flatten()!=0)&(err<=e_max)]
    return n_out1, n_out2#, idx_cleaned


#######################################################################################################################
idx_thr100 = umbralizar(100, [0, 1000], 'thr')[0]
idx_thr50 = umbralizar(50, [0, 1000], 'thr')[0]
idx_matched = umbralizar(0, [0, 1000], 'matched_filter')[0]
idx_sta_lta_1 = umbralizar(0, [0, 1000], 'sta_lta', [10, 100])[0]
idx_sta_lta_2 = umbralizar(0, [0, 1000], 'sta_lta', [10, 50])[0]
idx_vnet = umbralizar(0.5, [0, 1000], 'vnet')[0]

res1 = histograma(idx_thr100, idx_vnet, 'Threshold')
res2 = histograma(idx_thr50, idx_vnet, 'Threshold')
res3 = histograma(idx_matched, idx_vnet, 'Matched Filter')
res4 = histograma(idx_sta_lta_1, idx_vnet, 'STA/LTA', [10, 100])
res4 = histograma(idx_sta_lta_2, idx_vnet, 'STA/LTA', [10, 50])

n_out1_thr50, n_out2_thr50 = get_outlier(idx_thr50[:,:,0])
n_out1_thr100, n_out2_thr100 = get_outlier(idx_thr100[:,:,0])
n_out1_matched, n_out2_matched = get_outlier(idx_matched)
n_out1_sta_lta_1, n_out2_sta_lta_1 = get_outlier(idx_sta_lta_1)
n_out1_sta_lta_2, n_out2_sta_lta_2 = get_outlier(idx_sta_lta_2)
n_out1_vnet, n_out2_vnet=get_outlier(idx_vnet[:,:,0])

err_thr50 = (idx_thr50[:, :, 0] - tof['test'].reshape(-1, 121)).flatten()
err_thr100 = (idx_thr100[:, :, 0] - tof['test'].reshape(-1, 121)).flatten()
err_matched = (idx_matched[:, :] - tof['test'].reshape(-1, 121)).flatten()
err_sta_lta_1 = (idx_sta_lta_1[:, :] - tof['test'].reshape(-1, 121)).flatten()
err_sta_lta_2 = (idx_sta_lta_2[:, :] - tof['test'].reshape(-1, 121)).flatten()
err_vnet = (idx_vnet[:, :, 0] - tof['test'].reshape(-1, 121)).flatten()

error_data_filt={'umbral_50': err_thr50[np.abs(err_thr50)<40],
            'umbral_100': err_thr100[np.abs(err_thr100)<40],
            'filtro_adaptado': err_matched[np.abs(err_matched)<40],
            'sta_lta_1': err_sta_lta_1[np.abs(err_sta_lta_1)<40],
            'sta_lta_2': err_sta_lta_2[np.abs(err_sta_lta_2)<40],
            'modelo': err_vnet[np.abs(err_vnet)<40]}

outlier_count={'umbral_50': (n_out1_thr50,n_out2_thr50),
            'umbral_100':(n_out1_thr100, n_out2_thr100),
            'filtro_adaptado':(n_out1_matched, n_out2_matched),
            'sta_lta_1':(n_out1_sta_lta_1, n_out2_sta_lta_1),
            'sta_lta_2':(n_out1_sta_lta_2, n_out2_sta_lta_2),
            'modelo':(n_out1_vnet, n_out2_vnet)}
plot_error_boxplot(error_data_filt, outlier_count)

error_data={'umbral_50': err_thr50,
            'umbral_100': err_thr100,
            'filtro_adaptado': err_matched,
            'sta_lta_1': err_sta_lta_1,
            'sta_lta_2': err_sta_lta_2,
            'modelo': err_vnet}

plot_error_histogram_multi(error_data)

#######################################################################################################################

# err200 = err200.sum(axis=1)

#REMOTE CHANGES

# load_segmentation_losses(mask['test'],fmc['test'],model, umbral=0.5, model_label='Binary_loss')