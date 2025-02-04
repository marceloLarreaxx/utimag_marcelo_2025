import sys
sys.path.append(r'C:\Marcelo\utimag-marcelo')

import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import utils
import pickle
from imag2D.pintar import arcoiris_cmap
from imag3D.CNN_superficie.dataset.merge_datasets import merge
import imag3D.CNN_superficie.cnnsurf_funcs as fu
import imag3D.CNN_superficie.cnnsurf_plot as cnnplot
from keras.optimizers import Adam
import imag3D.CNN_superficie.keras_unet_mod.models.custom_vnet as cu
import tensorflow as tf
from importlib import reload
import time
# import matplotlib as mpl
# mpl.use('TkAgg')
reload(cu)
# reload(fu)
plt.ion()

# sys.path.append(r'C:\Marcelo\utimag-marcelo\imag3D\CNN_superficie')

# tf.keras.mixed_precision.set_global_policy('mixed_float16')

# cargar config file
cfg = utils.load_cfg(os.path.dirname(os.path.realpath(__file__)) + '/')

# junta todas las adquisiciones de la lista cfg['piezas']
fmc, tof, mask = merge(cfg, decimar=cfg['decimar']) # crear los datasets
print(fmc['train'].shape[0], fmc['val'].shape[0], fmc['test'].shape[0])

# -----------------------------------------------------------------------holaaaaa--------------perfecto---------------------------
# aquí comienza la definicion de la red ---------------------------------------------------------------------------
# with mirrored_strategy.scope():

kernel_regu = cfg['kernel_regu'] if 'kernel_regu' in cfg else None
bias_regu = cfg['bias_regu'] if 'bias_regu' in cfg else None

# esto es para usar 2 gpu --------------------------
# mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

# with mirrored_strategy.scope():
# --------------------------------------------------

# esto define métricas ------------------------------------------------------------------------------------------
n_out_1_metric = tf.keras.metrics.MeanMetricWrapper(fu.n_out_1, name='n_out_1')
n_out_2_metric = tf.keras.metrics.MeanMetricWrapper(fu.n_out_2, name='n_out_2')
# defina la metrica de error de indice
idx_error_metric = tf.keras.metrics.MeanMetricWrapper(fu.idx_error, u=0.5, name='idx_error')

# se crea el modelo ------------------------------------------------------------------------------------------------
model = cu.custom_vnet(fmc['train'].shape[1:], use_batch_norm=True, use_attention=cfg['attention'], num_classes=1,
                       num_layers=cfg['n_layers'], filters=cfg['n_filters'], pool_size=cfg['pool_size'],
                       conv_kernel_size=cfg['conv_kernel_size'],
                       conv_strides=cfg['conv_strides'], dropout=0.3, output_activation='sigmoid',
                       kernel_regularizer=kernel_regu, bias_regularizer=bias_regu)
model.summary()

# tf.keras.utils.get_custom_objects().clear() # linea medio caca para borrar una cosa que queda guardada y molesta

# compila el modelo, eligiendo la loss function y laas métricas
model.compile(optimizer=Adam(), loss=fu.tversky_loss, #loss='binary_crossentropy', #loss=fu.WeightedBinaryCrossEntropy((1, 1)), #
              metrics=[tf.keras.metrics.BinaryAccuracy(), idx_error_metric, n_out_1_metric, n_out_2_metric])

# Parámetros de entrenamiento/fit
lr = 0.0001 # learning rate
epochs = cfg['epochs']
batch_size = cfg['batch_size']
steps_per_epoch = cfg['steps_per_epoch'] if 'steps_per_epoch' in cfg else \
    int(np.ceil(fmc['train'].shape[0] / cfg['batch_size']))

# entrenamiento ------------------------------------------------------------------------------------------------------
if cfg['train_model']:
    print('n_train: ', fmc['train'].shape[0])
    print('n_test: ', fmc['test'].shape[0])
    callback1 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=6)
    #callback2 = tf.keras.callbacks.ModelCheckpoint(
    #filepath=cfg['model_save_path'], verbose=True)
    history = model.fit(fmc['train'], mask['train'], batch_size=batch_size, steps_per_epoch=steps_per_epoch, epochs=epochs,
                        validation_data=(fmc['val'], mask['val']), callbacks=[callback1])
    # fin del entrnamiento

    # plotea las loss durante el training
    plt.figure()
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.ylim(0, 1)
    plt.legend()

    plt.figure()
    plt.plot(history.history['idx_error'], label="Training idx error")
    plt.plot(history.history["val_idx_error"], label="Validation idx error")
    plt.legend()

    # graficar metricas de outlier


    # guarda el modelo --------------------------------------------------------------------------------
    model_name = input('Nombre para guardar el modelo: ')
    # model_path = cfg['model_name'] + '.h5'
    model.save(cfg['model_save_path'] + model_name + '.h5')
    utils.dict2txt(cfg, cfg['model_save_path'] + model_name + '.txt')

    # Guardando el historial del modelo----------------------------------------------------------------
    np.savez(cfg['model_save_path'] + model_name+'_history', **history.history)



def plot_random():
    fig, ax = plt.subplots()
    i, j = np.random.randint(fmc['train'].shape[0]), np.random.randint(11)
    ax.imshow(fmc['train'][i, j, :, :, 0].T, cmap=arcoiris_cmap)
    ax.set_aspect(0.02)
    ax.plot(tof['train'][i, j, :], 'k')
    ax.set_aspect(0.02)


def example(i, umbral1, umbral2, peco_interval, w_min=0, zoom=None, co=('magenta', 'cyan', 'red'), window_num=10,
            figsize=(16, 10)):
    """Pinta un ejemplo del test data, con el indice i"""
    x = fmc['test'][i, :, :, :, 0].numpy()
    n_samples = x.shape[-1]
    x = x.reshape((121, n_samples))
    idx1 = utils.first_thr_cross(x, peco_interval, umbral1, window_num)
    valid1 = idx1[:, 1] > w_min

    y = fmc['test'][i, :, :, :, 0]
    y = model(np.expand_dims(y, axis=0)).numpy()
    y = y[0, :, :, :, 0].reshape((121, n_samples))
    idx2 = utils.first_thr_cross(y, peco_interval, umbral2, window_num)
    valid2 = idx2[:, 1] > w_min

    n = np.arange(121)

    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(x.T, cmap=arcoiris_cmap)
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

    return idx1, idx2


def timing_fun(n=20):
    t = []
    for i in range(n):
        t0 = time.perf_counter()
        y = model.predict_on_batch(np.expand_dims(fmc['test'][0, :, :, :, :], axis=0))
        t.append((time.perf_counter() - t0)*1000)
    return t

# plt.ion()
# example(100, 100, 0.5, [0,1000] )
# example(110, 100, 0.5, [0,1000] )