import keras_tuner as kt
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
import imag3D.CNN_superficie.keras_unet_mod.models.custom_vnet as cu
import imag3D.CNN_superficie.cnnsurf_funcs as fu
from imag3D.CNN_superficie.dataset.merge_datasets import merge
import utils
import os
import sys

# cargar config file
cfg = utils.load_cfg(os.path.dirname(os.path.realpath(__file__)) + '/')

fmc, tof, mask = merge(cfg) # crear los datasets


mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

def build_model(hp, max_params=3e6):

    a = hp.Choice('pool_size_xy', [1, 2])
    b = hp.Choice('pool_size_t', [2, 4, 8])
    pool_size = (a, a, b)
    num_layers = hp.Int('layers', 2, 5, step=1)
    filters = hp.Choice('filters', [4, 8, 16])
    loss_func = hp.Choice('loss_func', ['binary_crossentropy','tversky_loss'])
    if loss_func == 'binary_crossentropy':
        loss=tf.keras.losses.binary_crossentropy
    else:
        loss= fu.tversky_loss
    conv_size_t = hp.Int('conv_size_t', 3, 21, step=3)
    conv_kernel_size = (3, 3, conv_size_t)
    with mirrored_strategy.scope():
        idx_error_metric = tf.keras.metrics.MeanMetricWrapper(fu.idx_error, u=0.5, name='idx_error')
        n_out_1_metric = tf.keras.metrics.MeanMetricWrapper(fu.n_out_1, name='n_out_1')
        n_out_2_metric = tf.keras.metrics.MeanMetricWrapper(fu.n_out_2, name='n_out_2')
        model = cu.custom_vnet(fmc['train'].shape[1:], use_batch_norm=True, use_attention=False, num_classes=1,
                               num_layers=num_layers,
                               filters=filters,
                               pool_size=pool_size,
                               conv_kernel_size=conv_kernel_size,
                               conv_strides=(1, 1, 1), dropout=0.3, output_activation='sigmoid',
                               kernel_regularizer=None, bias_regularizer=None)

        model.compile(optimizer=Adam(), loss=loss, metrics=[idx_error_metric, n_out_1_metric, n_out_2_metric])
    model_params_number = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    if model_params_number > max_params:
        raise kt.errors.FailedTrialError('demasiados parámetros')
    return model


# Crear un sintonizador con RandomSearch
tuner = kt.tuners.Hyperband(
    build_model,
    objective=kt.Objective('val_idx_error', 'min'),#kt.Objective('val_n_out_1', 'min'),kt.Objective('val_n_out_2', 'min')],
    directory=cfg['model_save_path'] + r'tuner10',
    project_name='cnn_surf',
    max_epochs=10,
    overwrite=True,
    hyperband_iterations=1,
    max_consecutive_failed_trials=20
)

tuner.search_space_summary()


#callback1 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
# Iniciar la búsqueda de hiperparámetros
tuner.search(fmc['train'], mask['train'], batch_size=16, steps_per_epoch=5,
                          validation_data=(fmc['val'], mask['val']))
# guardar el mejor modelo
best_model = tuner.get_best_models()[0]
best_model.save(cfg['model_save_path'] + 'best_model_tuner10.h5')


# Guardar los resultados de los mejores modelos en un txt (codigo sugerido por CHATGPT) ----------------------------

# Importante: Guarda una referencia al stdout original para restaurarlo más tarde
original_stdout = sys.stdout

try:
    # Abre el archivo en modo de escritura y redirige la salida estándar
    with open(cfg['model_save_path'] + 'testtuner10_best40.txt', 'w') as f:
        sys.stdout = f

        # Ejecuta results_summary()
        tuner.results_summary(40)

    import csv

    # Obtener los mejores ensayos
    best_trials = tuner.oracle.get_best_trials(num_trials=40)
    # Escribir métricas en un archivo CSV
    with open(cfg['model_save_path']+ 'best_trials_info_tuner10.csv', 'w', newline='') as csvfile:
        fieldnames = ['trial_id', 'hyperparameters', 'epoch_best_performance','idx_error', 'n_out_1', 'n_out_2', 'val_idx_error', 'val_n_out_1', 'val_n_out_2']  # Define los nombres de las columnas
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')

        # Escribe las métricas para cada prueba
        writer.writeheader()
        for trial in best_trials:
            writer.writerow({'trial_id': trial.trial_id,
                            'hyperparameters': trial.hyperparameters.values,
                            'epoch_best_performance': trial.best_step,
                            'idx_error': round(trial.metrics.metrics.get('idx_error').get_best_value(),2) if 'idx_error' in trial.metrics.metrics else None,
                            'n_out_1': round(trial.metrics.metrics.get('n_out_1').get_best_value(),2) if 'n_out_1' in trial.metrics.metrics else None,
                            'n_out_2': round(trial.metrics.metrics.get('n_out_2').get_best_value(),2) if 'n_out_2' in trial.metrics.metrics else None,
                            'val_idx_error': round(trial.metrics.metrics.get('val_idx_error').get_best_value(),2) if 'val_idx_error' in trial.metrics.metrics else None,
                            'val_n_out_1': round(trial.metrics.metrics.get('val_n_out_1').get_best_value(),2) if 'val_n_out_1' in trial.metrics.metrics else None,
                            'val_n_out_2': round(trial.metrics.metrics.get('val_n_out_2').get_best_value(),2) if 'val_n_out_2' in trial.metrics.metrics else None

                            })


finally:
    # Restaura la salida estándar original
    sys.stdout = original_stdout