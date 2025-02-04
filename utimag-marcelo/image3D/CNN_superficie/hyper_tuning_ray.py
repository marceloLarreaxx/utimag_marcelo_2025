import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
# from ray.rllib.integrations.keras import ReportCheckpointCallback
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import imag3D.CNN_superficie.keras_unet_mod.models.custom_vnet as cu
import imag3D.CNN_superficie.cnnsurf_funcs as fu
from imag3D.CNN_superficie.dataset.merge_datasets import merge
import utils
import os

# Cargar config file
cfg = utils.load_cfg(os.path.dirname(os.path.realpath(__file__)) + '/')

fmc, tof, mask = merge(cfg)  # Crear los datasets


def build_model(config):
    #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"],
     #                                                  cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

    a = config["pool_size_xy"]
    b = config["pool_size_t"]
    pool_size = (a, a, b)
    num_layers = config["layers"]
    filters = config["filters"]
    loss_func = config["loss_func"]
    if loss_func == 'binary_crossentropy':
        loss = tf.keras.losses.binary_crossentropy
    else:
        loss = fu.tversky_loss
    conv_size_t = config["conv_size_t"]
    conv_kernel_size = (3, 3, conv_size_t)
    # with mirrored_strategy.scope():
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

    model.compile(optimizer=Adam(learning_rate=config["lr"]), loss=loss,
                  metrics=[idx_error_metric, n_out_1_metric, n_out_2_metric])
    return model


def train_model(config):
    model = build_model(config)

    callback1 = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    model.fit(fmc['train'], mask['train'], batch_size=32, steps_per_epoch=100,
              validation_data=(fmc['val'], mask['val']), callbacks=[callback1])


ray.init(num_cpus=1, num_gpus=2)

hyperband = AsyncHyperBandScheduler(time_attr="training_iteration", max_t=400, grace_period=20)

tuner = tune.Tuner(
    tune.with_resources(train_model, resources={"cpu": 1, "gpu": 2}),
    tune_config=tune.TuneConfig(
        metric='val_idx_error',
        mode="min",
        scheduler=hyperband,
        num_samples=10,
    ),
    param_space={
        "threads": 2,
        "lr": tune.choice([0.0001, 0.001]),
        "pool_size_xy": tune.choice([1, 2]),
        "pool_size_t": tune.choice([2, 4, 8]),
        "layers": tune.randint(2, 4),
        "filters": tune.choice([4, 8, 16]),
        "loss_func": tune.choice(['binary_crossentropy', 'tversky_loss']),
        "conv_size_t": tune.randint(6, 21)
    },
)

analysis=tuner.fit()
best_trial = analysis.get_best_result()
print("Best trial config: {}".format(best_trial.config))


    #import csv

    # # Obtener los mejores ensayos
    # allbest_trials = analysis.get_best_trial("val_idx_error", "min", "all")
    #
    # # Guardar los resultados en un archivo de texto
    # with open(cfg['model_save_path'] + 'testtuner10_best.txt', 'w') as f:
    #     for trial in allbest_trials:
    #         f.write("Trial ID: {}\n".format(trial.trial_id))
    #         f.write("Hyperparameters: {}\n".format(trial.config))
    #         f.write("Epoch best performance: {}\n".format(trial.last_result["epoch_best_performance"]))
    #         f.write("idx_error: {}\n".format(round(trial.last_result["idx_error"], 2)))
    #         f.write("n_out_1: {}\n".format(round(trial.last_result["n_out_1"], 2)))
    #         f.write("n_out_2: {}\n".format(round(trial.last_result["n_out_2"], 2)))
    #         f.write("val_idx_error: {}\n".format(round(trial.last_result["val_idx_error"], 2)))
    #         f.write("val_n_out_1: {}\n".format(round(trial.last_result["val_n_out_1"], 2)))
    #         f.write("val_n_out_2: {}\n\n".format(round(trial.last_result["val_n_out_2"], 2)))
    #
    # # Guardar los resultados en un archivo CSV
    # with open(cfg['model_save_path'] + 'best_trials_info_tuner10.csv', 'w', newline='') as csvfile:
    #     fieldnames = ['trial_id', 'hyperparameters', 'epoch_best_performance', 'idx_error',
    #                   'n_out_1', 'n_out_2', 'val_idx_error', 'val_n_out_1', 'val_n_out_2']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
    #     writer.writeheader()
    #     for trial in allbest_trials:
    #         writer.writerow({
    #             'trial_id': trial.trial_id,
    #             'hyperparameters': trial.config,
    #             'epoch_best_performance': trial.last_result["epoch_best_performance"],
    #             'idx_error': round(trial.last_result["idx_error"], 2),
    #             'n_out_1': round(trial.last_result["n_out_1"], 2),
    #             'n_out_2': round(trial.last_result["n_out_2"], 2),
    #             'val_idx_error': round(trial.last_result["val_idx_error"], 2),
    #             'val_n_out_1': round(trial.last_result["val_n_out_1"], 2),
    #             'val_n_out_2': round(trial.last_result["val_n_out_2"], 2)
    #         })