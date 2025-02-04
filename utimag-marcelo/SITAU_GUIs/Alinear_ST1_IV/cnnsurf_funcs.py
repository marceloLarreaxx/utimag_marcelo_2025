import numpy as np
from keras.layers import Cropping3D
import keras
from matplotlib import pyplot as plt
import utils
import imag3D.utils_3d as utils_3d
from imag2D.pintar import arcoiris_cmap
from imag3D import ifaz_3d
import imag3D.Adquisiciones.procesar_adq_functions as aq
import pyopencl as cl
from scipy.spatial.transform import Rotation
from tensorflow.keras import backend as K
import tensorflow as tf
import imag3D.CNN_superficie.keras_unet_mod.models.custom_vnet as cu
import imag3D.Adquisiciones.geom_robot as gr

def crear_input_data(cfg, codepath, format, envelope=True):
    matrix, mat_vars = aq.load_adq(cfg, '1', format)
    cfg['n_samples'] = matrix.shape[2]

    # calcular coeficientes del filtro pasabanda
    bandpass_coef = aq.coeficientes_pasabanda(cfg)

    # escribir macros
    mac = utils.parameters_macros(cfg, utils.FLOAT_LIST, utils.INT_LIST)
    code = ''
    with open(codepath) as f:
        code += f.read()

    ctx, queue, mf = utils.init_rtx3080()
    prg = cl.Program(ctx, mac + code).build()

    buf = {'matrix': utils.DualBuffer(queue, None, matrix.shape, data_type=np.int16),
           'matrix_filt': utils.DualBuffer(queue, None, matrix.shape, data_type=np.float32),
           'matrix_imag': utils.DualBuffer(queue, None, matrix.shape, data_type=np.float32),
           'bandpass_coef': utils.DualBuffer(queue, bandpass_coef, None, data_type=np.float32),
           'hilb_coef': utils.DualBuffer(queue, utils.HILB_COEF, None, data_type=np.float32)}

    input_data = []

    adqs = cfg['adq_indexes'] if cfg['adq_indexes'] else range(cfg['n_adq'])
    for i in adqs:
        print(i)
        matrix, params = aq.load_adq(cfg, str(i + 1), format)
        buf['matrix'].c2g(matrix)
        args = utils.return_knl_args(buf, ['matrix', 'bandpass_coef', 'matrix_filt'])
        prg.filt_kernel_int16(queue, (cfg['n_elementos'], cfg['n_elementos']), None, *args).wait()
        matrix_filt = buf['matrix_filt'].g2c()
        if envelope:
            args = utils.return_knl_args(buf, ['matrix_filt', 'hilb_coef', 'matrix_imag'])
            prg.filt_kernel(queue, (cfg['n_elementos'], cfg['n_elementos']), None, *args).wait()
            matrix_imag = buf['matrix_imag'].g2c()
            temp = np.abs(matrix_filt + 1j * matrix_imag)
        else:
            temp = matrix_filt
        # aq.ejecutar_filtros(cfg, queue, buf, prg)

        # el filtro genera taps/2 ceros al principio y al final. Los quito para alivianar los datos
        m = int(cfg['taps']/2)
        sli = slice(m, cfg['n_samples'] - m)
        n_samples_new = cfg['n_samples'] - cfg['taps']  # redefinir n_samples
        temp = temp[:, :, sli]
        input_data.append(temp.reshape(121, 11, 11, n_samples_new))

    return np.concatenate(input_data)


def load_poses(cfg, params, format):
    if format == 'matlab':
        poses = params['poses']
        # pasar de metros a mm
        poses[:, 0:3] *= 1000
        z_toca = params['z_0'][0][0] * 1000
        # cuando no esté definida la pose central, se usa la primera, dado que se trata de un caso de
        # solo rotaciones????
        pose_central = 1000 * params['pose_central'][0] if 'pose_central' in params.keys() else poses[0, :]
        r_ext = cfg['radio_nominal'] if cfg['curv_sign'] > 0 else cfg['radio_externo_nominal']

        # centro del cilindro o esfera
        ori = gr.return_component_center(z_toca, cfg['array_gap'], r_ext, pose_central)


def transformar_pose(pose_0, mat_vars, cfg):
    """ Pasar de metros a mm, y cambiar los ejes. Pose_0 esta en ejes donde "z" apunta hacia abajo. Paso
    a unos con "z" apuntando hacia arriba, porque las funciones de calculo de tiempos de vuelo estan pensadas de
    ese modo"""

    # las poses vienen referidas a la pose_0, cuyo eje z apunta hacia abajo, y su origen esta a cierta altura
    # respecto al plano de referencia
    z_toca = 1000 * mat_vars['z_0'][0, 0]
    h_origen = cfg['radio_nominal']  # altura del eje del cilindro respecto al plano de apoyo, en el caso convexo
    z_trasla = z_toca + cfg['array_gap'] - h_origen  # traslacion para llevar el origen al eje del cilindro

    pose = np.concatenate((1000 * pose_0[0:3], pose_0[3:])) - np.array([0, 0, z_trasla, 0, 0, 0])
    # para hacer que el eje z apunta hacia arriba, hay que girar 180 grados alrededor del eje "y" o el "x"
    # elijo usar el "y", entonces cambian de signo las rotaciones alrededor de "x" y de "z".
    # Tambien cambia el signo de las componentes "x" y "z". Por otro lado, la componente "x" se puede poner
    # en 0 por simetria
    pose[0] = 0
    pose[2] = -pose[2]
    pose[3] = -pose[3]
    pose[5] = -pose[5]
    return pose


def compute_tof_model(mat_vars, cfg, filtro_offset=1):
    """Calcula el TOF teótico según la pose del array respecto al cilindro
    filtro_offset: si se han quitado los taps/2 ceros al principio de las señales, hay que tenerlo en cuenta
    para el t_idx"""

    # las poses vienen referidas a la pose_0, cuyo eje z apunta hacia abajo, y su origen esta a cierta altura
    # respecto al plano de referencia
    z_toca = 1000 * mat_vars['z_0'][0, 0]
    h_origen = cfg['radio_nominal']  # altura del eje del cilindro respecto a h_origen
    z_trasla = z_toca + cfg['array_gap'] - h_origen  # traslacion para llevar el origen al eje del cilindro

    # para hacer que el eje z apunta hacia arriba, hay que girar 180 grados alrededor del eje "y" o el "x"
    # elijo usar el "y", entonces cambian de signo las rotaciones alrededor de "x" y de "z". Tambien cambia el signo de
    # las componentes "x" y "z". Por otro lado, la componente "x" se puede poner en 0 por simetria
    # esto tambien tendria que hacerlo con las coordenadas del array
    array_coords = np.array(utils_3d.array_coordinates_list(cfg['nel_x'], cfg['nel_y'], cfg['pitch'], cfg['pitch']))
    array_coords[:, 0] = -array_coords[:, 0]

    # esto es para las adquisiciones en que el array estaba rotado 90 grados con respecto a los ejes del cilindro
    if cfg['rot_z90']:
        rot = Rotation.from_euler('z', 90)
        array_coords = rot.apply(array_coords)

    # distfun_pe = ifaz_3d.return_pulse_echo_cylinder_fun(array_coords, cfg['curv_sign'])
    if cfg['forma'] == 'cylinder':
        distfun = ifaz_3d.return_pitch_catch_cylfun_circmirror(array_coords, array_coords, cfg['curv_sign'])
    elif cfg['forma'] == 'plane':
        distfun = ifaz_3d.return_pitch_catch_plane_fun(array_coords, array_coords)
    else:
        raise Exception('forma no válida')

    dist = []
    pose = np.zeros((cfg['n_adq'], 6))
    for i in range(cfg['n_adq']):
        temp = mat_vars['poses'][i, :]
        pose[i, :] = np.concatenate((1000 * temp[0:3], temp[3:])) - np.array([0, 0, z_trasla, 0, 0, 0])
        # para hacer que el eje z apunta hacia arriba, hay que girar 180 grados alrededor del eje "y" o el "x"
        # elijo usar el "y", entonces cambian de signo las rotaciones alrededor de "x" y de "z". Tambien cambia el signo de
        # las componentes "x" y "z". Por otro lado, la componente "x" se puede poner en 0 por simetria
        pose[i, 0] = 0
        pose[i, 2] = -pose[i, 2]
        pose[i, 3] = -pose[i, 3]
        pose[i, 5] = -pose[i, 5]
        if cfg['forma'] == 'cylinder':
            temp = distfun(cfg['radio_nominal'], pose[i, 0:3] + np.array([0, 0, cfg['delta_z']]), ['xyz', pose[i, 3:]])
        elif cfg['forma'] == 'plane':
            temp = distfun(pose[i, 0:3] + np.array([0, 0, cfg['delta_z']]), ['xyz', pose[i, 3:]])

        dist.append(temp.reshape((121, 11, 11)))

    dist = np.concatenate(dist)

    # IMPORTANTE: lo del taps/2 !!!!
    t_idx = np.uint16(cfg['fs'] * dist / cfg['c1'] - cfg['idx_ret_ini'] - filtro_offset*cfg['taps']/2)
    return t_idx


def crear_mask_labels(input_data, t_idx, cfg):
    """Genera la "imagen" label, tipo máscara binaria, creando una franca de unos, desde el TOF teorico
    hacia adelante, una cantidad dada por cfg['label_width']"""

    # crear matriz de labels en base a t_idx
    labels = np.zeros(input_data.shape, dtype=np.int8)
    n_data = labels.shape[0]
    n_samples = labels.shape[-1]

    # creo una variable auxiliar con la misma shape que input_data, con el indice de samples en la ultima dimension
    aux = np.arange(n_samples)
    aux = aux.reshape((1, 1, 1, -1))
    aux = np.repeat(aux, 11, axis=2)
    aux = np.repeat(aux, 11, axis=1)
    aux = np.repeat(aux, n_data, axis=0)

    t_idx_aux = t_idx.reshape((n_data, 11, 11, 1))
    if 'label_width' in cfg.keys():
        selec = np.logical_and(t_idx_aux < aux, aux < t_idx_aux + cfg['label_width'])
    else:
        # el mask pinta tode debajo del eco de entrada
        selec = t_idx_aux < aux

    labels[selec] = 1

    return labels


def forzar_sample_number(matrix, n_samples):
    n = matrix.shape[-1]
    if n > n_samples:
        # recortar
        matrix = np.delete(matrix, slice(n_samples, n), axis=-1)

    elif n < n_samples:
        # agregar ceros
        temp = list(matrix.shape)
        temp[-1] = n_samples
        temp2 = np.zeros(temp)
        temp2[:, :, :, :n] = matrix
        matrix = temp2

    return matrix


def add_fliped_data(matrix):
    """Data augmentation"""
    temp1 = np.flip(matrix, axis=1)
    temp2 = np.flip(matrix, axis=2)
    return np.concatenate((matrix, temp1, temp2), axis=0)


def reduce_batch_size_random(matrix, idx):
    matrix_r = matrix[idx, ...]
    return matrix_r


def parificar(input_data, labels):
    """CHAPUZA:
    Es para pasar de (n, 11, 11, n_samples) a (n, 12, 12, n_samples).
    Copiar los ultimos A-scan para hacer que las dimensiones "elementos" sean par (12 en lugar de 11)"""

    n = input_data.shape[0]

    a = np.zeros((n, 12, 12, 1200))
    a[:, :11, :11, :] = input_data
    a[:, 11, 11, :] = input_data[:, 10, 10, :]

    b = np.zeros((n, 12, 12, 1200))
    b[:, :11, :11, :] = labels
    b[:, 11, 11, :] = labels[:, 10, 10, :]

    return a, b


def compute_t_offset(matrix, t_idx, window_width, max_offset=20):
    """ Buscar un offset óptimo de t_idx tal que en una ventana [offset + t_idx, offset + t_idx + window_width]
    sea máxima le "energía" """

    t_idx = np.expand_dims(t_idx, axis=-1)
    aux = np.arange(matrix.shape[-1]).reshape((1, 1, -1))
    aux = np.repeat(aux, 11, axis=0)
    aux = np.repeat(aux, 11, axis=1)

    ofs = np.arange(-max_offset, max_offset + 1)
    e = []
    for i in ofs:
        sel = np.logical_and(t_idx + i < aux, aux < t_idx + i + window_width)
        e.append(np.abs(matrix[sel]).sum())  # puse abs por si uso radio frecuencia
    j = np.argmax(np.array(e))
    return ofs[j]


def return_offset_fun(matrix, t_idx, kernel):
    assert kernel.size % 2 == 1  # debe ser impar
    hw = int((kernel.size - 1) / 2)  # half width

    # ideas prestada por chat gpt, para seleccionar una ventana en cada A-scan
    ax0_sel = np.arange(11)[:, None] # esto le agrega una dimension, queda (11, 1) el shape
    ax1_sel = np.arange(11)
    windowed_signals = np.lib.stride_tricks.sliding_window_view(matrix, window_shape=kernel.size, axis=2)


    kernel = np.expand_dims(kernel, axis=[0, 1])

    def fun(x):
        x = int(np.round(x))
        ofs_idx = np.clip(t_idx + x - hw, 0, matrix.shape[-1])  # para evitar porblemas de borde
        ventanas = windowed_signals[ax0_sel, ax1_sel, ofs_idx]
        q = np.abs(ventanas*kernel)
        return q.sum()

    return fun


def match_tensor_shapes(x1, x2):
    """Para resolver problema de concatenate en las skip layers"""
    n1 = x1.shape[3]  # numero de samples
    n2 = x2.shape[3]
    # x sera el tensor a cropear, el mayor de los dos
    if np.argmax([n1, n2]):
        q = [x1, x2]  # primero el pequeño, luego el grande
        dif = n2 - n1
    else:
        q = [x2, x1]
        dif = n1 - n2

    q[1] = Cropping3D(((0, 0), (0, 0), (0, dif)))(q[1])
    return q


def match_kernel_to_input(kernel_size, x):
    n1 = kernel_size[2]
    n2 = x.shape[3]
    if n1 > n2:
        q = list(kernel_size)
        q[2] = n2
    else:
        q = kernel_size
    return q


def return_sparse_fmc_selec(n_fmc, tx, nel_x=11, nel_y=11):
    """
    DEvuelve el array "selec" de indices para seleccionar las submatrices correspondientes
    a los emisores en la lista tx. Es para un batch de matrices fmc del tipo (batch, 11, 11, tiempo).
    Si el batch de fmc es input_data, entonces el subconjunto es input_data[selec, :, :, :]
    """

    linear_idx = [] # indices linear de los emisores en tx
    for ij in tx:
        temp, _ = utils_3d.array_ij2element(ij, nel_x, nel_y, 1, 1)  # el pitch no importa para esto
        linear_idx.append(temp)

    linear_idx = np.array(linear_idx)
    nel = nel_x*nel_y
    selec = linear_idx.copy()
    for i in range(1, n_fmc):
        # le suma de a 121 a los indices para las distintas fmc a lo largo del eje "batch"
        selec = np.append(selec, nel*i + linear_idx)

    return selec


#
# def return_idx_error_fun(peco_interval, u=0.5, window_num=10):
#     def idx_error(y_true, y_pred):
#         # forma (batch, 11, 11, samples, 1)
#         y_true = y_true.numpy()
#         y_pred = y_pred.numpy()
#         n_samples = y_true.shape[-2]  # el ultimo eje es el canal
#         idx_true = n_samples - 1 - y_true.sum(axis=-2)
#         idx_pred = utils.first_thr_cross(y_pred, peco_interval, u, window_num, axis=-2)
#         # idx_pred = idx_pred.reshape(-1, 121, 3)
#         err = idx_pred[:, :, :, 0, :] - idx_true
#         return err.mean(axis=(1, 2, 3))
#
#     return idx_error


def return_example(model, i, fmc, u1, u2, peco_interval, window_num=10, w_min=0, vnet_thr_inverso=False):
    x = fmc[i, :, :, :, 0]
    n_samples = x.shape[-1]
    if type(x) is not np.ndarray: # es tensor
        x = x.numpy()
    x = x.reshape((121, n_samples))
    idx1 = utils.first_thr_cross(x, peco_interval, u1, window_num)
    valid1 = idx1[:, 1] > w_min

    y = fmc[i, :, :, :, 0]
    y = model(np.expand_dims(y, axis=0)).numpy()
    y = y[0, :, :, :, 0].reshape((121, n_samples))
    if vnet_thr_inverso:
        y = 1 - np.flip(y, axis=0)
    idx2 = utils.first_thr_cross(y, peco_interval, u2, window_num)
    valid2 = idx2[:, 1] > w_min

    return x, idx1[:, 0], valid1, y, idx2[:, 0], valid2


def idx_error(y_true, y_pred, u=0.5):
    # forma (batch, 11, 11, samples, 1)
    n_samples = y_true.shape[-2]  # el ultimo eje es el canal
    idx_true = n_samples - 1 - tf.math.reduce_sum(y_true, axis=-2)
    idx_pred = tf.math.argmax(y_pred > u, axis=-2)


    err = tf.cast(idx_pred, tf.float32) - idx_true


    return tf.reduce_max(tf.math.abs(err), axis=(1, 2, 3))
@tf.keras.utils.register_keras_serializable(name='n_out_1')
def n_out_1(y_true, y_pred, u=0.5):
    # forma (batch, 11, 11, samples, 1)

    idx_pred = tf.math.argmax(y_pred > u, axis=-2)
    # contar ceros outlier 1 (nocruzar umbral)
    # Encontrar valores cero en el tensor
    zeros_mask = tf.equal(idx_pred, 0)

    # Contar ceros en el tensor
    n_out_1 = tf.reduce_sum(tf.cast(zeros_mask, tf.int32))


    return n_out_1

@tf.keras.utils.register_keras_serializable(name='n_out_2')
def n_out_2(y_true, y_pred, u=0.5, e_max=40):
    # forma (batch, 11, 11, samples, 1)
    n_samples = y_true.shape[-2]  # el ultimo eje es el canal
    idx_true = n_samples - 1 - tf.math.reduce_sum(y_true, axis=-2)
    idx_pred = tf.math.argmax(y_pred > u, axis=-2)
    err = tf.cast(idx_pred, tf.float32) - idx_true

    # contar ceros outlier 2 (error mayor que e_max)

    #Contar valores mayores que el umbral en el tensor
    n_out_2 = tf.reduce_sum(tf.cast(tf.greater(tf.math.abs(err), e_max), tf.int32))

    return n_out_2

def weighted_binary_crossentropy(target, output, w):

    target = tf.convert_to_tensor(target)
    output = tf.convert_to_tensor(output)
    weights = tf.convert_to_tensor(w, dtype=target.dtype)

    epsilon_ = tf.constant(tf.keras.backend.epsilon(), output.dtype.base_dtype)
    # output = tf.clip_by_value(output, epsilon_, 1.0 - epsilon_)

    # Compute cross entropy from probabilities.
    bce = weights[1] * target * tf.math.log(output + epsilon_)
    bce += weights[0] * (1 - target) * tf.math.log(1 - output + epsilon_)
    return -bce


@tf.keras.utils.register_keras_serializable(name='weighted binary crossentropy')
class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, w, **kwargs):
        super().__init__()
        self.w = w

    def call(self, y_true, y_pred):
        return weighted_binary_crossentropy(y_true, y_pred, self.w)

    def get_config(self):
        # Devuelve la configuración necesaria para reconstruir la función de pérdida
        config = super().get_config()
        config.update({'w': self.w})
        return config

################################
#         Tversky loss         #
################################
def tversky_loss(y_true, y_pred,delta = 0.7, smooth = 0.000001):
    """Tversky loss function for image segmentation using 3D fully convolutional deep networks
	Link: https://arxiv.org/abs/1706.05721
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    smooth : float, optional
        smoothing constant to prevent division by zero errors, by default 0.000001
    """

    axis = [1,2,3]
    # Calculate true positives (tp), false negatives (fn) and false positives (fp)
    tp = K.sum(y_true * y_pred, axis=axis)
    fn = K.sum(y_true * (1-y_pred), axis=axis)
    fp = K.sum((1-y_true) * y_pred, axis=axis)
    tversky_class = (tp + smooth)/(tp + delta*fn + (1-delta)*fp + smooth)
    # Average class scores
    tversky_loss = K.mean(1-tversky_class)

    return tversky_loss


@tf.keras.utils.register_keras_serializable(name='tversky loss')
class Tversky_loss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__()
        # self.delta = delta
        # self.smooth=smooth

    def call(self, y_true, y_pred):
        return tversky_loss(y_true, y_pred)

    def get_config(self):
        # Devuelve la configuración necesaria para reconstruir la función de pérdida
        config = super().get_config()
        # config.update({'delta': self.delta})
        # config.update({'smooth': self.smooth})
        return config

# @tf.keras.utils.register_keras_serializable(name='idx_error')
# class IdxError(tf.keras.metrics.Metric):
#
#     def __init__(self, u, **kwargs):
#         super().__init__(**kwargs)
#         self.u = u
#         self.n = tf.Variable(0, dtype=tf.float32)
#         self.idx_error = tf.Variable([0], dtype=tf.float32)
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         err = idx_error(y_true, y_pred, self.u)
#         self.idx_error.assign_add(tf.reduce_sum(tf.math.abs(err), axis=(1, 2, 3)))
#         self.n.assign_add(y_true.shape[0])
#
#     def result(self):
#         return self.idx_error/self.n
#
#     def reset_states(self):
#         self.n.assign(0)
#         self.idx_error.assign(0)

@tf.keras.utils.register_keras_serializable(name='idx_error')
class IdxError(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, u=0.5):
        super().__init__(idx_error, u=u)


def change_model_input_samples(cfg, model_name, n):
    """Si un modelo está entrenado para un input de (11, 11, n_samples), y queremos meterle algo
    con otro nro de samples n, creamos un modelo nuevo copiando los pesos del otro. Pero en el cuello
    de la red puede diferir el tamaño del kernel, entonces los "matcheo" artesanalmente.
    Lo he probado y no funciona muy bien...."""

    model = tf.keras.models.load_model(
        r'C:\Users\ggc\\PROYECTOS\UTIMAG\utimag\imag3D\CNN_superficie\trained_models\\' + model_name + '.h5',
        custom_objects={'tversky_loss': tf.keras.utils.get_custom_objects()['Custom>tversky loss'],
                        'idx_error': tf.keras.utils.get_custom_objects()['Custom>idx_error'],
                        'n_out_1': tf.keras.utils.get_custom_objects()['Custom>n_out_1'],
                        'n_out_2': tf.keras.utils.get_custom_objects()['Custom>n_out_2']})

    kernel_regu = cfg['kernel_regu'] if 'kernel_regu' in cfg else None
    bias_regu = cfg['bias_regu'] if 'bias_regu' in cfg else None
    new_shape = (11, 11, n, 1)
    new_model = cu.custom_vnet(new_shape, use_batch_norm=True, use_attention=cfg['attention'], num_classes=1,
                               num_layers=cfg['n_layers'], filters=cfg['n_filters'], pool_size=cfg['pool_size'],
                               conv_kernel_size=cfg['conv_kernel_size'],
                               conv_strides=cfg['conv_strides'], dropout=0.3, output_activation='sigmoid',
                               kernel_regularizer=kernel_regu, bias_regularizer=bias_regu)

    conf = model.get_config()
    new_conf = new_model.get_config()

    conv_list = []
    new_conv_list = []
    batch_list = []
    new_batch_list = []

    for i in range(len(model.layers)):  # todo: Puede ser que no tengan el mismo nro de layers!!!!!!!!!!!!!!!!!

        if conf['layers'][i]['class_name'] in ['Conv3D', 'Conv3DTranspose']:
            conv_list.append(model.layers[i])
        if new_conf['layers'][i]['class_name'] in ['Conv3D', 'Conv3DTranspose']:
            new_conv_list.append(new_model.layers[i])
        if conf['layers'][i]['class_name'] == 'BatchNormalization':
            batch_list.append(model.layers[i])
        if new_conf['layers'][i]['class_name'] == 'BatchNormalization':
            new_batch_list.append(new_model.layers[i])

    for b1, b2 in zip(batch_list, new_batch_list):
        b2.set_weights(b1.get_weights())

    # matcheo de tamaños de los kernels, invento arbitrario mío
    for c1, c2 in zip(conv_list, new_conv_list):
        ksize1 = c1.get_config()['kernel_size']
        ksize2 = c2.get_config()['kernel_size']
        if ksize2[-1] < ksize1[-1]:
            w = np.array(c1.get_weights())
            c2.set_weights(w[:, :, :, 0:ksize2[-1], :, :])
        elif ksize2[-1] > ksize1[-1]:
            w = np.array(c1.get_weights())
            new_w_shape = list(w.shape)
            new_w_shape[3] = ksize2[-1]
            new_w = np.zeros(new_w_shape, dtype=np.float32)
            new_w[:, :, :, 0:ksize1[-1], :, :] = w
            c2.set_weights(new_w)
        else:
            # tamaños iguales
            c2.set_weights(c1.get_weights())

    return new_model
