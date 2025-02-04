# -*- coding: utf-8 -*-
import os
import numpy as np
import pyopencl as cl
from scipy import signal
from dataclasses import dataclass
import matplotlib.widgets as mwidgets
import argparse
import json5

"""
Varias funciones y clases. 
"""

# definición de constantes

# MODIFICO ESTO EN EL PROYECTO DE MARCELO PARA QUE NO USE EL PATH DEL SISTEMA QUE APUNTA AL DE MARIO!!!!!!!

# chequear que exista la variable de entorno con el path del paquete utimag, UTIMAG_PATH
# if 'UTIMAG_PATH' in os.environ:
#     utimag_path = os.environ.get('UTIMAG_PATH')
# else:
#     raise Exception('Variable de entorno UTIMAG_PATH no definida')

utimag_path = r'C:\Marcelo\utimag-marcelo'


# ------------------- COEFCICIENTES FILTRO HILBERT ----------------------------------------------------------------
# coeficientes de filtro Hilbert que me pasó J. Camacho
# WARNING: este filtro está hardcoded
# tiene 63 coeficientes, por lo cual se debe usar taps=62
# TODO: usar distinto nro de taps para el filtro pasabando y para Hilbert
# modo de no obligar a que el pasabanda tenga necesariamente taps=62
q = np.loadtxt(utimag_path + r'\hilbert_63_16b.coe', delimiter=',')
# normalizar filtro para que tenga gananacia=1
temp = signal.freqz(q)
HILB_COEF = q / np.abs(temp[1]).max()

# --------------------NOMBRES DE MACROS PARA CÓDIGO OPEN-CL-----------------------------------------------------
# nombres de parámetros necesarios para el cálculo de imágenes, clasificados en enteros y floats, porque
# es necesario para generar los macros del código OpenCl. Los floats literales debe ser del tipo 4.6f, 5.f, etc
# o sea, hay que agregarles la "f"

with open(utimag_path + r'\CL_macros_names.json5') as f:
    macs = json5.load(f)

INT_LIST = macs['INT']
FLOAT_LIST = macs['FLOAT']

# ------------------------------------------------------------------------------------------------------------------


def db(x):
    return 20 * np.log10(x)


def normalizar(img):
    return img / img.max()


def rolling_rms(s, span):
    rms = np.zeros_like(s)
    for i in range(span, s.shape[-1] - span):
        temp = s[:, (i - span):(i + span)]
        temp = np.abs(temp) ** 2
        rms[:, i] = np.sqrt(temp.sum(axis=-1))
    return rms


def roi2shape_2d(roi, x_step, z_step):
    """Se supone un roi: (x_min, x_max, z_max, z_min). Notar que z está al revés que lo usual"""
    nz = int((roi[2] - roi[3])/z_step)
    nx = int((roi[1] - roi[0]) / x_step)
    return nz, nx


def roi2shape_3d(roi, x_step, y_step, z_step, cifras=2, use_float32=False):
    """Se supone un roi: (x_min, x_max, y_min, y_max, z_max, z_min). Notar que z está al revés que lo usual.
    IMPORTANTE: Uso np.around con 2 (o más) cifras decimales, porque al usar la funcion que rota los véritces del roi,
    (con rotaciones de Scipy), aparecen números como 4.999999, que debería ser 5."""

    if use_float32:
        # ALERTA ******************************************************************************
        # esta opcion es para el caso de rotacion interpolación, porque resulta que a veces
        # esta cuenta da diferente hecha en float64 y en float32. Porque a veces alguna resta
        # en float32 da 4.99999 por ej, en float 64 da 5, y al hacer floor, uqeda 4 en un caso y
        # 5 en el otro

        roi = np.array(roi, dtype=np.float32)
        nz = np.floor((roi[4] - roi[5])/z_step)
        nx = np.floor((roi[1] - roi[0]) / x_step)
        ny = np.floor((roi[3] - roi[2]) / y_step)
    else:
        nz = np.around(((roi[4] - roi[5])/z_step), cifras)
        nx = np.around(((roi[1] - roi[0]) / x_step), cifras)
        ny = np.around(((roi[3] - roi[2]) / y_step), cifras)

    return int(nz), int(nx), int(ny)


def init_rtx3080():
    # seleccionar un "device"
    plat = cl.get_platforms()  # lista de plataformas
    gpu = plat[0].get_devices()  # esta es la GPU
    ctx = cl.Context(gpu)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    return ctx, queue, mf


def load_cfg(script_path):
    """Para pasar el json5 desde la línea de comandos al ejecutar el script

    --configfile: ruta del json5
    --relativepath: con esta opción se pasa la ruta relativa a la ruta del script. En caso contrario hay que pasar la
    ruta absoluta del json5
    --samename: falta implementar

    Si no se pasa ningún argumento por línea de comandos, se pide un input de usuario, que debe ser
    ruta absoluta
    Return: diccionario de parámetros y opciones y lo que sea"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--configfile', '-c')
    parser.add_argument('--relativepath', '-r', action='store_true')
    args = parser.parse_args()

    if args.configfile is not None and not args.relativepath:
        fname = args.configfile
    elif args.relativepath:
        path = script_path + '/'
        fname = path + args.configfile
    else:
        fname = input('Config file: ')

    with open(fname) as f:
        cfg = json5.load(f)  # diccionario

    return cfg


def xz2index(x, z, roi, x_step, z_step):
    i = int((roi[2] - z) / z_step)
    j = int((x - roi[0]) / x_step)
    return i, j


def crop_xy_square(img, roi, center, d, x_step, y_step):

    """Recorta una sub imagen cuadrada centrada en "center", con mitad de lado "d" """
    assert x_step == y_step  # todo: que funciones pasos distintos en x e y
    m = int(d/x_step)
    i = int((roi[3] - center[1]) / y_step) # direccion Y, vertical, eje 0
    j = int((center[0] - roi[0]) / x_step)
    subroi = [center[0] - d, center[0] + d, center[1] - d, center[1] + d]
    return img[i-m:i+m, j-m:j+m], subroi


# def crop_xy_subroi(ax, img, roi_xy, x_step, y_step):
#
#     q = 0
#
#     def onselect(eclick, erelease):
#         x1, x2, y1, y2 = rect.extents
#
#         # OJO CON EL ORDEN DE INDICES RESPECTO DE X E Y !!!!!!
#         i1 = int((y2 - roi_xy[2]) / y_step)
#         i2 = int((y1 - roi_xy[2]) / y_step)
#         j1 = int((x1 - roi_xy[0]) / x_step)
#         j2 = int((x2 - roi_xy[0]) / x_step)
#         q = img[i1:i2, j1:j2]
#
#     rect = mwidgets.RectangleSelector(ax, onselect, interactive=True)
#     return q


def parameters_macros(pdic, float_list, int_list):
    """pdic: diccionario con parámetros numéricos
       float_list: lista de keys
       int_list: lista de keys

       Aquellos parametros no definidos en pdic, se les pone el valor arbitrario 0. Si no
       se definiesen fallaría la compilación del codigo open-cl por macros sin definir"""

    k = pdic.keys()
    s = ''
    not_set = []
    for key in float_list:
        if key in k:
            s += '#define ' + key.upper() + ' ' + str(np.float32(pdic[key])) + 'f' + '\n'
        else:
            s += '#define ' + key.upper() + ' ' + '0.f' + '\n'
            not_set.append(key)
    for key in int_list:
        if key in k:
            s += '#define ' + str(key).upper() + ' ' + str(pdic[key]) + '\n'
        else:
            s += '#define ' + str(key).upper() + ' ' + '0' + '\n'
            not_set.append(key)

    if len(not_set) > 0:
        print('WARNING: undefined parameters (default value: 0): ')
        print(not_set, '\n')

    return s


def ifaz_2d_macros(ifaz):
    """Escribe los macros que definen las funciones para calcular la interfaz"""
    if ifaz is None:
        # definir arbitrariamente como 0, igual no se van a usar
        mac = '#define z_ifaz(x)            0.f\n' + \
              '#define dz_ifaz(x, z)        0.f\n' + \
              '#define d2z_ifaz(x, z, dz)   0.f\n'
    elif ifaz == 'circle':
        mac = '#define z_ifaz(x)            sqrt(pown(p[2], 2) - pown(x - p[0], 2)) + p[1]\n' + \
              '#define dz_ifaz(x, z)        (p[0] - x) / (z - p[1])\n' + \
              '#define d2z_ifaz(x, z, dz)   (1 + pown(dz, 2)) / (p[1] - z)\n'
    elif ifaz == 'polynomial':
        mac = '#define z_ifaz(x)            p[0]*pown(x, 4) + p[1]*pown(x, 3)  + p[2]*pown(x, 2) + p[3]*x + p[4]\n' + \
              '#define dz_ifaz(x, z)        4*p[0]*pown(x, 3) + 3*p[1]*pown(x, 2)  + 2*p[2]*x + p[3]\n' + \
              '#define d2z_ifaz(x, z, dz)   12*p[0]*pown(x, 2) + 6*p[1]*x + 2*p[2]\n'
    else:
        raise Exception('interfaz no válida')

    return mac


class DualBuffer:

    def __init__(self, queue, nparray, shape, data_type, read_only=False):
        """Objeto que contiene un array numpy y el correspondiente Buffer de PyOpenCL con el tamaño
        adecuado"""
        self.queue = queue
        self.data_type = data_type
        mf = cl.mem_flags
        aux = mf.READ_ONLY if read_only else mf.READ_WRITE
        if nparray is None:
            self.cpu = np.zeros(shape, dtype=data_type)
            self.gpu = cl.Buffer(queue.context, aux, size=self.cpu.nbytes)
        else:
            self.cpu = nparray.astype(data_type)  # asegurarse de que el tipo de dato sea correcto
            assert nparray.flags['C_CONTIGUOUS']  # asegurarse de que tiene el memory layput correcto !!
            self.gpu = cl.Buffer(queue.context, aux | mf.COPY_HOST_PTR, hostbuf=self.cpu)

    def c2g(self, nparray):
        """Copiar numpy array desde el host al buffer de GPU"""
        if nparray is not None:
            assert nparray.flags['C_CONTIGUOUS']  # asegurarse de que tiene el memory layput correcto !!
            self.cpu = nparray.astype(self.data_type)
        cl.enqueue_copy(self.queue, self.gpu, self.cpu).wait()

    def g2c(self):
        """Copia desde el buffer de GPU al host"""
        cl.enqueue_copy(self.queue, self.cpu, self.gpu).wait()
        # el return tiene que ser una copia. Si se devuelve self.cpu, se está devolviendo
        # una referencia, y se puede armar quilombo, por que se puede modificar el buffer
        # involuntariamente
        return self.cpu.copy()

    def check_size(self):
        """Chequear si los tamaños son iguales"""
        if self.cpu.nbytes != self.gpu.size:
            raise Exception('Error: tamaños distintos')


def return_knl_args(buf_dict, knl_arg_names):
    """A partir de un diccionario de DualBuffers, y una lista de nombres, crea un alista de argumenstos
    para pasar a un kernel"""
    args = [buf_dict[k].gpu for k in knl_arg_names]
    return args


def fmc_subap_mask(matrix, subap):
    """ Crea array con la misma forma que matrix, con unos en la parte que corresponde a la subapertura
    creando a partir de la matriz FMC completa una barrido lineal FMC"""

    # definir máscara
    n = matrix.shape[0]
    a = np.zeros((n, n), dtype=np.float32)
    for k in range(subap):
        a += np.eye(n, k=k - subap + 1)
    a = np.expand_dims(a, 2)
    a = np.repeat(a, matrix.shape[2], axis=2)
    return a


def gaussian_pulse(freq, bw, t0, t_min, t_max, fs):
    assert t_min < t0 < t_max
    tau = 1 / (bw * freq)  # chequear esta formula, todo
    t = np.arange(t_min, t_max + 1/fs, 1 / fs)
    q = np.exp(-((t - t0) / (2 * tau)) ** 2 + 1j * (2 * np.pi) * freq * (t - t0))
    return q


def sum_gaussian(freq, bw, t_max, fs, delay):
    tau = 2.355 / (2*np.pi * bw * freq)  # chequear esta formula, todo
    t = np.arange(-t_max, t_max, 1 / fs)
    q = np.zeros_like(t, dtype=np.complex128)
    delay = delay - delay.mean()  # restar la media
    for t0 in delay:
        q += np.exp(-((t - t0) / (2 * tau)) ** 2 + 1j * (2 * np.pi) * freq * (t - t0))
    # q = np.zeros((t.size, delay.size), dtype=np.complex128)
    # for i in range(delay.size):
    #     q[:, i] = np.exp(-((t-delay[i])/(2*tau))**2 + 1j*(2*np.pi)*freq*(t - delay[i]))
    return q, t


# función para calcular un global_size como múltiplos de local_size (zvu ó zu), que cubra la img_shape (vuz ó uz)
# notar que se permutan los ejes zvu ---> vuz (zu ---> uz)
def loc2glob(local_size, img_shape):
    h = lambda n, m: m * np.short(np.ceil(n / float(m)))  # ej: si n = 35 y m = 4, 4*6 = 36, entonces h = 6
    global_size = (h(img_shape[0], local_size[0]), h(img_shape[1], local_size[1]))

    return global_size


def img_line_perfil(img, roi, x1, z1, x2, z2, dx, franja=None):
    """
    Calcula imagen sobre un recta que va desde (x1, z1) hasta (x2, z2)
    Supone que dx=dz, si son distinto falla, se puede arreglar
    """
    if franja is None:
        x = np.arange(x1, x2, dx)
        z = (x - x1) * (z2 - z1) / (x2 - x1) + z1
        ix = np.round((x - roi[0]) / dx)
        ix = ix.astype(np.int)
        iz = -np.round((z - roi[2]) / dx)
        iz = iz.astype(np.int)
        perfil = np.abs(img[iz, ix])
    else:
        x = np.arange(x1, x2, dx)
        # dos rectas paralelas
        z_min = (x - x1) * (z2 - z1) / (x2 - x1) + z1 - franja
        z_max = (x - x1) * (z2 - z1) / (x2 - x1) + z1 + franja
        ix = np.round((x - roi[0]) / dx)
        ix = ix.astype(np.int)
        iz_min = -np.round((z_min - roi[2]) / dx)
        iz_min = iz_min.astype(np.int)
        iz_max = -np.round((z_max - roi[2]) / dx)
        iz_max = iz_max.astype(np.int)
        perfil = np.zeros(x.size)
        for i in range(x.size):
            temp = img[iz_max[i]:iz_min[i], ix]
            perfil[i] = np.abs(temp).mean()

    return perfil / perfil.max(), x


def busca_grupos(data, param, op, noZeros):
    """
    Busca grupos contiguos en un vector que cumplan el criterio (excluyendo los valores 0):
    op = 0 : data >= param
    op = 1 : data < param
    op = 2 : abs(data[i]-data[i-1]) < param
    noZeros = 1 : no tienen en cuenta los elementos que valen cero
    devuelve
    g  = vector de tamaño data.size con el grupo al que pertenece cada elemento
         el grupo '0' significa que el elemento no cumple el criterio y no está en ningún grupo
    ng = número de elementos de cada grupo
    ix = elemento inicial y final de cada grupo (tamaño 2xn.size)
    """

    # todo: agregar la opción de unir intervalos separados por menos de M puntos.
    #  La idea es que si dos grupos están separados por 1 sólo elemento que no cumple, es un outlier

    g = np.zeros(data.size)  # Grupo al que pertenece cada elemento
    dentro = 0  # Bandera que indica si estoy dentro de un intervalo válido
    ixg = 0  # Indice al grupo actual
    n0 = 0  # Número de elementos que no pertenecen a ningún grupo
    ng = np.array([0])  # Número de elementos de cada grupo
    ix = np.array([[0], [data.size - 1]])  # elemento inicial y final de cada intervalo
    for i in range(data.size):  # para cada elemento del vector
        # Verifico si se cumple la condición según el operador 'op' y la bandera noZeros
        if (((data[i] > 0) and (noZeros)) or not (noZeros)) and \
                (((data[i] >= param) and (op == 0))
                 or ((data[i] >= param) and (op == 1))
                 or ((np.abs(data[i] - data[i - 1]) < param) and (op == 2))):
            if dentro == 0:  # Cumple condición y estaba fuera: comienza intervalo nuevo
                dentro = 1  # Bandera dentro de intervalo
                ixg = ixg + 1  # Ingremento el número de grupos
                n = 1  # Número de elementos del grupo actual
                g[i] = ixg  # Etiqueta del grupo al que pertenece el elemento
                ng = np.append(ng, 0)  # Incremento el 1 el vector de nº de elementos
                ix = np.append(ix, np.array([[i], [0]]), 1)  # Incremento y marco el inicio del intervalo
            else:
                # Se cumple la condición y estoy dentro de un intevalo : sigo en el mismo grupo
                g[i] = ixg  # Etiqueto el elemento
                n = n + 1  # Incremento el número de elementos del grupo
        else:  # El elemento no cumple con el criterio
            n0 = n0 + 1  # Incremento el núero de elementos que no cumplen el criterio
            if dentro == 1:  # Estoy dentro del intervalo y no cumple : se cierra el grupo
                dentro = 0  # Bajo la bandera dentro de grupo
                g[i] = 0  # Etiqueto el elemento en el grupo 0
                ng[ixg] = n  # Actualizo el número de elementos en el grupo que estoy cerrando
                ix[1, ixg] = i - 1  # Indice al último elemento del grupo que estoy cerrando
            else:
                # No cumple el criterio y está fuera de un intervalo : sigo fuera
                g[i] = 0  # Etiqueto el elemento en el grupo 0

    if dentro == 1:  # Terminé dentro de un intervalo. Lo cierro
        ng[ixg] = n  # Número de elementos del grupo
        ix[1, ixg] = i  # Índice al elemento final del grupo

    ng[0] = n0  # número de elementos que no cumplen el criterio

    return g, ng, ix


def min_samples_circ(x_0, radio, espesor, zgap, c1, c2, fs, m=3):
    """ Funcion para estimar el minimo tiempo de adquisicion necesarios para una interfaz circular
    con un espesor de pared (pensado para pieza tipo seccion de tubo)
    Args:
        x_0:
        radio:
        espesor:
        zgap:
        c1:
        c2:
        fs:
        m:

    Returns:

    """

    # dos casos
    # radio < x_0, tomamos la linea que une el centro del círculo con el primer elemento, y medimos
    # la parte en agua
    # radio >= x_0, tomamos la linea vertical que pasa por el primer elemento y cruza el círculo
    if radio < x_0:
        d1 = (np.hypot(x_0, zgap + radio) - radio)
    else:
        cuerda = radio - np.sqrt(radio ** 2 - x_0 ** 2)
        d1 = (zgap + cuerda)

    d2 = m * espesor
    tof1 = d1 / c1
    tof2 = d2 / c2  # m veces el espesor de pared
    tof = (d1 - zgap) / c1 + d2 / c2  # resto zgap que es camino de agua mínimo
    return 2 * fs * tof

    # Calcular nro mímimo de samples a adquirir para un tubo (zc, rmax, rmin) centrado bajo el array.
    # Se consideran dos rayos con recorridos largos, y se toma el tiempo máximo. Un rayo va hacia el punto
    # (0, zc + rmin) desde el primer elemento, y va por camino simétrico hasta el último. El otro rayo va y vuelve
    # el útimo elemento al punto (-rmin, zc)
    # """
    #
    # # punto de coordenadas (0, zc + rmin), tomando zc negativo
    # zf = pdic['zc'] + pdic['rmin']
    # rayo1 = av.fermat_gd_circ(0, 0, 0, zf, 0, pdic['c1'], pdic['c2'], 0, pdic['zc'], pdic['rmax'],
    #                           pdic['gamma'])
    #
    # # punto de coordenadas (-rmin, zc), tomando zc negativo
    # rayo2 = av.fermat_gd_circ(pdic['x_0'], 0, pdic['rmin'], 0, 0, pdic['zc'], 0, pdic['c1'], pdic['c2'],
    #                           pdic['rmax'], pdic['gamma'])
    #
    # # chequear si el rayo2 tiene sentido. Si cruza el círculo interno no vale.
    # test = rayo2['xe'] < pdic['rmin']
    # if test:
    #     # rayo2 no sirve
    #     tof = rayo1['tof']
    # else:
    #     tof = np.maximum(rayo1['tof'], rayo2['tof'])
    #
    # return 2 * pdic['fs'] * tof


def measure_profile_width(s, h, n):
    """
    Función para medir el ancho de la indicacion de un defecto en un perfil.
    La señal s tiene que tener un pico en su máximo absoluto. Se ajusta con una cuadrática en un intervalo
    alrededor del máximo de cuyo ancho es 2*n, y se calcula los puntos a altura relativa h para medir el ancho"""
    amax = np.max(s)
    imax = np.argmax(s)
    x = np.arange(imax - n, imax + n)
    y = s[imax - n: imax + n]
    p = np.polyfit(x, y, 2)
    # resolver cuadrática para el cruce
    pp = p - np.array([0, 0, h * amax])  # otro polinomia restando la altura del cruce
    r = np.roots(pp)
    return np.abs(r[1] - r[0])


def detect_first_echo_1d(s, rango, umbral, n):
    """
    Detectar el índice del primer eco con distintos métodos:

    Args:
        s: A-scan, array 1D
        rango: [i0, i1], intervalo en que se estima que está el primer eco
        umbral: [u0, u1, u2],
            u0: valor menor a 1, para definir un umbral relativo al máximo
            u1: umbral absoluto (del orden de 30 debe ser)
            u2: umbral para la "derivada" del A-scan, la variacion entre 2 samples se compara con esto
        n: número de veces consecutivas en que la variación debe superar al umbral para considerar que es un flanco

    Returns:
        idx[0]: Primer flanco positivo
        idx[1]: Primer flanco negativo
        idx[2]: Cruce de umbral absoluto
        idx[3]: Cruce de umbral relativo al máximo (umbral[0] * amplitud del pico)
        idx[4]: indice del máximo
        idx[5]: valor absoluto del máximo

    """

    assert len(umbral) == 3
    assert umbral[0] < 1  # umbral como fraccion del máximo

    w = np.zeros(6)  # q: (flanco positivo, flanco negativo, cruce uabs, cruce urel, argmax, max)
    # recortar la parte donde debe estar el primero eco
    i0, i1 = rango
    s = s[i0:i1]
    s_abs = np.abs(s)  # valor absoluto del A-scan
    amp_pico = s_abs.max()
    w[5] = amp_pico
    w[4] = np.argmax(s_abs) + i0   # indice del máximo absoluto dentro del rango
    # compara con umbral relativo al máximo. Busca el primer 1: 0000000000011111111111111111111111111111111
    w[3] = np.argmax(s_abs > umbral[0]*amp_pico) + i0  # indice del primer 1
    # compara con umbral absoluto
    # ¿que pasa si nunca supera el umbral abosluto? argmax dará 0...
    w[2] = np.argmax(s_abs > umbral[1]) + i0  # indice del primer 1

    ds = np.diff(s)
    i = 0
    w[0], w[1] = np.nan, np.nan  # inicializa con "nan" por si no encuentra el flanco, queda ese valor
    test = [False, False]

    while not (test[0] and test[1]) and (i + n) < s.size:
        q = ds[i: (i + n)]  # n samples consecutivas de la derivada
        if np.all(q >= umbral[2]):  # testea flanco positivo
            test[0] = True
            w[0] = i + i0
        if np.all(q <= -umbral[2]):  # testea flanco negativo
            test[1] = True
            w[1] = i + i0
        i += 1

    return w


def detect_first_echo(s, rango, umbral, n):
    return np.apply_along_axis(detect_first_echo_1d, -1, s, rango, umbral, n)


def first_thr_cross_1d(s, rango, umbral, window_size):
    i0, i1 = rango
    s = s[i0:i1]
    s_abs = np.abs(s)
    idx = np.argmax(s_abs > umbral)
    # si nunca se supera el umbral, idx da 0 el peso
    # TODO: usar otras medidas para calcular el peso, por ej: el valor RMS en la ventana (norma L2), o la suma de
    # TODO: valores absolutos (norma L1). Medir tambien en la ventana antes del eco
    if idx > 0:
        s_abs_window = s_abs[idx:(idx + window_size)]
        idx_max = np.argmax(s_abs_window)
        w = s_abs_window.max()
        #w = s_abs_window.max() / umbral  # divido por umbral para escalarlo a algo
    else:
        idx_max = 0 # valor arbitrario
        w = 0  # se asigna peso cero porque no se detectó nada

    # # calcular la recta tangente, e intersectar con y=0
    #
    # pendiente = s_abs[idx + 1] - s_abs[idx]
    # retroceso = s_abs[idx] / pendiente

    # q = []
    # for i in range(s.size - window_size):
    #     sw = s_abs[i:i+window_size] - umbral
    #     temp = sw[sw > 0]
    #     q.append(temp.sum()/window_size)
    #
    # q = np.array(q)
    # idx2 = np.argmax(q > otro_umbral)

    # como los idx son int64, pero w puede ser float, homogeneizo el tipo de datos a float
    return np.array([idx + i0, w, idx_max + idx + i0]).astype('float32')


def first_thr_cross(s, rango, umbral, window_max, axis=-1):
    # devuelve un array de Nx3.
    # Primera columna es el indice donde se cruza el umbral
    # La segunda es el valor del máximo en la ventana
    # La tercera es el indice de ese maximo
    return np.apply_along_axis(first_thr_cross_1d, axis, s, rango, umbral, window_max)


def first_thr_cross_umbral_por_canal(s, rango, umbral, window_size):
    assert s.shape[0] == umbral.size
    s = s[:, slice(*rango)]
    s_up = s > umbral.reshape((-1, 1))
    idx = np.argmax(s_up, axis=-1)
    w = np.zeros((idx.size, ))
    for i in range(s.shape[0]):
        if idx[i] > 0:
            sw = s[i, idx[i]:idx[i] + window_size]
            w[i] = sw.max()
        else:
            w[i] = 0

    # return np.concatenate([idx + rango[0], w], axis=0)
    return idx + rango[0], w


def thr_cross_integral_1d(s, umb):
    w = 0
    out = np.zeros_like(s)
    for i in range(s.size):
        up = s[i] > umb
        # q es una variable que va contanto el nro de veces consecutivas en que s es mayor al umbral
        # ni bien s cae debajo del umbral q se resetea a 0. La variable w calcula el valor medio de lo
        # que supera el umbral durante ese período
        # en que la señal se mantiene encima del umbral
        if up:
            w += s[i] - umb
        else:
            w = 0

        out[i] = w

    return np.array(out)


def thr_cross_integral(s, umb):
    return np.apply_along_axis(thr_cross_integral_1d, -1, s, umb)


# def thr_cross_integral_1d(s, umb1, umb2, method):
#     cond = True
#     i = 0
#     q = 0
#     w = 0
#     out = []
#     while cond and i < s.size:
#         up = s[i] > umb1
#         # q es una variable que va contanto el nro de veces consecutivas en que s es mayor al umbral
#         # ni bien s cae debajo del umbral q se resetea a 0. La variable w calcula el valor medio de lo
#         # que supera el umbral durante ese período
#         # en que la señal se mantiene encima del umbral
#         if up:
#             q += up
#             w += s[i] - umb1
#             out.append(w) # registra los valores de w
#         else:
#             q = 0
#             w = 0
#             out.append(w)
#         # si la señal se mantiene por encima del umbral un tiempo(nro de muestras) mayor a umb_t, entonces
#         # se para el loop, y se considera que ahi llego el primer eco
#         if method == 'tiempo':
#             cond = q <= umb2
#         elif method == 'amplitud':
#             # w = w / q if q > 0 else 0  # promediar si q es mayor a 0
#             cond = w <= umb2
#         i += 1
#
#     ts_idx = i - q  # tiempo (indice) de arrivo del primer eco (eco de la superficie)
#
#     return out, ts_idx


def transition_function_1d(s, rango, width):
    """Comaprar señal antes y despues de un tiempo, en una ventana atras, y otra por delante"""
    i0, i1 = rango
    s = s[i0:i1]
    s_abs = np.abs(s)
    q = []
    for i in range(width, s.size - width):
        aux1 = s[i-width:i]
        aux2 = s[i:i + width]
        q.append(aux2.max()-aux1.max())
    # q = width*[0] + q + width*[0]
    i0 = i0 + width
    i1 = i1 - width
    return np.array(q), i0, i1


def sklearn_reg2fun(reg, degree=2):
    """Funcion auxiliar para generar los coeficientes y la funcion del polinomio que interpola los puntos
    de la superficie, a partir del objeto LinearResgression de scikit learn"""
    coef = np.append(reg.coef_, reg.intercept_)

    if degree==2:
        def surf_fun(x, y):
            q = coef[5] + coef[0] * x + coef[1] * y + coef[2] * x ** 2 + coef[3] * y ** 2 + coef[4] * x * y
            grad_q = np.array([coef[0] + 2 * coef[2] * x + coef[4] * y, coef[1] + 2 * coef[3] * y + coef[4] * x])
            return q, grad_q

        c = coef.copy()

    else: # degree=1
        def surf_fun(x, y):
            q = coef[2] + coef[0] * x + coef[1] * y
            grad_q = np.array([coef[0], coef[1]])
            return q, grad_q

        c = np.array([coef[0], coef[1], 0, 0, 0, coef[2]])

    return surf_fun, c


def xy_scan_points(nx, step_x, ny, step_y, first_axis):
    """ Supone un escaneo que empieza en (0,0), avanza en step_x, luego en step_y
    , y luego retrcede en step_x, etc"""

    if first_axis == 'x':
        x = np.zeros(((nx + 1) * ny))
        y = np.zeros(((nx + 1) * ny))
        for i in range(ny):
            s = slice(i * (nx + 1), (i + 1) * (nx + 1))
            aux = np.arange(0, (nx + 1)*step_x, step_x)
            if i % 2:
                # retrocede
                x[s] = np.flip(aux)
            else:
                # avanza
                x[s] = aux

            y[s] = i*step_y

    else:
        x = np.zeros(((ny + 1) * nx))
        y = np.zeros(((ny + 1) * nx))
        for i in range(nx):
            s = slice(i*(ny+1), (i+1)*(ny+1))
            aux = np.arange(0, (ny + 1)*step_y, step_y)
            if i % 2:
                # retrocede
                y[s] = np.flip(aux)
            else:
                #avanza
                y[s] = aux

            x[s] = i * step_x

    return x, y


def circle_line_intersec(radio, p0, v):

    """
    Circulo centrado en (0,0), linea desde p0 con direccion v (para cada fila de los arrays)
    Args:
        radio:
        p0: array de shape (N, 2), de N puntos
        v: array de shape (N, 2), de N puntos

    Returns:
        q1, q2, las dos soluciones
    """

    p0, v = np.array(p0), np.array(v) # por si son listas

    a = np.sum(v**2, axis=1)  # coeficiente del termino cuadratico ( array de (N, ) )
    b = 2 * np.sum(p0 * v, axis=1)  # coeficiente del termino grado 1 ( array de (N, ) )
    c = np.sum(p0**2, axis=1) - radio**2  # coeficiente grado 0 ( array de (N, ) )

    discr = (b**2 - 4*a*c)  # (N, )
    if np.any(discr < 0):  # chequear que el discriminante sea positivo
        print('no hay interseccion (discriminante negativo)')
    discr_raiz = np.sqrt(discr)
    s1 = (-b - discr_raiz) / (2*a)
    s2 = (-b + discr_raiz) / (2*a)
    s1 = s1.reshape((-1, 1))
    s2 = s2.reshape((-1, 1))
    q1 = p0 + s1*v
    q2 = p0 + s2*v
    return q1, q2


def enderezar_matriz_zigzag(x, impares=True):
    """

    Args:
        x: 1 2 3 4
           4 3 2 1
           1 2 3 4
           4 3 2 1

    Returns:
        y: 1 2 3 4
           1 2 3 4
           1 2 3 4
           1 2 3 4
    """

    n_fil, n_col = x.shape
    if impares:
        filas = range(1, n_fil, 2)
    else:
        filas = range(0, n_fil, 2)

    q = np.fliplr(x[filas, :])
    y = x.copy()
    y[filas, :] = q
    return y


def return_tgc_fmc(matrix, idx_surf, fs, c2, db_mm, db_0):
    """todo: arreglar esto que esta mal"""
    nel = matrix.shape[0]
    assert idx_surf.size == nel
    aux = np.arange(matrix.shape[2]).reshape(1, 1, -1)  # 0 1 2 ... n_samples
    # repetir
    aux = np.repeat(aux, nel, axis=0)
    aux = np.repeat(aux, nel, axis=1)
    d = c2 * aux/fs
    idx_surf = np.expand_dims(idx_surf)
    d_surf = c2 * idx_surf.reshape(nel, nel, 1) / fs
    tgc = 10 ** ((db_mm * (d-d_surf) + db_0) / 20)
    return tgc


def dict2txt(x, fname):
    """Guardar el contenido del dict x a un txt"""
    with open(fname, 'w') as f:
        for key, value in x.items():
            f.write('%s:%s\n' % (key, value))


class VolumeData:

    def __init__(self, matrix, vol_type, dv, **kwargs):
        self.matrix = matrix
        self.vol_type = vol_type  # 'zxy' or 'xyt'
        self.dv = dv  # (dx, dy, dz) or (dx, dy, dt)

        if vol_type == 'xyt':
            self.fs = 1/self.dv[2]
            self.n_emi_x, self.n_emi_y, self.n_samples = self.matrix.shape
            self.x0 = self.n_emi_x * dv[0] / 2
            self.y0 = self.n_emi_y * dv[1] / 2
            self.roi = (-self.x0, self.x0, -self.y0, self.y0, self.n_samples * self.dv[2], 0)
        else:
            self.roi = kwargs['roi']

    def xyt2index(self, x, y, t):
        ix = int((x + self.x0) / self.dv[0])
        iy = int((y + self.y0) / self.dv[1])
        it = int(t / self.dv[2])
        return ix, iy, it

    def xyz2index(self, x, y, z):
        ix = int((x - self.roi[0]) / self.dv[0])
        iy = int((y - self.roi[2]) / self.dv[1])
        iz = int((self.roi[4] - z) / self.dv[2])
        return ix, iy, iz

    def return_ascan(self, x, y):
        if self.vol_type == 'xyt':
            i, j, _ = self.xyt2index(x, y, 0)
            ascan = self.matrix[i, j, :]
        else:
            i, j, _ = self.xyz2index(x, y, 0)
            ascan = self.matrix[:, i, j]

        return ascan

    def return_bscan(self, x, y):
        if self.vol_type == 'xyt':
            i, j, _ = self.xyt2index(x, y, 0)
            bscan_x = self.matrix[:, j, :].T
            bscan_y = self.matrix[i, :, :].T
        else:
            i, j, _ = self.xyz2index(x, y, 0)
            bscan_x = self.matrix[:, :, j]
            bscan_y = self.matrix[:, i, :]

        return bscan_x, bscan_y

    def return_cscan(self, z1, z2):
        if self.vol_type == 'xyt':
            _, _, i1 = self.xyt2index(0, 0, z1)
            _, _, i2 = self.xyt2index(0, 0, z2)
            cscan = np.abs(self.matrix[:, :, i1:i2]).max(axis=2)
        else:
            _, _, i2 = self.xyz2index(0, 0, z1)
            _, _, i1 = self.xyz2index(0, 0, z2)
            cscan = np.abs(self.matrix[i1:i2, :, :]).max(axis=0)

        return cscan