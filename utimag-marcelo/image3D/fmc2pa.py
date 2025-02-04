import numpy as np


def fmc_matrix2linear(fmc_adq, nel):
    """Tomar la captura FMC hecha con array matricial, y sumar a lo largo de "x", para crear una
    aquicisión como si fuera un array lineal"""

    # separar las adquisiciones de cada hilera "y". Si el array es de 4x32 (4 en x, 32 en y),
    # queda un array de (4, 32, 32, n_samples).
    # de modo que el primer índice indica la "sub-imagen 2D"
    # sub_adq = np.array([fmc_adq[i::nel, i::nel, :] for i in range(4)])

    m = int(fmc_adq.shape[0]/nel)
    n_samples = fmc_adq.shape[2]
    adq_linear = np.zeros((m, m, n_samples))
    for i in range(m):
        for j in range(m):
            temp = fmc_adq[nel*i:(nel*i + nel), nel*j:(nel*j + nel), :]   # (4, 4, n_samples)
            adq_linear[i, j, :] = np.sum(np.sum(temp, axis=0), axis=0)

    return adq_linear/nel


def fmc2pa_tx_2d(fmc_adq, delay, fs):
    """Toma adquisicion FMC de un array lineal y sintetiza emisión (una linea)"""
    # poner los delays (tienen que ser todos positivos)
    delay_index = np.around(fs * delay).astype(np.int)
    # crear nueva matrix usando el delay máximo para "agrandar" los A-scan
    nel, _, n_samples = fmc_adq.shape
    fmc_adq_plus = np.zeros((nel, nel, n_samples + delay_index.max()))
    # agregar delays poniendo ceros al principio de cada A-scan
    for i in range(nel):
        # para cada emisor, se retrasan las recepciones de todos
        k = delay_index[i]
        fmc_adq_plus[i, :, k:k + n_samples] = fmc_adq[i, :, :]

    # sumar emisiones
    pa_raw = fmc_adq_plus.sum(axis=0)
    return pa_raw


def pa_rx_linear(pa_raw, delay, fs):
    """Toma adquisicion PA raw de un array lineal y sintetiza recepción (una linea), con
    una sola ley focal (sin enfoque dinamico)"""
    # poner los delays (tienen que ser todos positivos)
    delay_index = np.around(fs * delay).astype(np.int)
    # crear nueva matrix usando el delay máximo para "agrandar" los A-scan
    nel, n_samples = pa_raw.shape
    pa_raw_plus = np.zeros((nel, n_samples + delay_index.max()))
    # agregar delays poniendo ceros al principio de cada A-scan
    for i in range(nel):
        # para cada emisor, se retrasan las recepciones de todos
        k = delay_index[i]
        pa_raw_plus[i, k:k + n_samples] = pa_raw[i, :]

    # sumar emisiones
    pa_line = pa_raw_plus.sum(axis=0)
    return pa_line


def fmc2pa_tx_2d_rows(fmc_adq, delay, nel, fs):
    """Supongo una FMC de un array matricial ordenada como (indice_lineal, indice_lineal, samples). Se sintetiza una
    adquisicion (una línea) en que cada hilera en dirección "y" emitió por separado y todas con la misma ley focal.
    Es decir, se hacen nel[0] adquisiciones iguales.
    nel: (nel_x, nel_y)"""

    assert delay.size == nel[1]
    # separar las adquisiciones de cada hilera "y". Si el array es de 4x32 (4 en x, 32 en y),
    # queda un array de (4, 32, 32, n_samples).
    # de modo que el primer índice indica la "sub-imagen 2D"
    sub_adq = np.array([fmc_adq[i::nel[0], i::nel[0], :] for i in range(4)])
    # poner los delays (tienen que ser todos positivos)
    delay_index = np.around(fs*delay).astype(np.int)
    # crear nueva matrix usando el delay máximo para "agrandar" los A-scan
    n_samples = fmc_adq.shape[2]
    sub_adq_plus = np.zeros((nel[0], nel[1], nel[1], n_samples + delay_index.max()))
    # agregar delays poniendo ceros al principio de cada A-scan
    for i in range(nel[1]):
        # para cada emisor, se retrasan las recepciones de todos
        k = delay_index[i]
        sub_adq_plus[:, i, :, k:k+n_samples] = sub_adq[:, i, :, :]

    # sumar emisiones
    pa_raw = sub_adq_plus.sum(axis=1)
    return pa_raw
