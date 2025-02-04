import numpy as np
from imag3D import utils_3d
from scipy.spatial import transform as tra
from sklearn.linear_model import LinearRegression
from scipy.optimize import root_scalar
from scipy.spatial.transform import Rotation
import imag3D.snell_fermat_3d as sf
import utils
from imag3D import utils_3d as u3d
import warnings


def surf_water_mask(surf_fun, roi, x_step, y_step, z_step):
    z, x, y = np.meshgrid(np.arange(roi[4], roi[5], -z_step),
                          np.arange(roi[0], roi[1], x_step),
                          np.arange(roi[2], roi[3], y_step), indexing='ij')

    return z < surf_fun(x, y)[0]


def surf_tgc(surf_fun, roi, x_step, y_step, z_step, db_mm, db_0):
    z, x, y = np.meshgrid(np.arange(roi[4], roi[5], -z_step),
                          np.arange(roi[0], roi[1], x_step),
                          np.arange(roi[2], roi[3], y_step), indexing='ij')

    # z_ifaz = np.expand_dims(surf_fun(x[0, :, :], y[0, :, :]), axis=0)
    z_ifaz = surf_fun(x, y)[0]
    tgc = 10 ** ((db_mm * (z_ifaz - z) + db_0) / 20)
    return tgc


def fit_first_echo_index(idx, weight, pitch, nel_x, nel_y, degree):
    """ Hace un ajuste por cuadrados mínimos de los índices del primer eco

    argumentos:
    idx: array 1-d (desenrrollado), con el indice del primer eco para cada elemento del array. Tiene que tener
    nel_x*nel_y elementos.
    El array-transductor se recorre en direccion x, o sea, definimos la dirección x como
    aquella en que se cuentan los elementos. De este modo, las coordenadas/indices (x, y) de los elementos se ordenan,
    en el caso de un ejemplo de de 3x4 elementos:
    ix = [0 1 2 3 0 1 2 3 0 1 2 3]
    iy = [0 0 0 0 1 1 1 1 2 2 2 2]

    weight: array del mismo tamaño que idx, con los pesos para el ajuste
    pitch: pitch del array, suponemos el mismo en x e y
    nel_x, nel_y: nro de elementos del array en cada dirección, ej: (11, 11)
    degree: grado del polinomio de ajuste, 1 o 2

    returns:
    coef: coeficientes del polinomio
    idx_fun: funcion fiteadora (x,y) ---> (índice, gradiente)
    reg: objeto de regresion lineal de scikit learn
    """

    n = nel_x * nel_y  # n_elementos
    ij_mesh = np.meshgrid(range(nel_x), range(nel_y), indexing='ij')

    # todo: ALERTA, PROBABLE ERROR si se aplica en caso de onda plana donde el hay que rotar primero el sistema
    # de coordenadas para que el vector de onda esté en el plano adecuado (phi distinto de 0)
    # en el caso en que phi es distinto de 0 y hay que cambiar de sistema de coordenadas, creo que idx_fun
    # hay que calcularla primero rotando xy_mesh para hacer el fit en el nuevo sistema de coordenadas
    # todo: FIN DE ALERTA

    # order='F' quiere decir que se recorre el array por columnas
    # por ej:
    # ij_mesh[0] = 0 0 0 0
    #              1 1 1 1
    #              2 2 2 2
    # como queremos que ij_mesh[0] corresponda a la coordenada x, entonces hay que recorrer por columnas
    # El objetivo es que quede ix = [0 1 2 0 1 2 0 1 2 0 1 2]
    # Otra alternativa sería:
    # ij_mesh = np.meshgrid(range(nel_x), range(nel_y), indexing='xy')
    # Entonces quedaría:
    # ij_mesh[0] =  0 1 2
    #               0 1 2
    #               0 1 2
    #               0 1 2
    # Y ahí hay que recorrer por filas, order='C'
    # Se podría usar entonces:
    # ix = ij_mesh[0].flatten()

    ix = ij_mesh[0].reshape((n, -1), order='F')
    iy = ij_mesh[1].reshape((n, -1), order='F')
    x0 = pitch * (nel_x - 1) / 2
    y0 = pitch * (nel_y - 1) / 2
    x = pitch * ix - x0
    y = pitch * iy - y0

    if degree == 2:
        xy = np.concatenate([x, y, x ** 2, y ** 2, x * y], axis=1)
    else:
        xy = np.concatenate([x, y], axis=1)

    reg = LinearRegression()
    reg.fit(xy, idx, weight)
    # idx_fit = reg.predict(ixy).reshape((nel_x, nel_y), order='F')
    coef = np.insert(reg.coef_, 0, reg.intercept_)

    if degree == 2:
        def idx_fun(x, y):
            q = coef[0] + coef[1] * x + coef[2] * y + coef[3] * x ** 2 + coef[4] * y ** 2 + coef[5] * x * y
            grad_q = np.array([coef[1] + 2 * coef[3] * x + coef[5] * y, coef[2] + 2 * coef[4] * y + coef[5] * x])
            return q, grad_q
    else:
        def idx_fun(x, y):
            q = coef[0] + coef[1] * x + coef[2] * y
            grad_q = np.array([coef[1], coef[2]])
            return q, grad_q

    return coef, idx_fun, reg


def pulse_echo_surface(idx_fun, c, fs, xy_mesh):
    idx_fit, grad = idx_fun(*xy_mesh)
    d = (c / 2) * idx_fit / fs  # distancias
    nxy = (c / 2) * grad / fs  # componentes (x, y) de la normal local a la superficie
    nxy_2 = nxy[0, :, :] ** 2 + nxy[1, :, :] ** 2
    grad_error = nxy_2 > 1
    nxy_2[grad_error] = 1  # achicamos a 1 para que la raiz no de "nan"
    nz = np.sqrt(1 - nxy_2)  # compnente z, normal hacia arriba
    nx = nxy[0, :, :]
    ny = nxy[1, :, :]
    # coordenadas de los puntos de entrada (ex, ey, ez)
    ex = xy_mesh[0] - d * nx
    ey = xy_mesh[1] - d * ny
    ez = -d * nz
    extras = {'normal': (nx, ny, nz), 'grad_error': grad_error}
    return ex, ey, ez, extras


def pitch_catch_surface(idx_fun, a0, c, fs, xy_mesh):
    """

    Args:
        idx_fun: función (x,y)----> indice del primer eco. Es el resultado de usar la
        función fit_first_echo_index
        a0: emisor
        c: velocidad en agua
        fs:
        xy_mesh: grilla en la apertura donde se quiere calcular los puntos de entrada e

    Returns:
        ex, ey, ez
    """
    # coordenadas relativas a a0
    xy_mesh_mov = (xy_mesh[0] - a0[0], xy_mesh[1] - a0[1])
    idx_fit, grad = idx_fun(*xy_mesh)
    d = c * idx_fit / fs  # distancias
    grad_d = c * grad / fs  # gradiente de las distancias
    grad_d_2 = grad_d[0, :, :] ** 2 + grad_d[1, :, :] ** 2  # norma/modulo del gradiente
    # qué pasa si grad_d_2 es mayor que 1?
    grad_error = grad_d_2 > 1
    grad_d_2[grad_error] = 1  # achicamos esos valores a 1 para que sqrt no de "nan"
    # cordenadas del punto q, que es la imagen especular del "emisor en (0,0,0)"
    qx = xy_mesh_mov[0] - d * grad_d[0, :, :]
    qy = xy_mesh_mov[1] - d * grad_d[1, :, :]
    qz = -d * np.sqrt(1 - grad_d_2)
    q = np.stack([qx, qy, qz], axis=0)
    norm_q = np.linalg.norm(q, axis=0)
    nx = qx / norm_q
    ny = qy / norm_q
    nz = qz / norm_q
    a = np.zeros_like(q)
    a[0, :, :] = xy_mesh_mov[0]
    a[1, :, :] = xy_mesh_mov[1]
    # a = a - np.reshape(a0, (3, 1, 1))
    # calcula la interseccion entre la recta que va desde q hasta a (receptor). Ese punto es
    # el punto de entrada e
    s = -0.5 * np.sum(q * q, axis=0) / np.sum((a - q) * q, axis=0)
    s = np.expand_dims(s, axis=0)
    e = s * (a - q) + q
    e = e + np.reshape(a0, (3, 1, 1))  # trasladar porque estaba usando a a0 como origen para estas cuentas
    ex, ey, ez = e[0, :, :], e[1, :, :], e[2, :, :]
    extras = {'normal': (nx, ny, nz), 'q': (qx, qy, qz), 'grad_error': grad_error}
    return ex, ey, ez, extras


def plane_wave_surface(idx_fun, x_0, theta, c, fs, xy_mesh):
    """

    Args:
        idx_fun:
        x_0: coordenada x de los elementos "más a a la derecha" del array
        theta:
        c:
        fs:
        xy_mesh:

    Returns:

    """
    idx_fit, grad = idx_fun(*xy_mesh)
    d = c * idx_fit / fs  # distancias
    grad_d = c * grad / fs  # gradiente de las distancias
    u = [np.sin(theta), -np.cos(theta)]

    if u[0] == 0:
        grad_d_2 = grad_d[0, :, :] ** 2 + grad_d[1, :, :] ** 2
        # qué pasa si grad_d_2 es mayor que 1?
        grad_error = grad_d_2 > 1
        grad_d_2[grad_error] = 1  # achicamos esos valores a 1 para que sqrt no de "nan"
        nz = np.sqrt(0.5 * (1 + np.sqrt(1 - grad_d_2)))
        nx = 0.5 * grad_d[0, :, :] / nz
        ny = 0.5 * grad_d[1, :, :] / nz

        qx = xy_mesh[0] - d * nx / nz
        qy = xy_mesh[1] - d * ny / nz
        qz = -d / nz
        ex = qx
        ey = qy
        ez = d * (1 / (2 * nz ** 2) - 1)
        res = None

    else:
        # todo: contemplar aparte el caso en que (ux - grad_d_x) = 0 (DIVISON POR CERO)
        b = grad_d[1, :, :] / (grad_d[0, :, :] - u[0])
        nx = np.zeros_like(xy_mesh[0])
        res = []
        ni, nj = xy_mesh[0].shape
        for i in range(ni):
            res.append([])
            for j in range(nj):
                fun = return_pw_nx_fun(theta, grad_d[:, i, j])
                nx_max = 1 / np.sqrt(2 * (1 + b[i, j] ** 2))
                res[i].append(root_scalar(fun, x0=-nx_max * 0.5, x1=nx_max * 0.5))
                nx[i, j] = res[i][j].root

        h = (1 + b ** 2) * nx ** 2  # variable auxiliar, tiene que ser <= 1
        grad_error = h > 1
        h[grad_error] = 1  # achicamos esos valores a 1 para que sqrt no de "nan"
        nz = np.sqrt(1 - h)
        ny = b * nx

        # todo: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # A partir de aquí está hecho segun la el vector q, que es reflexion especular
        # del receptor, y unas cosas que no me entiendo demasiado a mi mismo
        # todo: modificarlo según la nueva formulacion, que deberia dar lo mimsmo pero es
        # mas simple

        d_a = (xy_mesh[0] + np.sign(u[0]) * x_0) * u[0] - d
        udotn = u[0] * nx + u[1] * nz  # producto escalar de u con la normal
        g = d_a / udotn
        alfa = g / (2 * udotn)
        qx = xy_mesh[0] - nx * g
        qy = xy_mesh[1] - ny * g
        qz = - nz * g
        ex = qx + alfa * u[0]
        ey = qy
        ez = qz + alfa * u[1]

    extras = {'normal': (nx, ny, nz),
              'q': (qx, qy, qz),
              'res': res,
              'grad_error': grad_error}

    return ex, ey, ez, extras


def return_pw_nx_fun(theta, grad):
    """Funcion para generar la funcion cuya raiz es nx en el cálculo de la normal
    por el método de plane wave"""
    u = [np.sin(theta), -np.cos(theta)]
    a = grad[0] - u[0]
    assert a != 0  # si a=0 se usa otro método
    b = grad[1] / a

    def fun(nx):
        nx2 = nx ** 2
        return u[0] * nx2 + u[1] * nx * np.sqrt(1 - nx2 * (1 + b ** 2)) + 0.5 * a

    return fun


def idx_filter_3d(idx, valid, nel_x, nel_y):
    """ Suaviza los índices mediante un ajuste cuadrático.
     La variable idx (indice del primero eco) es de 1-dimension, y viene ordenada a lo largo de dirección "x".
     Es decir: si nel_x=4, entonces peco viene indexada como: 0 1 2 3 0 1 2 3 0 1 2 3 etc."""

    warnings.warn('esta funcion se usaba en el metodo "viejo" de calcular superficie con puslo-eco',
                  DeprecationWarning)
    # ATENCION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # para estos reshape tengo que usar order='F' porque quiero que el indice fila corresponda la dirección "x"
    # idx = idx.reshape((nel_x, -1), order='F')
    # el indexing debe ser 'ij'
    ij_mesh = np.meshgrid(range(nel_x), range(nel_y), indexing='ij')

    # filtar tof de ecos de interfaz
    n = nel_x * nel_y  # n_elementos
    n_valid = valid.sum()  # nro de elementos validos
    # pasar a unidimensional para el ajuste
    # como el indexing es 'ij', queda:
    # ix = [0 1 2 3 0 1 2 3 0 1 2 3 etc]
    # iy = [0 0 0 0 1 1 1 1 2 2 2 2 etc]
    ix = ij_mesh[0].reshape((n, -1), order='F')
    iy = ij_mesh[1].reshape((n, -1), order='F')
    ixy = np.concatenate([ix, iy, ix ** 2, iy ** 2, ix * iy, np.ones((n, 1))], axis=1)
    ix_valid = ij_mesh[0].reshape((n, -1), order='F')[valid]
    iy_valid = ij_mesh[1].reshape((n, -1), order='F')[valid]
    ixy_valid = np.concatenate([ix_valid, iy_valid, ix_valid ** 2, iy_valid ** 2,
                                ix_valid * iy_valid, np.ones((n_valid, 1))], axis=1)

    # coeficcient matrix para linear least squares sería ixy
    # idx_ = idx.data.reshape((n, -1), order='F')
    # q = np.linalg.lstsq(ixy, idx_)
    q = np.linalg.lstsq(ixy_valid, idx[valid])
    # idx_fit = np.matmul(ixy, q[0]).reshape((nel_x, nel_y), order='F')
    idx_fit = np.matmul(ixy, q[0]).reshape((nel_x, nel_y), order='F')
    resid = idx_fit - idx.reshape((nel_x, nel_y), order='F')
    return idx_fit, resid


def calc_ifaz_3d(d, nel_x, nel_y, pitch):
    """Calcular interfaz mediante método de pulso-eco.
    Args:
        d: array de distancias perpendiculares a la superficie, calculadas en base al eco de la misma
    """
    warnings.warn('esta funcion se usaba en el metodo "viejo" de calcular superficie con puslo-eco',
                  DeprecationWarning)
    # normal
    nx = np.diff(d, axis=0) / pitch
    nx = nx[:, 0:(nel_y - 1)]
    ny = np.diff(d, axis=1) / pitch
    ny = ny[0:(nel_x - 1), :]
    # nx = np.diff(d, axis=1) / pitch
    # nx = nx[0:(nel_y - 1), :]
    # ny = np.diff(d, axis=0) / pitch
    # ny = ny[:, 0:(nel_x-1)]
    nz = np.sqrt(1 - nx ** 2 - ny ** 2)  # puede dar nan, revisar, todo

    # indice 0: x, indice 1: y
    x_idx, y_idx = np.meshgrid(range(nel_x), range(nel_y), indexing='ij')

    # posiciones del array
    xa = pitch * (x_idx - (nel_x - 1) / 2)
    ya = pitch * (y_idx - (nel_y - 1) / 2)
    za = 0

    # puntos de la interfaz
    aux_x = slice(0, nel_x - 1)
    aux_y = slice(0, nel_y - 1)
    xs = - nx * d[0:-1, 0:-1] + xa[0:-1, 0:-1]
    ys = - ny * d[0:-1, 0:-1] + ya[0:-1, 0:-1]
    zs = - nz * d[0:-1, 0:-1] + za

    return (xs, ys, zs), (nx, ny, nz), (xa, ya, za)


def fit_surf_lstsq(points, model):
    # supongamos los puntos en formato "meshgrid". Entonces hay que hacer reshape y generar la matriz de
    # coeficientes para linear least squares
    nel_x, nel_y = points[0].shape
    xs = points[0].reshape((nel_x * nel_y, -1))
    ys = points[1].reshape((nel_x * nel_y, -1))
    zs = points[2].reshape((nel_x * nel_y, -1))

    # zs puede tener nanes, quitarlos
    temp = np.logical_not(np.isnan(zs))
    zs = np.expand_dims(zs[temp], axis=1)
    xs = np.expand_dims(xs[temp], axis=1)
    ys = np.expand_dims(ys[temp], axis=1)

    if model == 'plane':
        coef = np.concatenate([xs, ys, np.ones_like(xs)], axis=1)
    elif model == 'poly2':
        # todo: tal vez conviene cambiar el orden, para que los términos cuadráticos vengan primero
        coef = np.concatenate([xs, ys, xs ** 2, ys ** 2, xs * ys, np.ones_like(xs)], axis=1)
    else:
        raise Exception('Invalid model')

    fit = np.linalg.lstsq(coef, zs)
    zs_fit = np.matmul(coef, fit[0])

    def surf_fun(x, y):
        if model == 'plane':
            z = fit[0][0] * x + fit[0][1] * y + fit[0][2]
        if model == 'poly2':
            z = fit[0][0] * x + fit[0][1] * y + fit[0][2] * x ** 2 + fit[0][3] * y ** 2 + fit[0][4] * x * y + fit[0][5]
        return z

    return fit, zs_fit, surf_fun


def cylinder_array_normal_dist(nel_x, nel_y, pitch_x, pitch_y, za, r, rotvec):
    """Define un array con pitches (pitch_x, pitch_y), a partir de una traslacion (0, 0, za) y una rotacion
     según rotvec. El cilindro tiene eje colineal a Y, y radio r. Calcula para cada elemento del array
     la distancia perpendicular al cilindro

     OJO: PENSADO CON CILINDRO EN EJE Y"""

    # calcula vectores posicion de los elementos del array en su propio sistema de coordenadas
    a_prop = u3d.array_coordinates_list(nel_x, nel_y, pitch_x, pitch_y)
    # define rotacion
    rot = tra.Rotation.from_rotvec(rotvec)
    a_center = np.array((0, 0, za))
    a_cyl = [rot.apply(a) + a_center for a in a_prop]  # coordenadas en sistema cilindro
    # para cada elemento, definir el punto sobre el eje del cilindro, en un plano XZ
    # a_eje = [np.array((0, a[1], 0)) for a in a_cyl]
    # # dist = [np.linalg.norm(a - b) for a, b in zip(a_cyl, a_eje)]
    dist = [np.hypot(a[0], a[2]) for a in a_cyl]
    dist = np.array(dist).reshape((nel_x, nel_y), order='F')
    # al convertir la lista esa en un array, queda con un shape tal que hay que usar
    # order C porque en la lista a_prop primer se avanza en la direccion x
    return dist - r  # en realidad el valor absoluto, por si dist es menor a r, todo


def return_array_surf_normdist_fun(nel_x, nel_y, pitch_x, pitch_y, surf_type):
    # calcula vectores posicion de los elementos del array en su propio sistema de coordenadas
    a_prop = u3d.array_coordinates_list(nel_x, nel_y, pitch_x, pitch_y)

    if surf_type == 'cylinder':
        # OJO: PENSADO CON CILINDRO EN EJE Y !!!!
        def fun(x):
            # define rotacion
            rot = tra.Rotation.from_rotvec(x[2:])
            a_center = np.array((0, 0, x[0]))
            a_cyl = [rot.apply(a) + a_center for a in a_prop]  # coordenadas en sistema cilindro
            # para cada elemento, definir el punto sobre el eje del cilindro, en un plano XZ
            dist = [np.hypot(a[0], a[2]) for a in a_cyl]
            dist = np.array(dist).reshape((nel_x, nel_y), order='F')
            # al convertir la lista esa en un array, queda con un shape tal que hay que usar
            # order C porque en la lista a_prop primer se avanza en la direccion x
            return dist - x[1]

    elif surf_type == 'plane':
        def fun(x):
            # define rotacion
            rot = tra.Rotation.from_rotvec(x[1:])
            a_center = np.array((0, 0, x[0]))
            a_plane = [rot.apply(a) + a_center for a in a_prop]  # coordenadas en sistema del plano
            dist = [np.abs(a[2]) for a in a_plane]
            dist = np.array(dist).reshape((nel_x, nel_y), order='F')
            return dist

    return fun


def return_pulse_echo_plane_fun(a):
    """ plano XY """
    a = np.array(a)

    def fun(a0, rot):
        a0 = np.array(a0).reshape((1, 3))
        rotat = tra.Rotation.from_euler(rot[0], rot[1], degrees=True)  # 'xyz': ejes extrínsecos
        a_mov = a0 + rotat.apply(a)
        dist = np.abs(a_mov[:, 2])  # coordenada z
        return 2 * dist

    return fun


def return_pulse_echo_cylinder_fun(a, curv_sign):
    """

    Args:
        a: coordenadas de los elementos del array en sistema array
        curv_sign: 1 convexo, -1 concavo

    Returns:
        fun:
    """
    # CILINDRO EN EJE X.
    a = np.array(a)

    if curv_sign < 0:
        def fun(radio, a0, rot):
            a0 = np.array(a0).reshape((1, 3))
            rotat = tra.Rotation.from_euler(rot[0], rot[1], degrees=True)  # 'xyz': ejes extrínsecos
            a_cyl = rotat.apply(a) + a0  # coordenadas en sistema cilindro
            # para cada elemento, definir el punto sobre el eje del cilindro, en un plano YZ
            a_cyl[:, 0] = 0  # poner X=0
            # acá esta la cuestion de cual de las dos posibilididas se elige:
            # Suponemos que la superficie está abajo, es la parte inferior del cilindro
            # entonces el criterio será si el elemento esta arriba o abajo del plano z=0
            # Si el elemento esta justo en z=0 no habrá ningun eco probablement, dado que al array
            # deberia estar muy girada, para emitir medio "de costado"
            q = np.linalg.norm(a_cyl, axis=1)
            abajo = a_cyl[:, 2] <= 0  # el igual es arbitrario
            # para contemplar que algunos elementos pueden estar abajo y otros arriba, uso esta expresion:
            dist = (radio - q) * abajo + (radio + q) * (1 - abajo)
            return 2 * dist  # multiplico por 2 por el ida y vuelta

    else:  # caso convexo
        def fun(radio, a0, rot):
            a0 = np.array(a0).reshape((1, 3))
            rotat = tra.Rotation.from_euler(rot[0], rot[1], degrees=True)  # 'xyz': ejes extrínsecos
            a_cyl = rotat.apply(a) + a0  # coordenadas en sistema cilindro
            # para cada elemento, definir el punto sobre el eje del cilindro, en un plano YZ
            a_cyl[:, 0] = 0  # poner X=0
            dist = np.linalg.norm(a_cyl, axis=1)
            return 2 * (dist - radio)  # multiplico por 2 por el ida y vuelta

    return fun


def return_pulse_echo_sphere_fun(a, curv_sign):
    a = np.array(a)

    if curv_sign < 0:
        def fun(radio, a0, rot):
            a0 = np.array(a0).reshape((1, 3))
            rotat = tra.Rotation.from_euler(rot[0], rot[1], degrees=True)  # 'xyz': ejes extrínsecos
            a_sph = rotat.apply(a) + a0  # coordenadas en sistema esfera
            # para cada elemento, definir el punto sobre el eje del cilindro, en un plano YZ

            # acá esta la cuestion de cual de las dos posibilididas se elige:
            # Suponemos que la superficie está abajo, es la parte inferior de la esfera
            # entonces el criterio será si el elemento esta arriba o abajo del plano z=0
            # Si el elemento esta justo en z=0 no habrá ningun eco probablement, dado que al array
            # deberia estar muy girada, para emitir medio "de costado"
            q = np.linalg.norm(a_sph, axis=1)
            abajo = a_sph[:, 2] <= 0  # el igual es arbitrario
            # para contemplar que algunos elementos pueden estar abajo y otros arriba, uso esta expresion:
            dist = (radio - q) * abajo + (radio + q) * (1 - abajo)
            return 2 * dist  # multiplico por 2 por el ida y vuelta

    else:  # caso convexo
        def fun(radio, a0, rot):
            a0 = np.array(a0).reshape((1, 3))
            rotat = tra.Rotation.from_euler(rot[0], rot[1], degrees=True)  # 'xyz': ejes extrínsecos
            a_sph = rotat.apply(a) + a0  # coordenadas en sistema esfera
            dist = np.linalg.norm(a_sph, axis=1)
            return 2 * (dist - radio)  # multiplico por 2 por el ida y vuelta

    return fun


def return_pitch_catch_plane_fun(tx_coords, rx_coords):

    def fun(a0, rot):
        """

        Args:
            radio:
            a0: centro del array
            rot: angulos de euler

        Returns:

        """

        # rotar y trasladar elementos
        rotat = tra.Rotation.from_euler(rot[0], rot[1], degrees=True)
        a0 = np.array(a0).reshape((1, 3))
        tx_coords_mov = rotat.apply(tx_coords) + a0
        rx_coords_mov = rotat.apply(rx_coords) + a0
        # imagen especular del emisor, cambiando el signo de z
        tx_coords_mov_refl = tx_coords_mov.copy()
        tx_coords_mov_refl[:, 2] = -tx_coords_mov_refl[:, 2]

        n_tx = tx_coords.shape[0]
        dist = []
        for i in range(n_tx):
            dist.append(np.linalg.norm(tx_coords_mov_refl[i, :] - rx_coords_mov, axis=1))
        return np.concatenate(dist)

    return fun


def return_plane_wave_plane_fun(coords_sa, v_pw):

    assert v_pw.shape[1] == 3
    n_waves = v_pw.shape[0]
    n_elems = len(coords_sa)
    dist = np.zeros((n_waves, n_elems))
    # normalizar los vectores por is no lo están
    norma_v_pw = np.linalg.norm(v_pw, axis=1).reshape((-1, 1))
    v_pw = v_pw / norma_v_pw

    def fun(a0, rot):
        a0 = np.array(a0).reshape(1, 3)
        # rotar y trasladar elementos
        rotat = tra.Rotation.from_euler(rot[0], rot[1], degrees=True)
        v_pw_rot = rotat.apply(v_pw)
        e_mov = rotat.apply(coords_sa) + a0

        dist = []
        for i in range(n_waves):
            # reflejar centro del array y vector de onda: cambiar el signo de z y listo
            a0_refl = a0.copy()
            a0_refl[0, 2] = -a0_refl[0, 2]
            v_pw_refl = v_pw_rot[i, :].copy()
            v_pw_refl = v_pw_refl.reshape(1, 3)
            v_pw_refl[0, 2] = -v_pw_refl[0, 2]
            dist.append(np.sum((e_mov - a0_refl) * v_pw_refl, axis=1))

        return np.concatenate(dist)  # order C, quedan primero todos los elementos para la primera onda, luego para
        # la segunda onda, etc

    return fun


# def return_pitch_catch_plane_fun(a0, a):
#     """Una interfaz plana en el plano Z=0. Emite el elemento a0 = (x0, y0, 0) según sus coordenadas
#     en el sistema del Transducer Array (TA). Se aplica al TA una rotacion y traslación dada por mov.
#     Se calcula la imagen especular de los elementos receptores y con eso se calcula la distancia
#
#     Args:
#     a0: (x0, y0, 0), elemento emisor
#     a: lista de coordenadas de los elementos en el sistema arrray (se genera con utils_3d.array_coordinates_list)
#
#     Returns:
#     fun: funcion mov ---> distancia, donde mov: (z0, rotx, roty)
#     """
#
#     a0 = np.array(a0).reshape((1, 3))  # transformar por si viene como lista o tupla
#     a = np.array(a)
#     assert a.shape[1] == 3  # tiene que tener forma (N, 3)
#
#     # TODO: REFLEJAR EMISOR!!!!!!!!! MUCHO MAS BARATO!!!
#
#     def fun(mov):
#         # rotar y trasladar elementos
#         rot = tra.Rotation.from_euler('XY', mov[1:], degrees=True)
#         despla = np.array([0, 0, mov[0]])
#         a0_mov = rot.apply(a0) + despla
#         a_mov = rot.apply(a) + despla
#         # reflejar receptores, cambiaando signo de z
#         a_mov_refl = a_mov
#         a_mov_refl[:, 2] = -a_mov_refl[:, 2]
#         dist = np.linalg.norm(a_mov_refl - a0_mov, axis=1)
#         return dist
#
#     return fun


def return_pitch_catch_cyl_fun(coords, curv_sign):
    """
    CILINDRO EN EJE X. Todo: ver el caso cócavo. Esto está pensado pa convexo

    Args:
        coords: lista de 2-uplas (coords_tx, [coords_rx])
        curv_sign:

    Returns:

    """

    if curv_sign > 0:  # convexo
        def fun(radio, a0, rot):
            """

            Args:
                radio:
                a0: centro del array
                rot: angulos de euler

            Returns:

            """
            dist = []
            # rotar y trasladar elementos
            rotat = tra.Rotation.from_euler(rot[0], rot[1], degrees=True)
            a0 = np.array(a0).reshape((1, 3))
            for co_tx, co_rx in coords:
                co_tx_mov = rotat.apply(co_tx) + a0
                co_rx_mov = rotat.apply(co_rx) + a0
                # calcular vector entre el eje del cilindro (eje x) y el elemento tx
                vec = co_tx_mov.copy()
                # hay que proyectarlo sobre el plano YZ, poniendo 0 en la componente x
                vec[0, 0] = 0
                vec = vec / np.linalg.norm(
                    vec)  # normalizando, obtenemos el vector unitario ortogonal al plano tangente
                # TODO: acá es donde debe ser diferente el caso convexo, por el signo de la normal !!
                # en el punto de incidencia normal del elemento tx
                # ahora hay que reflejar los receptores rx respecto a ese plano
                # TODO: convieve reflejar el transmisor que es uno solo!!!!!!!!!!!!!!!!!!!!
                co_rx_mov_refl = co_rx_mov - 2 * vec * np.dot(co_rx_mov - radio * vec, vec.T)
                dist.append(np.linalg.norm(co_rx_mov_refl - co_tx_mov, axis=1))
            return np.concatenate(dist)

    else:  # concavo
        raise NotImplementedError

    return fun


def return_pitch_catch_sphere_fun(coords, curv_sign):
    """
    Args:
        coords: lista de 2-uplas (coords_tx, [coords_rx])
        curv_sign:

    Returns:

    """

    if curv_sign > 0:  # convexo
        def fun(radio, a0, rot):
            """

            Args:
                radio:
                a0: centro del array
                rot: angulos de euler

            Returns:

            """
            dist = []
            # rotar y trasladar elementos
            rotat = tra.Rotation.from_euler(rot[0], rot[1], degrees=True)
            a0 = np.array(a0).reshape((1, 3))
            for co_tx, co_rx in coords:
                co_tx_mov = rotat.apply(co_tx) + a0
                co_rx_mov = rotat.apply(co_rx) + a0
                # calcular vector entre el eje del cilindro (eje x) y el elemento tx
                vec = co_tx_mov.copy()
                vec = vec / np.linalg.norm(
                    vec)  # normalizando, obtenemos el vector unitario ortogonal al plano tangente
                # TODO: acá es donde debe ser diferente el caso convexo, por el signo de la normal !!
                # en el punto de incidencia normal del elemento tx
                # reflejar emisor en el plano tangente
                co_tx_mov_refl = -co_tx_mov + 2 * radio * vec
                dist.append(np.linalg.norm(co_tx_mov_refl - co_rx_mov, axis=1))
            return np.concatenate(dist)

    else:  # concavo
        raise NotImplementedError

    return fun


# *********************************************************************************************************************
# ******** funciones para de PITCH-CATCH CILINDRO con ESPEJO CIRCULAR *************************************************

def return_cyl_op_transform(v_op, curv_sign):
    """
    v_op es un punto del circulo, que define un eje optico. A partir de eso se definen dos versores n_op, t_op
    para un sistema de coordenadas VCS, tal que n_op tiene la direccion del eje optico, y apunta hacia "arriba",
    de modo que dependiendo de concavo/convexo tiene sentido igual u opuesto a v_op. Los ejes de este sistema son:
    0: x del WCS (eje del cilindro)
    1: ortogonal al eje óptico
    2: eje óptico, positivo hacia el lado de la fuente ("arriba")

    Calcula la transformacion de coordenadas cyl2op de los ejes del cilindro (WCS) a los ejes "opticos" (VCS),
    y su inversa op2cyl.
    Es solo la parte de rotacion. Luego hay que tener en cuenta que el orgen de VCS es el vertice v_op
    Args:
        v_op: vertice del eje optico
        curv_sign: -1 en caso concavo, 1 en caso convexo

    Returns:
        cyl2op, op2cyl: funciones, transofrmaciones de coordenadas
    """

    v_op_proj = v_op.copy()
    v_op_proj[:, 0] = 0
    # definir dos versores para calcular coordenadas sobre el eje optico
    # caso concavo: el versor es opuesto a v_op
    n_op = curv_sign * v_op_proj / np.linalg.norm(v_op_proj, axis=1).reshape(-1, 1)  # normaliza
    t_op = np.zeros_like(v_op_proj)  # vector perpendicaular a g0_proj
    t_op[:, 1] = n_op[:, 2]
    t_op[:, 2] = -n_op[:, 1]

    def cyl2op(cyl_coords):
        op_coords = np.zeros_like(cyl_coords)
        op_coords[:, 1] = np.sum(cyl_coords * t_op, axis=1)  # producto escalar
        op_coords[:, 2] = np.sum(cyl_coords * n_op, axis=1)  # producto escalar
        return op_coords

    def op2cyl(op_coords):
        cyl_coords = np.zeros_like(op_coords)
        # esta es una forma rara de multiplicar por la matriz de rotacion inversa, que es la traspuesta
        cyl_coords[:, 1] = t_op[:, 1] * op_coords[:, 1] + n_op[:, 1] * op_coords[:, 2]
        cyl_coords[:, 2] = t_op[:, 2] * op_coords[:, 1] + n_op[:, 2] * op_coords[:, 2]
        return cyl_coords

    return cyl2op, op2cyl


def compute_circ_mirror_image(source, n_probe, cyl_radio, curv_sign):
    """ Dado source (coordenadas en sistema cilindro) y la normal n_probe al array, calcular el punto
    de incidencia V (v_op) donde se define un eje óptico, y un plano con un espejo circular. Ahi se proyecta el source
    y se calculca su imagen segun el espejo"""

    n_probe_rep = np.repeat(n_probe.reshape(1, 3), source.shape[0], axis=0)  # array de (N, 3)
    g0_1, g0_2 = u3d.cylinder_line_intersec(cyl_radio, source, n_probe_rep)
    v_op = g0_1 if curv_sign > 0 else g0_2  # se elige segun convexo o concavo

    # calculamos las coordenadas de source en el eje optico
    v_op[:, 0] = 0  # proyectar sobre plano yz
    cyl2op, op2cyl = return_cyl_op_transform(v_op, curv_sign)
    source_op = cyl2op(source - v_op)
    # ahora calculamos la imagen, que en este caso es real
    source_imag_op = np.zeros_like(source_op)
    # ahora aplico las formulas para el espejo circular (ver Hetch)
    temp = 1 / source_op[:, 2]
    finv = (-1) * curv_sign * 2 / cyl_radio  # inverso del foco del espejo
    source_imag_op[:, 2] = 1 / (finv - temp)
    # acá aprece el termino de "lateral magnification"
    source_imag_op[:, 1] = (-source_imag_op[:, 2] / source_op[:, 2]) * source_op[:, 1]
    # ahora que vovler a coordenadas en el cilindro
    source_imag = op2cyl(source_imag_op) + v_op

    return source_imag


def compute_g_tangent_plane_mirror_imag(tx_coords_cyl, tx_cm_imag, rx_coords_cyl, cyl_radio, curv_sign):
    # proyectar sobre el plano YZ
    rx_proj = rx_coords_cyl.copy()
    rx_proj[:, 0] = 0
    # recta que pasa por el receptor y por la imagen real del emisor, cm: circular mirror
    vec = tx_cm_imag - rx_proj  # direccion de esa recta
    # itnerseccion con cilindro
    g1, g2 = u3d.cylinder_line_intersec(cyl_radio, rx_proj, vec)
    g = g1 if curv_sign > 0 else g2
    # calcular la imagen especular del emisor respecto al plano tangente en g, tx_pm_imag, pm: plane mirror
    n_cyl = curv_sign * g / np.linalg.norm(g, axis=1).reshape(-1, 1)
    tx_coords_cyl = tx_coords_cyl.reshape(1, 3)
    temp = np.sum(((g - tx_coords_cyl) * n_cyl), axis=1).reshape(-1, 1)  # producto escalar
    tx_pm_imag = tx_coords_cyl + 2 * n_cyl * temp
    return tx_pm_imag, g, n_cyl


def return_pitch_catch_cylfun_circmirror(tx_coords_sa, rx_coords_sa, curv_sign):

    def fun(cyl_radio, a0, rot):
        rotat = Rotation.from_euler(*rot, degrees=True)
        tx_coords_cyl = rotat.apply(tx_coords_sa) + a0
        rx_coords_cyl = rotat.apply(rx_coords_sa) + a0
        n_probe = rotat.apply([0., 0., -1.])
        # calcular imagenes reales de los emisores por el espejo circular
        tx_cm_imag = compute_circ_mirror_image(tx_coords_cyl, n_probe, cyl_radio, curv_sign)
        n_tx = tx_coords_sa.shape[0]
        d = []
        for i in range(n_tx):
            tx_pm_imag, _, _ = compute_g_tangent_plane_mirror_imag(tx_coords_cyl[i, :], tx_cm_imag[i, :],
                                                                   rx_coords_cyl, cyl_radio, curv_sign)
            d.append(np.linalg.norm(rx_coords_cyl - tx_pm_imag, axis=1))
            # g_proj.append(compute_g_circmirror(tx_mirror_imag[i, :], rx_coords_cyl, cyl_radio))
            # n_cyl.append(g/np.linalg.norm(g, axis=1))
            # vec = rx_coords_cyl - tx_coords_cyl[i, :]
            #
            # # hay un caso deforme en que vec tiene la direccion del eje x. En este caso hay que
            # # buscar el "punto medio" en el "plano radial"
            # caso_raro = np.isclose()
            #
            # # producto vectorial entre vec y g_proj, define el plano de incidencia
            # n_inc[:, 0] = vec[:, 1]*g_proj[:, 2] - vec[:, 2]*g_proj[:, 1]
            # n_inc[:, 1] = vec[:, 2] * g_proj[:, 0] - vec[:, 0] * g_proj[:, 2]
            # n_inc[:, 2] = vec[:, 0]*g_proj[:, ] - vec[:, 1]*g_proj[:, 0]
            # # ahora hay que intersectar el plano normal a n_inc que pasa por el emisor (o receptor)
            # # con la recta que pasa por g_proj en direccion X
            # w1 = np.sum(((tx_coords_cyl - g_proj) * g_proj), axis=1)  # producto escalar
            # w2 = w1 / n_inc[:, 0].reshape(-1, 1) # producto escalar de n_inc con el [1, 0, 0]
            # g.append(g_proj + w2 * np.array([[1, 0, 0]]))
        return np.concatenate(d)

    return fun


# *********************************************************************************************************************
# ******** funciones para de PITCH-CATCH ESFERA con ESPEJO CIRCULAR *************************************************
def return_sphere_op_transform(v_op, k, curv_sign):
    """
    - v_op es el punto donde intersecta la linea que pasa por le centro del array u tiene direccion k
    - k es la "dirección da la onda", que en el caso pitch catch es la normal al array y en el caso plano wave la direccion
    de la onda plana

    Hay que considerar el plano que contiene la normal en v_op y el vector k. La transformacion de coordenadas
    se define en base a eso. Este plano es el plano "espejo circular"
    v3 es el eje optico
    (v2, v3) definen el plano del espejo circular

    Los eje del sistema VCS son:

    v1: ortogonal a un plano "espejo circular" definido por v_op y k
    v2: tangencial al espejo circular
    v3: eje óptico, positivo hacia el lado de la fuente ("arriba")

    Calcula la transformacion de coordenadas sph2op de los ejes de la esfera (WCS) a los ejes "opticos" (VCS),
    y su inversa op2sph.
    Es solo la parte de rotacion. Luego hay que tener en cuenta que el orgen de VCS es el vertice v_op
    Args:
        v_op: vertice del eje optico
        k: "direccion de la onda"
        curv_sign: -1 en caso concavo, 1 en caso convexo
    """

    assert v_op.shape[1] == 3 and k.shape[1] == 3
    # definir terna
    v3 = v_op/np.linalg.norm(v_op, axis=1).reshape(-1, 1) * curv_sign
    v2 = k - np.sum(k*v3, axis=1).reshape(-1, 1) * v3
    # que pasa si v2=0 ?? o sea, k paralelo a la normal en el punto de incidencia
    # en ese caso se puede usar cualquier vector perpendicular a v3, por ej. el que tiene v2[0]=0
    if np.isclose(np.linalg.norm(v2), 0):
        v2 = np.array([[0, 1, -v3[0, 1]/v3[0, 2]]])

    v2 = v2/np.linalg.norm(v2, axis=1).reshape(-1, 1)

    # producto vectorial v2 x v3
    v1 = np.column_stack((v2[:, 1]*v3[:, 2] - v2[:, 2]*v3[:, 1],
                   v2[:, 2]*v3[:, 0] - v2[:, 0]*v3[:, 2],
                   v2[:, 0]*v3[:, 1] - v2[:, 1]*v3[:, 0]))

    # lista de matrices de transformacion desde VCS a WCS
    mat_list = [np.column_stack((v1[i, :], v2[i, :], v3[i, :])) for i in range(v1.shape[0])]

    def op2sph(vcs_coords):
        assert vcs_coords.shape[1] == 3
        wcs_coords = np.zeros_like(vcs_coords)
        for i, m in enumerate(mat_list):
            wcs_coords[i, :] = np.matmul(m, vcs_coords[i, :].T).T
        return wcs_coords

    def sph2op(wcs_coords):
        assert wcs_coords.shape[1] == 3
        vcs_coords = np.zeros_like(wcs_coords)
        for i, m in enumerate(mat_list):
            vcs_coords[i, :] = np.matmul(m.T, wcs_coords[i, :].T).T
        return wcs_coords

    return op2sph, sph2op, (v1, v2, v3)


def compute_sphere_mirror_image(tx_wcs_coords, n_probe, sph_rad, curv_sign):
    """Calcula la imagen del emisor generada por el espejo esferico, y devuelve sus coordenadas en WCS"""

    n_probe_rep = np.repeat(n_probe.reshape(1, 3), tx_wcs_coords.shape[0], axis=0)  # array de (N, 3)
    g0_1, g0_2 = u3d.sphere_line_intersec(sph_rad, tx_wcs_coords, n_probe_rep)
    v_op = g0_1 if curv_sign > 0 else g0_2  # se elige segun convexo o concavo

    op2sph, sph2op, _ = return_sphere_op_transform(v_op, n_probe, curv_sign)
    tx_vcs = sph2op(tx_wcs_coords - v_op)

    # ahora calculamos la imagen, que en este caso es real
    tx_imag_vcs = np.zeros_like(tx_vcs)
    # ahora aplico las formulas para el espejo circular (ver Hetch)
    temp = 1 / tx_vcs[:, 2]
    finv = (-1) * curv_sign * 2 / sph_rad  # inverso del foco del espejo
    tx_imag_vcs[:, 2] = 1 / (finv - temp)
    # acá aprece el termino de "lateral magnification"
    tx_imag_vcs[:, 1] = (-tx_imag_vcs[:, 2] / tx_vcs[:, 2]) * tx_vcs[:, 1]
    # ahora que vovler a coordenadas en WCS
    tx_imag_wcs = op2sph(tx_imag_vcs) + v_op
    return tx_imag_wcs


def return_pitch_catch_sphere_fun_circmirror(tx_coords_pcs, rx_coords_pcs, curv_sign):

    def fun(sph_rad, a0, rot):

        rotat = Rotation.from_euler(*rot, degrees=True)
        tx_coords_wcs = rotat.apply(tx_coords_pcs) + a0
        rx_coords_wcs = rotat.apply(rx_coords_pcs) + a0

        # OJO!: acá suponemos que estamos usando un sistema PCS en el cual la normal es "negativa"
        n_probe = rotat.apply([0., 0., -1.]).reshape(-1, 3)  # WARNING: puede haber problemas con el sentido del eje Z en PCS?!?!?!?!?

        # calcular imagenes reales de los emisores por el espejo circular
        tx_imag_wcs = compute_sphere_mirror_image(tx_coords_wcs, n_probe, sph_rad, curv_sign)
        n_tx = tx_coords_wcs.shape[0]
        n_rx = rx_coords_wcs.shape[0]
        d = []
        if curv_sign:
            for i in range(n_tx):
                p0 = tx_imag_wcs[i, :].reshape((-1, 3))
                p0 = np.repeat(p0, n_rx, axis=0) # repetir las coordenadas de la imagen especular n_rx veces como filas
                v = rx_coords_wcs - p0 # direccion de las lineas que hay intersectar con la esfera, todas naces en p0
                q1, q2 = utils_3d.sphere_line_intersec(sph_rad, p0, v)
                g = q2
                d.append(np.linalg.norm(tx_coords_wcs[i, :] - g, axis=1) + np.linalg.norm(rx_coords_wcs - g, axis=1))

        else:
            # TODO: en el caso concavo creo que dependiendo si la imagen es real o virtual hay que elegir q1 o q2
            raise NotImplementedError

        return np.concatenate(d)

    return fun




# ***************************************************************************************************************
# *************** FUNCIONES PARA PLANE WAVE CILINDRO ************************************************************
def calc_cyl_plane_wave_gp(a0, v_pw, radio):
    """
    TODO: esta funcion es media al pedo, es casi lo mismo que cylinder_line_interesec

    Calcula el punto Gp en que la recta que pasa por el centro del array
    y tiene direccion v_pw cruza el círculo de radio r.
    Args:
        a0: centro del array
        v_pw: direccion de la onda plana en el sistema cilindro (eje X), hay obtenerlo rotando el vector del sistema
        array, que es el que se define al disparar

    Returns:
        gp1, gp2: las dos soluciones, gp1 la mas cercana a a0
        """

    # asegurarse de que sean numpy arrays
    a0 = np.array(a0).reshape(1, 3)  # es necesario para la funcion cylinder_line_intersec
    v_pw = np.array(v_pw).reshape(1, 3)

    assert v_pw[0, 2] < 0  # chequar que la onda plana vaya hacia abajo
    gp1, gp2 = u3d.cylinder_line_intersec(radio, a0, v_pw)

    return gp1, gp2


def calc_cyl_plane_wave_focus_yz(a0, v_pw, radio, curv_sign):
    """
    Calcular la fuente virtual H, generada por el espejo circular, en el plano de la proyeccion, al incidir
    la onda plana.

    Args:
        a0: centro del array
        v_pw: vector incidente normalizado
        radio: radio del espejo
        curv_sign: covexo:1, concavo:-1
    Returns:
        hm, gp

    """
    # primer calcular gp
    gp1, gp2 = calc_cyl_plane_wave_gp(a0, v_pw, radio)
    gp = gp1 if curv_sign > 0 else gp2  # se elige segun convexo o concavo
    gp[0, 0] = 0
    cyl2op, op2cyl = return_cyl_op_transform(gp, curv_sign)
    # calcular coordenadas de v_pw en VCS (sistema del eje optico)
    v_pw_op = cyl2op(v_pw.reshape(1, 3))
    f = (-1) * curv_sign * radio / 2  # posicion del foco
    hm_yz_op = f * np.array([0, -v_pw_op[0, 1] / v_pw_op[0, 2], 1])
    hm_yz = op2cyl(hm_yz_op.reshape(1, 3)) + gp

    return hm_yz, gp


def calc_cyl_plane_wave_reflected_vector(a0, hm_yz, e, v_pw, radio, curv_sign):
    """

    Args:
        hm_yz: (3, ) o (1, 3)  ?
        e: (N, 3), array con las coordenadas de los elementos
        v_pw: (3, )
        radio:
        curv_sign:

    Returns:

    """

    a0 = np.array(a0).reshape(1, 3)
    hm_yz = np.array(hm_yz).reshape(1, 3)
    v_pw = np.array(v_pw).reshape(1, 3)
    v_pw = v_pw / np.linalg.norm(v_pw)  # normalizar v_pw

    # primero calculamo g_yz, proyeccion de g. Para ello se calcula la interseccion de la recta
    # entre hm_yz y e_yz (proyeccion de e)

    v = hm_yz[0, 1:] - e[:, 1:]  # vector 2d desde hm_yz hasta e_yz
    aux = utils.circle_line_intersec(radio, e[:, 1:], v)
    # g_yz_2d es (N, 2)
    g_yz_2d = aux[0] if curv_sign > 0 else aux[1]  # convexo o concavo
    g_yz = np.zeros_like(e)
    g_yz[:, 1:] = g_yz_2d

    # una vez calculada el punto g proyectado en el plano YZ, se puede calcular la normal en ese punto: norm_g
    # hay una normal para cada elemento e[i, :]

    norm_g = g_yz / (np.linalg.norm(g_yz, axis=1).reshape((-1, 1)))  # vector normal en el punto de incidencia (G)

    # calcular vector reflejado, uno por cada elemento
    n_elems = e.shape[0]
    aux = norm_g * np.repeat(v_pw, n_elems, axis=0)
    dot_nvpw = np.sum(aux, axis=1).reshape(-1, 1)
    v_pw_n = np.sum(dot_nvpw, axis=1).reshape(-1, 1) * norm_g  # componente de v_pw en direccion de la normal
    v_refl = v_pw - 2 * v_pw_n

    # calcular el reflejo del centro del array (a0), para cada norm_g (para cada elemento)
    aux = (norm_g * (a0 - g_yz)).sum(axis=1).reshape(-1, 1)  # producto escalar
    a0_refl = a0 - 2 * aux * norm_g

    return a0_refl, v_refl


def return_plane_wave_cyl_fun(coords_sa, v_pw, curv_sign):
    """
    coords_sa: array de coordenadas de los elementos en el sistema array, shape (n_elems, 3)
    (se genera con utils_3d.array_coordinates_list, hay que transformar la lista en array)
    v_pw: array de vectores de direccionde la onda, cada fila es un vector, shape (n_waves, 3)
    """

    assert v_pw.shape[1] == 3
    n_waves = v_pw.shape[0]
    # normalizar los vectores por is no lo están
    norma_v_pw = np.linalg.norm(v_pw, axis=1).reshape((-1, 1))
    v_pw = v_pw / norma_v_pw

    def fun(radio, a0, rot):
        a0 = np.array(a0).reshape(1, 3)
        # rotar y trasladar elementos
        rotat = tra.Rotation.from_euler(rot[0], rot[1], degrees=True)
        v_pw_rot = rotat.apply(v_pw)
        e_mov = rotat.apply(coords_sa) + a0

        dist = []
        for i in range(n_waves):
            hm_yz, gp = calc_cyl_plane_wave_focus_yz(a0, v_pw_rot[i, :], radio, curv_sign)
            # g, _, _, _, _ = calc_cyl_plane_wave_reflected_vector(hm_yz, e_mov, v_pw_rot[i, :], radio)
            # # tiempo de ida de la onda plana, supondiendo que en t=0 pasa por el centro del array
            # dist_ida = np.dot(g - a0, v_pw_rot[i, :])
            # dist.append(np.linalg.norm(e_mov - g, axis=1) + dist_ida)
            a0_refl, v_refl = calc_cyl_plane_wave_reflected_vector(a0, hm_yz, e_mov, v_pw_rot[i, :], radio, curv_sign)
            dist.append(np.sum((e_mov - a0_refl) * v_refl, axis=1))

        return np.concatenate(dist)  # order C, quedan primero todos los elementos para la primera onda, luego para
        # la segunda onda, etc

    return fun


# ************ FUNCIONES PARA PLANE WAVE ESFERA **********************************************************************
# def return_sphere_op_transform(v_op, v_pw, curv_sign):
#     """v_op es el punto donde intersecta la linea uqe pasa por le centro del array u tiene direccion v_pw
#     Hay que considerar el plano que contiene la normal en ese punto y el vector v_pw. La transformacion de coordenadas
#     se define en base a eso.
#     v3 es el eje optico
#     (v2, v3) definen el plano del espejo circular
#     """
#     assert v_op.shape == (1, 3) and v_pw.shape == (1, 3)
#     # definir terna
#     v3 = v_op/np.linalg.norm(v_op, axis=1)
#     v2 = v_pw - np.sum(v_pw*v3, axis=1) * v3
#     # que pasa si v2=0 ?? o sea, v_pw paralelo a la normal en el punto de incidencia
#     # en ese caso se puede usar cualquier vector perpendicular a v3, por ej. el que tiene v2[0]=0
#     if np.isclose(np.linalg.norm(v2), 0):
#         v2 = np.array([[0, 1, -v3[0, 1]/v3[0, 2]]])
#         v2 = v2/np.linalg.norm(v2, axis=1)
#
#     # producto vectorial v2 x v3
#     v1 = np.array([[v2[0, 1]*v3[0, 2] - v2[0, 2]*v3[0, 1]],
#                    [v2[0, 2]*v3[0, 0] - v2[0, 0]*v3[0, 2]],
#                    [v2[0, 0]*v3[0, 1] - v2[0, 1]*v3[0, 0]]])
#     v1 = v1.reshape(1, 3)
#     mat = np.concatenate((v1, v2, v3), axis=0).T
#
#     def op2sph(op_coords):
#         assert op_coords.shape[1] == 3
#         return np.matmul(mat, op_coords.T).T
#
#     return op2sph, (v1, v2, v3)


def calc_sphere_plane_wave_focus(a0, v_pw, radio, curv_sign):
    # buscar interseccion con la esfera
    gp1, gp2 = utils_3d.sphere_line_intersec(radio, a0, v_pw)
    v_op = gp1 if curv_sign > 0 else gp2  # se elige segun convexo o concavo
    # primero determinar el plano de trabajo
    op2sph, _, (v1, v2, v3) = return_sphere_op_transform(v_op, v_pw, curv_sign)
    # tangente del angulo que forma el vector de onda con el eje optico
    tang = np.sum(v_pw*v2, axis=1)/np.sum(v_pw*v3, axis=1)
    f = (-1) * curv_sign * radio / 2  # posicion del foco
    hm_op = f * np.array([[0, -tang[0], 1]])
    hm_sph = op2sph(hm_op) + v_op
    return hm_sph, v_op


def calc_sphere_plane_wave_reflected_vector(a0, hm, e, v_pw, radio, curv_sign):
    """
    """

    a0 = np.array(a0).reshape(1, 3)
    v_pw = np.array(v_pw).reshape(1, 3)
    v_pw = v_pw / np.linalg.norm(v_pw)  # normalizar v_pw

    v = hm - e
    aux = utils_3d.sphere_line_intersec(radio, e, v)
    g = aux[0] if curv_sign > 0 else aux[1]  # convexo o concavo
    norm_g = g / (np.linalg.norm(g, axis=1).reshape((-1, 1)))  # vector normal en el punto de incidencia (G)

    # calcular vector reflejado, uno por cada elemento
    n_elems = e.shape[0]
    aux = norm_g * np.repeat(v_pw, n_elems, axis=0)
    dot_nvpw = np.sum(aux, axis=1).reshape(-1, 1)
    v_pw_n = np.sum(dot_nvpw, axis=1).reshape(-1, 1) * norm_g  # componente de v_pw en direccion de la normal
    v_refl = v_pw - 2 * v_pw_n

    # calcular el reflejo del centro del array (a0), para cada norm_g (para cada elemento)
    aux = (norm_g * (a0 - g)).sum(axis=1).reshape(-1, 1)  # producto escalar
    a0_refl = a0 - 2 * aux * norm_g

    return a0_refl, v_refl


def return_plane_wave_sphere_fun(coords_sa, v_pw, curv_sign):
    assert v_pw.shape[1] == 3
    n_waves = v_pw.shape[0]
    # normalizar los vectores por is no lo están
    norma_v_pw = np.linalg.norm(v_pw, axis=1).reshape((-1, 1))
    v_pw = v_pw / norma_v_pw

    def fun(radio, a0, rot):
        a0 = np.array(a0).reshape(1, 3)
        # rotar y trasladar elementos
        rotat = tra.Rotation.from_euler(rot[0], rot[1], degrees=True)
        v_pw_rot = rotat.apply(v_pw)
        e_mov = rotat.apply(coords_sa) + a0

        dist = []
        for i in range(n_waves):
            temp = v_pw_rot[i, :].reshape(1, 3)
            hm_sph, v_op = calc_sphere_plane_wave_focus(a0, temp, radio, curv_sign)
            a0_refl, v_refl = calc_sphere_plane_wave_reflected_vector(a0, hm_sph, e_mov, temp, radio, curv_sign)
            dist.append(np.sum((e_mov - a0_refl) * v_refl, axis=1))
        return np.concatenate(dist)

    return fun


# ********************************************************************************************************************
# ********* funciones para calcular las reflexiones en esfera en forma exacta usando ray casting e interpolación *****
def circle_ray_cast(z_source, angs, radio, curv_sign):
    """ Una fuente en z, respecto al centro de la esfera / círculo. Angulos correspondientes a los puntos de reflexion.
    Devuelve las coordenaddas de esos puntos y el correspondiente vector direccion del rayo reflejado"""
    # crear array de puntos sobre la esfera, considera plano x=0
    n_pts = angs.size
    angs = np.radians(angs)
    g = np.zeros((n_pts, 3))
    g[:, 1] = radio * np.sin(angs)
    g[:, 2] = radio * np.cos(angs)
    g *= curv_sign  # si es concavo, los puntos pasan para abaja a y<0. Lo s paso a y<0 para que los rayos reflejados
    # se dirijan hacia y>0. Porque luego voy a interpolar con valores de y>0
    ng = curv_sign * g/np.linalg.norm(g, axis=1).reshape((-1, 1))  # normal en g
    source = np.array([0, 0, z_source]).reshape((1, 3))
    v_refl = sf.incident2reflected(g - source, ng)
    return g, v_refl


def interpolate_rays(r, z, z_source, g, v_refl, curv_sign):
    # calcular coordenadas cilindricas (r, z)
    # r = np.hypot(elem_coords[:, 0], elem_coords[:, 1])
    # z = elem_coords[:, 2]
    # valores de x donde los rayos intersecan la recta horizontal que pasa por z
    y_rays = g[:, 1] + (z - g[:, 2])*v_refl[:, 1]/v_refl[:, 2]

    # si es circulo convaco los y_rays son negativos!! En este caso los hacemos positivos
    # para la interpolacion?? o le cambio el signo a r???
    # r *= curv_sign

    assert np.isclose(y_rays[0], 0)  # siempre hay tener el 0, por si r es muy pequeño
    i = np.argmax(r < y_rays) - 1 # indice donde r empieza a ser mayor que y_rays, de modo
    # que r esta entre y_rays[i+1] y y_rays[i]
    delta_y = r - y_rays[i]
    # calcular los tiempos de vuelo en los puntos de interseccion
    source = np.array([0, 0, z_source]).reshape((1, 3))
    pts = np.zeros_like(g)
    pts[:, 1] = y_rays
    pts[:, 2] = z
    d = np.linalg.norm(g - source, axis=1) + np.linalg.norm(pts - g, axis=1)
    # interpolacion lineal
    d_interp = delta_y * (d[i+1] - d[i])/(y_rays[i+1] - y_rays[i]) + d[i]
    g_interp = delta_y * (g[i+1, :] - g[i, :])/(y_rays[i+1] - y_rays[i]) + g[i, :]
    return d_interp, g_interp, d, y_rays


def return_sphere_pitch_catch_interporays_fun(tx_coords_pcs, rx_coords_pcs, angs, curv_sign):

    def fun(radio, a0, rot):
        rotat = Rotation.from_euler(*rot, degrees=True)
        tx_coords_wcs = rotat.apply(tx_coords_pcs) + a0
        rx_coords_wcs = rotat.apply(rx_coords_pcs) + a0

        n_tx = tx_coords_pcs.shape[0]
        n_rx = rx_coords_pcs.shape[0]
        d = []
        for i in range(n_tx):
            q = tx_coords_wcs[i, :].reshape((1, 3))
            h = np.linalg.norm(q)
            q = np.repeat(q, n_rx, axis=0)
            # producto escalar entre el receptor y el vector en la direccion del emisor
            # (rr, zz) son las coordenadas cilindricas de rx sobre el eje definido por tx
            zz = np.sum(rx_coords_wcs * q/h, axis=1).reshape((-1, 1))
            rr = np.linalg.norm(rx_coords_wcs - zz*(q/h), axis=1)
            # calcular los rayos reflejados que salen de tx
            g, v_refl = circle_ray_cast(h, angs, radio, curv_sign)

            for j in range(n_rx):
                d_interp, _, _, _ = interpolate_rays(rr[j], zz[j], h, g, v_refl, curv_sign)
                d.append(d_interp)
        return np.array(d)

    return fun


def return_cyl_pitch_catch_interporays_fun(tx_coords_pcs, rx_coords_pcs, angs, curv_sign):

    def fun(radio, a0, rot):
        rotat = Rotation.from_euler(*rot, degrees=True)
        tx_coords_wcs = rotat.apply(tx_coords_pcs) + a0
        rx_coords_wcs = rotat.apply(rx_coords_pcs) + a0

        n_tx = tx_coords_pcs.shape[0]
        n_rx = rx_coords_pcs.shape[0]
        rx_yz = rx_coords_wcs.copy()
        rx_yz[:, 0] = 0 # proyectar en plano yz, x=0
        d = []
        for i in range(n_tx):
            q = tx_coords_wcs[i, :].reshape((1, 3))
            # proyectar sobre plano yz (cilindro en eje x)
            q_yz = q.copy()
            q_yz[0, 0] = 0
            h = np.linalg.norm(q)
            q_yz = np.repeat(q_yz, n_rx, axis=0)
            # producto escalar entre el receptor y el vector en la direccion del emisor
            # (rr, zz) son las coordenadas polares de rx sobre el eje definido por tx (todos proyectados sobre
            # el plano yz)
            zz = np.sum(rx_yz * q_yz/h, axis=1).reshape((-1, 1))
            temp = rx_yz - zz*(q_yz/h)
            # guardo el signo de la componente y (en el sistema ajustado al emisor) del receptor
            sign_y_rx = np.sign(temp[:, 1])
            rr = np.linalg.norm(temp, axis=1)
            # calcular los rayos reflejados que salen de tx
            g, v_refl = circle_ray_cast(h, angs, radio, curv_sign)

            for j in range(n_rx):
                _, g_interp, _, _ = interpolate_rays(rr[j], zz[j], h, g, v_refl, curv_sign)
                # como estoy tomando los g siempre con y>0 por simetria, en el caso en que el receptor tenga esté
                # "en la otra mitad" hay que tomar los g con y<0
                g_interp[1] *= sign_y_rx[j]  # con esto le cambio el signo si es necesario
                # hay que cambiar las coordenadas g, porque estan referidos a un eje z que pasa por el emisor
                cos_ang = q[0, 1]/np.linalg.norm(q)
                sin_ang = q[0, 2]/np.linalg.norm(q)
                gy = g_interp[1] * sin_ang + g_interp[2] * cos_ang
                gz = -g_interp[1] * cos_ang + g_interp[2] * sin_ang
                g_ = np.array([0, gy, gz])
                # plano tangente en g_
                ng_ = g_ / np.linalg.norm(g_)  # normal al plano, depende de rx
                # reflejar emisor en ese plano
                tx_mirror = q - 2 * np.sum((q - g_) * ng_) * ng_
                d.append(np.linalg.norm(tx_mirror - rx_coords_wcs[j, :]))

        return np.array(d)

    return fun


# ********* funciones para pitch catch cilindro con subapertura que depende del punto de reflexion *******************

def find_receive_points_cylinder(tx_coords, a0, rot, cyl_radio, concavo=True):
    """ Hay unos elementos emisores "tx_coords". Se mueve el array y se miran los rayos que salen de
     esos elementos con direccion de la normal del array. Se busca donde intersectan esos rayos al cilindro (puntos q2).
     Ahí se refleja el rayo, y se ve donde el rayo reflejado interseca el plano del array (punto rx_point).
      Tambien se calcula la imagen especular del emisor respecto al plano tangente"""

    assert tx_coords.shape[1] == 3
    a0 = np.array(a0)
    tx_coords_mov = a0.reshape(1, 3) + rot.apply(tx_coords)
    # OJO!: acá suponemos que estamos usando un sistema PCS en el cual la normal es "negativa"
    n_array = rot.apply([[0, 0, -1]])  # nomral a la apertura del array
    # intersectar rayo con cilindro
    v_inc = np.repeat(n_array, tx_coords_mov.shape[0], axis=0)
    q1, q2 = u3d.cylinder_line_intersec(cyl_radio, tx_coords_mov, v_inc)
    if concavo:
        # se usa q2
        n_cyl = np.zeros_like(q2)
        n_cyl[:, 1:] = -q2[:, 1:]
        n_cyl = n_cyl / np.linalg.norm(n_cyl, axis=1).reshape(-1, 1)
        # el rayo incidente v_inc tiene la direccion de n_array, hya que reflejarlo segun n_cyl
        # proyecto el vector incidente sobre la normal
        v_inc_proj_n_cyl = n_cyl * (np.sum(v_inc * n_cyl, axis=1).reshape(-1, 1))
        v_refl = v_inc - 2 * v_inc_proj_n_cyl
    else:
        raise Exception('falta implementar caso convexo')

    # reflejar la fuente en el plano tangente
    v = q2 - tx_coords_mov
    v_proj_n_cyl = n_cyl * (np.sum(v * n_cyl, axis=1).reshape(-1, 1))
    coords_mirror = tx_coords_mov + 2 * v_proj_n_cyl

    # intersectar rayos reflejados con el plano del array
    w1 = np.sum(((tx_coords_mov - q2) * n_array), axis=1)  # producto escalar
    w2 = (w1 / np.sum(v_refl * n_array, axis=1)).reshape(-1, 1)
    rx_point = q2 + w2 * v_refl

    return rx_point, coords_mirror, q2, n_cyl


def receive_subap(rx_coords, subap_radio, nel_x, nel_y, pitch_x, pitch_y):
    """rx_coords son coordenadas en el sistema del array, hay que buscar los elementos del array
    que estan dentro de un radio alrededor de cada elemento receptor"""
    a = u3d.array_coordinates_list(nel_x, nel_y, pitch_x, pitch_y)
    a = np.array(a)
    n_rx = rx_coords.shape[0]
    rx_subap_coords = []
    rx_subap_linidx = []
    mask = []
    for m in range(n_rx):
        mask.append(np.linalg.norm(a - rx_coords[m, :], axis=1) <= subap_radio)
        rx_subap_coords.append(a[mask[m], :])
        rx_subap_linidx.append(np.nonzero(mask[m])[0])
    # x0, y0 = pitch_x*(nel_x - 1)/2, pitch_y*(nel_y - 1)/2
    # i = (rx_coords[:, 0] + x0)/ pitch_x
    # j = (rx_coords[:, 1] + y0)/ pitch_y
    # i = np.round(i).astype(int)
    # j = np.round(j).astype(int)

    return rx_subap_linidx, rx_subap_coords, mask


def return_pitch_catch_cyl_fun_emision_0_grados(tx_coords, subap_radio, nel_x, nel_y, pitch_x, pitch_y):
    array_coords = np.array(u3d.array_coordinates_list(nel_x, nel_y, pitch_x, pitch_y))

    def fun(cyl_radio, a0, rot):
        rotat = Rotation.from_euler(*rot, degrees=True)
        rotat_inv = rotat.inv()
        rx_point, coords_mirror, q2, n_cyl = find_receive_points_cylinder(tx_coords, a0, rotat, cyl_radio)
        # pasar de coordenadas en sistema cilindro a coordenadas en sistema
        rx_coords_sa = rotat_inv.apply(rx_point - a0)
        coords_mirror_sa = rotat_inv.apply(coords_mirror - a0)
        rx_subap_linidx, rx_subap_coords, mask = receive_subap(rx_coords_sa, subap_radio, nel_x, nel_y,
                                                               pitch_x, pitch_y)
        d = []
        for i in range(coords_mirror_sa.shape[0]):
            d.append(np.linalg.norm(array_coords - coords_mirror_sa[i, :], axis=1))
        d = np.concatenate(d)
        mask = np.concatenate(mask)
        return d, mask

    return fun


# ******************************************************************************************************************
def mov_surf_points(e, a0, rot, n=None):
    """

    Args:
        e: (ex, ey, ez) tuple o list, donde ex.shape es (nel_x, nel_y)
        n: (nx, ny, nz), tuple o list, donde ex.shape es (nel_x, nel_y)
        a0: coordenadas del centro del array
        rot: rotacion de scipy. todo ALERTA TERNA IZQUIERDA: estoy usando una terna izquierda para el Sistema Array,
        esto puede crear problemas con las rotaciones, tener en cuenta!!!

    Returns:
        (ex_mov, ey_mov, ez_mov): coordenadas de los puntos trasladados y rotados
    """

    # hay que poner los puntos en forma (n_puntos, 3)
    n_points = e[0].size
    temp = (e[0].reshape((n_points, 1), order='C'),
            e[1].reshape((n_points, 1), order='C'),
            e[2].reshape((n_points, 1), order='C'))
    e_n3 = np.hstack(temp)
    e_n3_rot = rot.apply(e_n3)

    # rotar normales
    if n is not None:
        temp = (n[0].reshape((n_points, 1), order='C'),
                n[1].reshape((n_points, 1), order='C'),
                n[2].reshape((n_points, 1), order='C'))
        n_n3 = np.hstack(temp)
        n_n3_rot = rot.apply(n_n3)
    else:
        n_n3_rot = None

    # trasladar
    e_n3_rot += np.array(a0)

    return e_n3_rot, n_n3_rot
