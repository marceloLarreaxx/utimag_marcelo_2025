import numpy as np
from scipy.spatial import transform as tra
from sklearn.linear_model import LinearRegression
from scipy.optimize import root_scalar
from scipy.spatial.transform import Rotation
import imag3D.snell_fermat_3d as sf
import utils
from imag3D import utils_3d as u3d


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
    #idx_fit = reg.predict(ixy).reshape((nel_x, nel_y), order='F')
    coef = np.insert(reg.coef_, 0, reg.intercept_)

    if degree == 2:
        def idx_fun(x, y):
            q = coef[0] + coef[1]*x + coef[2]*y + coef[3]*x**2 + coef[4]*y**2 + coef[5]*x*y
            grad_q = np.array([coef[1] + 2*coef[3]*x + coef[5]*y, coef[2] + 2*coef[4]*y + coef[5]*x])
            return q, grad_q
    else:
        def idx_fun(x, y):
            q = coef[0] + coef[1]*x + coef[2]*y
            grad_q = np.array([coef[1], coef[2]])
            return q, grad_q

    return coef, idx_fun, reg


def pulse_echo_surface(idx_fun, c, fs, xy_mesh):
    idx_fit, grad = idx_fun(*xy_mesh)
    d = (c / 2) * idx_fit / fs  # distancias
    nxy = (c / 2) * grad / fs  # componentes (x, y) de la normal local a la superficie
    nxy_2 = nxy[0, :, :] ** 2 + nxy[1, :, :] ** 2
    grad_error = nxy_2 > 1
    nxy_2[grad_error] = 1 # achicamos a 1 para que la raiz no de "nan"
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
    grad_d = c * grad / fs # gradiente de las distancias
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
        nz = np.sqrt(0.5*(1 + np.sqrt(1 - grad_d_2)))
        nx = 0.5 * grad_d[0, :, :]/nz
        ny = 0.5 * grad_d[1, :, :] / nz
        qx = xy_mesh[0] - d * nx / nz
        qy = xy_mesh[1] - d * ny / nz
        qz = -d/nz
        ex = qx
        ey = qy
        ez = d*(1/(2*nz**2) - 1)
        res = None

    else:
        # todo: contemplar aparte el caso en que (ux - grad_d_x) = 0 (DIVISON POR CERO)
        b = grad_d[1, :, :]/(grad_d[0, :, :] - u[0])
        nx = np.zeros_like(xy_mesh[0])
        res = []
        ni, nj = xy_mesh[0].shape
        for i in range(ni):
            res.append([])
            for j in range(nj):
                fun = return_pw_nx_fun(theta, grad_d[:, i, j])
                nx_max = 1 / np.sqrt(2 * (1 + b[i, j] ** 2))
                res[i].append(root_scalar(fun, x0=-nx_max*0.5, x1=nx_max*0.5))
                nx[i, j] = res[i][j].root

        h = (1+b**2) * nx**2  # variable auxiliar, tiene que ser <= 1
        grad_error = h > 1
        h[grad_error] = 1 # achicamos esos valores a 1 para que sqrt no de "nan"
        nz = np.sqrt(1 - h)
        ny = b*nx
        d_a = (xy_mesh[0] + np.sign(u[0])*x_0) * u[0] - d
        udotn = u[0]*nx + u[1]*nz  # producto escalar de u con la normal
        g = d_a/udotn
        alfa = g/(2*udotn)
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
        return u[0]*nx2 + u[1]*nx * np.sqrt(1 - nx2 * (1 + b ** 2)) + 0.5 * a

    return fun


def idx_filter_3d(idx, valid, nel_x, nel_y):
    """ Suaviza los índices mediante un ajuste cuadrático.
     La variable idx (indice del primero eco) es de 1-dimension, y viene ordenada a lo largo de dirección "x".
     Es decir: si nel_x=4, entonces peco viene indexada como: 0 1 2 3 0 1 2 3 0 1 2 3 etc."""

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

    # normal
    nx = np.diff(d, axis=0) / pitch
    nx = nx[:, 0:(nel_y - 1)]
    ny = np.diff(d, axis=1) / pitch
    ny = ny[0:(nel_x-1), :]
    # nx = np.diff(d, axis=1) / pitch
    # nx = nx[0:(nel_y - 1), :]
    # ny = np.diff(d, axis=0) / pitch
    # ny = ny[:, 0:(nel_x-1)]
    nz = np.sqrt(1 - nx**2 - ny**2)  # puede dar nan, revisar, todo

    # indice 0: x, indice 1: y
    x_idx, y_idx = np.meshgrid(range(nel_x), range(nel_y), indexing='ij')

    # posiciones del array
    xa = pitch * (x_idx - (nel_x - 1)/2)
    ya = pitch * (y_idx - (nel_y - 1)/2)
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
        coef = np.concatenate([xs, ys, xs**2, ys**2, xs*ys, np.ones_like(xs)], axis=1)
    else:
        raise Exception('Invalid model')

    fit = np.linalg.lstsq(coef, zs)
    zs_fit = np.matmul(coef, fit[0])

    def surf_fun(x, y):
        if model == 'plane':
            z = fit[0][0]*x + fit[0][1]*y + fit[0][2]
        if model == 'poly2':
            z = fit[0][0] * x + fit[0][1] * y + fit[0][2] * x**2 + fit[0][3] * y**2 + fit[0][4] * x * y + fit[0][5]
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
    a_cyl = [rot.apply(a) + a_center for a in a_prop] # coordenadas en sistema cilindro
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
            a_cyl = [rot.apply(a) + a_center for a in a_prop] # coordenadas en sistema cilindro
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


def return_pulse_echo_cylinder_fun(a, concavo=False):
    # CILINDRO EN EJE X. Todo: ver el caso cócavo. Esto está pensado pa convexo
    a = np.array(a)

    if concavo:
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
            abajo = a_cyl[:, 2] <= 0 # el igual es arbitrario
            # para contemplar que algunos elementos pueden estar abajo y otros arriba, uso esta expresion:
            dist = (radio - q) * abajo + (radio + q) * (1 - abajo)
            return 2*dist   # multiplico por 2 por el ida y vuelta

    else: # caso convexo
        def fun(radio, a0, rot):
            a0 = np.array(a0).reshape((1, 3))
            rotat = tra.Rotation.from_euler(rot[0], rot[1], degrees=True)  # 'xyz': ejes extrínsecos
            a_cyl = rotat.apply(a) + a0  # coordenadas en sistema cilindro
            # para cada elemento, definir el punto sobre el eje del cilindro, en un plano YZ
            a_cyl[:, 0] = 0  # poner X=0
            dist = np.linalg.norm(a_cyl, axis=1)
            return 2*(dist - radio)  # multiplico por 2 por el ida y vuelta

    return fun


def return_pitch_catch_plane_fun(a0, a):
    """Una interfaz plana en el plano Z=0. Emite el elemento a0 = (x0, y0, 0) según sus coordenadas
    en el sistema del Transducer Array (TA). Se aplica al TA una rotacion y traslación dada por mov.
    Se calcula la imagen especular de los elementos receptores y con eso se calcula la distancia

    Args:
    a0: (x0, y0, 0), elemento emisor
    a: lista de coordenadas de los elementos en el sistema arrray (se genera con utils_3d.array_coordinates_list)

    Returns:
    fun: funcion mov ---> distancia, donde mov: (z0, rotx, roty)
    """

    a0 = np.array(a0).reshape((1, 3))  # transformar por si viene como lista o tupla
    a = np.array(a)
    assert a.shape[1] == 3  # tiene que tener forma (N, 3)

    def fun(mov):
        # rotar y trasladar elementos
        rot = tra.Rotation.from_euler('XY', mov[1:], degrees=True)
        despla = np.array([0, 0, mov[0]])
        a0_mov = rot.apply(a0) + despla
        a_mov = rot.apply(a) + despla
        # reflejar receptores, cambiaando signo de z
        a_mov_refl = a_mov
        a_mov_refl[:, 2] = -a_mov_refl[:, 2]
        dist = np.linalg.norm(a_mov_refl - a0_mov, axis=1)
        return dist

    return fun


def return_pitch_catch_cyl_fun(coords):
    """
    CILINDRO EN EJE X. Todo: ver el caso cócavo. Esto está pensado pa convexo

    Args:
        coords: lista de 2-uplas (coords_tx, [coords_rx])

    Returns:

    """

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
            vec = vec/np.linalg.norm(vec)  # normalizando, obtenemos el vector unitario ortogonal al plano tangente
            # en el punto de incidencia normal del elemento tx
            # ahora hay que reflejar los receptores rx respecto a ese plano
            # TODO: convieve reflejar el transmisor que es uno solo!!!!!!!!!!!!!!!!!!!!
            co_rx_mov_refl = co_rx_mov - 2 * vec * np.dot(co_rx_mov - radio*vec, vec.T)
            dist.append(np.linalg.norm(co_rx_mov_refl - co_tx_mov, axis=1))
        return np.concatenate(dist)

    return fun


# ******** funciones para version alternativa de pitch-catch cilindro **************

def return_cyl_op_transform(g0, curv_sign):
    """

    Args:
        g0: vertice del eje optico
        curv_sign: -1 en caso concavo, 1 en caso convexo

    Returns:

    """
    g0_proj = g0.copy()
    g0_proj[:, 0] = 0
    # definir dos versores para calcular coordenadas sobre el eje optico
    # caso convexo: el versor es opuesto a g0
    g0_proj_norm = -curv_sign * g0_proj / np.linalg.norm(g0_proj, axis=1).reshape(-1, 1)  # normaliza
    k0 = np.zeros_like(g0_proj)  # vector perpendicaular a g0_proj
    k0[:, 1] = g0_proj_norm[:, 2]
    k0[:, 2] = -g0_proj_norm[:, 1]

    def cyl2op(source):
        source_op = np.zeros_like(source)
        # (source - g0_proj): cambiar origen de coordenadas al vertice del espejo (g0)
        source_trasla = (source - g0_proj)
        source_op[:, 1] = np.sum(source_trasla * k0, axis=1)  # producto escalar
        source_op[:, 2] = np.sum(source_trasla * g0_proj_norm, axis=1)  # producto escalar
        return source_op

    def op2cyl(source_op):
        source = np.zeros_like(source_op)
        # esta es una forma rara de multiplicar por la matriz de rotacion inversa, que es la traspuesta
        source[:, 1] = k0[:, 1] * source_op[:, 1] + g0_proj_norm[:, 1] * source_op[:, 2]
        source[:, 2] = k0[:, 2] * source_op[:, 1] + g0_proj_norm[:, 2] * source_op[:, 2]
        return source + g0_proj

    return cyl2op, op2cyl


def compute_circ_mirror_image(source, n_probe, cyl_radio, concavo):
    """ Dado source (coordenadas en sistema cilindro) y la normal n_probe al array, calcular el punto
    de incidencia G0 donde se define un eje óptico, y un plano con un espejo circular. Ahi se proyecta el source
    y se calculca su imagen segun el espejo"""

    n_probe_rep = np.repeat(n_probe.reshape(1, 3), source.shape[0], axis=0)  # array de (N, 3)
    g0_1, g0_2 = u3d.cylinder_line_intersec(cyl_radio, source, n_probe_rep)

    if concavo:
        # calculamos las coordenadas de source en el eje optico
        cyl2op, op2cyl = return_cyl_op_transform(g0_2, -1)
        source_op = cyl2op(source)
        # ahora calculamos la imagen, que en este caso es real
        source_imag_op = np.zeros_like(source_op)
        # ahora aplico las formulas para el espejo circular (ver Hetch)
        temp = 1/source_op[:, 2]
        finv = -2/cyl_radio  # inverso del foco del espejo
        source_imag_op[:, 2] = 1/(finv - temp)
        # acá aprece el termino de "lateral magnification"
        source_imag_op[:, 1] = (-source_imag_op[:, 2] / source_op[:, 2]) * source_op[:, 1]
        # ahora que vovler a coordenadas en el cilindro
        source_imag = op2cyl(source_imag_op)

    else:
        raise NotImplemented

    return source_imag


def compute_g_tangent_plane_mirror_imag(tx_coords_cyl, tx_cm_imag, rx_coords_cyl, cyl_radio):
    # proyectar sobre el plano YZ
    rx_proj = rx_coords_cyl.copy()
    rx_proj[:, 0] = 0
    # recta que pasa por el receptor y por la imagen real del emisor, cm: circular mirror
    vec = tx_cm_imag - rx_proj  # direccion de esa recta
    # itnerseccion con cilindro
    _, g = u3d.cylinder_line_intersec(cyl_radio, rx_proj, vec)
    # calcular la imagen especular del emisor respecto al plano tangente en g, tx_pm_imag, pm: plane mirror
    n_cyl = -g / np.linalg.norm(g, axis=1).reshape(-1, 1)
    tx_coords_cyl = tx_coords_cyl.reshape(1, 3)
    temp = np.sum(((g - tx_coords_cyl) * n_cyl), axis=1).reshape(-1, 1)  # producto escalar
    tx_pm_imag = tx_coords_cyl + 2 * n_cyl * temp
    return tx_pm_imag, g, n_cyl


def return_pitch_catch_cylfun_circmirror(tx_coords_sa, rx_coords_sa, concavo):

    if concavo:
        def fun(cyl_radio, a0, rot):
            rotat = Rotation.from_euler(*rot, degrees=True)
            tx_coords_cyl = rotat.apply(tx_coords_sa) + a0
            rx_coords_cyl = rotat.apply(rx_coords_sa) + a0
            n_probe = rotat.apply([0., 0., -1.])
            # calcular imagenes reales de los emisores por el espejo circular
            tx_cm_imag = compute_circ_mirror_image(tx_coords_cyl, n_probe, cyl_radio, concavo)
            n_tx = tx_coords_sa.shape[0]
            d = []
            for i in range(n_tx):
                tx_pm_imag, _, _ = compute_g_tangent_plane_mirror_imag(tx_coords_cyl[i, :], tx_cm_imag[i, :],
                                                                       rx_coords_cyl, cyl_radio)
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
    else:
        raise NotImplemented

    return fun


def find_receive_points_cylinder(tx_coords, a0, rot, cyl_radio, concavo=True):
    """ Hay unos elementos emisores "tx_coords". Se mueve el array y se miran los rayos que salen de
     esos elementos con direccion de la normal del array. Se busca donde intersectan esos rayos al cilindro (puntos q2).
     Ahí se refleja el rayo, y se ve donde el rayo reflejado interseca el plano del array (punto rx_point).
      Tambien se calcula la imagen especular del emisor respecto al plano tangente"""

    assert tx_coords.shape[1] == 3
    a0 = np.array(a0)
    tx_coords_mov = a0.reshape(1, 3) + rot.apply(tx_coords)
    n_array = rot.apply([[0, 0, -1]]) # nomral a la apertura del array
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
        v_inc_proj_n_cyl = n_cyl * (np.sum(v_inc*n_cyl, axis=1).reshape(-1, 1))
        v_refl = v_inc - 2 * v_inc_proj_n_cyl
    else:
        raise Exception('falta implementar caso convexo')

    # reflejar la fuente en el plano tangente
    v = q2 - tx_coords_mov
    v_proj_n_cyl = n_cyl * (np.sum(v*n_cyl, axis=1).reshape(-1, 1))
    coords_mirror = tx_coords_mov + 2*v_proj_n_cyl

    # intersectar rayos reflejados con el plano del array
    w1 = np.sum(((tx_coords_mov - q2) * n_array), axis=1) # producto escalar
    w2 = (w1 / np.sum(v_refl*n_array, axis=1)).reshape(-1, 1)
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


def return_pitch_catch_cyl_fun_subap(tx_coords,  subap_radio, nel_x, nel_y, pitch_x, pitch_y):

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


# *************** FUNCIONES PARA PLANE WAVE CILINDRO ************************************************************
def calc_gp(a0, v_pw, radio):
    """
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
    # gp1, gp2 = utils.circle_line_intersec(radio, a0, v_pw) # son 2-uplas
    # gp1, gp2 = np.insert(gp1, 0, 0), np.insert(gp2, 0, 0)  # agregar eje x

    # TODO: TEMA CONVCAVO / CONVEXO
    # si el centro del array (a0) está por arriba del cilindro, hay que tomar la raiz negativa, porque v_pw
    # apunta hacia abajo??
    # En el caso cóncavo, consideramos la mitad inferior del cilindro, y el centro del array adentro del cilindro.
    # y hay que uzar la raiz positiva
    return gp1, gp2


def calc_hm_yz(a0, v_pw, radio, curv_sign):
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
    # # primer calcular gp
    # gp1, gp2 = calc_gp(a0, v_pw, radio)
    # gp = gp1 if curv_sign > 0 else gp2 # se elige segun convexo o concavo
    # cyl2op, op2cyl = return_cyl_op_transform(gp, curv_sign)
    # # calcular coordenadas de v_pw en VCS (sistema del eje optico)
    # v_pw_op = cyl2op(v_pw.reshape(1, 3))
    # tan_beta = v_pw_op[1]/v_pw_op[0]
    # f = curv_sign * radio/2  # posicion del foco
    # hm_yz_op = f * np.array([0, 1, -tan_beta])
    # hm_yz = op2cyl(hm_yz_op)

    # hm_yz[0, 0] = gp[0, 0]  # ponerlo en el mismo plano que gp, plano del circulo ESTO ES ARBITRARIO!!!! CAMBIAR???

    # primer calcular gp
    gp, _ = calc_gp(a0, v_pw, radio)  # POR AHORA LO HAGO PARA EL CASO CONVEXO, TODO: CASO CONCAVO
    norm_p = gp.copy()  # gp es un punto en 3D, la normal es su proyeccion sobre el plano YZ
    norm_p[0, 0] = 0
    norm_p = norm_p/np.linalg.norm(norm_p, axis=1)  # normal en gp
    tange_p = np.array([0, norm_p[0, 2], -norm_p[0, 1]])  # vector tangente en gp
    tan_alfa = np.vdot(v_pw, tange_p) / np.vdot(v_pw, norm_p)
    hm_yz = (radio/2) * (norm_p + tan_alfa*tange_p)

    return hm_yz, gp


def calc_cyl_plane_wave_reflected_vector(a0, hm_yz, e, v_pw, radio):
    """

    Args:
        hm_yz: (3, ) o (1, 3)  ?
        e: (N, 3), array con las coordenadas de los elementos
        v_pw: (3, )
        radio:

    Returns:

    """

    a0 = np.array(a0).reshape(1, 3)
    hm_yz = np.array(hm_yz).reshape(1, 3)
    v_pw = np.array(v_pw).reshape(1, 3)
    v_pw = v_pw / np.linalg.norm(v_pw) # normalizar v_pw

    # primero calculamo g_yz, proyeccion de g. Para ello se calcula la interseccion de la recta
    # entre hm_yz y e_yz (proyeccion de e)

    v = hm_yz[0, 1:] - e[:, 1:]  # vector 2d desde hm_yz hasta e_yz
    g_yz_2d, _ = utils.circle_line_intersec(radio, e[:, 1:], v)  # POR AHORA LO HAGO PARA EL CASO CONVEXO, TODO: CASO CONCAVO
    # g_yz_2d es (N, 2)
    g_yz = np.zeros_like(e)
    g_yz[:, 1:] = g_yz_2d

    # una vez calculada el punto g proyectado en el plano YZ, se puede calcular la normal en ese punto: norm_g
    # hay una normal para cada elemento e[i, :]

    norm_g = g_yz/(np.linalg.norm(g_yz, axis=1).reshape((-1, 1)))   # vector normal en el punto de incidencia (G)

    # calcular vector reflejado, uno por cada elemento
    n_elems = e.shape[0]
    aux = norm_g * np.repeat(v_pw, n_elems, axis=0)
    dot_nvpw = np.sum(aux, axis=1).reshape(-1, 1)
    v_pw_n = np.sum(dot_nvpw, axis=1).reshape(-1, 1) * norm_g  # componente de v_pw en direccion de la normal
    v_refl = v_pw - 2*v_pw_n

    # calcular el reflejo del centro del array (a0), para cada norm_g (para cada elemento)
    aux = (norm_g * (a0 - g_yz)).sum(axis=1).reshape(-1, 1)  # producto escalar
    a0_refl = a0 - 2*aux*norm_g

    # # para calcular G hay que intersectar con cilindro la recta con direccion v_refl que pasa por E
    # # ASUNTO CASO CONVEXO:
    # # hay que poner -v_refl por como está definida la funcion
    # # al elegir la primera solución, con -v_refl estamos elegiendo la correcta para el caso convexo
    # g, _ = u3d.cylinder_line_intersec(radio, e, -v_refl)
    #
    # # ahora usa la relacion de la reflexion
    # g_hm_yz = hm_yz - g_yz  # (N, 3)
    # g_hm_x = -dot_nvpw * np.sum(norm_g * g_hm_yz, axis=1).reshape((-1, 1))  # producto escalar de cada fila con cada fila, (N, )
    # g_hm = np.concatenate((g_hm_x, g_hm_yz[:, 1:]), axis=1)
    # hm = g + g_hm
    # return a0_refl, g, hm, v_refl, norm_g, g_yz
    return a0_refl, v_refl


def return_plane_wave_cyl_fun(coords_sa, v_pw):
    """
    coords_sa: array de coordenadas de los elementos en el sistema array, shape (n_elems, 3)
    (se genera con utils_3d.array_coordinates_list, hay que transformar la lista en array)
    v_pw: array de vectores de direccionde la onda, cada fila es un vector, shape (n_waves, 3)
    """

    assert v_pw.shape[1] == 3
    n_waves = v_pw.shape[0]
    n_elems = len(coords_sa)
    dist = np.zeros((n_waves, n_elems))
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
            hm_yz, gp = calc_hm_yz(a0, v_pw_rot[i, :], radio, 1)
            # g, _, _, _, _ = calc_cyl_plane_wave_reflected_vector(hm_yz, e_mov, v_pw_rot[i, :], radio)
            # # tiempo de ida de la onda plana, supondiendo que en t=0 pasa por el centro del array
            # dist_ida = np.dot(g - a0, v_pw_rot[i, :])
            # dist.append(np.linalg.norm(e_mov - g, axis=1) + dist_ida)
            a0_refl, v_refl = calc_cyl_plane_wave_reflected_vector(a0, hm_yz, e_mov, v_pw_rot[i, :], radio)
            dist.append(np.sum((e_mov - a0_refl)*v_refl, axis=1))

        return np.concatenate(dist)  # order C, quedan primero todos los elementos para la primera onda, luego para
        # la segunda onda, etc

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
