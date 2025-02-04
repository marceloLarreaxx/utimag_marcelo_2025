"""Funciones para calcular refracción en 3D.

Definiciones (algunos nombres usados varias veces):
    a: coordenadas de la fuente

"""

import numpy as np
import pyvista as pv


class Ray3D:
    """Clase para representar rayos que se refractan en interfaz.
    Attributes:
        a: coordenadas de la fuente, (xa, ya, za)
        e: coordenadas del punto de entrada (xe, ye, ze)
        f: coordenadas del  foco, (xf, yf, zf)
        ne: vector unitario normal a la interfaz en el punto de entrada. Apunta
            hacia el segundo medio, (dz_dx, dz_dy, -1)
        tof: tiempo de vuelo total
        t_ae: tiempo de vuelo hasta el punto de entrada
    """

    def __init__(self, a, e, f, ne, tof, t_ae):
        self.a = np.array(a)
        self.e = np.array(e)
        self.f = np.array(f)
        self.ne = np.array(ne)
        self.tof = tof
        self.t_ae = t_ae

    @classmethod
    def compute(cls, a, f, e_xy_0, c1, c2, ifaz_fun, method, gamma=10, max_iter=1000,
                dtof_min=10 ** (-6), epsi=1/100):

        q = fermat_3d(a, f, e_xy_0, c1, c2, ifaz_fun, method, gamma=gamma, max_iter=max_iter,
                  dtof_min=dtof_min, epsi=epsi)
        tof = q[1]
        e_z, grad, _ = ifaz_fun(q[0][0], q[0][1])
        ne = np.append(grad, -1)
        t_ae = np.linalg.norm(a - q[0])/c1
        return cls(a, q[0], f, ne, tof, t_ae)

    def incident_unit_vector(self):
        return (self.e - self.a) / np.linalg.norm(self.e - self.a)

    def refracted_unit_vector(self):
        return (self.f - self.e) / np.linalg.norm(self.f - self.e)

    def draw(self, plo, color='green', line_width=2, incident=True):
        inc_ray_pv = pv.Line(self.a, self.e)
        refrac_ray_pv = pv.Line(self.e, self.f)
        if incident:
            plo.add_mesh(inc_ray_pv, color=color, line_width=line_width, show_scalar_bar=False)
        actor = plo.add_mesh(refrac_ray_pv, color=color, line_width=line_width, show_scalar_bar=False)
        return actor

    def draw_back_extension(self, plo, color='b', line_width=2, length=30):
        back_point = self.e - length*self.refracted_unit_vector()
        plo.add_mesh(pv.Line(self.e, back_point), color=color, line_width=line_width, show_scalar_bar=False)
        return back_point


def sl_depth2xyz(e_sl, vec_sl, r):
    """Calcula las coordenadas (x, y, z) de un foco sobre una scan line (SL).

    Args:
        e_sl: punto de entrada de la SL
        vec_sl: vector unitario en la dirección de SL (si no es unitario igual la funcion normaliza)
        r: profundidad del foco (distancia a e_sl)

    Returns:
        (x,y,z): coordenadas del foco
        """
    e_sl = np.array(e_sl)
    vec_sl = np.array(vec_sl)
    # normalizar
    vec_sl = vec_sl/np.linalg.norm(vec_sl)
    return e_sl + r * vec_sl


def return_3d_ifaz_fun(surf_params, surf_type, eje):
    """Devuelve una función z, grad_z = f(x, y) que describe la interfaz/superficie.

    Args:
        surf_params: parámetros de la superficie. Si es un cilindro: (x_centro, z_centro, radio)
        surf_type (string): 'cylinder', 'plane', ....todo

    Returns:
        fun: (x,y) ----> z, [dz_dx, dz_dy]
    """

    if surf_type == 'cylinder':
        # c: (xc, zc, radio)
        c = surf_params
        # cilindro con eje paralelo a eje y
        if eje=='y':
            def fun(x, y):
                z = c[1] + np.sqrt(c[2] ** 2 - (x - c[0]) ** 2)
                dz_dx = (c[0] - x) / (z - c[1])
                dz_dy = 0
                grad = np.array([dz_dx, dz_dy])  # vector gradiente
                d2z_dx = -(dz_dx + 1)/(z - c[1]) # derivada segunda respecto a x
                # las otras derivadas segundas son 0, porque z no depende de y
                hess = np.array([[d2z_dx, 0], [0, 0]])  # hessiano

                # x = x.reshape(-1, 1) # una columna
                # z = c[1] + np.sqrt(c[2] ** 2 - (x - c[0]) ** 2)
                # dz_dx = (c[0] - x) / (z - c[1])
                # dz_dy = np.zeros_like(dz_dx)
                # grad = np.concatnate((dz_dx, dz_dy), axis=1) # vector gradiente
                # d2z_dx = -(dz_dx + 1)/(z - c[1]) # derivada segunda respecto a x
                # # las otras derivadas segundas son 0, porque z no depende de y
                # hess = np.array([[d2z_dx, 0], [0, 0]])  # hessiano
                return z, grad, hess

        else: # eje=x
            def fun(x, y):
                z = c[1] + np.sqrt(c[2] ** 2 - (y - c[0]) ** 2)
                dz_dy = (c[0] - y) / (z - c[1])
                dz_dx = 0
                grad = np.array([dz_dx, dz_dy])  # vector gradiente
                d2z_dy = -(dz_dy + 1) / (z - c[1])  # derivada segunda respecto a x
                # las otras derivadas segundas son 0, porque z no depende de y
                hess = np.array([[0, 0], [0, d2z_dy]])  # hessiano

                # x = x.reshape(-1, 1) # una columna
                # z = c[1] + np.sqrt(c[2] ** 2 - (x - c[0]) ** 2)
                # dz_dx = (c[0] - x) / (z - c[1])
                # dz_dy = np.zeros_like(dz_dx)
                # grad = np.concatnate((dz_dx, dz_dy), axis=1) # vector gradiente
                # d2z_dx = -(dz_dx + 1)/(z - c[1]) # derivada segunda respecto a x
                # # las otras derivadas segundas son 0, porque z no depende de y
                # hess = np.array([[d2z_dx, 0], [0, 0]])  # hessiano
                return z, grad, hess

    elif surf_type == 'plane':
        p = surf_params

        def fun(x, y):
            z = p[0] * x + p[1] * y + p[2]
            grad = np.array([p[0], p[1]])
            hess = np.array([[0, 0], [0, 0]])  # al pedo, por completitud nomás
            return z, grad, hess

    elif surf_type == 'poly2':
        p = surf_params

        def fun(x, y):
            # TODO: poner orden de los coeficientes razonable, compatible con caso plano
            z = p[0] * x**2 + p[1] * y**2 + p[2]*x*y + p[3]*x + p[4]*y + p[5]
            dz_dx = 2*p[0]*x + p[2]*y + p[3]
            dz_dy = 2 * p[1] * y + p[2] * x + p[4]
            grad = np.array([dz_dx, dz_dy])
            hess = np.array([[2*p[0], p[2]], [p[2], 2*p[1]]])
            return z, grad, hess

    elif surf_type == 'cone':
        # c: (xc, zc, radio_base, largo)
        c = surf_params

        def fun(x, y):
            k = (c[2] / c[3])
            r = k * y
            sq = np.sqrt(r ** 2 - (x - c[0]) ** 2)
            dz_dx = (c[0] - x) / sq
            dz_dy = r * k / sq  # vale para el semieje "y" positivo, y z > zc
            grad = np.array([dz_dx, dz_dy])  # vector gradiente
            # chequear estas derivadas segundas, sospecho algún error todo
            d2z_dx = -(dz_dx**2 + 1)/sq
            d2z_dy = (2*k**2 - dz_dy**2) / sq
            dz_dydx = -dz_dx * dz_dy / sq
            hess = np.array([[d2z_dx, dz_dydx], [dz_dydx, d2z_dy]])
            return z, grad, hess

    else:
        raise Exception('surf_type not valid')

    return fun


def return_tof_3d_fun(a, f, c1, c2, ifaz_fun):
    """Devuelve una función t, grad_t = f(x, y), que calcula el tiempo de vuelo y su gradiente.

    Args:
        a (Union[tuple, list, np.array]): vector de coordenadas de la fuente
        f (Union[tuple, list, np.array]): vector de coordenadas del punto focal
        c1, c2 (floats): velocidades de propagación
        ifaz_fun (function): función generada por "return_ifaz_fun"

    Returns:
        fun: (x, y) -------> tof, [dtof_dx, dtof_y] (tof: time of flight)
    """

    # transformar a y f a np arrays
    a, f = np.array(a), np.array(f)

    def fun(x, y):
        z, grad, hess = ifaz_fun(x, y)
        e = np.array([x, y, z])
        d1 = np.linalg.norm(a - e)
        d2 = np.linalg.norm(f - e)
        t = d1/c1 + d2/c2

        # derivadas primeras
        de_dx = np.array([1, 0, grad[0]])
        de_dy = np.array([0, 1, grad[1]])
        dd1_dx = np.dot(e - a, de_dx) / d1
        dd1_dy = np.dot(e - a, de_dy) / d1
        dd2_dx = np.dot(e - f, de_dx) / d2
        dd2_dy = np.dot(e - f, de_dy) / d2
        dt_dx = dd1_dx/c1 + dd2_dx/c2
        dt_dy = dd1_dy/c1 + dd2_dy/c2
        grad_t = np.array([dt_dx, dt_dy])

        # derivadas segundas de punto de entrada e
        d2e_dx = np.array([0, 0, hess[0, 0]])
        d2e_dy = np.array([0, 0, hess[1, 1]])
        d2e_dydx = np.array([0, 0, hess[0, 1]])

        # hessianos de d1 y d2
        hess_d1 = np.zeros((2, 2))
        hess_d2 = np.zeros((2, 2))

        hess_d1[0, 0] = (np.dot(de_dx, de_dx) + np.dot(e - a, d2e_dx) - dd1_dx**2)/d1
        hess_d1[1, 1] = (np.dot(de_dy, de_dy) + np.dot(e - a, d2e_dy) - dd1_dy**2)/d1
        hess_d1[0, 1] = (np.dot(de_dy, de_dx) + np.dot(e - a, d2e_dydx) - dd1_dx*dd1_dy)/d1
        hess_d1[1, 0] = hess_d1[0, 1]  # matriz simétrica
        hess_d2[0, 0] = (np.dot(de_dx, de_dx) + np.dot(e - f, d2e_dx) - dd2_dx**2)/d2
        hess_d2[1, 1] = (np.dot(de_dy, de_dy) + np.dot(e - f, d2e_dy) - dd2_dy**2)/d2
        hess_d2[0, 1] = (np.dot(de_dy, de_dx) + np.dot(e - f, d2e_dydx) - dd2_dx*dd2_dy)/d2
        hess_d2[1, 0] = hess_d2[0, 1]  # matriz simétrica

        hess_t = hess_d1/c1 + hess_d2/c2
        return t, grad_t, hess_t

    return fun


def fermat_3d(a, f, e_xy_0, c1, c2, ifaz_fun, method, gamma=10, max_iter=1000, dtof_min=10 ** (-6), epsi=1/100):
    """ Calcula el tiempo de vuelo (tof) entre a y f, minimizando por Gradient Descent.

    Args:
        a (Union[tuple, list, np.array]): vector de coordenadas de la fuente
        f (Union[tuple, list, np.array]): vector de coordenadas del punto focal
        e_xy_0 (Union[tuple, list, np.array]): coordenadas x e y inciales para comenzar la iteración
        c1, c2 (floats): velocidades de propagación
        ifaz_fun (function): función generada por "return_ifaz_fun"
        method: 'gradient_descent', 'newton', 'nwgd' (mezcla de Newton con Gradient Descent)
        gamma: rate de descenso
        max_iter: número máximo de iteraciones
        dtof_min: tolerancia (para el módulo del vector gradiente, que debe converger a 0)
        epsi: umbral para el determinante del hessiano (se usa en 'nwgd')

    Returns:
        (x, y): punto de entrada
        t: tof
        log: listas con los valores de t y gradiente a lo largo de la iteración
    """

    # transformar a y f a np arrays
    a, f = np.array(a), np.array(f)
    log = {'e': [],
           'grad_t': [],
           'det_hess': [],
           'convergence': None}

    tof_fun = return_tof_3d_fun(a, f, c1, c2, ifaz_fun)
    x, y = e_xy_0
    t, grad_t, hess_t = tof_fun(x, y)
    log['e'].append([x, y])
    log['grad_t'].append(grad_t)
    q = np.linalg.norm(grad_t)
    newton_fail = False
    i = 0

    if method == 'gradient_descent':

        while (q > dtof_min) and (i < max_iter):
            x = x - grad_t[0] * gamma
            y = y - grad_t[1] * gamma
            t, grad_t, _ = tof_fun(x, y)
            q = np.linalg.norm(grad_t)
            i += 1
            log['e'].append([x, y])
            log['grad_t'].append(grad_t)

    if method == 'newton':

        e_xy = np.array([x, y])
        det_hess = hess_t[0, 0] * hess_t[1, 1] - hess_t[0, 1] * hess_t[1, 0]
        det_min = 10**(-6)

        while (q > dtof_min) and (i < max_iter):
            if det_hess <= det_min:
                newton_fail = True
                print('determinante del hessiano es nulo')
                print('grad_t: ', grad_t)
                print('hess_t: ', hess_t)
                break
            else:
                inv_hess = np.array([[hess_t[1, 1], -hess_t[0, 1]], [-hess_t[1, 0], hess_t[0, 0]]]) / det_hess
                e_xy = e_xy - np.matmul(inv_hess, grad_t)  # newton step

            t, grad_t, hess_t = tof_fun(*e_xy)
            det_hess = hess_t[0, 0]*hess_t[1, 1] - hess_t[0, 1]*hess_t[1, 0]
            q = np.linalg.norm(grad_t)
            i += 1
            log['e'].append(e_xy)
            log['grad_t'].append(grad_t)
            log['det_hess'].append(det_hess)

        x, y = e_xy

    if method == 'nwgd':

        e_xy = np.array([x, y])
        det_hess = hess_t[0, 0] * hess_t[1, 1] - hess_t[0, 1] * hess_t[1, 0]

        while (q > dtof_min) and (i < max_iter):
            if det_hess <= epsi:
                # e_xy[0] = e_xy[0] - grad_t[0] * gamma
                # e_xy[1] = e_xy[1] - grad_t[1] * gamma
                print('gd')
                e_xy = e_xy - gamma*np.matmul(np.eye(2), grad_t)
            else:
                # inverso del hessiano (Newton)
                print('nw')
                inv_hess = np.array([[hess_t[1, 1], -hess_t[0, 1]], [-hess_t[1, 0], hess_t[0, 0]]]) / det_hess
                e_xy = e_xy - np.matmul(inv_hess, grad_t)  # newton step

            t, grad_t, hess_t = tof_fun(*e_xy)
            det_hess = hess_t[0, 0] * hess_t[1, 1] - hess_t[0, 1] * hess_t[1, 0]
            q = np.linalg.norm(grad_t)
            i += 1
            log['e'].append(e_xy)
            log['grad_t'].append(grad_t)
            log['det_hess'].append(det_hess)

        x, y = e_xy

    e = np.array([x, y, ifaz_fun(x, y)[0]])  # punto de entrada como array (x, y, z)
    log['convergence'] = False if (i == max_iter or newton_fail) else True

    return e, t, log


def return_snell_3d_fun(a, e_xy, theta_refrac, c1, c2, ifaz_fun):
    """Devuelve función que calcula la expresión de Snell en función del punto de entrada. A esta
    función hay que buscarle la raiz, para encontrar el punto de entrada. La función se define en
    términos de la fuente y  de la dirección de refracción. """

    ze, grad_ifaz = ifaz_fun(*e_xy)
    # convertir en np arrays
    a = np.array(a)
    e = np.array([e_xy[0], e_xy[1], ze])
    ne = np.array([-grad_ifaz[0], -grad_ifaz[1], 1])

    # calcular vector unitario incidente
    norm_inc = np.linalg.norm(e - a)
    v_inc = (e - a)/norm_inc
    # calcular producto vectorial con el vector normal ne
    # normalizar primer ne
    ne = ne/np.linalg.norm(ne)
    oe = np.cross(v_inc, ne)
    sin_ang_inc = np.linalg.norm(oe)  # seno del angulo incidente (TODO: chequear el tema del signo!!!)

    # expresiómn de Snell
    sin_ang_refrac = np.sin(theta_refrac)
    s = sin_ang_refrac - (c2 / c1) * sin_ang_inc
    return s


def snell_3d_root():
    pass


def refrac_basis_vectors(v, ne):
    """ Devuele un terna de vectores unitarios (ne, oe, pe), tales que uno es la
    normal al plano de incidencia/refracción, y otro está contenido en ese plano y
    es ortogonal a la normal ne. También devuelve el seno del ángulo entre v y ne

        v: vector incidente o refractado.
        oe: vector ortogonal al plano de incidencia/refracción
        pe: vector ortogonal a v y a oe, contenido en el plano de incidencia/refracción

    Esta funcion normaliza todos los vectores
    """

    v = v / np.linalg.norm(v)
    ne = ne / np.linalg.norm(ne)

    oe = np.cross(v, ne)
    # que pasa si la incidencia es normal ---> oe da vector 0
    if np.isclose(np.abs(oe), 0).all():
        sin_ang = 0
        pe = None
        print('incidencia normal')
    else:
        sin_ang = np.linalg.norm(oe)
        oe = oe / sin_ang
        pe = np.cross(ne, oe)

    # pe = v - np.dot(v, ne)
    # pe = pe / np.linalg(pe)
    # oe = np.cross(v, ne)

    return ne, oe, pe, sin_ang


# def incident2refrac(v_inc, ne, c1, c2):
#     """ Calcula rayo refractado a partir del rayo incidente y la normal en el
#     punto de entrada
#
#     ALERTA: NO HACE FALTA USAR EL PRODUCTO VECTORIAL, CON PRODUCTOS ESCALARES DE PUEDEN
#     CALCULAR LOS VECTORES !!!! O SEA QUE LA FUNCION refrac_basis_vectors es al pedo!! TODO"""
#
#     # convertir en np arrays por si son listas o tuples
#     v_inc, ne = np.array(v_inc), np.array(ne)
#     assert np.isclose(np.linalg.norm(ne), 1)
#     # calcular vector unitario incidente
#     v_inc = v_inc / np.linalg.norm(v_inc)
#     ne, oe, pe, sin_ang_inc = refrac_basis_vectors(v_inc, ne)
#     sin_ang_refrac = (c2/c1) * sin_ang_inc  # Snell
#     cos_ang_refrac = np.cos(np.arcsin(sin_ang_refrac))
#     v_refrac = sin_ang_refrac * pe - cos_ang_refrac * ne
#
#     return v_refrac, sin_ang_refrac, sin_ang_inc


def incident2refrac(v_inc, ne, c1, c2):
    """ Calcula rayo refractado a partir del rayo incidente y la normal en el
    punto de entrada"""

    # convertir en np arrays por si son listas o tuples
    v_inc, ne = np.array(v_inc), np.array(ne)
    ne = ne / np.linalg.norm(ne)  # normalizar por is las dudas
    # calcular vector unitario incidente
    v_inc = v_inc / np.linalg.norm(v_inc)
    cos_ang_inc = -np.dot(v_inc, ne)
    if np.isclose(cos_ang_inc, 1): # incidencia normal
        v_refrac = v_inc
        sin_ang_inc = 0
        sin_ang_refrac = 0
    else:
        pe = v_inc + cos_ang_inc*ne
        pe = pe / np.linalg.norm(pe) # vector en plano de incidencia, ortogonal a ne
        sin_ang_inc = np.sqrt(1 - cos_ang_inc**2)
        sin_ang_refrac = (c2/c1) * sin_ang_inc  # Snell
        if sin_ang_refrac > 1: # pasado el angulo crítico
            # todo, CHAPUZA: esto es una chapuza para evitar "nan". !!!!!!!!!!!!!!! revisar
            sin_ang_refrac =1

        cos_ang_refrac = np.cos(np.arcsin(sin_ang_refrac))
        v_refrac = sin_ang_refrac * pe - cos_ang_refrac * ne

    return v_refrac, sin_ang_refrac, sin_ang_inc


def incident2refrac_muchos(v_inc, ne, c1, c2):
    """ Calcula rayo refractado a partir del rayo incidente y la normal en el
    punto de entrada"""

    # convertir en np arrays por si son listas o tuples
    v_inc, ne = np.array(v_inc), np.array(ne)
    ne = ne / np.linalg.norm(ne, axis=1).reshape(-1, 1)  # normalizar por is las dudas
    # calcular vector unitario incidente
    v_inc = v_inc / np.linalg.norm(v_inc, axis=1).reshape(-1, 1)
    v_refrac = np.zeros_like(v_inc)
    cos_ang_inc = -(v_inc*ne).sum(axis=1)
    cos_ang_inc = cos_ang_inc.reshape(-1, 1)
    sin_ang_inc = np.zeros_like(cos_ang_inc)
    sin_ang_refrac = np.zeros_like(cos_ang_inc)

    inc_normal = np.isclose(cos_ang_inc, 1).flatten() # incidencia normal
    v_refrac[inc_normal, :] = v_inc[inc_normal, :]
    sin_ang_inc[inc_normal] = 0
    sin_ang_refrac[inc_normal] = 0

    pe = v_inc + cos_ang_inc*ne
    pe = pe / (np.linalg.norm(pe, axis=1) + 1e-6).reshape(-1, 1)  # vector en plano de incidencia, ortogonal a ne. SUMO "epsilon=1e-6" para
    # evitar division por 0 !!!
    sin_ang_inc = np.sqrt(1 - cos_ang_inc**2) # esto dará nans
    sin_ang_refrac = (c2/c1) * sin_ang_inc  # Snell

    ang_crit = sin_ang_refrac > 1
    # todo, CHAPUZA: esto es una chapuza para evitar "nan". !!!!!!!!!!!!!!! revisar
    sin_ang_refrac[ang_crit] = 1

    cos_ang_refrac = np.cos(np.arcsin(sin_ang_refrac))
    v_refrac = sin_ang_refrac * pe - cos_ang_refrac * ne

    return v_refrac, sin_ang_refrac, sin_ang_inc


def refrac2incident(v_refrac, ne, c1, c2):
    """Calcula el rayo incidente en función del refractado y la normal en el
    punto de entrada

    Args:
        v_refrac: vector refractado, apuntando hacia el segundo medio
        ne: normal a la interfaz, apuntando hacia el primer medio
        c1:
        c2:

    Returns:
        v_inc: vector incidente
        sin_ang_refrac
        sin_ang_inc
    """

    # transformar a np.array y normalizar
    v_refrac = np.array(v_refrac)
    v_refrac = v_refrac / np.linalg.norm(v_refrac)

    ne, oe, pe, sin_ang_refrac = refrac_basis_vectors(v_refrac, ne)
    sin_ang_inc = (c1 / c2) * sin_ang_refrac  # Snell
    cos_ang_inc = np.cos(np.arcsin(sin_ang_inc))
    v_inc = sin_ang_inc * pe - cos_ang_inc * ne

    return v_inc, sin_ang_refrac, sin_ang_inc


def incident2reflected(v_inc, ne):
    """v_inc: rayo incidente, (no hace falta que este normalizado).
    Tiene que ser "contrario" a la normal
    ne: normal a la superficie"""

    # transformar en array por si son listas o tuplas
    assert v_inc.shape[1] == 3
    assert ne.shape[1] == 3
    v_inc = np.array(v_inc)
    ne = np.array(ne)
    aux = np.sum(v_inc*ne, axis=1).reshape((-1, 1))
    v_refl = v_inc - 2 * aux * ne
    # implemetacion ineficiente y mala con producto vectorial al pedo ------------
    # # chequear incidencia normal:
    # oe = np.cross(v_inc, ne)
    # # que pasa si la incidencia es normal ---> oe da vector 0
    # if np.isclose(np.abs(oe), 0).all():
    #     v_refl = -v_inc
    # else:
    #     ne, _, pe, _ = refrac_basis_vectors(v_inc, ne)
    #     # ahora ne esta normalizado
    #     # v_refl = np.dot(v_inc, ne)*ne - np.dot(v_inc, pe)*pe
    #     v_refl = np.dot(v_inc, pe) * pe - np.dot(v_inc, ne) * ne

    return v_refl/np.linalg.norm(v_refl).reshape((-1, 1))


def ray_plane_intersec(p, v, a, n):
    """p: punto por el que pasa el rayo
       v: vector unitario en direccion del rayo
       a: punto en el plano
       n: normal al plano"""

    # transformar en array por si son listas o tuplas
    p = np.array(p)
    v = np.array(v)
    a = np.array(a)
    n = np.array(n)
    q = np.dot(a - p, n)
    w = q/np.dot(v, n)
    return p + w*v


def cylinder_reflect(a, x, y, r, convex=True):
    """Un sale sale de a y se refleja en cilindro en punto (x, y, z)
    a: tiene que estar arriba de la superficie"""

    if convex:
        assert a[2] > r
        z = np.sqrt(r ** 2 - x ** 2)
        n = np.array((x, 0, z))  # normal hacia afuera/arriba
    else:
        assert a[2] > -r
        z = -np.sqrt(r ** 2 - x ** 2)
        n = -np.array((x, 0, z))  # normal hacia adentro/arriba

    p = np.array((x, y, z))
    v_inc = p - np.array(a)
    v_refl = incident2reflected(v_inc, n)
    return v_refl, p


def project_ifaz_normal_onto_inspection_plane(ne, n_insp):
    """ Devuelve 2 de vectores unitarios ortogonales contenidos en el plano de inspección, cuya normal es
    n_insp. El vector ne_proj tiene la dirección de la proyección de la normal sobre el plano.

    n_projected está normalizado
    """

    ne, n_insp = np.array(ne), np.array(n_insp)
    ne = ne / np.linalg.norm(ne)
    n_insp = n_insp / np.linalg.norm(n_insp)  # normalizar

    # proyeción de la normal sobre el plano de inspeccion
    ne_proj = ne - np.dot(ne, n_insp) * n_insp
    ne_proj_unit = ne_proj / np.linalg.norm(ne_proj)  # normalizar
    # vector unitario contenido en el plano de inspeccion y ortogonal a ne_prjected
    t_insp = np.cross(n_insp, ne_proj_unit)
    t_insp = t_insp / np.linalg.norm(t_insp)  # normalizar

    return ne_proj, t_insp


def return_scan_line_fun(ang, n_insp, a0, c1, c2, ifaz_fun, aprox_2d=False):
    """Genera un función para calcular la distancia entre el punto del plano del array a donde
    cruza un rayo en el plano de inspección y el centro del array. Minimizar esta función
    para hallar el punto de entrada óptimo.
    El plano de inspección debe contener al eje z, de modo que n_insp tiene que ser tipo [nx, nz, 0]
    La variable "s" parametriza la curva intersección, entre el plano y la interfaz"""

    assert n_insp[2] == 0

    if aprox_2d:
        def fun(s):
            x = s * n_insp[1]
            y = -s * n_insp[0]
            z, grad_z = ifaz_fun(x, y)
            e = np.array([x, y, z])
            ne = np.array([-grad_z[0], -grad_z[1], 1])
            ne_proj, t_insp = project_ifaz_normal_onto_inspection_plane(ne, n_insp)
            ne_proj_unit = ne_proj / np.linalg.norm(ne_proj)  # normalizar
            v_refrac = np.sin(ang) * t_insp - np.cos(ang) * ne_proj_unit
            v_inc = e - a0
            v_inc = v_inc / np.linalg.norm(v_inc)
            # calcular el vector refractado como si la normal fuese la proyección sobre el plano
            v_refrac, sin_ang_refrac, sin_ang_inc = incident2refrac(v_inc, ne_proj, c1, c2)
            cost = np.abs(sin_ang_refrac - np.sin(ang))
            return cost, (e, ne, v_inc, v_refrac)

    else:

        def fun(s):
            x = s * n_insp[1]
            y = -s * n_insp[0]
            z, grad_z = ifaz_fun(x, y)
            e = np.array([x, y, z])
            ne = np.array([-grad_z[0], -grad_z[1], 1])
            ne_proj, t_insp = project_ifaz_normal_onto_inspection_plane(ne, n_insp)
            ne_proj_unit = ne_proj / np.linalg.norm(ne_proj)  # normalizar
            v_refrac = np.sin(ang) * t_insp - np.cos(ang) * ne_proj_unit
            v_inc, _, _ = refrac2incident(v_refrac, ne, c1, c2)
            b = -e[2]/v_inc[2] * v_inc + e
            dist = np.linalg.norm(a0 - b)
            return dist, (b, e, ne, v_inc, v_refrac)

    return fun


if __name__ == '__main__':

    c1, c2 = 1.48, 5.3
    a = np.array([0, 0, 0])
    e = []
    f = []
    s = []
    log = []
    inc_ray_pv = []
    refrac_ray_pv = []
    x_step = 5

    # interfaz plana
    ifaz_fun = return_3d_ifaz_fun((0.1400, 0.85, -20), surf_type='plane')
    z0, grad_ifaz0, _ = ifaz_fun(0, 0)
    ifaz0 = [0, 0, z0]
    ifaz_normal = [-grad_ifaz0[0], -grad_ifaz0[1], 1]
    ifaz_mesh = pv.Plane(ifaz0, ifaz_normal, i_size=100, j_size=100)

    for i in range(30):

        f.append(np.array([x_step * i + 0.2, 0, -50]))
        e_temp, tof, log_temp = fermat_3d(a, f[i], (0, 0), c1, c2, ifaz_fun, method='gradien_descent')
        e.append(e_temp)
        v_inc = e[i] - a
        s.append(incident2refrac(v_inc, ifaz_normal, c1, c2))
        log.append(log_temp)

        inc_ray_pv.append(pv.Line(a, e[i]))
        refrac_ray_pv.append(pv.Line(e[i], f[i]))

    # comparación con Snell
    v_refrac_fermat = np.array(f) - np.array(e)
    v_refrac_fermat = v_refrac_fermat / np.linalg.norm(v_refrac_fermat, axis=1).reshape((-1,1))
    v_refrac_snell = np.array([q[0] for q in s])
    print(np.abs(v_refrac_snell - v_refrac_fermat).max())

    # pintar
    p = pv.Plotter()
    for i in range(30):
        p.add_mesh(inc_ray_pv[i], color='red', show_scalar_bar=False)
        p.add_mesh(refrac_ray_pv[i], color='blue', show_scalar_bar=False)

    p.add_mesh(ifaz_mesh, color='white', opacity=0.5, show_scalar_bar=False)
    p.add_axes(box=True)
    p.show()
