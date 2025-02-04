import abc
import numpy as np
import pyopencl as cl
import matplotlib.pyplot as plt
from matplotlib import patches
import pyvista as pv
import imag3D.snell_fermat_3d as sf
import utils


def xyz2index(vector, roi, x_step, y_step, z_step, round_idx=True):
    ix = (vector[0] - roi[0]) / x_step
    iy = (vector[1] - roi[2]) / y_step
    iz = (roi[4] - vector[2]) / z_step
    if round_idx:
        ix, iy, iz = int(ix), int(iy), int(iz)
    return iz, ix, iy


def array_coordinates_list(nel_x, nel_y, pitch_x, pitch_y):

    x_0 = pitch_x * (nel_x - 1) / 2
    y_0 = pitch_y * (nel_y - 1) / 2
    a_xy = [(pitch_x * i - x_0, pitch_y * j - y_0) for j in range(nel_y) for i in range(nel_x)]
    a_xy = a_xy[0:nel_x * nel_y]  # recortar los repetidos
    # vectores en 3d (elementos en el plano z=0)
    a = [np.array([x, y, 0]) for x, y in a_xy]

    return a


def array_ij2element(e_ij, nel_x, nel_y, pitch_x, pitch_y):

    # indice lineal contando los elementos en la direccion del eje x
    linear_index = np.uint16(e_ij[0] + e_ij[1]*nel_x)

    # coordenadas 3D del elementos
    x_0 = pitch_x * (nel_x - 1) / 2
    y_0 = pitch_y * (nel_y - 1) / 2

    coords = np.array([pitch_x * e_ij[0] - x_0, pitch_y * e_ij[1] - y_0, 0], dtype=np.float32)

    return linear_index, coords


def index_vecindad(e_ij, matriz_vecindad):
    """
    Args:
        e_ij: índices del elemento emisor, no pueden ser del borde del Transarray
        matriz_vecindad: matriz binaria centrada en el emisor, con unos en los receptores. Ejemplos:
                    111   010
                    111   111
                    111   010

    Returns:
        rx_elems: array de indices de los elementos que se activan en elem_xy. Estan numerados
        leyendo en orden C, es decir, a lo largo del eje Y. Para indexar el array de tiempos de vuelo
        hay que transformar a indice lineal usando array_ij2element
    """

    rx_elems = list(np.nonzero(matriz_vecindad))  # indices de los elementos vecinos que se activan
    # Transformo a lista para poder modificarlos (tuple es inmutable)
    # Ej:   0 1 0   rx_elems = [indice fila, indice columna] = [[0, 1, 1, 1, 2], [1, 0, 1, 2, 1]]
    #       1 1 1
    #       0 1 0

    # sumar los indices del emisor para pasar a indices absolutos. Hay que restarle el indice del elemento
    # central de la matriz vecindad, que es (1, 1) en el ejemplo. Para que esto tenga sentido la matriz_vecindad
    # debe tener tamaño impar en ambos ejes, porque si no no existe el elemnto central
    rx_elems[0] += e_ij[0] - int(matriz_vecindad.shape[0]/2)
    rx_elems[1] += e_ij[1] - int(matriz_vecindad.shape[1]/2)
    # quiero que el resultado se una lista de pares de indices [(i1, j1), (i2, j2), (i3, j3), etc]
    # para eso uso la funcion zip
    return list(zip(rx_elems[0], rx_elems[1]))


def coord_pitch_catch_subap(tx_elems_ij, matriz_vecindad, nel_x, nel_y, pitch_x, pitch_y):
    """
    Args:
        tx_elems_ij: lista de indices de transmisores (output de index_vecindad)
        matriz_vecindad:
        pitch_x, pitch_y:
        nel_x:
        nel_y:

    Returns:
        coords: lista de 2-uplas (coords_tx, coords_rx), coords_tx con shape
        (1, 3) y coords_rx con shape (n_rx, 3), n_rx: numero de receptores
        ej: [(tx0, [rx00, rx01, rx02, ...]),
             (tx1, [rx10, rx11, rx12, ...])]

        linidx: lista de 2-uplas (linear index de tx, linear index de rx). En el mismo
        formato que coords
    """

    n_rx = np.count_nonzero(matriz_vecindad)
    coords, linidx = [], []
    for tx_ij in tx_elems_ij:
        linidx_tx, coords_tx = array_ij2element(tx_ij, nel_x, nel_y, pitch_x, pitch_y)
        rx_elems_ij = index_vecindad(tx_ij, matriz_vecindad)

        coords_rx = np.zeros((n_rx, 3))
        linidx_rx = np.zeros(n_rx, dtype=np.uint16)
        for m, rx_ij in enumerate(rx_elems_ij):
            q = array_ij2element(rx_ij, nel_x, nel_y, pitch_x, pitch_y)
            coords_rx[m, :] = q[1]
            linidx_rx[m] = q[0]

        coords.append((coords_tx, coords_rx))
        linidx.append((linidx_tx, linidx_rx))

    return coords, linidx


# def coord_vecindad(e_ij, elem_xy, matriz_vecindad):
#     """
#
#     Args:
#         e_ij: índices del elemento emisor
#         elem_xy: meshgrid
#         matriz_vecindad: matriz binaria centrada en el emisor, con unos en los receptores. Ejemplos:
#                     111   010
#                     111   111
#                     111   010
#
#     Returns:
#         a_rx: array de (n_rx, 3) donde n_rx es el numero de receptores
#         idx: array de indices lineales de los elementos que se activan en elem_xy
#     """
#
#     idx_rx = list(np.nonzero(matriz_vecindad))  # indices de los elementos vecinos que se activan
#     # Transformo a lista para poder modificarlos (tuple es inmutable)
#
#     # OJO!!: en idx_rx vienen los indices ordenados como si el array se lee por columnas
#     # por eso tengo que usar orden C en los siguiente.
#
#     # sumar los indices del emisor para pasar a indices absolutos. Hay que restarle el indice del elemento
#     # central de la matriz vecindad, que es (1, 1) en el ejemplo. Para que esto tenga sentido la matriz_vecindad
#     # debe tener tamaño impar en ambos ejes, porque si no no existe el elemnto central
#     idx_rx[0] += e_ij[0] - int(matriz_vecindad.shape[0]/2)
#     idx_rx[1] += e_ij[1] - int(matriz_vecindad.shape[1]/2)
#     # trasnformar a indice lineal
#     idx_lin_rx = np.ravel_multi_index(idx_rx, elem_xy[0].shape, order='C')
#     a_rx = np.zeros((idx_rx[0].size, 3))  # array de coordenads de receptores
#     # coordenadas xy de receptores
#     # x_vecindad = elem_xy[0][idx_rx[0], idx_rx[1]].reshape((-1, ), order='F')
#     # y_vecindad = elem_xy[1][idx_rx[0], idx_rx[1]].reshape((-1, ), order='F')
#     a_rx[:, 0] = elem_xy[0].flatten(order='C')[idx_lin_rx]
#     a_rx[:, 1] = elem_xy[1].flatten(order='C')[idx_lin_rx]
#     return a_rx, idx_lin_rx


def overlap_sum_xdir(img1, img2, delta_x, x_step):
    n = int(delta_x/x_step)
    m = img1.shape[1] - n
    ovlap = (img1[:, n:, :] + img2[:, 0:m, :])/2
    return np.concatenate((img1[:, 0:n, :], ovlap, img2[:, m:, :]), axis=1)


def array_top_view(ax, p_dict, roi, x_lines, y_lines, element):

    array_coords = array_coordinates_list(p_dict['nel_x'], p_dict['nel_y'], p_dict['pitch_x'], p_dict['pitch_y'])
    x_a = [a[0] for a in array_coords]
    y_a = [a[1] for a in array_coords]
    ax.plot(x_a, y_a, 'ko')

    # rect = patches.Rectangle((roi[0], roi[2]), roi[1] - roi[0], roi[3] - roi[2], linewidth=1, fill=0)
    # ax.add_artist(rect)
    ax.set_aspect('equal')

    for x in x_lines:
        ax.vlines(x, roi[2], roi[3], 'gray')
    for y in y_lines:
        ax.hlines(y, roi[0], roi[1], 'gray')
    for e in element:
        ax.plot(array_coords[e][0], array_coords[e][1], 'o')

    return ax


def plot_ray_pv(plo, ray, surfun):
    """ray: 3-upla de las que devuelve la función snell_fermat.fermat_3d"""


class TransducerArray:

    def __init__(self, nel_x, nel_y, pitch, wx, wy, a0, rot):
        """
        Args:
            nel_x:
            nel_y:
            pitch:
            wx:
            wy:
            a0: vector que indica el centro del array
            rot: Rotation de scipy (scipy.spatial.transform.Rotation)
        """
        self.n_elements = nel_x*nel_y
        self.nel_x = nel_x
        self.nel_y = nel_y
        self.pitch = pitch
        self.wx = wx
        self.wy = wy
        self.a0 = np.array(a0)
        self.rot = rot

    def return_elem_coords(self, mov=False):
        # centros de los elementos, array centrado en 0, sobre plano XY
        elem_coord = array_coordinates_list(self.nel_x, self.nel_y, self.pitch, self.pitch)
        if mov:
            # rotar y trasladar
            return [self.a0 + self.rot.apply(q) for q in elem_coord]
        else:
            return elem_coord

    def ij2element(self, i, j, mov=False):
        idx, elem = array_ij2element((i, j), self.nel_x, self.nel_y, self.pitch, self.pitch)
        if mov:
            return idx, self.rot.apply(elem) + self.a0
        else:
            return idx, elem

    def pv_plot(self, plo, elem=[], elem_color='black'):
        """
        Dibuja el array en 3D, rotado y trasladado
        elem: lista [[3,5], [2,6], etc], estos aparecen "iluminados"
        """

        # define rectangulo en plano XY centrado en 0, que represent aun elemento del array
        elem0_rectangle = [np.array([-self.wx / 2, self.wy / 2, 0]),
                           np.array([self.wx / 2, self.wy / 2, 0]),
                           np.array([self.wx / 2, -self.wy / 2, 0]),
                           np.array([-self.wx / 2, -self.wy / 2, 0])]

        # centros de los elementos, en sistema array
        elem_center = self.return_elem_coords()
        elem_pv = []
        for i in range(self.nel_x):
            elem_pv.append([])
            for j in range(self.nel_y):
                idx, _ = self.ij2element(i, j)
                vertex = [self.rot.apply(p + elem_center[idx]) + self.a0 for p in elem0_rectangle]
                elem_pv[i].append(pv.Rectangle(vertex))
                elem_color = 'red' if [i, j] in elem else elem_color
                plo.add_mesh(elem_pv[i][j], color=elem_color)

        return elem_pv

    def plot_ray(self, plo, elem, f, e_xy_0, c1, c2, surf_fun, method):
        """ elem: (i, j)
        """
        idx, a = self.ij2element(*elem)
        ray = sf.Ray3D.compute(a, f, e_xy_0, c1, c2, surf_fun, method)
        ray.draw(plo)
        return ray

    def compute_rays(self, f, e_xy_0, c1, c2, surf_fun, method):
        """Calcular los rayos desde todos los elementos hacia foco f"""
        elem_coords = self.return_elem_coords()
        rays = [sf.Ray3D.compute(a, f, e_xy_0, c1, c2, surf_fun, method) for a in elem_coords]
        return rays

    def plot_array_face_2d(self, elem=[]):
        """
        elem: lista [[3,5], [2,6], etc], estos aparecen "iluminados"
        Dibujar el array en el plano del array, con elementos seleccionados para estar pintados
        de verde, activos
        """

        # centros de los elementos, en sistema array
        elem_center = self.return_elem_coords()
        d_x = self.nel_x * self.pitch / 2 + self.wx
        d_y = self.nel_y * self.pitch / 2 + self.wy
        # hacer que la figura tenga las mimsa proporciones que el array,
        # el 25 es porque está en pulgadas la cosa
        fig, ax = plt.subplots(figsize=(2, 8))
        # for i in range(self.n_elements):
        for i in range(self.nel_x):
            for j in range(self.nel_y):
                idx, _ = self.ij2element(i, j)
                vex0 = (-self.wx/2 + elem_center[idx][0], -self.wy/2 + elem_center[idx][1])

                # definir color
                if [i, j] in elem:
                    elem_color = 'green'
                else:
                    elem_color = 'tan'

                ax.add_patch(patches.Rectangle(vex0, self.wx, self.wy, color=elem_color))
                ax.set_aspect('equal')

                ax.set_xlim(-d_x, d_x)
                ax.set_ylim(-d_y, d_y)
                plt.axis('off')  # para no ver los ejes
        return ax


def two_sphere_intersec(a1, a2, r1, r2):
    """a1, a2: centros de las esferas
        r1, r2: radios
    """
    a1 = np.array(a1)
    a2 = np.array(a2)
    d = np.linalg.norm(a1 - a2)
    q = (r1**2 - r2**2 + d**2)/(2*d)
    circ_rad = np.sqrt(r1**2 - q**2)  # radio del circulo interseccion
    vec12 = a2 - a1
    vec12 = vec12/np.linalg.norm(vec12)
    circ_center = a1 + q*vec12
    return circ_center, circ_rad, vec12


def sphere_circle_intersec(rs, rc, x, y):
    """
    Supongamos una esfera centrada en el origen y un circulo en el plano YZ centrado en (xc, yc, 0)
    rs: radio de la esfera
    rc, xc, yc: radio y centro del circulo
    """
    yq = (rs**2 - rc**2 - x**2 + y**2)/(2*y)
    zq = np.sqrt(rs**2 - x**2 - yq**2)
    return zq


def cylinder_xdir_fun(x, y, radio, concave):
    z_temp = np.sqrt(radio**2 - y**2)
    z = -z_temp if concave else z_temp
    n = -np.array([0, y, z])
    n = n / np.linalg.norm(n)
    return z, n


def cylinder_line_intersec(radio, p0, v):
    # cilindro con eje X

    """
    Cilindro en eje X, linea desde p0 con direccion v
    Args:
        radio:
        p0: (N, 3), array de puntos 3D, cada fila un punto
        v: (N, 3), array de vectores de direccion de las rectas

    Returns:
        q1, q2, las dos soluciones
    """

    p0, v = np.array(p0).reshape(-1, 3), np.array(v).reshape(-1, 3)

    # proyectar en plano YZ
    p0_yz, v_yz = p0[:, 1:], v[:, 1:]

    a = np.sum(v_yz**2, axis=1)  # coeficiente del termino cuadratico (array de (N, ))
    b = 2 * np.sum(p0_yz * v_yz, axis=1)  # coeficiente del termino grado 1 (array de (N, ))
    c = np.sum(p0_yz**2, axis=1) - radio**2  # coeficiente grado 0 (array de (N, ))

    discr = (b**2 - 4*a*c)  # (N, )
    if np.any(discr < 0):  # chequear que el discriminante sea positivo
        print('no hay interseccion (discriminante negativo)')

    discr_raiz = np.sqrt(discr)
    s1 = (-b - discr_raiz) / (2*a)  # (N, 1)
    s2 = (-b + discr_raiz) / (2*a)  # (N, 1)
    s1 = s1.reshape((-1, 1))
    s2 = s2.reshape((-1, 1))
    q1 = p0 + s1*v
    q2 = p0 + s2*v
    return q1, q2


def sphere_line_intersec(radio, p0, v):

    """
    Esfera centrada en origen. Linea desde p0 con direccion v
    Args:
        radio:
        p0: (N, 3), array de puntos 3D, cada fila un punto
        v: (N, 3), array de vectores de direccion de las rectas

    Returns:
        q1, q2, las dos soluciones
    """

    p0, v = np.array(p0).reshape(-1, 3), np.array(v).reshape(-1, 3)

    a = np.sum(v**2, axis=1)  # coeficiente del termino cuadratico (array de (N, ))
    b = 2 * np.sum(p0 * v, axis=1)  # coeficiente del termino grado 1 (array de (N, ))
    c = np.sum(p0**2, axis=1) - radio**2  # coeficiente grado 0 (array de (N, ))

    discr = (b**2 - 4*a*c)  # (N, )
    if np.any(discr < 0):  # chequear que el discriminante sea positivo
        print('no hay interseccion (discriminante negativo)')

    discr_raiz = np.sqrt(discr)
    s1 = (-b - discr_raiz) / (2*a)  # (N, 1)
    s2 = (-b + discr_raiz) / (2*a)  # (N, 1)
    s1 = s1.reshape((-1, 1))
    s2 = s2.reshape((-1, 1))
    q1 = p0 + s1*v
    q2 = p0 + s2*v
    return q1, q2


def project_point_on_plane(a, n, p):
    """ Proyecta el vector "a" en el plano con normal "n" que pasar por "p" """


def fmc2plane_wave(fmc_acq, fs, theta, elem_xy, order='F'):
    """
    Esta funcion considera una onda plana con dirección (ux, uz). Para que la dirección esté
    en el plano XZ, en general hay que hacer primero un cambio de coordenadas rotando alrededor del
    eje Z. Entonces elem_xy hay que pasarlos en ese sistema
    Args:
        fmc_acq: array (emisor, receptor, tiempo)
        fs: frecuencia de muestreo
        theta: angulo respecto al eje z negativo, el vector dirección es
        (np.sin(theta), 0, -np.cos(theta))
        elem_xy: par de arrays (nel_x, nel_y). Coordenadas de los elementos
        en el sistema tal que el vector de onda está en el plano XZ
        order: el orden en que se leen elem_xy para "desenrollarlo" y que quede compatible con fmc_acq

    Returns:
       pw_acq: array (receptor, tiempo)
    """

    delay = elem_xy[0]*np.sin(theta)
    # cambiar la forma
    n = elem_xy[0].size
    n_samples = fmc_acq.shape[2]
    delay = delay.reshape((n, 1), order=order)
    delay = delay - delay.min()  # para que sea todos positivos
    delay_index = np.around(fs * delay).astype(np.int16)
    pw_acq_0 = np.zeros((n, n, n_samples + delay_index.max()), dtype=fmc_acq.dtype)

    # agregar delays poniendo ceros al principio de cada A-scan
    for i in range(n):
        # para cada emisor, se retrasan las recepciones de todos
        k = delay_index[i, 0]
        pw_acq_0[i, :, k:k + n_samples] = fmc_acq[i, :, :]

    # sumar emisores
    pw_acq = pw_acq_0.sum(axis=0)
    # recortar para que quede la misma cantidad de samples original
    pw_acq = pw_acq[:, 0:n_samples]

    return pw_acq, delay_index


def stitch_3d_xy(pos_xy, image_list, roi, x_step, y_step, cscan=False, normalizar=True):
    """

    Args:
        pos_xy: list [[x1, x2, x3 etc], [y1, y2, y3 etc]] del centro del array, medidos en el sistema del array
        image_list: lista de imagenes a coser
        image_shape:
        roi:
        x_step:
        y_step:

    Returns:
        img_total
    """

    pos_xy = np.array(pos_xy) # 2 filas, tantas columnas como adquiciciones (len(image_list))
    n_adq = len(image_list)
    # assert pos_xy.shape[1] == n_adq
    x_min, y_min = pos_xy.min(axis=1)
    x_max, y_max = pos_xy.max(axis=1)
    roi_total = [x_min + roi[0], x_max + roi[1],
                 y_min + roi[2], y_max + roi[3],
                 roi[4], roi[5]]

    # supongo que las posiciones del array son multiplos de x_step e y_step, para evitar problemas de redondeo
    nx_tot = int((roi_total[1] - roi_total[0]) / x_step)
    ny_tot = int((roi_total[3] - roi_total[2]) / y_step)

    if cscan:
        nx, ny = image_list[0].shape  # tienen que ser todas del mismo tañaño
        img_total = np.zeros((nx_tot, ny_tot), dtype=image_list[0].dtype)
        # definir una lista de "imagenes" que valen 1 en todos los pixeles, y stichear eso también, de modo
        # que en los solapes de 2, y luego dividimosr por eso matriz para normalizar. Esto es muuuy ineficiente
        # en cuanto uso de memoria...pero pa salir del paso...
        norm_list = [np.ones_like(image_list[0]) for i in range(n_adq)]
        norm_total = np.zeros_like(img_total)

        for i in range(n_adq):
            ix = int((pos_xy[0, i] + roi[0] - roi_total[0])/x_step)
            iy = int((pos_xy[1, i] + roi[2] - roi_total[2]) / y_step)
            img_total[ix:ix+nx, iy:iy+ny] += image_list[i] # todo: REVISAR tema de la SUMA en las partes de solape
            # porque los cscan no se suman coherentemente
            norm_total[ix:ix + nx, iy:iy + ny] += norm_list[i]
    else:
        nz, nx, ny = image_list[0].shape  # tienen que ser todas del mismo tañaño
        nz_tot = nz
        img_total = np.zeros((nz_tot, nx_tot, ny_tot), dtype=image_list[0].dtype)
        # definir una lista de "imagenes" que valen 1 en todos los pixeles, y stichear eso también, de modo
        # que en los solapes de 2, y luego dividimosr por eso matriz para normalizar. Esto es muuuy ineficiente
        # en cuanto uso de memoria...pero pa salir del paso...
        norm_list = [np.ones_like(image_list[0]) for i in range(n_adq)]
        norm_total = norm_total = np.zeros_like(img_total)

        for i in range(n_adq):
            ix = int((pos_xy[0, i] + roi[0] - roi_total[0])/x_step)
            iy = int((pos_xy[1, i] + roi[2] - roi_total[2]) / y_step)
            img_total[:, ix:ix+nx, iy:iy+ny] += image_list[i]
            norm_total[:, ix:ix + nx, iy:iy + ny] += norm_list[i]

        if normalizar:
            img_total = img_total/norm_total

    return img_total, roi_total


def compute_cscan(img, surf_fun, z1, z2, roi, x_step, y_step, z_step):
    assert z2 < z1
    if surf_fun is None:
        iz1 = int(np.round((roi[4] - z1)/z_step))
        iz2 = int(np.round((roi[4] - z2)/z_step))
        cscan = img[iz1:iz2, :, :].max(axis=0)
    else:
        nx, ny = img.shape[1:3]
        cscan = np.zeros((nx, ny))
        for i in range(nx):
            x = i*x_step + roi[0]
            for j in range(ny):
                y = j * y_step + roi[2]
                zs = surf_fun(x, y)[0]
                iz1 = int(np.round((roi[4] - (z1 + zs)) / z_step))
                iz2 = int(np.round((roi[4] - (z2 + zs)) / z_step))
                cscan[i, j] = img[iz1:iz2, i, j].max()

    return cscan


def fix_nan_columns(img):
    """ las imagenes calculadas con array virtual a veces tienen columnas(direccion z) de nanes, debidos a que
    algun elemento virtual de esa linea de imagen es nan por problema de convergenca. Esto reemplaza esa columna
    por el promedio de sus vecinas"""
    a = np.nonzero(np.isnan(img[0, :, :])) # (indice fila, indices columna)
    for m in range(a[0].size):
        i, j = a[0][m], a[1][m]
        print(i,j)
        # si justo una columna contigua es de nanes cagamos...todo
        img[:, i, j] = (img[:, i+1, j] + img[:, i, j+1] + img[:, i-1, j] + img[:, i, j-1])/4

    return img


def return_xyz_mesh(roi, step):
    """esta mesh responde a la forma de las imagenes"""
    dx, dy, dz = step
    z, x, y = np.meshgrid(np.arange(roi[4], roi[5], -dz),
                          np.arange(roi[0], roi[1], dx),
                          np.arange(roi[2], roi[3], dy), indexing='ij')
    return x, y, z