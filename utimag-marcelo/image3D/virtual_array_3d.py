import numpy as np
import pyvista as pv
import imag3D.snell_fermat_3d as sf
import pyopencl as cl
import imag3D.utils_3d as uti


def compute_virtual_source_3d(r1, r2, tof1, tof2, t_ae, c1, c2):

    tk = t_ae * (1 - (c1 / c2)**2)
    rad1, rad2 = c2 * (tof1 - tk), c2 * (tof2 - tk)

    # https: // math.stackexchange.com / questions / 256100
    # / how-can-i-find-the-points-at-which-two-circles-intersect
    a = (rad1**2 - rad2**2 + (r2 - r1)**2) / (2 * (r2 - r1))
    error = rad1**2 - a**2
    # error < 0 indica que si no hay solución a la intersección. Esto sucede si la línea es
    # justo el rayo que sale de la fuente, en ese caso error sería idealmente 0? Por cuestión
    # numérica puede dar un valor negativo casi nulo, entonces fuerzo a hv a ser nulo. En este
    # caso particular la fuente virtual dar resultado exacto (REVISAR bien este caso, todo)
    if error < 0:
        hv = 0
    else:
        hv = np.sqrt(rad1**2 - a**2)
    rv = r1 + a

    def vs_tof_fun(r):
        return tk + np.hypot(r - rv, hv) / c2

    return (rv, hv, tk), vs_tof_fun, error


class VirtualSource3D:

    def __init__(self, a, e_sl, vec_sl, r1, r2, c1, c2):
        self.a = np.array(a)  # fuente real
        vec_sl = np.array(vec_sl)
        # normalizar
        self.vec_sl = vec_sl/np.linalg.norm(vec_sl)
        self.e_sl = np.array(e_sl)
        self.r1 = r1  # primer foco, se mide a lo largo de un eje con origen en el punto de entrada
        # positivo hacia abajo
        self.r2 = r2  # segundo foco
        self.ray1 = None
        self.ray2 = None
        self.c1 = c1
        self.c2 = c2
        self.t_ae = None
        self.tk = None
        self.rv = None
        self.hv = None
        self.tof_to = None
        self.error = None

    def plot_vs_circle(self, plo, color='red'):
        vs_center = self.return_vs_center()
        # # vector perpendicular a scan line, que pasa por el círculo
        # if self.vec_sl[2] != 0:
        #     rad_vector = np.array([1, 0, -self.vec_sl[0] / self.vec_sl[2]])
        #     rad_vector = self.rv * rad_vector / np.linalg.norm(rad_vector)
        # else:
        #     # en este caso la scan line es horizontal (en plano xy)
        #     rad_vector = self.rv * np.array([0, 0, 1])

        # circ_point = vs_center + rad_vector
        vs_circle_pv = pv.Disc(center=vs_center.tolist(), inner=self.hv - 0.1, outer=self.hv + 0.1,
                               normal=self.vec_sl.tolist(), c_res=40)
        # vs_circle_pv = pv.CircularArcFromNormal(vs_center, normal=self.vec_sl, polar=circ_point, angle=360)
        # vs_circle_pv = pv.CircularArcFromNormal(vs_center.tolist(), normal=self.vec_sl.tolist(), polar=(self.hv, 0, 0), angle=360)
        plo.add_mesh(vs_circle_pv, color=color)
        return vs_circle_pv

    def plot_scan_line(self, plo, length, color='white', line_width=2):
        pv_scan_line = pv.Line(self.e_sl, self.e_sl + length * self.vec_sl)
        plo.add_mesh(pv_scan_line, color=color, line_width=line_width)
        return pv_scan_line

    def compute(self, ifaz_fun, e_xy_0=(0, 0), gamma=10):
        f1 = self.e_sl + self.r1*self.vec_sl
        f2 = self.e_sl + self.r2*self.vec_sl
        self.ray1 = sf.Ray3D.compute(self.a, f1, e_xy_0, self.c1, self.c2, ifaz_fun, gamma=gamma,
                                     method='gradient_descent')
        self.ray2 = sf.Ray3D.compute(self.a, f2, e_xy_0, self.c1, self.c2, ifaz_fun, gamma=gamma,
                                     method='gradient_descent')
        self.t_ae = self.ray1.t_ae  # podría usarse ray2
        (self.rv, self.hv, self.tk), self.tof_to, self.error = compute_virtual_source_3d(self.r1, self.r2,
                                                                                         self.ray1.tof, self.ray2.tof,
                                                                                         self.t_ae, self.c1, self.c2)
        if self.error < 0:
            print('error: no hay solución')
            print('argumento de la raiz:', self.error)

        return self.ray1, self.ray2

    def return_vs_center(self):
        return sf.sl_depth2xyz(self.e_sl, self.vec_sl, self.rv)

    def return_vspoint(self):
        # vector que pasa por uno de los puntos de entrada y es perpendicular a la scan line
        p = self.ray1.e - self.e_sl + self.vec_sl * (np.dot(self.e_sl - self.ray1.e, self.vec_sl))
        w = p / np.linalg.norm(p)
        vspoint = self.return_vs_center() + w * self.hv
        return vspoint


def tof_error_analysis(roi, p_dict, p_surf, cl_code, q=0.9):
    print('inicializar GPU')
    # seleccionar un "device"
    plat = cl.get_platforms()  # lista de plataformas
    gpu = plat[0].get_devices()  # esta es la GPU
    print(gpu)
    ctx = cl.Context(gpu)
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    prg = {'exact': cl.Program(ctx, cl_code['exact']).build(),
           'av': cl.Program(ctx, cl_code['av']).build()}

    # -------- ARRAYS Y BUFFERS ----------------------------
    tofmap = {'exact': [],
              'av': []}
    errormap = []
    zv = np.zeros((p_dict['nx'], p_dict['ny'], p_dict['n_elementos']), dtype=np.float32)
    hv = np.zeros_like(zv)
    tk = np.zeros_like(zv)
    cosas = np.zeros((4, p_dict['nx'], p_dict['ny'], p_dict['n_elementos']), dtype=np.float32)
    tofmap_size = 4 * p_dict['nz'] * p_dict['nx'] * p_dict['ny']
    gpu_buf = {'source': cl.Buffer(ctx, mf.READ_WRITE, size=12),  # float32, 4 bytes
               'elem': cl.Buffer(ctx, mf.READ_WRITE, size=2),  # uint16, 2 bytes
               'p_surf': cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=p_surf),
               'tofmap': cl.Buffer(ctx, mf.READ_WRITE, size=tofmap_size),
               'tofmap_av': cl.Buffer(ctx, mf.READ_WRITE, size=tofmap_size),
               'zv': cl.Buffer(ctx, mf.READ_WRITE, size=zv.nbytes),
               'hv': cl.Buffer(ctx, mf.READ_WRITE, size=zv.nbytes),
               'tk': cl.Buffer(ctx, mf.READ_WRITE, size=zv.nbytes),
               'cosas': cl.Buffer(ctx, mf.READ_WRITE, size=cosas.nbytes)}

    for n in range(p_dict['n_elementos']):
        # convertir indice lineal a (i,j)
        i = n % p_dict['nel_x']
        j = int(n / p_dict['nel_x'])
        _, source = uti.array_ij2element((i, j), p_dict['nel_x'], p_dict['nel_y'],
                                         p_dict['pitch'], p_dict['pitch'])

        tofmap['exact'].append(np.zeros((p_dict['nz'], p_dict['nx'], p_dict['ny']), dtype=np.float32))
        tofmap['av'].append(np.zeros((p_dict['nz'], p_dict['nx'], p_dict['ny']), dtype=np.float32))

        cl.enqueue_copy(queue, gpu_buf['source'], source)
        cl.enqueue_copy(queue, gpu_buf['elem'], np.uint16(n))
        prg['exact'].tofmap_refrac(queue, tofmap['exact'][n].shape, None, gpu_buf['source'], gpu_buf['tofmap'],
                                   gpu_buf['p_surf'])
        cl.enqueue_copy(queue, tofmap['exact'][n], gpu_buf['tofmap'])
        prg['av'].a_virt(queue, (p_dict['nx'], p_dict['ny']), None, gpu_buf['zv'], gpu_buf['hv'], gpu_buf['tk'],
                                gpu_buf['p_surf'])
        cl.enqueue_copy(queue, zv, gpu_buf['zv'])
        cl.enqueue_copy(queue, hv, gpu_buf['hv'])
        cl.enqueue_copy(queue, tk, gpu_buf['tk'])
        cl.enqueue_copy(queue, cosas, gpu_buf['cosas'])
        prg['av'].tofmap_refrac_av(queue, tofmap['av'][n].shape, None, gpu_buf['elem'], gpu_buf['zv'], gpu_buf['hv'],
                                gpu_buf['tk'], gpu_buf['tofmap_av'], gpu_buf['p_surf'])
        cl.enqueue_copy(queue, tofmap['av'][n], gpu_buf['tofmap_av'])

        temp = (tofmap['exact'][n] - tofmap['av'][n]) * 1000  # pasado a nanosegundos
        errormap.append(temp)

    # convertir en np array
    errormap = np.array(errormap)
    tofmap['exact'] = np.array(tofmap['exact'])
    tofmap['av'] = np.array(tofmap['av'])

    results = {'tofmap': tofmap,
               'cosas': cosas,
               'virtual_array': (zv, hv, tk),
               'error': errormap,
               'error_mean': np.mean(errormap, axis=0),
               'error_std': np.std(errormap, axis=0),
               'error_q': np.quantile(np.abs(errormap), 0.9, axis=0)}

    return results
