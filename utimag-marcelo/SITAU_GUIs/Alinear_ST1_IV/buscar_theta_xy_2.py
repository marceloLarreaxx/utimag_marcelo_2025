# abrir el robot
# conectar sitau, con opcion de no
# pasar por pose intermedia y sacar burbujas
# definir pose inicial e ir hasta ahi, cerca de la placa
# hacer loop del barrido con opcion sin sitau
# si mide guardar los ascan
# pintar con matplotlib mientras mide, preguntar a chat gpt como refrescar las curvas del plot
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtWidgets import QApplication

from SITAU2 import st2lib
from imag2D.pintar import arcoiris_cmap
from utils import first_thr_cross
import SITAU1ethernet.stfplib_py.stfplib as stfplib
from robot.robot_gui_control import RobotGUI
import os
import robot_helpers as rh
import sitau_helper as sh
import time
import pyqtgraph as pg
from scipy.spatial import transform as tra
from scipy.spatial.transform import Rotation

IP_ROBOT = '192.168.2.11'
IP_PC = '192.168.2.13'
SPEEDL_TIMEOUT = 10
STEPS_BACK = 0
MAX_FORCE = 30
COS45 = np.cos(np.pi / 4)

bin_path = r'C:\MarceloLarrea\utimag_Marcelo\SITAU1ethernet\stfplib_py\BIN\\'
DEFAULT_THR = 50
DEFAULT_INDEX_LINE = [0, 0, 0, 0, 0]
DEFAULT_WINDOW_NUM = 10
ASCAN_ELEMENTS = [0, 10, 60, 110, 120]  # CHEQUEAR !!!!!!!
COLORS = ['r', 'b', 'w', 'y', 'g']
# HOME_POSE = np.array([-0.9*np.cos(np.pi/4), -0.9*np.cos(np.pi/4), 0.4, 0, 0, -np.pi/4])
HOME_POSE = np.array([-750 * np.cos(np.pi / 4), -750 * np.cos(np.pi / 4), 300, 0, 0, 45])
# HOME_POSE = np.array([-0.384, -0.329, 0.238, 0, 0, -np.pi / 4])
TCP_OFFSET_0 = [31, 0.1, 262, 180, 0, 0]

robot = rh.InterpreterHelper(IP_ROBOT, IP_PC)

robot.start_interpreter_mode()
robot.connect()
time.sleep(0.5)
robot.start_listening()
time.sleep(0.5)
# EJECUTAR EN EL ROBOT EL THREAD QUE HACE QUE PARE SI SE LO TOCA
robot.execute_command(rh.ur_threads['contact_test'].format(MAX_FORCE))
robot.force_thread(MAX_FORCE)
robot.set_offset_tcp_0(TCP_OFFSET_0)

t = 10
robot.go_home(t)
time.sleep(t + 1)

sitau = stfplib.C_STFPLIB()  # accedo a librerìas en self.sitau para poder utilizar funciones
sitau_h = sh.SitauHelper()

########## INICIAMOS BARRIDO ##########

z_theta_start_deg = 0
z_theta_end_deg = 90
z_step = 15

theta_z = np.arange(z_theta_start_deg, z_theta_end_deg + z_step, z_step)
theta_x = 15
# theta_x = np.array(theta_x)

measure_SITAU = False

plt.ion()

# Adding zero radians to the arrays if necessary
if 0.0 not in theta_z:
    pos_to_neg = np.where(np.diff(np.sign(theta_z)))[0]
    theta_z = np.insert(theta_z, pos_to_neg + 1, 0.0)

# nx = theta_x.size
nz = theta_z.size
n_focal_laws = 0

sweep_ascan = []
if measure_SITAU:
    sitau_h.open_sitau()
    n_focal_laws = sitau_h.config_focal_laws()
    n_samples = sitau.ST_GetAScanDataNumber()
    sweep_ascan = np.zeros(shape=(nx, n_focal_laws, n_samples))

    # app = QApplication([])
    w = pg.PlotWidget()
    w.show()
    [w.getPlotItem().addItem(pg.PlotDataItem()) for i in ASCAN_ELEMENTS]
    ascan_plots = w.getPlotItem().curves
    # app.exec_()

# theta_xy = np.zeros(shape=(nx, 2))

# ESTABLEZCO POSICIÓN INICIAL (QUE PERMITA REMOVER BURBUJAS DEL ARRAY)
z = 30
robot.set_home([HOME_POSE[0], HOME_POSE[1], z, HOME_POSE[3], HOME_POSE[4], HOME_POSE[5]])
t = 8
x = HOME_POSE[0]
y = HOME_POSE[1]

pose0 = [0, 0, z, 0, 0, 0]  # misma que home pero más abajo en el eje z
robot.move_from_home(pose0,t)
time.sleep(t + 1)

# PASAMOS A LA NUEVA POSICIÓN HOME
# robot.set_home([HOME_POSE[0], HOME_POSE[1], z, HOME_POSE[3], HOME_POSE[4], HOME_POSE[5]])

t = 2
for i in range(nz):
    rot_vec = tra.Rotation.from_euler('zx', [theta_z[i], theta_x], degrees=True)
    temp = rot_vec.as_euler('xyz', degrees=True)
    pose = np.concatenate(([0,0,0], temp))

        # if j % 2 == 0:
        #     print('no')
        #     pose = [0, 0, 0, theta_x[j], 0, theta_z[i]]
        #     print([np.round([theta_x[j], theta_z[i]], 2)], 't = ' + str(t))
        # else:
        #     print('si')
        #     pose = [0, 0, 0, theta_x[j], 0, theta_z[i]]
        #     print([np.round([theta_x[j], theta_z[i]], 2)], 't = ' + str(t))
        #     if measure_SITAU:
        #         theta_xy[j + i * nx] = [theta_x[i], theta_z[j]]

    robot.move_from_home(pose, t, block=True)
    if measure_SITAU:
        range_x = np.arange(n_samples)
        data_i = sitau_h.measure()
        sweep_ascan[j + i * nx, :, :] = data_i
    time.sleep(t + 0.3)
print("----------------------------------")
pose = [0, 0, 0, 0, 0, theta_z[-1]]
robot.move_from_home(pose, t, block=True)
#######################################

