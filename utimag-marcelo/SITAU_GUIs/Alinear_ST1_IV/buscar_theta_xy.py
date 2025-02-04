# abrir el robot
# conectar sitau, con opcion de no
# pasar por pose intermedia y sacar burbujas
# definir pose inicial e ir hasta ahi, cerca de la placa
# hacer loop del barrido con opcion sin sitau
# si mide guardar los ascan
# pintar con matplotlib mientras mide, preguntar a chat gpt como refrescar las curvas del plot
import matplotlib.pyplot as plt
import numpy as np
from SITAU2 import st2lib
from imag2D.pintar import arcoiris_cmap
from utils import first_thr_cross
import SITAU1ethernet.stfplib_py.stfplib as stfplib
from robot.robot_gui_control import RobotGUI
import os
import robot_helpers as rh
import sitau_helper as sh
import time
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

robot.go_home(10)

sitau = stfplib.C_STFPLIB()  # accedo a librerìas en self.sitau para poder utilizar funciones
sitau_h = sh.SitauHelper()

########## INICIAMOS BARRIDO ##########

y_theta_start_deg = 0
y_theta_end_deg = 0
y_step = 1

x_theta_start_deg = -20
x_theta_end_deg = 20
x_step = 2

theta_x = np.arange(x_theta_start_deg, x_theta_end_deg + x_step, x_step)
theta_y = np.arange(y_theta_start_deg, y_theta_end_deg + y_step, y_step)

measure_SITAU = True

plt.ion()

# Adding zero radians to the arrays if necessary
if 0.0 not in theta_x:
    pos_to_neg = np.where(np.diff(np.sign(theta_x)))[0]
    theta_x = np.insert(theta_x, pos_to_neg + 1, 0.0)
    theta_y = np.insert(theta_y, pos_to_neg + 1, 0.0)

nx = theta_x.size
ny = theta_y.size
n_focal_laws = 0

sweep_ascan = []
if measure_SITAU:
    sitau_h.open_sitau()
    n_focal_laws = sitau_h.config_focal_laws()
    n_samples = sitau.ST_GetAScanDataNumber()
    sweep_ascan = np.zeros(shape=(nx * ny, n_focal_laws, n_samples))
    fig, ax = plt.subplots()
    lines = [ax.plot([], []) for i in range(5)]


theta_xy = np.zeros(shape=(nx * ny, 2))

# ESTABLEZCO POSICIÓN INICIAL (QUE PERMITA REMOVER BURBUJAS DEL ARRAY)
z = 30
robot.set_home([HOME_POSE[0], HOME_POSE[1], z, HOME_POSE[3], HOME_POSE[4], HOME_POSE[5]])
t = 8
x = HOME_POSE[0]
y = HOME_POSE[1]

pose0 = [0, 0, 30, 0, 0, 0]  # misma que home pero más abajo en el eje z
# robot.move_to_pose(pose0, t)
robot.move_from_home(pose0,t)
time.sleep(t + 5)

# PASAMOS A LA NUEVA POSICIÓN HOME
# robot.set_home([HOME_POSE[0], HOME_POSE[1], z, HOME_POSE[3], HOME_POSE[4], HOME_POSE[5]])
robot.go_home(8)

t = 1
for i in range(ny):
    for j in range(nx):
        if i % 2 == 0:
            pose = [0, 0, 0, theta_x[j], theta_y[i], 0]
            print([np.round([theta_x[j], theta_y[i], 2])], 't = ' + str(t))
            if measure_SITAU:
                theta_xy[j + i * nx] = [theta_x[j], theta_y[i]]
        else:
            pose = [0, 0, 0, np.flip(theta_x)[j], theta_y[i], 0]
            print([np.round([np.flip(theta_x)[j], theta_y[i]], 2)], 't = ' + str(t))
            if measure_SITAU:
                theta_xy[j + i * nx] = [np.flip(theta_x)[j], theta_y[i]]

        robot.move_from_home(pose, t, block=True)
        if measure_SITAU:
            range_x = np.arange(n_samples)
            data_i = sitau_h.measure()
            sweep_ascan[j + i * nx, :, :] = data_i
            # for k in range(n_focal_laws):
            #     lines[k][0].set_ydata(np.abs(data_i[k, :]))
            #     lines[k][0].set_xdata(range_x)
            #     plt.draw()
            # idx_echo = self.alinearGui.detect_first_echo(data_i)
            # t_idx = idx_echo[:, 0]
            # for k1, vl in enumerate(self.alinearGui.vertical_lines):
            #     vl.setValue(t_idx[k1])
        time.sleep(t + 0.3)
    print("----------------------------------")

if measure_SITAU:
    fl_name1 = "A_scan_from"+str(x_theta_start_deg)+"_to_"+str(x_theta_end_deg)+"_"+str(x_step)+"steps"
    fl_name2 = fl_name1 + "_degrees"
    np.save(fl_name1, sweep_ascan)
    np.save(fl_name2, theta_xy)


#######################################

