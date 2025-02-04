import time

import methods
import robot_helpers as rh
import numpy as np
from scipy.spatial.transform import Rotation
import SITAU1ethernet.stfplib_py.stfplib as stfplib
import sitau_helper as sh
import imag3D.ifaz_3d as ifaz3d
import imag3D.utils_3d as u3d
import os
from datetime import datetime

#################### CONFIGURACUÓN UR10e ####################

IP_ROBOT = '192.168.2.11'
IP_PC = '192.168.2.13'

# TCP_OFFSET_0 = [31, 0.1, 262, -178, -0.5, 6]  # VALOR DE OFFSET DETERMINADO POR BÚSQUEDA DE BARRIDOS

# TCP_OSSET SEGUNDA CORECCIÓN DE PARALELISMO
TCP_OFFSET_0 = [31, 0.1, 262, -177.8, 2.6, 6.1]

robot = rh.InterpreterHelper(IP_ROBOT, IP_PC)
robot.start_interpreter_mode()
robot.connect()
robot.start_listening()
robot.set_tcp_offset(TCP_OFFSET_0)
sweep = True
# robot.go_home(10)
# time.sleep(10)

shape = 1   # 1: plane; 2: cylinder; 3 sphere

n = 200
pose_combinations2 = []
########## lista de rangos de movimiento para plano ##########
home_init = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\home_delta_z.npy')
robot.set_home(home_init)  # DETERMINO HOME ACTUAL
delta_z = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\delta_z.npy')

if shape == 1:
    # dz = (-76, -49); rx = (0, 20); ry = (0, 20); limry = -68    # VALORES UTILIZADOS EN PLANO BASE
    dz = (-89, -82); rx = (0, 2); ry = (0, 2); limry = -86  # VALORES UTILIZADOS EN PLANO FIBRA
    # lst = [(-24, 3, 5), (-20, 2, 1), (-24, 0, 0)]
    while len(pose_combinations2) < n / 2:
        pose_combinations = methods.get_list_of_positions(n, dz, rx, ry)
        pose_combinations2 = methods.filter_list(pose_combinations, limry, 2)
        print(len(pose_combinations2))
    # pose_combinations2 = [(-76, 0, 0), (-74, 4, 3), (-75, 3, 2)]
######### lista de rangos de movimiento para cilindro ########
if shape == 2:
    # dz = (-75, -47); dy = (0, 0.5);  ry = (0, 20); rz = (0, 30); limry = -70  # VALORES UTILIZADOS EN CILINDRIO 12mm
    # dz = (-53, -25); dy = (0, 6); ry = (0, 20); rz = (0, 90); limry = -50  # VALORES UTILIZADOS EN CILINDRIO 35mm
    # dz = (-66, -46); dy = (0, 3); ry = (0, 20); rz = (0, 90); limry = -62  # VALORES UTILIZADOS EN CILINDRIO CÓNCAVO 25mm
    # dz = (-51, -40); dy = (0, 3); ry = (0, 20);  rz = (0, 30); limry = -48  # VALORES UTILIZADOS EN CILINDRIO CÓNCAVO 40mm
    dz = (-43, -68); dy = (0, 3); ry = (0, 20);  rz = (0, 90); limry = -64  # VALORES UTILIZADOS EN CILINDRIO CÓNCAVO 25mm invertido

    while len(pose_combinations2) < n / 2:
        pose_combinations = methods.get_list_of_positions2(n, dz, dy, rz, ry)
        pose_combinations2 = methods.filter_list2(pose_combinations, limry, 3, 25, 15)
        print(len(pose_combinations2))
    # pose_combinations2 = [(-69.266, 0, -0.02, 0), (-64.4, 0.017, 0, 17.8), (-66.11, 0.01, 35.8, 15.36)] # Datos de prueba nada más
    # pose_combinations2 = [(-69.266, 0, -0.02, 0), (-68.4, 0.017, 10, 17.8), (-56.11, 0.01, 24, 40)]
    # pose_combinations2 = [(-46.244, 0, -0.02, 0), (-40.431, 0.01, -0.02, 0), (-42.047, 0.01, 0, 5.08), (-39.194, 0.0, 0, 15.05), (-40.027, 0.02, 88, 10)]  #Prueba para cóncavo 40mm

######### lista de rangos de movimiento para esfera ########
if shape == 3:
    dz = (-60, -71.6); dx = (0, 2); dy = (0, 2)
    pose_combinations2 = methods.get_list_of_positions3(n, dx, dy, dz)
##############################################################

#############################################################

#################### CONFIGURACUÓN SITAU ####################

tx_list = [[0, 0], [5, 5], [10, 10], [0, 5], [5, 0], [10, 0], [0, 10], [10, 5], [5, 10]]
ascan_elements = []
for tx in tx_list:
    linear_index, _ = u3d.array_ij2element(tx, 11, 11, 1, 1)
    ascan_elements.append(linear_index)

sitau = stfplib.C_STFPLIB()  # accedo a librerìas en self.sitau para poder utilizar funciones
sitau_h = sh.SitauHelper()
measure_SITAU = True
if measure_SITAU:
    config_1 = False  # True: define leyes focales donde emite el elemento i y el mismo recibe; False: emite elemento i y reciben todos
    acquisition_time = 50
    gain = 40
    sitau_h.open_sitau()
    sitau_h.set_ascan_elements(ascan_elements)
    n_focal_laws = sitau_h.config_focal_laws(config_1)
    sitau.ST_SetAcqTime(acquisition_time)
    n_samples = sitau.ST_GetAScanDataNumber()
    n_ch = sitau.ST_GetChannelNumber()
    sitau.ST_SetGain(gain)
    # data_ascan = np.zeros(shape=(n_focal_laws, n_ch, n_samples))
    ##### CREANDO DIRECTORIO DE ALMACENAMIENTO #####
    dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    full_path = os.path.join(os.getcwd(), dir_name)
    os.makedirs(full_path)
    ##### CREANDO Y ALMACENANDO DICCIONARIO DE PARÁMETROS DE MEDICIÓN ####
    dict_param = {'pose_combinations': pose_combinations2,
                  'ascan_elements': ascan_elements,
                  'acquisition_time': acquisition_time,
                  'gain': gain,
                  'n_focal_laws': n_focal_laws,
                  'n_samples': n_samples,
                  'n_ch': n_ch,
                  'home_init': home_init,
                  'range_z': dz,
                  'delta_z': delta_z}
    dict_name = os.path.join(full_path, "measure_parameters")
    np.savez(dict_name, **dict_param)

#############################################################

######### COMBINACIONES DE DESPLAZAMIENTO Y ROTACIONES ########

if sweep:
    robot.go_home(10)
    time.sleep(18)
    t = 3
    for i in range(len(pose_combinations2)):
        pose_i = pose_combinations2[i]
        print('pose_i: {}'.format(pose_i))
        if shape == 1:
            pose = [0, 0, pose_i[0], pose_i[1], pose_i[2], 0]
        elif shape == 2:
            pose = [0, pose_i[1], pose_i[0], 0, pose_i[3], pose_i[2]]
        elif shape == 3:
            pose = [pose_i[0], pose_i[1], pose_i[2], 0, 0, 0]
        print("Measure: {} / Pose: {}".format(i, pose))
        robot.move_from_home(pose, t, block=True)
        data_ascan = np.zeros(shape=(n_focal_laws, n_ch, n_samples), dtype=np.int16)
        if measure_SITAU:
            acq_counter = sitau.ST_Trigger(2)
            for j in range(n_focal_laws):
                _, data_ascan[j, :, :] = sitau.ST_GetBuffer_LastImage(j)
            # fl_name = os.path.join(full_path, "pose_" + str(i) + "_" + str(np.round(pose_combinations2[i], 2)))
            fl_name = os.path.join(full_path, str(i+1))
            np.save(fl_name, data_ascan)
        print('----------0----------')

    time.sleep(5)
    robot.go_home(8)
###############################################################

#################### LOADING DICTIONARIES ####################

# dict_data = np.load("dict_name")
# dict_data = {ky: dict_data[ky] for ky in dict_data}
