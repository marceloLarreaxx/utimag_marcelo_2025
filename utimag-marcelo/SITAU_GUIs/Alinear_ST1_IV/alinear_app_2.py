import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget
from PyQt5.QtCore import QFile, Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt5 import uic, QtGui
import pyqtgraph as pg
from SITAU2 import st2lib
from imag2D.pintar import arcoiris_cmap
from utils import first_thr_cross
import SITAU1ethernet.stfplib_py.stfplib as stfplib
from robot.robot_gui_control import RobotGUI
import os
import robot_helpers as rh
import time
from scipy.spatial.transform import Rotation
import imag3D.ifaz_3d as ifaz3d
import imag3D.utils_3d as u3d

IP_ROBOT = '192.168.2.11'
IP_PC = '192.168.2.13'
SPEEDL_TIMEOUT = 10
STEPS_BACK = 0
MAX_FORCE = 50
COS45 = np.cos(np.pi / 4)

bin_path = r'C:\MarceloLarrea\utimag_Marcelo\SITAU1ethernet\stfplib_py\BIN\\'
DEFAULT_THR = 50
DEFAULT_INDEX_LINE = [0, 0, 0, 0, 0]
DEFAULT_WINDOW_NUM = 10
ASCAN_ELEMENTS = [0, 10, 60, 110, 120]  # CHEQUEAR !!!!!!!
COLORS = ['r', 'cyan', 'w', 'y', 'g']
# HOME_POSE = np.array([-0.9*np.cos(np.pi/4), -0.9*np.cos(np.pi/4), 0.4, 0, 0, -np.pi/4])
coef_home = 725
HOME_POSE = np.array([-coef_home * np.cos(np.pi / 4), -coef_home * np.cos(np.pi / 4), 300, 0, 0, 45])
# HOME_POSE = np.array([-0.384, -0.329, 0.238, 0, 0, -np.pi / 4])
# TCP_OFFSET_0 = [31, 0.1, 262, 180, 0, 0]

# TCP_OSSET PRIMERA CORECCIÓN DE PARALELISMO
# TCP_OFFSET_0 = [31, 0.1, 262, -177.6, 2.4, 0.1]

# TCP_OSSET SEGUNDA CORECCIÓN DE PARALELISMO
TCP_OFFSET_0 = [31, 0.1, 262, -177.8, 2.6, 6.1]
# TCP_OFFSET_0 = [31, 0.1, 262, -177.8, 3.5, 6.1]

# TCP_OFFSET_0 = [31, 0.1, 262, -178, -0.5, 6]  # VALOR DE OFFSET DETERMINADO POR BÚSQUEDA DE BARRIDOS
PROBE_GAP = 0.75
CYLINDER_RAD = 35 / 2  # RADIO EN MILÍMETROS
CYLINDER_RAD_2 = 11.2 / 2  # RADIO EN MILÍMETROS
CYLINDER_RAD_U25 = 12.5
CYLINDER_RAD_U25_EXT = 20
CYLINDER_RAD_U40 = 20.5
CYLINDER_RAD_U40_EXT = 40.5
SPHERE_RAD = 19/2
# PLANO_7_11 = 7.11
DELTA_Z = 0


class AlinearGui(QMainWindow):

    def __init__(self):
        super().__init__()

        self.move(80, 100)

        #self.sitau = stfplib.C_STFPLIB()  # accedo a librerìas en self.sitau para poder utilizar funciones
        self.ui = uic.loadUi('alinear_gui_2.ui', self)
        # self.sitau = st2lib.ST2Lib()  # crear objeto sitau que tiene las funciones de control
        # crear el timer que lee periodicamente el buffer y conectarlo a update_plots
        self.timer = QTimer()
        self.timer.timeout.connect(self.measure_and_plot)

        # crear objetos del A-scan plot. Para eso hace una lista en la que cada elementos es un PlotDataItem
        # que se agrega al PlotItem del ascan_widget
        [self.ascan_widget.getPlotItem().addItem(pg.PlotDataItem()) for i in ASCAN_ELEMENTS]
        self.ascan_plots = self.ascan_widget.getPlotItem().curves

        # lista de botones que hay que desabilitar cuando se está adquiriendo
        self.disable_widget_list = [self.opensys_pushb, self.closesys_pushb, self.start_pushb,
                                    self.timer_spinbox]
        # conexiones varias
        self.start_pushb.clicked.connect(self.start_acquisition)
        self.opensys_pushb.clicked.connect(self.open_sitau)
        self.stop_pushb.clicked.connect(self.stop_acquisition)
        self.closesys_pushb.clicked.connect(self.close_sitau)

        self.thr_spinbox.setValue(DEFAULT_THR)  # establecemos valor inicial de threshold
        self.thr_spinbox.valueChanged.connect(self.set_thr_line_pos)

        # ETIQUETAS
        self.idx_label0.setText(str(DEFAULT_INDEX_LINE[0]))
        self.idx_label10.setText(str(DEFAULT_INDEX_LINE[1]))
        self.idx_label60.setText(str(DEFAULT_INDEX_LINE[2]))
        self.idx_label110.setText(str(DEFAULT_INDEX_LINE[3]))
        self.idx_label120.setText(str(DEFAULT_INDEX_LINE[4]))

        self.thr_line = pg.InfiniteLine(pos=self.thr_spinbox.value(), angle=0, movable=True)
        self.ascan_widget.addItem(self.thr_line)
        self.thr_line.sigPositionChanged.connect(self.get_thr_position)
        # self.thr_line.sigPositionChanged.connect(lambda: self.get_thr_position(0))

        [self.ascan_widget.addItem(
            pg.InfiniteLine(pos=DEFAULT_INDEX_LINE[i], angle=90, pen=pg.mkPen(COLORS[i], width=2))) for i in
            range(len(ASCAN_ELEMENTS))]
        self.vertical_lines = self.ascan_widget.getPlotItem().items[6:11]

        # Botón para abrir ventana de robot
        self.robot_win = RobotWindowGui(self)
        self.openRobotManager_pushb.clicked.connect(self.open_robot_window)

        # Botón para abrir ventana de fmc
        self.fmc_win = FmcView(self, self.robot_win)
        self.openFmcManager_pushb.clicked.connect(self.open_fmc_window)

        self.show()

    def measure(self, t=1):
        acq_counter = self.sitau.ST_Trigger(t)
        n_samples = self.sitau.ST_GetAScanDataNumber()
        data = np.zeros((self.n_focal_laws, n_samples), dtype=np.int16)
        for i in range(self.n_focal_laws):
            result, data[i, :] = self.sitau.ST_GetBuffer_LastImage(i)
        return data

    def open_robot_window(self):
        self.robot_win.show()

    def open_fmc_window(self):
        self.fmc_win.show()

    def open_sitau(self):
        if self.sitau.ST_OpenSys(bin_path, "192.168.2.10", 6002, 6008) < 0:
            self.sitau.ST_CloseSys()
            del self.sitau
            os._exit(1)
        # robot_directory = r'C:\MarceloLarrea\utimag_Marcelo\robot\\'
        # self.robot_window = RobotGUI(path=robot_directory)

    def start_acquisition(self):
        for x in self.disable_widget_list:
            x.setEnabled(False)
        # adquirir
        self.config_focal_laws(ASCAN_ELEMENTS)
        self.sitau.ST_SetAcqTime(self.adquisition_time_spinbox.value())
        self.sitau.ST_SetGain(self.gain_spinbox.value())
        self.timer.start(self.timer_spinbox.value())  # setea el periodo del timer
        # self.sitau.Start()

    def stop_acquisition(self):
        for x in self.disable_widget_list:
            x.setEnabled(True)
        self.timer.stop()
        # self.sitau.Stop()

    def config_focal_laws(self, ascan_elements):
        n_ch = self.sitau.ST_GetChannelNumber()
        self.sitau.ST_DeleteFocalLaws()
        delay = np.zeros(n_ch, dtype=np.float32)
        for i in ascan_elements:
            tx_enable = np.zeros(n_ch, dtype=np.int32)
            tx_enable[i] = 1
            self.sitau.ST_AddFocalLaw(tx_enable, tx_enable, delay, delay, n_ch)
        self.n_focal_laws = self.sitau.ST_GetFocalLaw_Number()
        return self.n_focal_laws

    def measure_and_plot(self, t=1):
        acq_counter = self.sitau.ST_Trigger(t)
        n_samples = self.sitau.ST_GetAScanDataNumber()
        data = np.zeros((self.n_focal_laws, n_samples), dtype=np.int16)
        for i in range(self.n_focal_laws):
            result, data[i, :] = self.sitau.ST_GetBuffer_LastImage(i)
            self.ascan_plots[i].setData(np.abs(data[i, :]), pen=pg.mkPen(COLORS[i], width=2))
        idx_echo = self.detect_first_echo(data)
        # print(idx_echo)
        t_idx = idx_echo[:, 0]
        # print(t_idx)

        self.idx_label0.setText(str(int(t_idx[0])))
        self.idx_label10.setText(str(int(t_idx[1])))
        self.idx_label60.setText(str(int(t_idx[2])))
        self.idx_label110.setText(str(int(t_idx[3])))
        self.idx_label120.setText(str(int(t_idx[4])))

        min_idx = min(t_idx)
        max_idx = max(t_idx)
        dif_idx = max_idx - min_idx

        # self.label_disp.setText(str(type(dif_idx)) + "-" + str(dif_idx))

        thr_dif = self.error_idx_spinbox.value()
        # self.comm.error_dif.emit(dif_idx)  # !!!!! EMITIENDO DATA
        if dif_idx <= thr_dif:
            self.aligned_label.setText("Aligned")
            self.aligned_label.setStyleSheet("background-color: lightgreen")
        else:
            self.aligned_label.setText("Non-Aligned")
            self.aligned_label.setStyleSheet("background-color: rgb(255, 0, 0)")

            # mean_diff_echo = np.mean(np.diff(t_idx))
        # print(mean_diff_echo)

        for i, x in enumerate(self.vertical_lines):
            x.setValue(t_idx[i])
        return acq_counter

    def detect_first_echo(self, data):
        # i1, i2 = self.lr.getRegion()
        i1 = self.lr_min_spinbox.value()
        i2 = self.lr_max_spinbox.value()
        # franja para seleccionar un intervalo en el A-scan
        # self.lr = pg.LinearRegionItem(values=(int(i1), int(i2)))
        # self.ascan_widget.addItem(self.lr)
        umbral = self.thr_line.value()
        window_num = DEFAULT_WINDOW_NUM  ################## MODIFICAR
        idx = first_thr_cross(data, (int(i1), int(i2)), umbral, window_num)
        return idx

    def get_thr_position(self):
        y_pos = self.thr_line.getYPos()
        self.thr_spinbox.setValue(round(y_pos))
        # print(round(y_pos,2))

    def set_thr_line_pos(self):
        y_pos = self.thr_spinbox.value()
        self.thr_line.setValue(y_pos)

    def close_sitau(self):
        self.sitau.ST_CloseSys()

    def closeEvent(self, event):
        self.sitau.ST_CloseSys()


class LoopWorker(QThread):
    def __init__(self, robotWindowGui, alinearGui, parent=None):
        super().__init__(parent)

        self.theta_xz = None
        self.sweep_xz = None
        self.ascan_elements = None
        self.theta_xy = None
        self.sweep_ascan_dict = None
        self.sweep_ascan = None
        self.robotWindowGui = robotWindowGui
        self.alinearGui = alinearGui
        self.stop_sweep = False

        self.home = HOME_POSE
        self.robotWindowGui.stop_sweep_btn.clicked.connect(self.sweep_stop)

    def run(self):
        global n_samples
        self.stop_sweep = False
        # self.alinearGui.sitau.ST_SetAcqTime(self.alinearGui.adquisition_time_spinbox.value())
        # self.alinearGui.sitau.ST_SetGain(self.alinearGui.gain_spinbox.value())

        y_theta_start_deg = self.robotWindowGui.y_deg_init_spinbox.value()
        y_theta_end_deg = self.robotWindowGui.y_deg_last_spinbox.value()
        y_step = self.robotWindowGui.y_deg_step_spinbox.value()

        x_theta_start_deg = self.robotWindowGui.x_deg_init_spinbox.value()
        x_theta_end_deg = self.robotWindowGui.x_deg_last_spinbox.value()
        x_step = self.robotWindowGui.x_deg_step_spinbox.value()

        theta_x = np.arange(x_theta_start_deg, x_theta_end_deg + x_step, x_step)
        theta_y = np.arange(y_theta_start_deg, y_theta_end_deg + y_step, y_step)
        theta_z = []

        # Adding zero radians to the arrays if necessary
        if 0.0 not in theta_x:
            pos_to_neg = np.where(np.diff(np.sign(theta_x)))[0]
            theta_x = np.insert(theta_x, pos_to_neg + 1, 0.0)
            theta_y = np.insert(theta_y, pos_to_neg + 1, 0.0)

        nx = theta_x.size
        ny = theta_y.size
        nz = 0

        if self.robotWindowGui.sweep_xz_check.isChecked():
            theta_z = np.arange(y_theta_start_deg, y_theta_end_deg + y_step, y_step)
            print(theta_z)
            nz = theta_z.size

        n_focal_laws = 0
        if self.robotWindowGui.measure_check.isChecked():
            if self.robotWindowGui.sweep_xz_check.isChecked():
                self.ascan_elements = self.get_ascan_elements_sweep_xz()
                n_focal_laws = self.alinearGui.config_focal_laws(self.ascan_elements)
                n_samples = self.alinearGui.sitau.ST_GetAScanDataNumber()
                self.sweep_xz = np.zeros(shape=(nz, n_focal_laws, n_samples))
                self.theta_xz = np.zeros(shape=(nz, 2))
            else:
                n_focal_laws = self.alinearGui.config_focal_laws(ASCAN_ELEMENTS)
                n_samples = self.alinearGui.sitau.ST_GetAScanDataNumber()
                # self.sweep_ascan = np.zeros(shape=(nx, ny, n_focal_laws, n_samples))
                self.sweep_ascan = np.zeros(shape=(nx * ny, n_focal_laws, n_samples))
                self.theta_xy = np.zeros(shape=(nx * ny, 2))

        # REDEFINIMO HOME
        z = 47
        self.robotWindowGui.set_home([self.home[0], self.home[1], z, self.home[3], self.home[4], self.home[5]])
        self.robotWindowGui.robot_interpreter.set_home(
            [self.home[0], self.home[1], z, self.home[3], self.home[4], self.home[5]])
        self.robotWindowGui.go_home()

        time.sleep(10)

        # LLEVÁNDOLO A POSICIÓN INICIAL DE BARRIDO
        t = 8
        if self.robotWindowGui.sweep_xz_check.isChecked():
            pose = [0, 0, 0, theta_x[0], 0, theta_z[0]]
        else:
            pose = [0, 0, 0, theta_x[0], theta_y[0], 0]
        self.robotWindowGui.robot_interpreter.move_from_home(pose, t, block=True)
        time.sleep(4)

        self.robotWindowGui.lbl_sweep_state.setText("Sweep in progress...")
        self.robotWindowGui.lbl_sweep_state.setStyleSheet("color: red;")
        t = self.robotWindowGui.sweep_time_spinb.value()
        t2 = self.robotWindowGui.measure_time_spinb.value()

        #############################################################################
        if self.robotWindowGui.sweep_xz_check.isChecked():
            theta_x = theta_x[0]  # solo nos interesa una inclinacón en x
            # t_idx_plot = np.zeros((self.ascan_elements.size, 1))
            t_idx_plot = np.zeros((n_focal_laws, 1))
            for i in range(nz):
                # Acá representamos la rotación requerida:
                # inicializamos una rotación compuesta (primero en z y luego en x)
                rot_zx = Rotation.from_euler('zx', [theta_z[i], theta_x], degrees=True)
                # Luego obtenemos los ángulos de Euler qwe representan la rotacion rot_zx pero expresada en términos de rotaciones en 'xyz'
                temp = rot_zx.as_euler('xyz', degrees=True)
                pose = np.concatenate((np.array([0, 0, 0]), temp))
                print([np.round([theta_x, theta_z[i]], 2)], 't = ' + str(t))

                self.robotWindowGui.robot_interpreter.move_from_home(pose, t, block=True)
                if self.robotWindowGui.measure_check.isChecked():
                    data_i = self.alinearGui.measure()
                    # self.sweep_ascan[i, :, :] = data_i
                    idx_echo = self.alinearGui.detect_first_echo(data_i)
                    t_idx = idx_echo[:, 0]
                    t_idx2 = np.array([t_idx])
                    # for k in range(n_focal_laws):
                    #     print(t_idx_plot[:,i+1][k])
                    #     print(self.robotWindowGui.ascan_plots[k].shape)
                    #     self.robotWindowGui.ascan_plots[k].setData(t_idx_plot[k,1:])
                    for k in range(n_focal_laws):
                        self.robotWindowGui.ascan_plots[k].setData(np.abs(data_i[k, :]))

                    min_idx = min(t_idx)
                    max_idx = max(t_idx)
                    dif_idx = max_idx - min_idx

                    self.sweep_xz[i, :, :] = data_i
                    self.theta_xz[i, :] = np.array([theta_x, theta_z[i]])
                    self.robotWindowGui.diference_lbl.setText(str(dif_idx))
                time.sleep(t + 0.3)
            print("----------------------------------")
            pose = [0, 0, 0, 0, 0, theta_z[-1]]
            self.robotWindowGui.robot_interpreter.move_from_home(pose, t, block=True)
        else:
            for i in range(ny):
                for j in range(nx):
                    if self.stop_sweep:
                        return
                    if i % 2 == 0:
                        pose = [0, 0, 0, theta_x[j], theta_y[i], 0]
                        print([np.round([theta_x[j], theta_y[i]], 2)], 't = ' + str(t))
                        if self.robotWindowGui.measure_check.isChecked():
                            self.theta_xy[j + i * nx] = [theta_x[j], theta_y[i]]
                    else:
                        pose = [0, 0, 0, np.flip(theta_x)[j], theta_y[i], 0]
                        print([np.round([np.flip(theta_x)[j], theta_y[i]], 2)], 't = ' + str(t))
                        if self.robotWindowGui.measure_check.isChecked():
                            self.theta_xy[j + i * nx] = [np.flip(theta_x)[j], theta_y[i]]

                    self.robotWindowGui.robot_interpreter.move_from_home(pose, t, block=True)
                    # self.robotWindowGui.robot_interpreter.move_from_home1(home1, pose, t, block=True)
                    if self.robotWindowGui.measure_check.isChecked():
                        data_i = self.alinearGui.measure()
                        self.sweep_ascan[j + i * nx, :, :] = data_i
                        for k in range(n_focal_laws):
                            self.alinearGui.ascan_plots[k].setData(np.abs(data_i[k, :]),
                                                                   pen=pg.mkPen(COLORS[k], width=2))
                        idx_echo = self.alinearGui.detect_first_echo(data_i)
                        t_idx = idx_echo[:, 0]
                        for k1, vl in enumerate(self.alinearGui.vertical_lines):
                            vl.setValue(t_idx[k1])
                    time.sleep(t2)
                print("----------------------------------")
        #############################################################################
        self.robotWindowGui.lbl_sweep_state.setText("Sweep completed")
        self.robotWindowGui.lbl_sweep_state.setStyleSheet("color: green;")
        fl_name1 = self.robotWindowGui.file_name_text_edit.toPlainText()
        fl_name2 = fl_name1 + "_degrees"
        if self.robotWindowGui.sweep_xz_check.isChecked():
            np.save(fl_name1, self.sweep_xz)
            np.save(fl_name2, self.theta_xz)
        else:
            np.save(fl_name1, self.sweep_ascan)
            np.save(fl_name2, self.theta_xy)

    def sweep_stop(self):
        self.stop_sweep = True

    def get_ascan_elements_sweep_xz(self):
        ascan_str = self.robotWindowGui.focal_laws_text_edit.toPlainText()
        ascan_str_list = ascan_str.split(',')
        ascan_int_list = [int(e) for e in ascan_str_list]
        ascan_np_list = np.array(ascan_int_list)
        return ascan_np_list


class RobotWindowGui(QMainWindow):
    move_keys_commands = {Qt.Key_R: 'mov_thrd = run zp()',
                          Qt.Key_F: 'mov_thrd = run zm()',
                          Qt.Key_A: 'mov_thrd = run yp()',
                          Qt.Key_D: 'mov_thrd = run ym()',
                          Qt.Key_W: 'mov_thrd = run xp()',
                          Qt.Key_S: 'mov_thrd = run xm()',
                          Qt.Key_O: 'mov_thrd = run rzp()',
                          Qt.Key_L: 'mov_thrd = run rzm()',
                          Qt.Key_U: 'mov_thrd = run ryp()',
                          Qt.Key_Y: 'mov_thrd = run rym()',
                          Qt.Key_J: 'mov_thrd = run rxp()',
                          Qt.Key_H: 'mov_thrd = run rxm()'}

    # self.interpreterHelper = rh.InterpreterHelper(IP_ROBOT, IP_PC)

    def __init__(self, alinearGui):
        super().__init__()

        # self.sweep_ascan = None

        self.pose_combinations = None
        self.delta_z = 0.0
        self.ui = uic.loadUi('manejo_robot.ui', self)
        self.move(900, 100)
        self.alinearGui = alinearGui
        # self.fmcView = fmcView
        self.robot_interpreter = rh.InterpreterHelper(IP_ROBOT, IP_PC)
        self.home = HOME_POSE
        ###################### CHECKBOXES #####################
        # self.new_home_check.stateChanged.connect(self.set_current_delta_z)
        # self.new_home_cylinder_check.stateChanged.connect(self.set_current_delta_z_2)
        ####################### BOTONES #######################
        self.ui.connect_robot_pushb.clicked.connect(self.connect_robot)
        self.ui.check_tcp_offset_pushb.clicked.connect(self.get_and_show_tcp_offset)
        self.ui.set_tcp_offset_pushb.clicked.connect(self.set_offset_tcp)
        self.ui.go_home_pushb.clicked.connect(self.go_home)
        self.sweep_theta_pushb.clicked.connect(self.start_loop)
        self.save_home_p_pushb.clicked.connect(self.save_home_p)
        self.set_initial_home_btn.clicked.connect(self.set_initial_home)
        self.save_delta_z_pushb.clicked.connect(self.save_delta_z)
        self.reconnect_robot_pushb.clicked.connect(self.reconnect_robot)
        #########################################################

        self.worker = None
        self.ascan_elements = self.get_ascan_elements_sweep_xz()

        [self.plot_sweep_idx.getPlotItem().addItem(pg.PlotDataItem()) for i in self.ascan_elements]
        self.ascan_plots = self.plot_sweep_idx.getPlotItem().curves

    def set_current_delta_z(self, state):
        if state == 2:
            home_init = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\home_delta_z.npy')
            delta_z = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\delta_z.npy')
            delta_z = delta_z.item()
            z = home_init[2]
        else:
            z = 0
            delta_z = 0
        print('DELTA_Z!!! {}'.format(delta_z))
        print('TYPE DELTA_Z!!! {}'.format(type(delta_z)))
        self.fmcView.home_z_lbl.setText('home_z: {}'.format(z))
        self.fmcView.set_new_delta_z_spinb.setValue(delta_z)  # establecemos valor inicial de delta_z

    # def set_current_delta_z_2(self, state):
    #     if state == 2:
    #         home_init = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\home_delta_z_cyl_1.npy')
    #         delta_z = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\delta_z_cyl.npy')
    #         delta_z = delta_z.item()
    #         z = home_init[2]
    #     else:
    #         z = 0
    #         delta_z = 0
    #     self.fmcView.home_z_lbl.setText('home_z: {}'.format(z))
    #     self.fmcView.set_new_delta_z_spinb.setValue(delta_z)  # establecemos valor inicial de delta_z

    def get_ascan_elements_sweep_xz(self):
        ascan_str = self.focal_laws_text_edit.toPlainText()
        ascan_str_list = ascan_str.split(',')
        print(ascan_str_list)
        ascan_int_list = [int(e) for e in ascan_str_list]
        ascan_np_list = np.array(ascan_int_list)
        return ascan_np_list

    def get_list_of_positions(self):
        n = 20  # número de valores en cada rango
        dz = (12, 28)
        rx = (0, 27)
        ry = (0, 27)
        dz_range = np.random.uniform(dz[0], dz[1], n)
        rx_range = np.random.uniform(rx[0], rx[1], n)
        ry_range = np.random.uniform(ry[0], ry[1], n)
        self.pose_combinations = list(zip(dz_range, rx_range, ry_range))

    def start_loop(self):
        if self.worker is None or not self.worker.isRunning():
            self.worker = LoopWorker(self, self.alinearGui)
            self.worker.start()

    def save_home_p(self):
        s = self.ui.home_p_lbl.text()
        np.save("home_p.npy", s)
        # s1 = np.load('home_p.npy')

    def is_aligned(self, message):
        self.ui.is_aligned_lbl.setText(message)
        if message == "Aligned":
            self.robot_interpreter.execute_command('socket_send_string(to_str(get_actual_tcp_pose()))')
            s = self.robot_interpreter.listen_conn.recv(128).decode()
            s = rh.format_pose_string(s)
            self.ui.home_p_lbl.setText("Home_p position would be: " + str(s))
        elif message == "Non-Aligned":
            self.ui.home_p_lbl.setText("--")

    def connect_robot(self):
        self.robot_interpreter.start_interpreter_mode()
        self.robot_interpreter.connect()
        time.sleep(0.5)
        self.robot_interpreter.start_listening()
        time.sleep(0.5)
        self.get_actual_tcp_pose()
        self.make_ur_threads()
        tcp_offset_0 = TCP_OFFSET_0
        self.set_offset_tcp_0(tcp_offset_0)
        ########## OBTENIENDO VALOR DE FUERZA ACTUAL ##########
        self.read_force_and_z()
        ########## INICIALIZO LISTA DE COMBINACIONES z-rx-ry#######
        self.get_list_of_positions()

    def reconnect_robot(self):
        self.robot_interpreter.reconnect()

    def get_actual_tcp_pose(self):
        self.robot_interpreter.execute_command('socket_send_string(to_str(get_actual_tcp_pose()))')
        s = self.robot_interpreter.listen_conn.recv(128).decode()
        # print('get_actual_tcp_pose: {}'.format(s))
        s = rh.format_pose_string(s)
        self.ui.txt_actual_tcp.setText("Current TCP position: " + str(s))

    def go_home(self):
        t = 10
        if self.new_home_check.isChecked():
            if self.new_home_cylinder_check.isChecked():
                home_init = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\home_delta_z.npy')
            else:
                home_init = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\home_delta_z.npy')
            home_init = self.return_rotvec_pose(home_init)
        else:
            home_init = self.return_rotvec_pose(self.home)
        trg_pose_str = 'movej({},t={})'.format(home_init, t)
        self.robot_interpreter.execute_command(trg_pose_str)
        self.get_actual_tcp_pose()

    def get_and_show_tcp_offset(self):
        s = self.robot_interpreter.get_tcp_offset()
        self.ui.txt_offset_tcp.setText("Current TCP offset: " + str(s))

    def set_offset_tcp_0(self, tcp_pose0):
        tcp_pose_str = self.return_rotvec_pose(tcp_pose0)
        tcp_pose_str = 'set_tcp(' + tcp_pose_str + ')'
        self.robot_interpreter.execute_command(tcp_pose_str)
        self.get_and_show_tcp_offset()

    def set_offset_tcp(self):
        x = self.x_spinbox.value()
        y = self.y_spinbox.value()
        z = self.z_spinbox.value()
        a1 = self.a1_spinbox.value()
        a2 = self.a2_spinbox.value()
        a3 = self.a3_spinbox.value()

        values = [x, y, z, a1, a2, a3]
        tcp_pose_str = self.return_rotvec_pose(values)
        tcp_pose_str = 'set_tcp(' + tcp_pose_str + ')'

        self.robot_interpreter.execute_command(tcp_pose_str)
        self.get_and_show_tcp_offset()
        # self.ui.txt_offset_tcp.setText(tcp_pose_str)

    def set_home(self, home):
        self.home = home

    @staticmethod
    def return_rotvec_pose(trg_pose):
        trg_pose = np.array(trg_pose, dtype=float)
        trg_pose[0:3] = trg_pose[0:3] / 1000  # Cambio los tres primeros elementos a metros
        r = Rotation.from_euler('xyz', trg_pose[3:], degrees=True)
        rotvec = r.as_rotvec()
        trg_pose2 = np.concatenate((trg_pose[0:3], rotvec))
        trg_pose_str = 'p[' + ','.join(map(str, np.around(trg_pose2, 3))) + ']'
        # print(trg_pose_str)
        return trg_pose_str
        # self.execute_command(trg_pose_str)

    def get_z_displacement(self):
        new_force = self.force_spinb.value()
        self.robot_interpreter.execute_command('kill force_thrd')
        self.robot_interpreter.execute_command(rh.ur_threads['contact_test'].format(20))
        self.robot_interpreter.execute_command('force_thrd = run contact_test()')
        ########## HACIA POSICIÓN INICIAL ##########
        z0 = self.initial_z_spinb.value()
        delta_z = self.delta_z_spinb.value()
        t = 8
        pose = [self.home[0], self.home[1], z0, self.home[3], self.home[4], self.home[5]]
        print(pose)
        # self.robot_interpreter.move_from_home(pose, t, block=False)
        self.z_displacement_lbl.setText('z current position: ' + str(z0) + ' mm')
        self.robot_interpreter.move_to_pose(pose, t, block=False)
        ########## DESPLAZAMIENTO POR PASOS ##########
        # time.sleep(t)
        zi = z0
        t = 2
        bool_warn = True
        while bool_warn:
            zi = zi - delta_z
            pose = [self.home[0], self.home[1], zi, self.home[3], self.home[4], self.home[5]]
            self.z_displacement_lbl.setText('z current position: ' + str(zi) + ' mm')
            self.robot_interpreter.move_to_pose(pose, t, block=True)
            time.sleep(t)
            str_warn = self.robot_interpreter.listen_conn.recv(128).decode()
            if str_warn == 'contact halt':
                bool_warn = False
        self.z_displacement_lbl.setText('z final position: ' + str(zi) + ' mm')
        # restablezcon max force
        # self.robot_interpreter.execute_command('kill force_thrd')
        # self.robot_interpreter.execute_command(rh.ur_threads['contact_test'].format(MAX_FORCE))
        # self.robot_interpreter.execute_command('force_thrd = run contact_test()')

    def set_initial_home(self):
        ########## Getting current z position ##########
        actual_tcp = self.robot_interpreter.get_actual_tcp_pose()
        x, y, z = actual_tcp[0:3]
        self.set_home([x, y, z, self.home[3], self.home[4], self.home[5]])
        self.initial_home_lbl.setText(str(np.around(self.home, 2)))

    def get_relative_pose_to_home(self):
        if self.new_home_cylinder_check.isChecked():
            home_init = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\home_delta_z.npy')
        else:
            home_init = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\home_delta_z.npy')
        print('HOME_INIT: {}'.format(home_init))
        home_rot = Rotation.from_euler('z', np.pi / 4)  # Represento una rotación de 45 grados en el eje z
        actual_tcp = self.robot_interpreter.get_actual_tcp_pose()  # Posición actual del array
        actual_rot = Rotation.from_rotvec(
            actual_tcp[3:])  # Instancia de Rotation a partir del vector de rotación actual
        rot_rel = home_rot.inv() * actual_rot
        rot_rel = rot_rel.as_euler('xyz', degrees=True)
        #########
        xyz_rel = np.array(actual_tcp[0:3]) - np.array(home_init[0:3])
        pose_rel = np.around(np.concatenate((xyz_rel, rot_rel)), 2)
        self.pose_rel_lbl.setText('Relative position: ' + str(np.round(xyz_rel, 2)))
        self.orientation_rel_lbl.setText('Relative orientation: ' + str(np.round(rot_rel, 2)))
        return xyz_rel, rot_rel

    def get_relative_pose_to_home_II(self):
        home_init = self.home
        home_rot = Rotation.from_euler('z', np.pi / 4)
        actual_tcp = self.robot_interpreter.get_actual_tcp_pose()
        actual_rot = Rotation.from_rotvec(actual_tcp[3:])
        rot_rel = home_rot.inv() * actual_rot
        rot_rel = rot_rel.as_euler('xyz', degrees=True)
        #########
        xyz_rel = np.array(actual_tcp[0:3]) - np.array(home_init[0:3])
        pose_rel = np.round(np.concatenate((xyz_rel, rot_rel)), 2)
        self.pose_rel_lbl.setText('Relative position: ' + str(np.round(xyz_rel, 2)))
        self.orientation_rel_lbl.setText('Relative orientation: ' + str(np.round(rot_rel, 2)))
        return xyz_rel, rot_rel

    def get_relative_pose_to_home_III(self):
        if self.new_home_cylinder_check.isChecked():
            home_init = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\home_delta_z.npy')
        else:
            home_init = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\home_delta_z.npy')
        z_new = float(self.fmcView.home_z_lbl.text().split(':')[1])
        home_init[2] = z_new
        print('HOME_INIT: {}'.format(home_init))
        home_rot = Rotation.from_euler('z', np.pi / 4)  # Represento una rotación de 45 grados en el eje z
        actual_tcp = self.robot_interpreter.get_actual_tcp_pose()  # Posición actual del array
        actual_rot = Rotation.from_rotvec(
            actual_tcp[3:])  # Instancia de Rotation a partir del vector de rotación actual
        rot_rel = home_rot.inv() * actual_rot
        rot_rel = rot_rel.as_euler('xyz', degrees=True)
        #########
        xyz_rel = np.array(actual_tcp[0:3]) - np.array(home_init[0:3])
        pose_rel = np.around(np.concatenate((xyz_rel, rot_rel)), 2)
        self.pose_rel_lbl.setText('Relative position: ' + str(np.round(xyz_rel, 2)))
        self.orientation_rel_lbl.setText('Relative orientation: ' + str(np.round(rot_rel, 2)))
        return xyz_rel, rot_rel

    def read_force_and_z(self):
        ########## Getting current z position ##########
        self.robot_interpreter.execute_command('socket_send_string(to_str(force()))')
        current_force = self.robot_interpreter.listen_conn.recv(128).decode()
        self.current_force_lbl.setText('Current force: ' + current_force + ' [N]')
        actual_tcp = self.robot_interpreter.get_actual_tcp_pose()
        if self.new_home_check.isChecked():
            if self.new_home_cylinder_check.isChecked():
                home_init = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\home_delta_z.npy')
            else:
                home_init = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\home_delta_z.npy')
        else:
            home_init = self.home
        self.delta_z = np.round(home_init[2] - actual_tcp[2], 3)
        self.z_displacement_lbl.setText('Δz: ' + str(self.delta_z) + ' [mm]')

    def save_delta_z(self):
        delta_z = self.delta_z + PROBE_GAP
        np.save('delta_z', delta_z)
        np.save('home_delta_z', self.home)

    def make_ur_threads(self):
        """Define los threads que se ejecutan para mover el robot, y que se matan para
        parar el movimiento. También está el contact_test thread para parar si hay contacto"""

        time.sleep(0.1)
        self.robot_interpreter.clear()
        # tra_speed = self.ui.slid_tra_speed.value()/1000 # de mm a metros
        # rot_speed = (np.pi/180) * self.ui.slid_rot_speed.value() # pasar de grados a radianes
        # acc = self.ui.slid_acc.value()/1000
        tra_speed = 50 / 1000  # de mm a metros
        rot_speed = (np.pi / 180) * 30  # pasar de grados a radianes
        acc = 50 / 1000
        self.robot_interpreter.execute_command(rh.ur_threads['contact_test'].format(MAX_FORCE))

        # esto del COS45 es por el sistema de coordenadas del robot que esta a 45 grados de la mesa
        # velocidades de traslacion
        self.robot_interpreter.execute_command(rh.ur_threads['xp'].format(
            tra_speed * COS45, tra_speed * COS45, acc, SPEEDL_TIMEOUT))
        self.robot_interpreter.execute_command(rh.ur_threads['xm'].format(
            -tra_speed * COS45, -tra_speed * COS45, acc, SPEEDL_TIMEOUT))
        self.robot_interpreter.execute_command(rh.ur_threads['yp'].format(
            -tra_speed * COS45, tra_speed * COS45, acc, SPEEDL_TIMEOUT))
        self.robot_interpreter.execute_command(rh.ur_threads['ym'].format(
            tra_speed * COS45, -tra_speed * COS45, acc, SPEEDL_TIMEOUT))
        self.robot_interpreter.execute_command(rh.ur_threads['zp'].format(tra_speed, acc, SPEEDL_TIMEOUT))
        self.robot_interpreter.execute_command(rh.ur_threads['zm'].format(-tra_speed, acc, SPEEDL_TIMEOUT))
        # velocidades de rotacion
        self.robot_interpreter.execute_command(rh.ur_threads['rxp'].format(
            rot_speed * COS45, rot_speed * COS45, acc, SPEEDL_TIMEOUT))
        self.robot_interpreter.execute_command(rh.ur_threads['rxm'].format(
            -rot_speed * COS45, -rot_speed * COS45, acc, SPEEDL_TIMEOUT))
        self.robot_interpreter.execute_command(rh.ur_threads['ryp'].format(
            -rot_speed * COS45, rot_speed * COS45, acc, SPEEDL_TIMEOUT))
        self.robot_interpreter.execute_command(rh.ur_threads['rym'].format(
            rot_speed * COS45, -rot_speed * COS45, acc, SPEEDL_TIMEOUT))
        self.robot_interpreter.execute_command(rh.ur_threads['rzp'].format(rot_speed, acc, SPEEDL_TIMEOUT))
        self.robot_interpreter.execute_command(rh.ur_threads['rzm'].format(-rot_speed, acc, SPEEDL_TIMEOUT))

        # ejecuta el contact_test thread
        self.robot_interpreter.execute_command('thrd = run contact_test()')

    def keyPressEvent(self, event):
        if event.key() in RobotWindowGui.move_keys_commands.keys():
            if not event.isAutoRepeat():
                self.robot_interpreter.execute_command(RobotWindowGui.move_keys_commands[event.key()])
                self.read_force_and_z()
                if self.new_home_check.isChecked():
                    self.get_relative_pose_to_home()
                else:
                    self.get_relative_pose_to_home_II()

    def keyReleaseEvent(self, event):
        if event.key() in RobotWindowGui.move_keys_commands.keys():
            if not event.isAutoRepeat():
                # stop_acc = self.ui.slid_stop_acc.value() / 1000
                stop_acc = 100 / 1000
                stop_command = 'kill mov_thrd stopl({}) socket_send_string(to_str(get_actual_tcp_pose()))'.format(
                    stop_acc)
                self.robot_interpreter.execute_command(stop_command)
                msg = self.robot_interpreter.listen_conn.recv(128).decode()
                print(msg)
                self.read_force_and_z()
                if self.new_home_check.isChecked():
                    self.get_relative_pose_to_home()
                else:
                    self.get_relative_pose_to_home_II()
                try:
                    s = rh.format_pose_string(msg)
                    self.ui.txt_actual_tcp.setText(str(s))
                except:
                    # chapuza todo
                    self.ui.txt_actual_tcp.setText(msg)
                    self.robot_interpreter.disconnect()
                    self.connect_robot()
                    self.make_ur_threads()


class FmcView(QMainWindow):
    def __init__(self, alinearGui, robotWindowGui):
        super().__init__()

        self.idx_plotitem = None
        self.alinearGui = alinearGui
        self.robotWindowGui = robotWindowGui
        self.ui = uic.loadUi('fmc_image_view_gui.ui', self)
        self.fmc_imageview.getView().setAspectLocked(ratio=20)
        colors = [(255, 255, 255), (52, 152, 219), (244, 208, 63), (220, 118, 51), (255, 0, 0)]
        cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 5), color=colors)
        self.fmc_imageview.setColorMap(cmap)
        ####################### CHECKBOX #######################
        self.cylinder_check.stateChanged.connect(self.set_current_delta_z)
        self.cylinder_U_check.stateChanged.connect(self.set_current_delta_z_2)
        self.plane2_check.stateChanged.connect(self.plane2_selected)
        self.sphere_check.stateChanged.connect(self.sphere_selected)
        ####################### SPINBOX #######################
        if self.robotWindowGui.new_home_cylinder_check.isChecked():
            home_init = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\home_delta_z.npy')
            delta_z = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\delta_z.npy')
        else:
            home_init = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\home_delta_z.npy')
            delta_z = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\delta_z.npy')
        delta_z = delta_z.item()
        z = home_init[2]
        self.home_z_lbl.setText('home_z: {}'.format(z))
        self.set_new_delta_z_spinb.setValue(delta_z)
        print('AFTER INITIALIZING:')
        print('delta_z: {}'.format(delta_z))
        print('z: {}'.format(z))
        self.previous_value = delta_z
        self.set_new_delta_z_spinb.valueChanged.connect(self.set_delta_z_home)
        ####################### BOTONES #######################
        self.acquire_btn.clicked.connect(self.acquire)
        self.plot_ascan_btn.clicked.connect(self.plot_loaded_ascan)
        self.set_new_delta_z_btn.clicked.connect(self.set_new_delta_z)
        ######################### MENÚ ########################
        self.actionLoad_from_ascan.triggered.connect(self.load_ascan)
        #########################################################

    def sphere_selected(self, state):
        if state == 2:
            self.cylinder_U_check.setDisabled(True)
            self.cylinder_check.setDisabled(True)
            self.shape_cilu_comboBox.setDisabled(True)
            self.plane2_check.setDisabled(True)
        else:
            self.cylinder_U_check.setDisabled(False)
            self.cylinder_check.setDisabled(False)
            self.shape_cilu_comboBox.setDisabled(False)
            self.plane2_check.setDisabled(False)

    def plane2_selected(self, state):
        if state == 2:
            self.cylinder_U_check.setDisabled(True)
            self.cylinder_check.setDisabled(True)
            self.shape_cilu_comboBox.setDisabled(True)
            self.sphere_check.setDisabled(True)
        else:
            self.cylinder_U_check.setDisabled(False)
            self.cylinder_check.setDisabled(False)
            self.shape_cilu_comboBox.setDisabled(False)
            self.sphere_check.setDisabled(False)

    def set_current_delta_z(self, state):
        if state == 2:
            self.cylinder_U_check.setDisabled(True)
            self.plane2_check.setDisabled(True)
            self.shape_cilu_comboBox.setDisabled(True)
            self.sphere_check.setDisabled(True)
            home_init = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\home_delta_z.npy')
            delta_z = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\delta_z.npy')
            delta_z = delta_z.item()
            z = home_init[2]
        else:
            self.cylinder_U_check.setDisabled(False)
            self.plane2_check.setDisabled(False)
            self.shape_cilu_comboBox.setDisabled(False)
            self.sphere_check.setDisabled(False)
            home_init = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\home_delta_z.npy')
            delta_z = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\delta_z.npy')
            delta_z = delta_z.item()
            z = home_init[2]
        print('DELTA_Z!!! {}'.format(delta_z))
        print('TYPE DELTA_Z!!! {}'.format(type(delta_z)))
        self.set_new_delta_z_spinb.setValue(delta_z)  # establecemos valor inicial de delta_z
        self.home_z_lbl.setText('home_z: {}'.format(z))

    def set_current_delta_z_2(self, state):
        if state == 2:
            self.cylinder_check.setDisabled(True)
            self.plane2_check.setDisabled(True)
            self.sphere_check.setDisabled(True)
            home_init = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\home_delta_z.npy')
            delta_z = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\delta_z.npy')
            delta_z = delta_z.item()
            z = home_init[2]
        else:
            self.cylinder_check.setDisabled(False)
            self.plane2_check.setDisabled(False)
            self.sphere_check.setDisabled(False)
            home_init = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\home_delta_z.npy')
            delta_z = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\delta_z.npy')
            delta_z = delta_z.item()
            z = home_init[2]
        print('DELTA_Z!!! {}'.format(delta_z))
        print('TYPE DELTA_Z!!! {}'.format(type(delta_z)))
        self.set_new_delta_z_spinb.setValue(delta_z)  # establecemos valor inicial de delta_z
        self.home_z_lbl.setText('home_z: {}'.format(z))

    def set_delta_z_home(self, value):
        # if self.cylinder_check.isChecked() or self.cylinder_U_check.isChecked():
        #     delta_z0 = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\delta_z_cyl.npy')
        # else:
        #     delta_z0 = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\delta_z.npy')
        print('----------0----------')
        # delta_z0 = self.set_new_delta_z_spinb.value()
        print('FROM: set_delta_z_home')
        # print('delta_z0: {}'.format(delta_z0))
        # delta_z1 = self.set_new_delta_z_spinb.value()
        # print('delta_z1: {}'.format(delta_z1))
        delta_dif = value - self.previous_value
        print('delta_dif: {}'.format(delta_dif))
        home_z = float(self.home_z_lbl.text().split(':')[1])
        print('home_z: {}'.format(home_z))
        home_z = float(self.home_z_lbl.text().split(':')[1]) + delta_dif
        print('home_z: {}'.format(home_z))
        self.home_z_lbl.setText('home_z: {}'.format(str(np.round(home_z, 2))))
        self.previous_value = value
        print('----------0----------')

    def set_new_delta_z(self):
        delta_z = self.set_new_delta_z_spinb.value()
        np.save("delta_z_corregido.npy", delta_z)

    def acquire(self):
        self.config_focal_law()
        self.alinearGui.sitau.ST_SetAcqTime(self.alinearGui.adquisition_time_spinbox.value())
        self.alinearGui.sitau.ST_SetGain(self.alinearGui.gain_spinbox.value())
        acq_counter = self.alinearGui.sitau.ST_Trigger(2)
        self.result, self.ascan = self.alinearGui.sitau.ST_GetBuffer_LastImage(0)
        print(self.ascan.shape)
        ###################################################################
        self.update_plot()
        ###################################################################
        tof_idx = self.get_tof_2()
        self.plot_idx(tof_idx)
        print('--------------------0--------------------')

    def config_focal_law(self):
        n_ch = self.alinearGui.sitau.ST_GetChannelNumber()
        self.alinearGui.sitau.ST_DeleteFocalLaws()
        delay = np.zeros(n_ch, dtype=np.float32)
        i = self.tx_index_spinb.value()
        tx_enable = np.zeros(n_ch, dtype=np.int32)
        tx_enable[i] = 1
        rx_enable = np.ones(n_ch, dtype=np.int32)
        self.alinearGui.sitau.ST_AddFocalLaw(tx_enable, rx_enable, delay, delay, n_ch)

    def update_plot(self):
        img = np.abs(self.ascan)
        self.fmc_imageview.setImage(img, scale=[1, 1], levels=(0, img.max()))

    def plot_idx(self, t_idx):
        imv = self.fmc_imageview.getView()
        # idx = self.detect_first_echo()
        if self.idx_plotitem is not None:
            imv.removeItem(self.idx_plotitem)
        self.idx_plotitem = pg.PlotDataItem(t_idx, pen=pg.mkPen(width=3, color=(0, 255, 0)))
        imv.addItem(self.idx_plotitem)

    def get_tof(self, fs=40, c1=1.48):
        ##### OBTENGO MI POSICÓN z ACTUAL #####
        xyz_rel, rot_rel = self.robotWindowGui.get_relative_pose_to_home()
        z = xyz_rel[2]
        print('xyz_rel: {}'.format(xyz_rel))
        print('rot_rel: {}'.format(rot_rel))
        print('z: {}'.format(z))
        delta_z = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\delta_z.npy')
        # delta_z_cyl = np.load('temp/delta_z_cyl.npy')
        #################################################
        rad_cyl_u = 0;
        rad_cyl_u_ext = 0
        if self.shape_cilu_comboBox.currentText() == "CIL_U25":
            rad_cyl_u = CYLINDER_RAD_U25
            rad_cyl_u_ext = CYLINDER_RAD_U25_EXT
        elif self.shape_cilu_comboBox.currentText() == "CIL_U40":
            rad_cyl_u = CYLINDER_RAD_U40
            rad_cyl_u_ext = CYLINDER_RAD_U40_EXT
        #################################################
        if self.cylinder_check.isChecked():
            print('delta_z: {}'.format(delta_z))
            z1 = z + delta_z
            print('z1: {}'.format(z1))
            z1 = z1 - CYLINDER_RAD
            print('CYLINDER_RAD: {}'.format(CYLINDER_RAD))
        elif self.cylinder_U_check.isChecked():
            print('delta_z: {}'.format(delta_z))
            # print('delta_z: {}'.format(delta_z))
            z1 = z + delta_z
            # z1 = z + delta_z
            print('z1: {}'.format(z1))
            z1 = z1 - rad_cyl_u_ext
            print('CYLINDER_RAD_U_EXT: {}'.format(rad_cyl_u_ext))
        else:  # para el caso del plano
            print('delta_z: {}'.format(delta_z))
            z1 = z + delta_z
        print('z1(2): {}'.format(z1))
        print('CYLINDER_RAD_U: {}'.format(rad_cyl_u))
        ##### CÁLCULO DE TOF TEÓRICOS #####
        array_coords = np.array(u3d.array_coordinates_list(11, 11, 1, 1))
        i = self.tx_index_spinb.value()
        array_coords[:, 0] = array_coords[:,
                             0] * -1  # todo MODIFICAR EN BASE AL TCP OFFSET!!!!!!! rotacion 180 alrededir del eje y
        tx_coords = array_coords[i, :].reshape((1, 3))
        rx_coords = array_coords
        if self.cylinder_check.isChecked():
            fun = ifaz3d.return_pitch_catch_cylfun_circmirror(tx_coords, rx_coords, 1)
            d = fun(CYLINDER_RAD, np.array([0, xyz_rel[1], z1]), ['xyz', rot_rel])
        elif self.cylinder_U_check.isChecked():
            fun = ifaz3d.return_pitch_catch_cylfun_circmirror(tx_coords, rx_coords, -1)
            d = fun(rad_cyl_u, np.array([0, xyz_rel[1], z1]), ['xyz', rot_rel])
        else:
            fun = ifaz3d.return_pitch_catch_plane_fun(tx_coords, rx_coords)
            d = fun(np.array([0, 0, z1]), ['xyz', rot_rel])  # función especular
        tof_idx = d * fs / c1
        return tof_idx

    def get_tof_2(self, fs=40, c1=1.48):
        print('----------0----------')
        print('FROM get_tof_2:')
        ##### OBTENGO MI POSICÓN z ACTUAL #####
        xyz_rel, rot_rel = self.robotWindowGui.get_relative_pose_to_home()  # obtengo el home establecido en la medicion inicialm del delta
        # xyz_rel, rot_rel = self.get_relative_pose_to_home_III()
        z = xyz_rel[2]
        print('xyz_rel: {}'.format(xyz_rel))
        print('rot_rel: {}'.format(rot_rel))
        print('z: {}'.format(z))
        delta_z = self.set_new_delta_z_spinb.value()  # valor actualmente en el spinbox
        #################################################
        rad_cyl_u = 0; rad_cyl_u_ext = 0
        if self.shape_cilu_comboBox.currentText() == "CIL_U25":
            rad_cyl_u = CYLINDER_RAD_U25
            rad_cyl_u_ext = CYLINDER_RAD_U25_EXT
        elif self.shape_cilu_comboBox.currentText() == "CIL_U40":
            rad_cyl_u = CYLINDER_RAD_U40
            rad_cyl_u_ext = CYLINDER_RAD_U40_EXT
        #################################################
        if self.cylinder_check.isChecked():
            print('delta_z: {}'.format(delta_z))
            z1 = z + delta_z
            print('z1: {}'.format(z1))
            z1 = z1 - CYLINDER_RAD_2
            print('CYLINDER_RAD: {}'.format(CYLINDER_RAD_2))
        elif self.cylinder_U_check.isChecked():
            print('delta_z: {}'.format(delta_z))
            z1 = z + delta_z
            print('z1: {}'.format(z1))
            if not self.inverted_check.isChecked():
                z1 = z1 - rad_cyl_u_ext

            print('CYLINDER_RAD_U_EXT: {}'.format(rad_cyl_u_ext))
            print('CYLINDER_RAD_U: {}'.format(rad_cyl_u))
        elif self.plane2_check.isChecked():
            print('delta_z: {}'.format(delta_z))
            z1 = z + delta_z
            print('z1: {}'.format(z1))
            PLANO_2 = self.plane2_spin.value()
            z1 = z1 - PLANO_2
            print('PLANO: {}'.format(PLANO_2))
        elif self.sphere_check.isChecked():
            print('delta_z: {}'.format(delta_z))
            z1 = z + delta_z
            print('z1: {}'.format(z1))
            z1 = z1 - SPHERE_RAD
            print('PLANO: {}'.format(SPHERE_RAD))
        else:  # para el caso del plano
            print('delta_z: {}'.format(delta_z))
            z1 = z + delta_z
        print('z1(2): {}'.format(z1))
        ##### CÁLCULO DE TOF TEÓRICOS #####
        array_coords = np.array(u3d.array_coordinates_list(11, 11, 1, 1))
        i = self.tx_index_spinb.value()
        array_coords[:, 0] = array_coords[:, 0] * -1  # todo MODIFICAR EN BASE AL TCP OFFSET!!!!!!! rotacion 180 alrededir del eje y
        tx_coords = array_coords[i, :].reshape((1, 3))
        rx_coords = array_coords
        if self.cylinder_check.isChecked():
            if self.interporayos_check.isChecked():
                angs = 30 * (np.linspace(0, 1, 200)) ** 0.7
                fun = ifaz3d.return_cyl_pitch_catch_interporays_fun(tx_coords, rx_coords, angs, 1)
            else:
                fun = ifaz3d.return_pitch_catch_cylfun_circmirror(tx_coords, rx_coords, 1)
            d = fun(CYLINDER_RAD_2, np.array([0, xyz_rel[1], z1]), ['xyz', rot_rel])

        elif self.cylinder_U_check.isChecked():
            if self.inverted_check.isChecked():
                curv = 1
                rad = rad_cyl_u_ext
            else:
                curv = -1
                rad = rad_cyl_u
            if self.interporayos_check.isChecked():
                angs = 30 * (np.linspace(0, 1, 200)) ** 0.7
                fun = ifaz3d.return_cyl_pitch_catch_interporays_fun(tx_coords, rx_coords, angs, curv)
            else:
                fun = ifaz3d.return_pitch_catch_cylfun_circmirror(tx_coords, rx_coords, curv)
            d = fun(rad, np.array([0, xyz_rel[1], z1]), ['xyz', rot_rel])
        elif self.sphere_check.isChecked():
            if self.interporayos_check.isChecked():
                angs = 30 * (np.linspace(0, 1, 200)) ** 0.7
                fun = ifaz3d.return_sphere_pitch_catch_interporays_fun(tx_coords, rx_coords, angs, 1)
            else:
                fun = ifaz3d.return_pitch_catch_sphere_fun_circmirror(tx_coords, rx_coords, 1)
            d = fun(SPHERE_RAD, np.array([0, xyz_rel[1], z1]), ['xyz', rot_rel])
        else:
            fun = ifaz3d.return_pitch_catch_plane_fun(tx_coords, rx_coords)
            d = fun(np.array([0, 0, z1]), ['xyz', rot_rel])  # función especular
        tof_idx = d * fs / c1
        return tof_idx

    def get_relative_pose_to_home_III(self):  # currently not in use
        if self.robotWindowGui.new_home_cylinder_check.isChecked():
            home_init = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\home_delta_z.npy')
        else:
            home_init = np.load(r'C:\MarceloLarrea\utimag_Marcelo\SITAU_GUIs\Alinear_ST1_IV\home_delta_z.npy')
        z_new = float(self.home_z_lbl.text().split(':')[1])
        home_init[2] = z_new
        print('HOME_INIT: {}'.format(home_init))
        home_rot = Rotation.from_euler('z', np.pi / 4)  # Represento una rotación de 45 grados en el eje z
        actual_tcp = self.robotWindowGui.robot_interpreter.get_actual_tcp_pose()  # Posición actual del array
        actual_rot = Rotation.from_rotvec(
            actual_tcp[3:])  # Instancia de Rotation a partir del vector de rotación actual
        rot_rel = home_rot.inv() * actual_rot
        rot_rel = rot_rel.as_euler('xyz', degrees=True)
        #########
        xyz_rel = np.array(actual_tcp[0:3]) - np.array(home_init[0:3])
        pose_rel = np.around(np.concatenate((xyz_rel, rot_rel)), 2)
        # self.pose_rel_lbl.setText('Relative position: ' + str(np.round(xyz_rel, 2)))
        # self.orientation_rel_lbl.setText('Relative orientation: ' + str(np.round(rot_rel, 2)))
        return xyz_rel, rot_rel

    def load_ascan(self):
        filename = QFileDialog.getOpenFileName(self, 'Load acquisition from .mat')[
            0]  # Devuelve una tuple, en el lugar 0 está la ruta
        print(filename)
        self.loaded_ascan = np.load(filename)
        #####################################
        self.tx_index_lbl.setEnabled(False)
        self.tx_index_spinb.setEnabled(False)
        self.acquire_btn.setEnabled(False)
        ####################################
        self.plot_ascan_btn.setEnabled(True)
        self.ascan_i_spinb.setEnabled(True)

    def plot_loaded_ascan(self):
        i = self.ascan_i_spinb.value()
        # self.ascan = self.loaded_ascan[i, :, :]
        self.ascan = self.loaded_ascan
        self.update_plot()
        # self.plot_idx()


if __name__ == '__main__':
    app = QApplication([])
    window = AlinearGui()
    window.show()
    app.exec_()
