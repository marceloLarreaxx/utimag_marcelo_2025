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

ASCAN_ELEMENTS = [0, 10, 60, 110, 120]
bin_path = r'C:\MarceloLarrea\utimag_Marcelo\SITAU1ethernet\stfplib_py\BIN\\'

class SitauHelper:
    def __init__(self):
        self.n_focal_laws = None
        self.sitau = stfplib.C_STFPLIB()  # accedo a librerìas en self.sitau para poder utilizar funciones
        self.ascan_elements = ASCAN_ELEMENTS

    def config_focal_laws(self, config_1=True):
        n_ch = self.sitau.ST_GetChannelNumber()
        self.sitau.ST_DeleteFocalLaws()
        delay = np.zeros(n_ch, dtype=np.float32)
        for i in self.ascan_elements:
            tx_enable = np.zeros(n_ch, dtype=np.int32)
            tx_enable[i] = 1
            if config_1:
                rx_enable = tx_enable.copy()
            else:
                rx_enable = np.ones(n_ch, dtype=np.int32)
            self.sitau.ST_AddFocalLaw(tx_enable, rx_enable, delay, delay, n_ch)
        self.n_focal_laws = self.sitau.ST_GetFocalLaw_Number()
        return self.n_focal_laws

    def measure(self, t=1):
        acq_counter = self.sitau.ST_Trigger(t)
        n_samples = self.sitau.ST_GetAScanDataNumber()
        data = np.zeros((self.n_focal_laws, n_samples), dtype=np.int16)
        print('from measure: data.shape: {}'.format(data.shape))
        for i in range(self.n_focal_laws):
            print('self.sitau.ST_GetBuffer_LastImage(i): size: {}'.format(len(self.sitau.ST_GetBuffer_LastImage(i))))
            result, data[i, :] = self.sitau.ST_GetBuffer_LastImage(i)
        return data

    def open_sitau(self):
        if self.sitau.ST_OpenSys(bin_path, "192.168.2.10", 6002, 6008) < 0:
            self.sitau.ST_CloseSys()
            del self.sitau
            os._exit(1)

    def set_ascan_elements(self, new_ascan_elements):
        self.ascan_elements = new_ascan_elements
