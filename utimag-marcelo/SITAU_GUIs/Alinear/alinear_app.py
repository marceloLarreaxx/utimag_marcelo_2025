import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget
from PyQt5.QtCore import QFile, Qt, QTimer
from PyQt5 import uic, QtGui
import pyqtgraph as pg
from SITAU2 import st2lib
from imag2D.pintar import arcoiris_cmap
from utils import first_thr_cross

OPENSYS_ARGS = ('st2lib\\BIN\\', '192.168.2.10', 55902, 50460, 0)
BSCAN_ASPECT_RATIO = 20
DEFAULT_THR = 50
DEFAULT_WINDOW_NUM = 10
ASCAN_ELEMENTS = [0, 10, 60, 110, 120]  # CHEQUEAR !!!!!!!

arcoiris_cmap_pg = pg.ColorMap(None, (255*arcoiris_cmap.colors).astype(int))


class AlinearGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi('alinear_gui_2.ui', self)

        # self.sitau = st2lib.ST2Lib()  # crear objeto sitau que tiene las funciones de control

        # crear el timer que lee periodicamente el buffer y conectarlo a update_plots
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)

        # crear objetos del A-scan plot. Para eso hace una lista en la que cada elementos es un PlotDataItem
        # que se agrega al PlotItem del ascan_widget
        self.ascan_plots = [self.ascan_widget.getPlotItem().addItem(pg.PlotDataItem()) for i in ASCAN_ELEMENTS]
        print(self.ascan_plots)

        # franja para seleccionar un intervalo en el A-scan
        self.lr = pg.LinearRegionItem(values=(0, 1000))
        self.ascan_widget.addItem(self.lr)

        # esto es para que no quede muy alargada la imagen, dadao que suele ser de 128 x 2000 aprox
        self.bscan_widget.getView().setAspectLocked(ratio=BSCAN_ASPECT_RATIO)
        # escala de colores
        self.bscan_widget.setColorMap(arcoiris_cmap_pg)

        # lista de botones que hay que desabilitar cuando se está adquiriendo
        self.disable_widget_list = [self.opensys_pushb, self.closesys_pushb, self.start_pushb,
                                    self.tx_spinbox, self.timer_spinbox, self.pulseamp_spinbox]

        # conexiones varias
        self.start_pushb.clicked.connect(self.start_acquisition)

        # self.lr.sigRegionChangeFinished.connect(self.plot_first_echo_idx)

        self.img = []

        self.show()

    def open_sitau2(self, default_xml='FMC_config.xml', default_fl='FP_un_disparo.txt'):
        err1 = self.sitau.LoadXMLFileConfig(default_xml)
        err2 = self.sitau.OpenSys(*OPENSYS_ARGS)
        err3 = self.sitau.SetEmissionFocalLawFile(default_fl)
        pass

    def start_acquisition(self):
        for x in self.disable_widget_list:
            x.setEnabled(False)
        # setear la amplitud del pulso
        self.sitau.SetPulseAmplitude(self.pulseamp_spinbox.value)
        # adquirir
        self.timer.start(self.timer_spinbox.value)  # setea el periodo del timer
        # self.sitau.Start()

    def stop_acquisition(self):
        for x in self.disable_widget_list:
            x.setEnabled(True)
        # self.sitau.Stop()

    def update_plots(self):
        self.img, index_trigger, index_focal_law, err = self.sitau.GetBuffer_LastAcquiredFocalLaw()
        # pintar
        self.bscan_widget.setImage(np.abs(self.img), levels=(0, self.img.max()))
        self.curves.setData(self.img[ASCAN_ELEMENTS, :])
        for i in ASCAN_ELEMENTS:
            self.ascan_plots[i].setData(self.img[i, :])

    def detect_first_echo(self):
        i1, i2 = self.ascan_widget.lr.getRegion()
        umbral = self.ascan_widget.thr_line.value()
        window_num = DEFAULT_WINDOW_NUM ################## MODIFICAR
        idx = first_thr_cross(self.img, (int(i1), int(i2)), umbral, window_num)
        return idx

    def close_sitau2(self):
        self.sitau.CloseSys()

    def closeEvent(self, event):
        self.sitau.CloseSys()


if __name__ == '__main__':
    app = QApplication([])
    window = AlinearGui()
    window.show()
    app.exec_()