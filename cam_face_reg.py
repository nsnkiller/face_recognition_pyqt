from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread

import sys
import cv2
import numpy as np

from camera_face_reg.Face_Reg import Ui_Form
from camera_face_reg.faces_util import Faces, FacesError


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, faces):
        super().__init__()
        self._run_flag = True
        self._faces_flag = False
        self.faces = faces
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def run(self):
        """
            the entry func of video thread
            capture from camera
        """
        while self._run_flag:
            ret, cv_img = self.cap.read()
            # if the 'face recognition' button pressed
            if self._faces_flag:
                cv_img = self.faces.faces_process(cv_img)
            if ret:
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        self.cap.release()

    def set_faces_flag(self, flag):
        self._faces_flag = flag

    def get_faces_flag(self):
        return self._faces_flag

    def stop(self):
        """
        Sets run flag to False and waits for thread to finish
        """
        self._run_flag = False
        cv2.destroyAllWindows()
        self.wait()


class App(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(App, self).__init__(parent)
        self.setupUi(self)
        self.thread = None

        try:
            self.faces = Faces()
        except FacesError as e:
            print(e)
            sys.exit(-1)

        # buttons slot/signal
        self.OpenCamera_pushButton.clicked.connect(self.open_camera)
        self.CloseCamera_pushButton.clicked.connect(self.close_camera)
        self.FaceReg_pushButton.clicked.connect(self.face_reg)
        self.addFace_pushButton.clicked.connect(self.add_faces)
        self.genFaceLib_pushButton.clicked.connect(self.gen_faces_lib)
        self.getFaceInfo_pushButton.clicked.connect(self.get_faces_names)

    def open_camera(self):
        """
            signal function for open camera button
        """
        if self.thread is None:
            # create the video capture thread
            self.thread = VideoThread(self.faces)
            # connect its signal to the update_image slot
            self.thread.change_pixmap_signal.connect(self.update_image)
            # start the thread
            self.thread.start()
            self.image_label.show()

    def close_camera(self):
        """
            signal function for close camera button
        """
        if self.thread is not None:
            self.thread.stop()
            self.thread = None
            self.image_label.hide()

    def face_reg(self):
        """
            signal function for face recognition button
        """
        if self.thread is not None:
            if self.thread.get_faces_flag():
                self.thread.set_faces_flag(False)
                self.FaceReg_pushButton.setText("开启识别")
            else:
                self.thread.set_faces_flag(True)
                self.FaceReg_pushButton.setText("关闭识别")

    def add_faces(self):
        """
            signal function for add faces button
        """
        openfile_names = QFileDialog.getOpenFileNames(self, 'Select one or more files to open', '', 'Images files(*.jpg)')
        self.faces.faces_encoding_from_files(openfile_names[0])
        self.faces.faces_serialization()

    def gen_faces_lib(self):
        """
            signal function for gen faces library button
        """
        from  PyQt5 import QtCore
        print(QtCore.PYQT_VERSION_STR)

    def get_faces_names(self):
        """
            signal function for get faces button
        """
        print(self.faces.get_faces_info())

    def closeEvent(self, event):
        if self.thread is not None:
            self.thread.stop()
            self.thread = None
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """
            Updates the image_label with a new opencv image
        Args:
            cv_img: a image frame from video camera capture
        """
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """
            Convert from an opencv image to QPixmap
        Args:
            cv_img: a image frame from video camera capture
        """
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        # p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())