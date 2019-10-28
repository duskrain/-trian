# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tiaozhuan.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


# from haoP import Ui_nihao

# 第二界面
class Ui_nihao(QtWidgets.QWidget):
    def __init__(self):
        super(Ui_nihao, self).__init__()
        self.setObjectName("nihao")
        self.resize(400, 300)
        self.setWindowTitle("消息")
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(100, 80, 181, 91))
        font = QtGui.QFont()
        font.setPointSize(55)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label.setText("你好！")


# 第一界面
class Ui_PageB(QtWidgets.QWidget):
    def __init__(self):
        super(Ui_PageB, self).__init__()
        self.setObjectName("nihao")
        self.resize(400, 300)
        self.setWindowTitle("消息")
        self.changeP = QtWidgets.QPushButton(self)
        self.changeP.setGeometry(QtCore.QRect(100, 100, 181, 71))
        self.changeP.setObjectName("changeP")
        self.changeP.setText("跳转")

        self.changeP.clicked.connect(self.haoPa)

    def haoPa(self):
        self.haoN = Ui_nihao()
        self.haoN.show()
        print("@@@@@@@@@@@@@@@@@@@@@@@@@")
        # self.haoN.exec_()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    widget = Ui_PageB()
    widget.show()
    sys.exit(app.exec_())