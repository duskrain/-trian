#!/usr/bin/env python3
# Author: winterssy <winterssy@foxmail.com>

import cv2
import dlib

from PyQt5.QtCore import QTimer, QThread, pyqtSignal, QRegExp, Qt
from PyQt5.QtGui import QImage, QPixmap, QIcon, QTextCursor, QRegExpValidator
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow, QMessageBox, QTableWidgetItem, QAbstractItemView
from PyQt5.uic import loadUi

import os
import webbrowser
import logging
import logging.config
import sqlite3
import sys
import threading
import queue
import multiprocessing
import winsound

from configparser import ConfigParser
from datetime import datetime


# 找不到已训练的人脸数据文件
class TrainingDataNotFoundError(FileNotFoundError):
    pass


# 找不到数据库文件
class DatabaseNotFoundError(FileNotFoundError):
    pass


class CoreUI(QMainWindow):
    database = './FaceBase.db'
    trainingData = './recognizer/trainingData.yml'
    cap = cv2.VideoCapture()
    captureQueue = queue.Queue()  # 图像队列
    alarmQueue = queue.LifoQueue()  # 报警队列，后进先出
    logQueue = multiprocessing.Queue()  # 日志队列
    receiveLogSignal = pyqtSignal(str)  # LOG信号


    def __init__(self):
        super(CoreUI, self).__init__()
        loadUi('./ui/Core2.ui', self)
        self.setWindowIcon(QIcon('./icons/icon.png'))
        self.setFixedSize(1298, 621)
        pix = QPixmap('./icons/title2.png')
        self.pictureLabel.setPixmap(pix)


        # 图像捕获
        self.isExternalCameraUsed = False
        self.useExternalCameraCheckBox.stateChanged.connect(
            lambda: self.useExternalCamera(self.useExternalCameraCheckBox))
        self.faceProcessingThread = FaceProcessingThread()
        self.startWebcamButton.clicked.connect(self.startWebcam)

        # 数据库
        self.initDbButton.setIcon(QIcon('./icons/warning.png'))
        self.initDbButton.clicked.connect(self.initDb)

        self.timer = QTimer(self)  # 初始化一个定时器
        self.timer.timeout.connect(self.updateFrame)
        # 功能开关
        self.faceTrackerCheckBox.stateChanged.connect(
            lambda: self.faceProcessingThread.enableFaceTracker(self))
        self.faceRecognizerCheckBox.stateChanged.connect(
            lambda: self.faceProcessingThread.enableFaceRecognizer(self))
        # self.panalarmCheckBox.stateChanged.connect(lambda: self.faceProcessingThread.enablePanalarm(self))

        # 直方图均衡化
        self.equalizeHistCheckBox.stateChanged.connect(
            lambda: self.faceProcessingThread.enableEqualizeHist(self))

        # 调试模式
        self.debugCheckBox.stateChanged.connect(lambda: self.faceProcessingThread.enableDebug(self))
        self.confidenceThresholdSlider.valueChanged.connect(
            lambda: self.faceProcessingThread.setConfidenceThreshold(self))
        self.autoAlarmThresholdSlider.valueChanged.connect(
            lambda: self.faceProcessingThread.setAutoAlarmThreshold(self))

        self.alarmSignalThreshold = 10

        # 日志系统
        self.receiveLogSignal.connect(lambda log: self.logOutput(log))
        self.logOutputThread = threading.Thread(target=self.receiveLog, daemon=True)
        self.logOutputThread.start()

        # 建立线程信号的槽连接
        # self.faceProcessingThread.trigger.connect(self.show_userInform)
        self.clearCheckButton.clicked.connect(self.clearInfor)
        self.inforCheckButton.setCheckable(True)
        self.faceProcessingThread.faceid_trigger.connect(self.show_userInform)



    # 检查数据库状态
    def initDb(self):
        try:
            if not os.path.isfile(self.database):
                raise DatabaseNotFoundError
            if not os.path.isfile(self.trainingData):
                raise TrainingDataNotFoundError

            conn = sqlite3.connect(self.database)
            cursor = conn.cursor()
            cursor.execute('SELECT Count(*) FROM users')
            result = cursor.fetchone()
            dbUserCount = result[0]
        except DatabaseNotFoundError:
            logging.error('系统找不到数据库文件{}'.format(self.database))
            self.initDbButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：未发现数据库文件，你可能未进行人脸采集')
        except TrainingDataNotFoundError:
            logging.error('系统找不到已训练的人脸数据{}'.format(self.trainingData))
            self.initDbButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：未发现已训练的人脸数据文件，请完成训练后继续')
        except Exception as e:
            logging.error('读取数据库异常，无法完成数据库初始化')
            self.initDbButton.setIcon(QIcon('./icons/error.png'))
            self.logQueue.put('Error：读取数据库异常，初始化数据库失败')
        else:
            cursor.close()
            conn.close()
            if not dbUserCount > 0:
                logging.warning('数据库为空')
                self.logQueue.put('warning：数据库为空，人脸识别功能不可用')
                self.initDbButton.setIcon(QIcon('./icons/warning.png'))
            else:
                self.logQueue.put('Success：数据库状态正常，发现用户数：{}'.format(dbUserCount))
                self.initDbButton.setIcon(QIcon('./icons/success.png'))
                self.initDbButton.setEnabled(False)
                self.faceRecognizerCheckBox.setToolTip('须先开启人脸跟踪')
                self.faceRecognizerCheckBox.setEnabled(True)

    # 是否使用外接摄像头
    def useExternalCamera(self, useExternalCameraCheckBox):
        if useExternalCameraCheckBox.isChecked():
            self.isExternalCameraUsed = True
        else:
            self.isExternalCameraUsed = False

    # 打开/关闭摄像头
    def startWebcam(self):
        if not self.cap.isOpened():
            if self.isExternalCameraUsed:
                camID = 1
            else:
                camID = 0
            self.cap.open(camID)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, frame = self.cap.read()
            if not ret:
                logging.error('无法调用电脑摄像头{}'.format(camID))
                self.logQueue.put('Error：初始化摄像头失败')
                self.cap.release()
                self.startWebcamButton.setIcon(QIcon('./icons/error.png'))
            else:
                self.faceProcessingThread.start()  # 启动OpenCV图像处理线程
                self.timer.start(5)  # 启动定时器
                # self.panalarmThread.start()  # 启动报警系统线程
                self.startWebcamButton.setIcon(QIcon('./icons/success.png'))
                self.startWebcamButton.setText('关闭摄像头')

        else:
            text = '如果关闭摄像头，人脸识别等功能将禁用。'
            informativeText = '<b>是否继续？</b>'
            ret = CoreUI.callDialog(QMessageBox.Warning, text, informativeText, QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No)

            if ret == QMessageBox.Yes:
                self.faceProcessingThread.stop()
                if self.cap.isOpened():
                    if self.timer.isActive():
                        self.timer.stop()
                    self.cap.release()

                self.realTimeCaptureLabel.clear()
                self.realTimeCaptureLabel.setText('<font color=red>摄像头未开启</font>')
                self.startWebcamButton.setText('摄像头已关闭')
                self.startWebcamButton.setEnabled(False)
                self.startWebcamButton.setIcon(QIcon())

    # 定时器，实时更新画面
    def updateFrame(self):
        if self.cap.isOpened():
            # ret, frame = self.cap.read()
            # if ret:
            #     self.showImg(frame, self.realTimeCaptureLabel)
            if not self.captureQueue.empty():
                captureData = self.captureQueue.get()
                realTimeFrame = captureData.get('realTimeFrame')
                self.displayImage(realTimeFrame, self.realTimeCaptureLabel)

    # 显示图片
    def displayImage(self, img, qlabel):
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # default：The image is stored using 8-bit indexes into a colormap， for example：a gray image
        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:  # rows[0], cols[1], channels[2]
            if img.shape[2] == 4:
                # The image is stored using a 32-bit byte-ordered RGBA format (8-8-8-8)
                # A: alpha channel，不透明度参数。如果一个像素的alpha通道数值为0%，那它就是完全透明的
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        # img.shape[1]：图像宽度width，img.shape[0]：图像高度height，img.shape[2]：图像通道数
        # QImage.__init__ (self, bytes data, int width, int height, int bytesPerLine, Format format)
        # 从内存缓冲流获取img数据构造QImage类
        # img.strides[0]：每行的字节数（width*3）,rgb为3，rgba为4
        # strides[0]为最外层(即一个二维数组所占的字节长度)，strides[1]为次外层（即一维数组所占字节长度），strides[2]为最内层（即一个元素所占字节长度）
        # 从里往外看，strides[2]为1个字节长度（uint8），strides[1]为3*1个字节长度（3即rgb 3个通道）
        # strides[0]为width*3个字节长度，width代表一行有几个像素

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        qlabel.setPixmap(QPixmap.fromImage(outImage))
        qlabel.setScaledContents(True)  # 图片自适应大小

    def receiveLog(self):
        while True:
            data = self.logQueue.get()
            if data:
                self.receiveLogSignal.emit(data)
            else:
                continue

    # LOG输出
    def logOutput(self, log):
        # 获取当前系统时间
        time = datetime.now().strftime('[%Y/%m/%d %H:%M:%S]')
        log = time + ' ' + log + '\n'

        self.logTextEdit.moveCursor(QTextCursor.End)
        self.logTextEdit.insertPlainText(log)
        self.logTextEdit.ensureCursorVisible()  # 自动滚屏

    #展示会员信息
    def show_userInform(self, msg):
        print(msg)

        if self.inforCheckButton.isChecked():
            conn = sqlite3.connect(self.database)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE face_id=?", (msg,))
            result = cursor.fetchall()
            if result:
                stu_id = result[0][0]
                cn_name = result[0][2]
                year = result[0][4]
                month = result[0][5]
                date = result[0][6]
                logtime = result[0][8]

                self.stuIDLineEdit.setText(stu_id)
                self.cnNameLineEdit.setText(cn_name)
                self.yearLineEdit.setText(year)
                self.monthLineEdit.setText(month)
                self.dateLineEdit.setText(date)
                self.logTimeLineEdit.setText(logtime)
                self.inforCheckButton.setCheckable(False)
            else:
                print("no face_id")

    #清除用户信息
    def clearInfor(self):
        self.stuIDLineEdit.clear()
        self.cnNameLineEdit.clear()
        self.yearLineEdit.clear()
        self.monthLineEdit.clear()
        self.dateLineEdit.clear()
        self.logTimeLineEdit.clear()
        self.inforCheckButton.setCheckable(True)

    # 窗口关闭事件，关闭OpenCV线程、定时器、摄像头
    def closeEvent(self, event):
        if self.faceProcessingThread.isRunning:
            self.faceProcessingThread.stop()
        if self.timer.isActive():
            self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        event.accept()
    @staticmethod
    def callDialog(icon, text, informativeText, standardButtons, defaultButton=None):
        msg = QMessageBox()
        msg.setWindowIcon(QIcon('./icons/icon.png'))
        msg.setWindowTitle('OpenCV Face Recognition System - Core')
        msg.setIcon(icon)
        msg.setText(text)
        msg.setInformativeText(informativeText)
        msg.setStandardButtons(standardButtons)
        if defaultButton:
            msg.setDefaultButton(defaultButton)
        return msg.exec()
# OpenCV线程
class FaceProcessingThread(QThread):
    faceid_trigger = pyqtSignal(int)

    def __init__(self):
        super(FaceProcessingThread, self).__init__()
        self.isRunning = True

        self.isFaceTrackerEnabled = True
        self.isFaceRecognizerEnabled = False
        self.isPanalarmEnabled = True

        self.isDebugMode = False
        self.confidenceThreshold = 70 #数值可调控
        self.autoAlarmThreshold = 65



        self.isEqualizeHistEnabled = False

    # 是否开启人脸跟踪
    def enableFaceTracker(self, coreUI):
        if coreUI.faceTrackerCheckBox.isChecked():
            self.isFaceTrackerEnabled = True
            coreUI.statusBar().showMessage('人脸跟踪：开启')
        else:
            self.isFaceTrackerEnabled = False
            coreUI.statusBar().showMessage('人脸跟踪：关闭')

    # 是否开启人脸识别
    def enableFaceRecognizer(self, coreUI):
        if coreUI.faceRecognizerCheckBox.isChecked():
            if self.isFaceTrackerEnabled:
                self.isFaceRecognizerEnabled = True
                coreUI.statusBar().showMessage('人脸识别：开启')
            else:
                CoreUI.logQueue.put('Error：操作失败，请先开启人脸跟踪')
                coreUI.faceRecognizerCheckBox.setCheckState(Qt.Unchecked)
                coreUI.faceRecognizerCheckBox.setChecked(False)
        else:
            self.isFaceRecognizerEnabled = False
            coreUI.statusBar().showMessage('人脸识别：关闭')

        # 是否开启调试模式
    def enableDebug(self, coreUI):
        if coreUI.debugCheckBox.isChecked():
            self.isDebugMode = True
            coreUI.statusBar().showMessage('调试模式：开启')
        else:
            self.isDebugMode = False
            coreUI.statusBar().showMessage('调试模式：关闭')

    # 设置置信度阈值
    def setConfidenceThreshold(self, coreUI):
        if self.isDebugMode:
            self.confidenceThreshold = coreUI.confidenceThresholdSlider.value()
            coreUI.statusBar().showMessage('置信度阈值：{}'.format(self.confidenceThreshold))

        # 直方图均衡化
    def enableEqualizeHist(self, coreUI):
        if coreUI.equalizeHistCheckBox.isChecked():
            self.isEqualizeHistEnabled = True
            coreUI.statusBar().showMessage('直方图均衡化：开启')
        else:
            self.isEqualizeHistEnabled = False
            coreUI.statusBar().showMessage('直方图均衡化：关闭')


    def run(self):

        global face_id

        faceCascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

        # 帧数、人脸ID初始化
        frameCounter = 0
        currentFaceID = 0

        # 人脸跟踪器字典初始化
        faceTrackers = {}

        isTrainingDataLoaded = False
        isDbConnected = False

        while self.isRunning:
            if CoreUI.cap.isOpened():
                ret, frame = CoreUI.cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # 是否执行直方图均衡化
                if self.isEqualizeHistEnabled:
                    gray = cv2.equalizeHist(gray)
                faces = faceCascade.detectMultiScale(gray, 1.3, 5, minSize=(90, 90))

                # 预加载数据文件
                if not isTrainingDataLoaded and os.path.isfile(CoreUI.trainingData):
                    recognizer = cv2.face.LBPHFaceRecognizer_create()
                    recognizer.read(CoreUI.trainingData)
                    isTrainingDataLoaded = True
                if not isDbConnected and os.path.isfile(CoreUI.database):
                    conn = sqlite3.connect(CoreUI.database)
                    cursor = conn.cursor()
                    isDbConnected = True

                captureData = {}
                realTimeFrame = frame.copy()
                alarmSignal = {}

                # 人脸跟踪
                # Reference：https://github.com/gdiepen/face-recognition
                if self.isFaceTrackerEnabled:

                    # 要删除的人脸跟踪器列表初始化
                    fidsToDelete = []

                    for fid in faceTrackers.keys():
                        # 实时跟踪
                        trackingQuality = faceTrackers[fid].update(realTimeFrame)
                        # 如果跟踪质量过低，删除该人脸跟踪器
                        if trackingQuality < 7:
                            fidsToDelete.append(fid)

                    # 删除跟踪质量过低的人脸跟踪器
                    for fid in fidsToDelete:
                        faceTrackers.pop(fid, None)

                    for (_x, _y, _w, _h) in faces:
                        isKnown = False
                        if self.isFaceRecognizerEnabled:
                            cv2.rectangle(realTimeFrame, (_x, _y), (_x + _w, _y + _h), (232, 138, 30), 2)
                            face_id, confidence = recognizer.predict(gray[_y:_y + _h, _x:_x + _w])
                            logging.debug('face_id：{}，confidence：{}'.format(face_id, confidence))


                            if self.isDebugMode:
                                CoreUI.logQueue.put('Debug -> face_id：{}，confidence：{}'.format(face_id, confidence))

                            # 从数据库中获取识别人脸的身份信息
                            try:

                                cursor.execute("SELECT * FROM users WHERE face_id=?", (face_id,))
                                print(str(face_id))

                                result = cursor.fetchall()
                                if result:

                                    en_name = result[0][3]
                                    print(en_name)
                                    self.faceid_trigger.emit(int(face_id))

                                else:
                                    raise Exception
                            except Exception as e:
                                logging.error('读取数据库异常，系统无法获取Face ID为{}的身份信息'.format(face_id))
                                CoreUI.logQueue.put('Error：读取数据库异常，系统无法获取Face ID为{}的身份信息'.format(face_id))
                                en_name = ''
                            # 若置信度评分小于置信度阈值，认为是可靠识别
                            if confidence < self.confidenceThreshold:
                                isKnown = True
                                cv2.putText(realTimeFrame, en_name, (_x - 5, _y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 97, 255), 2)
                            else:
                                # 若置信度评分大于置信度阈值，该人脸可能是陌生人
                                cv2.putText(realTimeFrame, 'unknown', (_x - 5, _y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2)
                                # 若置信度评分超出自动报警阈值，触发报警信号
                                if confidence > self.autoAlarmThreshold:
                                    # 检测报警系统是否开启
                                    if self.isPanalarmEnabled:
                                        alarmSignal['timestamp'] = datetime.now().strftime('%Y%m%d%H%M%S')
                                        alarmSignal['img'] = realTimeFrame
                                        CoreUI.alarmQueue.put(alarmSignal)
                                        logging.info('系统发出了报警信号')


                        # 帧数自增
                        frameCounter += 1

                        # 每读取10帧，检测跟踪器的人脸是否还在当前画面内
                        if frameCounter % 10 == 0:
                            # 这里必须转换成int类型，因为OpenCV人脸检测返回的是numpy.int32类型，
                            # 而dlib人脸跟踪器要求的是int类型
                            x = int(_x)
                            y = int(_y)
                            w = int(_w)
                            h = int(_h)

                            # 计算中心点
                            x_bar = x + 0.5 * w
                            y_bar = y + 0.5 * h

                            # matchedFid表征当前检测到的人脸是否已被跟踪
                            matchedFid = None

                            for fid in faceTrackers.keys():
                                # 获取人脸跟踪器的位置
                                # tracked_position 是 dlib.drectangle 类型，用来表征图像的矩形区域，坐标是浮点数
                                tracked_position = faceTrackers[fid].get_position()
                                # 浮点数取整
                                t_x = int(tracked_position.left())
                                t_y = int(tracked_position.top())
                                t_w = int(tracked_position.width())
                                t_h = int(tracked_position.height())

                                # 计算人脸跟踪器的中心点
                                t_x_bar = t_x + 0.5 * t_w
                                t_y_bar = t_y + 0.5 * t_h

                                # 如果当前检测到的人脸中心点落在人脸跟踪器内，且人脸跟踪器的中心点也落在当前检测到的人脸内
                                # 说明当前人脸已被跟踪
                                if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and
                                        (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                                    matchedFid = fid

                            # 如果当前检测到的人脸是陌生人脸且未被跟踪
                            if not isKnown and matchedFid is None:
                                # 创建一个人脸跟踪器
                                tracker = dlib.correlation_tracker()
                                # 锁定跟踪范围
                                tracker.start_track(realTimeFrame, dlib.rectangle(x - 5, y - 10, x + w + 5, y + h + 10))
                                # 将该人脸跟踪器分配给当前检测到的人脸
                                faceTrackers[currentFaceID] = tracker
                                # 人脸ID自增
                                currentFaceID += 1

                    # 使用当前的人脸跟踪器，更新画面，输出跟踪结果
                    for fid in faceTrackers.keys():
                        tracked_position = faceTrackers[fid].get_position()

                        t_x = int(tracked_position.left())
                        t_y = int(tracked_position.top())
                        t_w = int(tracked_position.width())
                        t_h = int(tracked_position.height())

                        # 在跟踪帧中圈出人脸
                        cv2.rectangle(realTimeFrame, (t_x, t_y), (t_x + t_w, t_y + t_h), (0, 0, 255), 2)
                        cv2.putText(realTimeFrame, 'tracking...', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                                    2)

                captureData['originFrame'] = frame
                captureData['realTimeFrame'] = realTimeFrame
                CoreUI.captureQueue.put(captureData)

            else:
                continue


   #停止OpenCV线程
    def stop(self):
        self.isRunning = False
        self.quit()
        self.wait()


if __name__ == '__main__':
    logging.config.fileConfig('./config/logging.cfg')
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('./icons/super.icon'))  # 画一个logo 黄色微笑脸
    window = CoreUI()
    window.show()
    sys.exit(app.exec())
