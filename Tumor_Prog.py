

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication,QWidget,QInputDialog,QFileDialog
import cv2
import datetime
import tensorflow as tf
import datetime
from PIL import Image
from Safwen.gradcam import GradCAM
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16,Xception
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import random
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
from detectron2.utils.visualizer import ColorMode


class Ui_MainWindow(object):
    def test(self):
        filename=QFileDialog.getOpenFileName()
        name=filename[0]
        start = datetime.datetime.now()
        keyboard = np.zeros((600, 1000, 3), np.uint8)

        class_labels = ["0 - Non-Tumorous","1 - Tumorous"]

        interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        img = Image.open(name)
        img = img.resize((128, 128))

        img = np.reshape(img, [1, 128, 128, 3])
        img = np.interp(img, (img.min(), img.max()), (0, +1))
        input_data = np.array(img, dtype=np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()
      

        output_data = interpreter.get_tensor(output_details[0]['index'])
        for pred in output_data:
            result = np.argmax(pred)
        label = class_labels[result]
        end = datetime.datetime.now()
        elapsed = end - start
        elapsed=elapsed.total_seconds()
        if label == "0 - No Hemmorhage":
            color=(0,255,0)
        else:
            color=(0,0,255)
        h=np.max(pred)
        prob="{:.2f} %".format(h*100)
        roi = cv2.imread(name)
        roi = cv2.resize(roi, (700, 700))
        keyboard = cv2.resize(keyboard, (700, 100))
        cv2.putText(keyboard,"Category: "+ label,(10,30),1,2,color,2)
        cv2.putText(keyboard,"Latency: "+str(elapsed)+" seconds",(10,60),1,2,color,2)
        cv2.putText(keyboard,"Probability: "+str(prob),(10,90),1,2,color,2)
        output = np.vstack([keyboard, roi])
        cv2.imshow('Test_Result', output)
        cv2.waitKey(0)

    def heatmap(self):
        
  

        labels=  {'0': 0, '1': 1}
        class_labels=["0 - Non-Tumorous","1 - Tumorous"]

        filename=QFileDialog.getOpenFileName()
        name=filename[0]
        from tensorflow.keras.preprocessing import image



        new_model = load_model("trainn_5.h5")
       

        keyboard = np.zeros((600, 1000, 3), np.uint8)
        image_path = name
        path=name

        test_img_load = load_img(image_path, target_size=(128,128,3))
        test_img = image.img_to_array(test_img_load)
        test_img = np.expand_dims(test_img, axis=0)
        test_img /= 255

        label_map_inv = {v:k  for k,v in labels.items()}

        result = new_model.predict(test_img)
       

        prediction = result.argmax(axis=1)
      
        i = label_map_inv[int(prediction)]
        label=class_labels[(int(i))]
     



        image = load_img(path, target_size=(128, 128))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        orig = cv2.imread(path)
        resized = cv2.resize(orig, (128, 128))


        cam = GradCAM(new_model, int(i))
        heatmap = cam.compute_heatmap(test_img)


        heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
        (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.3)


        
        
        

        output = np.hstack([orig, output])
        keyboard = cv2.resize(keyboard, (output.shape[1], 100))
        
        
        key=np.vstack([keyboard,output])
        key = imutils.resize(key, height=800)
        if label == "0 - No Hemmorhage":
            color=(0,255,0)
        else:
            color=(0,0,255)

        cv2.putText(key, label, ((int(((key.shape[1])/2))-180), 40), 1, 2, color, 2)
        cv2.imwrite('Results/Heatmap.png',key)
        cv2.imshow("Output", key)
        cv2.waitKey(0)

    def seg(self):


        filename=QFileDialog.getOpenFileName()
        name=filename[0]
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
        cfg.MODEL.WEIGHTS = "model_tumor.pth"
        predictor = DefaultPredictor(cfg)
        MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[0]='TUMOR'



        path=name
        im = cv2.imread(path)
        outputs = predictor(im)




        v = Visualizer(im[:, :, ::-1], 
                   metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW)   # remove the colors of unsegmented pixels
    
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))


        pp=v.get_image()[:, :, ::-1]
        pp=cv2.resize(pp,(im.shape[1],im.shape[0]))
        output=np.hstack([im,pp])
        output=imutils.resize(output,height=800)


        cv2.imwrite('Results/Segmentation.png',output)
        cv2.imshow('Segmentation',output)

        cv2.waitKey(0)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 617)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(-100, -50, 1481, 741))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("medical-stethoscope-heart-pulse-logo-vector-22224424.jpg"))
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(90, 450, 151, 71))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(330, 450, 151, 71))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(580, 450, 151, 71))
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(320, 10, 471, 71))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(290, 60, 471, 71))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(220, 540, 471, 71))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")

        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(710, 560, 89, 25))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(10, 560, 89, 25))
        self.pushButton_5.setObjectName("pushButton_5")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.pushButton_2.clicked.connect(self.test)
        self.pushButton.clicked.connect(self.heatmap)
        self.pushButton_3.clicked.connect(self.seg)
        self.pushButton_4.clicked.connect(QtCore.QCoreApplication.instance().quit)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "HEATMAP"))
        self.pushButton_2.setText(_translate("MainWindow", "TEST"))
        self.pushButton_3.setText(_translate("MainWindow", "TUMOR DETECTION"))
        self.label_2.setText(_translate("MainWindow", "BRAIN TUMOR      "))
        self.label_3.setText(_translate("MainWindow", "Diagnosis Application"))
        self.label_4.setText(_translate("MainWindow", "Developped By : Safwen_Naimi"))
        self.pushButton_4.setText(_translate("MainWindow", "QUIT"))
        self.pushButton_5.setText(_translate("MainWindow", "INFO"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
