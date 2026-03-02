import sys
import os
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QFileDialog,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QCheckBox,
    QSpinBox,  
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import uic
import imageio
from PIL import Image
import numpy as np
import cv2 as cv
import shutil
from segmentyanes import *
from sensor import *
from feature_extraction import run_feature_extraction
import glob
import subprocess
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from fpdf import FPDF
import time
import threading  
import resources_rc

# Try importing picamera2.
try:
    from picamera2.previews.qt import QPicamera2
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except Exception:
    QPicamera2 = None
    Picamera2 = None
    PICAM_AVAILABLE = False

try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)

    IN1, IN2, IN3, IN4 = 17, 18, 27, 22
    pins = [IN1, IN2, IN3, IN4]

    for p in pins:
        GPIO.setup(p, GPIO.OUT)
        GPIO.output(p, 0)

    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False

SEQ = [
    [1,0,0,0],
    [1,1,0,0],
    [0,1,0,0],
    [0,1,1,0],
    [0,0,1,0],
    [0,0,1,1],
    [0,0,0,1],
    [1,0,0,1]
]

step_delay = 0.002
motor_position = 0 

def move_motor(steps, direction):
    global motor_position
    if not GPIO_AVAILABLE:
        print(f"[SIM] Motor {'naik' if direction==1 else 'turun'} {steps} steps")
        motor_position += direction * steps
        return

    seq = SEQ if direction == 1 else list(reversed(SEQ))
    for _ in range(abs(steps)):
        for s in seq:
            for pin, val in zip(pins, s):
                GPIO.output(pin, val)
            time.sleep(step_delay)
        motor_position += direction

    for p in pins:
        GPIO.output(p, 0)


class PDFWithHeaderFooter(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("Poppins", "", "add-on/Poppins-Bold.ttf", uni=True)
        self.add_font("Inter", "", "add-on/Inter_18pt-SemiBold.ttf", uni=True)
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    def header(self):
        self.set_font("Poppins", "", 20)
        self.set_fill_color(4, 21, 98)
        self.set_xy(0, 16)
        self.cell(165, 3, fill=True)
        self.image("/home/microscope/malaria/MalaScope/add-on/logo.png", x=170, y=10, w=30)
        self.ln(10)
        self.cell(0, 10, "Segmentation and Detection Report", ln=True, align="L")
        self.ln(5)

    def footer(self):
        self.set_y(-20)
        self.set_font("Poppins", "", 15)
        self.set_text_color(100)
        self.cell(0, 10, "MalaScope, 2025", align="L")
        self.set_xy(65, 281)
        self.set_fill_color(4, 21, 98)
        self.cell(170, 2, fill=True)

    def generate_result(self, imagePath, detectPath, cells, mal, parPath):
        self.add_page()
        self.set_font("Inter", size=12)
        self.set_xy(18, 50)
        self.set_text_color(120)
        self.cell(170, 10, f"Report generated on {self.timestamp}", ln=True)

        self.image(imagePath, x=18, y=70, w=88, h=49.5)
        self.image(detectPath, x=100, y=70, w=88, h=49.5)

        self.set_font_size(10)
        self.set_text_color(150)
        self.set_xy(18, 123)
        self.multi_cell(
            170, 5,
            "Green bounding boxes indicate normal red blood cells, while red bounding boxes indicate malaria-infected cells.",
        )

        self.set_text_color(0)
        self.set_xy(18, 135)
        self.set_font_size(14)
        self.cell(170, 10, f"Total red blood cells detected: {cells}", border=1, ln=True)
        self.set_x(18)
        self.cell(170, 10, f"Malaria infected cells detected: {mal}", border=1, ln=True)

        if os.path.exists(parPath) and os.path.isdir(parPath):
            image_files = os.listdir(parPath)[:8]
            for index, filename in enumerate(image_files):
                col = index % 4
                row = index // 4
                x = 40 + col * 32
                y = 163 + row * 32
                file_path = os.path.join(parPath, filename)
                self.image(file_path, x=x, y=y, w=30, h=30)

        self.set_text_color(255)
        if mal != 0:
            self.set_xy(18, 230)
            self.set_fill_color(255, 0, 0)
            self.multi_cell(
                170, 6,
                "Based on our system's detection results, the patient is identified as having malaria and requires further clinical evaluation and appropriate treatment.",
                border=1, fill=True,
            )
        else:
            self.set_xy(18, 170)
            self.set_fill_color(0, 255, 0)
            self.multi_cell(
                170, 6,
                "Our system's detection results indicate no signs of malaria infection in the patient. However, continued monitoring or further clinical evaluation may be recommended if symptoms persist.",
                border=1, fill=True,
            )

        self.output("/home/microscope/malaria/MalaScope/Report.pdf")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("Main_Program.ui", self)

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.work_dir = os.path.join(self.base_dir, "work_files")

        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, "cluster"), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, "extracted_cell"), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, "predicted_cell"), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, "predicted_cell", "Parasitized"), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, "predicted_cell", "Uninfected"), exist_ok=True)

        self.using_picam = PICAM_AVAILABLE
        self.picam2 = None
        self.qpicamera2 = None

        if PICAM_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                self.picam2.configure(
                    self.picam2.create_preview_configuration({"size": (480, 270)})
                )
                self.qpicamera2 = QPicamera2(self.picam2, width=480, height=270, keep_ar=True)
            except Exception:
                self.using_picam = False
                self.picam2 = None
                self.qpicamera2 = None

        if not self.using_picam:
            self.cap = cv.VideoCapture(0)
            self.timer = QTimer()
            self.timer.timeout.connect(self._update_frame)

        self.stackedWidget = self.findChild(QStackedWidget, "stackedWidget")

        self.mainPage = self.findChild(QPushButton, "mainBtn")
        self.segmentPage = self.findChild(QPushButton, "rbcBtn")
        self.detectPage = self.findChild(QPushButton, "malBtn")
        self.aboutPage = self.findChild(QPushButton, "abtBtn")
        self.close_app = self.findChild(QPushButton, "closeBtn")

        self.distVal = self.findChild(QLabel, "distVal")
        self.imageSource = [
            self.findChild(QRadioButton, "camInput"),
            self.findChild(QRadioButton, "fileInput"),
        ]
        self.getButton = self.findChild(QPushButton, "getBtn")
        self.inputIm = self.findChild(QLabel, "rawImage")
        self.kmeansButton = self.findChild(QPushButton, "kmeansBtn")
        self.layout = QVBoxLayout()

        self.clusterText = self.findChild(QLabel, "clustText")
        self.selectCluster = [
            self.findChild(QCheckBox, "clust1"),
            self.findChild(QCheckBox, "clust2"),
            self.findChild(QCheckBox, "clust3"),
            self.findChild(QCheckBox, "clust4"),
            self.findChild(QCheckBox, "clust5"),
            self.findChild(QCheckBox, "clust6"),
        ]
        self.clusterIm = [
            self.findChild(QLabel, "clust1Im"),
            self.findChild(QLabel, "clust2Im"),
            self.findChild(QLabel, "clust3Im"),
            self.findChild(QLabel, "clust4Im"),
            self.findChild(QLabel, "clust5Im"),
            self.findChild(QLabel, "clust6Im"),
        ]
        self.extractButton = self.findChild(QPushButton, "extBtn")
        self.extractedIm = self.findChild(QLabel, "cellsExtract")
        self.rbcValText = self.findChild(QLabel, "rbcText")
        self.sepOverlap = self.findChild(QPushButton, "overlapBtn")
        self.saveCells = self.findChild(QPushButton, "saveBtn")
        self.detectButton = self.findChild(QPushButton, "detectBtn")

        self.detectText = self.findChild(QLabel, "detectText")
        self.detectIm = self.findChild(QLabel, "detectIm")
        self.visualIm = [
            self.findChild(QLabel, "vizImage_1"),
            self.findChild(QLabel, "vizImage_2"),
            self.findChild(QLabel, "vizImage_3"),
            self.findChild(QLabel, "vizImage_4"),
            self.findChild(QLabel, "vizImage_5"),
            self.findChild(QLabel, "vizImage_6"),
            self.findChild(QLabel, "vizImage_7"),
            self.findChild(QLabel, "vizImage_8"),
        ]
        self.pdfGenButton = self.findChild(QPushButton, "pdfBtn")

        self.spinBox_fast = self.findChild(QSpinBox, "spinBox_fast")
        self.spinBox_fine = self.findChild(QSpinBox, "spinBox_fine")
        self.label_position = self.findChild(QLabel, "label_position")

        self.btn_fast_up   = self.findChild(QPushButton, "btn_fast_up")
        self.btn_fast_down = self.findChild(QPushButton, "btn_fast_down")
        self.btn_fine_up   = self.findChild(QPushButton, "btn_fine_up")
        self.btn_fine_down = self.findChild(QPushButton, "btn_fine_down")

        if self.spinBox_fast:
            self.spinBox_fast.setRange(1, 99999)
            self.spinBox_fast.setValue(2000)
        if self.spinBox_fine:
            self.spinBox_fine.setRange(1, 99999)
            self.spinBox_fine.setValue(100)

        if self.btn_fast_up:
            self.btn_fast_up.clicked.connect(self.fast_up)
        if self.btn_fast_down:
            self.btn_fast_down.clicked.connect(self.fast_down)
        if self.btn_fine_up:
            self.btn_fine_up.clicked.connect(self.fine_up)
        if self.btn_fine_down:
            self.btn_fine_down.clicked.connect(self.fine_down)

        self.setStyles()

        self.imageSource[0].toggled.connect(self.cameraInputToggled)
        self.imageSource[1].toggled.connect(self.externalFileToggled)
        self.getButton.clicked.connect(self.takeImage)
        self.kmeansButton.clicked.connect(self.kmeansProcess)
        self.extractButton.clicked.connect(self.extractCells)
        self.sepOverlap.clicked.connect(self.separateOverlap)
        self.saveCells.clicked.connect(self.saveExtractedCells)
        self.detectButton.clicked.connect(self.detectCells)
        self.pdfGenButton.clicked.connect(self.generatePDF)

        self.mainPage.clicked.connect(self.moveMainPage)
        self.segmentPage.clicked.connect(self.moveSegmentPage)
        self.detectPage.clicked.connect(self.moveDetectPage)
        self.aboutPage.clicked.connect(self.moveAboutPage)
        self.close_app.clicked.connect(self.closeApp)

        if self.using_picam and self.picam2 is not None:
            try:
                self.picam2.start()
            except Exception:
                pass

        self.sensor = MagnificationSensor()
        self.update_position()  
        
    #Motor    
    def fast_up(self):
        steps = self.spinBox_fast.value()
        threading.Thread(target=move_motor, args=(steps, 1), daemon=True).start()
        self.label_position.setText(f"Bergerak Naik {steps} steps...")
        # Update posisi 
        QTimer.singleShot(int(steps * step_delay * 8 * 1000) + 500, self.update_position)

    def fast_down(self):
        steps = self.spinBox_fast.value()
        threading.Thread(target=move_motor, args=(steps, -1), daemon=True).start()
        self.label_position.setText(f"Bergerak Turun {steps} steps...")
        QTimer.singleShot(int(steps * step_delay * 8 * 1000) + 500, self.update_position)

    def fine_up(self):
        steps = self.spinBox_fine.value()
        threading.Thread(target=move_motor, args=(steps, 1), daemon=True).start()
        self.label_position.setText(f"Bergerak Naik {steps} steps...")
        QTimer.singleShot(int(steps * step_delay * 8 * 1000) + 500, self.update_position)

    def fine_down(self):
        steps = self.spinBox_fine.value()
        threading.Thread(target=move_motor, args=(steps, -1), daemon=True).start()
        self.label_position.setText(f"Bergerak Turun {steps} steps...")
        QTimer.singleShot(int(steps * step_delay * 8 * 1000) + 500, self.update_position)

    def update_position(self):
        if self.label_position:
            self.label_position.setText(f"Position: {motor_position} step")

    #Kamera
    def cameraInputToggled(self, checked):
        if checked:
            self.distance = self.sensor.read_distance()
            self.distVal.setText(f"Lens to Object Dist : {self.distance} mm")
            time.sleep(0.2)
            if self.using_picam and self.qpicamera2 is not None:
                if not self.qpicamera2.parent():
                    self.layout.setContentsMargins(0, 0, 0, 0)
                    self.layout.addWidget(self.qpicamera2)
                    self.inputIm.setLayout(self.layout)
                self.inputIm.clear()
            else:
                if not hasattr(self, "cap") or not self.cap.isOpened():
                    self.cap = cv.VideoCapture(0)
                self.timer.start(30)
                self.inputIm.clear()

    def externalFileToggled(self, checked):
        if checked:
            self.distVal.setText("Camera is not active")
            if self.using_picam and self.qpicamera2 is not None and self.qpicamera2.parent():
                self.layout.removeWidget(self.qpicamera2)
                self.qpicamera2.setParent(None)
            else:
                if hasattr(self, "timer") and self.timer.isActive():
                    self.timer.stop()
                if hasattr(self, "cap") and self.cap.isOpened():
                    try:
                        self.cap.release()
                    except Exception:
                        pass
            self.inputIm.clear()

    def closeEvent(self, event):
        if self.using_picam and self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
        else:
            if hasattr(self, "timer") and self.timer.isActive():
                self.timer.stop()
            if hasattr(self, "cap") and self.cap.isOpened():
                try:
                    self.cap.release()
                except Exception:
                    pass
        super().closeEvent(event)

    def takeImage(self):
        self.imagePath = None
        self.fileValue = False
        if self.imageSource[0].isChecked():
            if self.using_picam and self.picam2 is not None:
                cfg = self.picam2.create_still_configuration(main={"size": (480, 270)})
                self.picam2.switch_mode_and_capture_file(
                    cfg, "image_taken.jpg", signal_function=self.on_capture_done
                )
                return
            else:
                if not hasattr(self, "cap") or not self.cap.isOpened():
                    self.cap = cv.VideoCapture(0)
                ret, frame = self.cap.read()
                if ret:
                    image_path = os.path.join(self.work_dir, "image_taken.jpg")
                    cv.imwrite(image_path, frame)
                    self.imagePath = image_path
                    self.displayImage(self.imagePath)
                    return
                else:
                    self.inputIm.setText("Failed to capture image from webcam.")
                    return

        elif self.imageSource[1].isChecked():
            file_dialog = QFileDialog()
            file_dialog.setFileMode(QFileDialog.ExistingFile)
            file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
            sample_dir = os.path.join(self.base_dir, "sample_raw")
            if os.path.exists(sample_dir):
                file_dialog.setDirectory(sample_dir)

            if file_dialog.exec_():
                selected_files = file_dialog.selectedFiles()
                self.imagePath = selected_files[0]
                self.fileValue = True
                process_path = os.path.join(self.work_dir, "raw_image.jpg")
                shutil.copy(self.imagePath, process_path)

        if self.imagePath is not None:
            self.pixmap = QPixmap(self.imagePath)
            labelW = self.inputIm.width()
            labelH = self.inputIm.height()
            scaled_pixmap = self.pixmap.scaled(labelW, labelH, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.inputIm.setPixmap(scaled_pixmap)
            self.inputIm.setAlignment(Qt.AlignCenter)

        if not self.imageSource[0].isChecked() and not self.imageSource[1].isChecked():
            self.inputIm.setText("No image taken or selected.")

    def on_capture_done(self, picam2):
        self.imagePath = os.path.join(self.work_dir, "image_taken.jpg")
        image = Image.open(self.imagePath)
        image.save(self.imagePath)
        self.displayImage(self.imagePath)
        if self.qpicamera2.parent():
            self.layout.removeWidget(self.qpicamera2)
            self.qpicamera2.setParent(None)

    def _update_frame(self):
        if not hasattr(self, "cap"):
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        labelW = self.inputIm.width()
        labelH = self.inputIm.height()
        scaled_pixmap = pixmap.scaled(labelW, labelH, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.inputIm.setPixmap(scaled_pixmap)
        self.inputIm.setAlignment(Qt.AlignCenter)

    def displayImage(self, imagePath):
        self.pixmap = QPixmap(imagePath)
        labelW = self.inputIm.width()
        labelH = self.inputIm.height()
        scaled_pixmap = self.pixmap.scaled(labelW, labelH, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.inputIm.setPixmap(scaled_pixmap)
        self.inputIm.setAlignment(Qt.AlignCenter)

    def kmeansProcess(self):
        self.moveSegmentPage()
        self.segmentPage.setChecked(True)
        self.clusterText.setText("Please wait, doing k-means clustering...")
        QApplication.processEvents()

        if self.imageSource[0].isChecked():
            self.imagePath = os.path.join(self.work_dir, "image_taken.jpg")
        elif self.imageSource[1].isChecked():
            self.imagePath = os.path.join(self.work_dir, "raw_image.jpg")

        if self.imagePath is not None:
            self.raw_image = imageio.imread(self.imagePath)
            self.raw_image = cv.cvtColor(self.raw_image, cv.COLOR_BGR2RGB)
            self.hsv_clean_image, _ = convert_hsv_circular(self.raw_image, v_thresh=20)
            kmeans_val = 6
            self.segmented_images, self.labels_full = kmeans_segmentation(
                self.hsv_clean_image, kmeans_val, use_preprocessing=True, v_thresh=20
            )

        for idx, segment_image in enumerate(self.segmented_images):
            clusterPath = os.path.join(self.work_dir, "cluster", f"cluster_{idx+1}.jpg")
            cv.imwrite(clusterPath, cv.cvtColor(segment_image, cv.COLOR_RGB2BGR))
            self.pixmap = QPixmap(clusterPath)
            labelW = self.clusterIm[idx].width()
            labelH = self.clusterIm[idx].height()
            scaled_pixmap = self.pixmap.scaled(labelW, labelH, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.clusterIm[idx].setPixmap(scaled_pixmap)
            self.clusterIm[idx].setAlignment(Qt.AlignCenter)

        self.clusterText.setText("k-means clustering done.")

    def extractCells(self):
        self.moveExtractPage()
        QApplication.processEvents()

        self.selected_cluster = [
            i for i, checkbox in enumerate(self.selectCluster) if checkbox.isChecked()
        ]
        rgb_clean_image = cv.cvtColor(self.hsv_clean_image, cv.COLOR_HSV2RGB)

        self.rbc_only_image, self.filtered_mask, self.binary_mask = remove_unwanted_cells_extended(
            self.segmented_images, self.selected_cluster, rgb_clean_image
        )
     
        rbc_only_image_gray = cv.cvtColor(self.rbc_only_image, cv.COLOR_RGB2GRAY)
        edge_map, contour_edge = sobel_edge_detect(rbc_only_image_gray)
        cells_detected = draw_bounding_boxes(self.rbc_only_image, contour_edge)
        
        #Tambah Label
        contours_for_label, _ = extract_contours(rbc_only_image_gray, edge_map)
        for idx, contour in enumerate(contours_for_label, start=1):
            x, y, w, h = cv.boundingRect(contour)
            center_x = x+w //2
            center_y = y+h //2
            label_text = str(idx)
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (lw, lh), _ = cv.getTextSize(label_text, font, font_scale, thickness)
            padding = 4
            lx = max(0, min(center_x - lw // 2 - padding, cells_detected.shape[1] - lw - 2 * padding))
            ly = max(lh + 2 * padding, min(center_y - lh // 2 - padding, cells_detected.shape[0]))
            cv.rectangle(cells_detected,
                         (lx, ly - lh - padding),
                         (lx + lw + 2 * padding, ly + padding),
                         (0, 0, 0), -1)
            cv.putText(cells_detected, label_text, (lx + padding, ly - padding),
                       font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)
         
        detectPath = os.path.join(self.work_dir, "detect_cells.jpg")
        cv.imwrite(detectPath, cells_detected)

        self.pixmap = QPixmap(detectPath)
        labelW = self.extractedIm.width()
        labelH = self.extractedIm.height()
        scaled_pixmap = self.pixmap.scaled(labelW, labelH, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.extractedIm.setPixmap(scaled_pixmap)
        self.extractedIm.setAlignment(Qt.AlignCenter)

        contours, _ = extract_contours(rbc_only_image_gray, edge_map)
        self.extracted_cells = []
        self.cell_masks_list = []
        self.bounding_boxes_sep = []
        for i, contour in enumerate(contours):
            contour_mask = np.zeros(self.rbc_only_image.shape[:2], dtype=np.uint8)
            cv.drawContours(contour_mask, [contour], -1, 255, -1)
            x, y, w, h = cv.boundingRect(contour)
            cell = self.rbc_only_image[y: y + h, x: x + w]
            cropped_mask = contour_mask[y: y + h, x: x + w]
            masked_cell = cv.bitwise_and(cell, cell, mask=cropped_mask)
            self.extracted_cells.append((masked_cell, x, y))
            self.cell_masks_list.append(cropped_mask)
            self.bounding_boxes_sep.append((x, y, w, h))

        self.rbcValText.setText(
            f"{len(self.extracted_cells)} Red Blood Cells detected with edge detection. If there is any overlapping cells, please click Separate Cells."
        )

    def separateOverlap(self):
        self.rbcValText.setText("Separating overlapping cells using BO-FRS + GMM...")
        QApplication.processEvents()

        opened_mask = bounded_opening(self.filtered_mask, num_openings=3)
        bofrs_results = bounded_opening_frs(opened_mask, num_openings=3)
        cropped_cells, bounding_boxes, cell_masks = separate_overlapping_rbc_with_gmm(
            bofrs_results, self.rbc_only_image
        )

        self.extracted_cells = []
        self.cell_info = []
        self.cell_masks_list = []
        self.bounding_boxes_sep = []

        for idx, (cell_img, bbox) in enumerate(zip(cropped_cells, bounding_boxes)):
            x, y, w, h = bbox
            self.extracted_cells.append((cell_img, x, y))
            self.cell_info.append({"filename": f"cell_{idx}.png", "bbox": [x, y, w, h]})
            self.cell_masks_list.append(cell_masks[idx])
            self.bounding_boxes_sep.append(bbox)

        copy_rbc = self.rbc_only_image.copy()
        for idx, bbox in enumerate(bounding_boxes, start=1):
            x, y, w, h = bbox
            cv.rectangle(copy_rbc, (x, y), (x + w, y + h), (0, 255, 0), 5)

            center_x = x + w // 2
            center_y = y + h // 2
            label_text = str(idx)
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (lw, lh), _ = cv.getTextSize(label_text, font, font_scale, thickness)
            padding = 4
            lx = max(0, min(center_x - lw // 2 - padding, copy_rbc.shape[1] - lw - 2 * padding))
            ly = max(lh + 2 * padding, min(center_y - lh // 2 - padding, copy_rbc.shape[0]))
            cv.rectangle(copy_rbc,
                         (lx, ly - lh - padding),
                         (lx + lw + 2 * padding, ly + padding),
                         (0, 0, 0), -1)
            cv.putText(copy_rbc, label_text,
                       (lx + padding, ly - padding),
                       font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)
            
        sepPath = os.path.join(self.work_dir, "after_sep.jpg")
        cv.imwrite(sepPath, copy_rbc)

        self.pixmap = QPixmap(sepPath)
        labelW = self.extractedIm.width()
        labelH = self.extractedIm.height()
        scaled_pixmap = self.pixmap.scaled(labelW, labelH, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.extractedIm.setPixmap(scaled_pixmap)
        self.extractedIm.setAlignment(Qt.AlignCenter)

        self.rbcValText.setText(
            f"Separation completed! {len(self.extracted_cells)} individual cells detected using BO-FRS + GMM method."
        )

    def saveExtractedCells(self):
        self.cellsPath = os.path.join(self.work_dir, "extracted_cell")
        for filename in os.listdir(self.cellsPath):
            file_path = os.path.join(self.cellsPath, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        self.cell_info = []
        for idx, (cells_image, x, y) in enumerate(self.extracted_cells):
            h, w = cells_image.shape[:2]
            filename = os.path.join(self.cellsPath, f"cell_{idx}.png")
            cv.imwrite(filename, cells_image)
            self.cell_info.append({"filename": f"cell_{idx}.png", "bbox": [x, y, w, h]})

        extracted_cells_count = len(self.extracted_cells)

        # ── Ekstraksi fitur sekaligus saat save ──
        self.rbcValText.setText("Saving cells and extracting features, please wait...")
        QApplication.processEvents()

        excel_path = os.path.join(self.work_dir, "cell_features.xlsx")
        try:
            df_features, cell_labels, filter_stats = run_feature_extraction(
                extracted_cells=self.extracted_cells,
                bounding_boxes=self.bounding_boxes_sep,
                cell_masks=self.cell_masks_list,
                img_shape=self.rbc_only_image.shape,
                output_csv_path=None,  
            )
            if not df_features.empty:
                df_features.to_excel(excel_path, index=False)
                self.df_features = df_features
                total_passed = filter_stats.get("passed", len(df_features))
                self.rbcValText.setText(
                    f"{extracted_cells_count} cells saved. "
                    f"{total_passed} quality cells, "
                    f"{len(df_features.columns) - 1} features extracted. "
                    f"Saved to cell_features.xlsx"
                )
            else:
                self.rbcValText.setText(
                    f"{extracted_cells_count} cells saved. Feature extraction returned no results."
                )
        except Exception as e:
            print(f"[ERROR] Feature extraction in save: {e}")
            self.rbcValText.setText(
                f"{extracted_cells_count} cells saved. Feature extraction failed: {e}"
            )

    def detectCells(self):
        self.moveDetectPage()
        self.detectPage.setChecked(True)
        self.detectText.setText("Running feature selection (Mutual Information)...")
        QApplication.processEvents()

        if not hasattr(self, "df_features") or self.df_features.empty:
            self.detectText.setText(
                "No feature data found.\nPlease run Extract → Separate → Save first."
            )
            return

        try:
            df = self.df_features.copy()

            # IDA 
            area_threshold    = df["Area"].quantile(0.33)
            cp_ratio_threshold = df["CP_Ratio"].quantile(0.67)
            df["IDA_Label"] = (
                (df["Area"] < area_threshold) &
                (df["CP_Ratio"] > cp_ratio_threshold)
            ).astype(int)
            ida_count    = int(df["IDA_Label"].sum())
            normal_count = len(df) - ida_count

            # MI
            exclude_cols  = ["Cell_Label", "X", "Y", "IDA_Label"]
            feature_cols  = [c for c in df.columns if c not in exclude_cols]
            X_feat = df[feature_cols].fillna(0)
            y_feat = df["IDA_Label"]
            mi_scores = mutual_info_classif(X_feat, y_feat, random_state=42)

            mi_results = pd.DataFrame({
                "Feature":  feature_cols,
                "MI_Score": mi_scores,
            }).sort_values("MI_Score", ascending=False).reset_index(drop=True)

            def _categorize(name):
                if "CP_" in name:       return "Central Pallor"
                elif "GLCM_" in name:   return "GLCM Texture"
                elif "Color_" in name:  return "Color Moment"
                else:                   return "Morfologi"

            mi_results["Category"] = mi_results["Feature"].apply(_categorize)

            mi_threshold      = 0.01
            selected_features = mi_results[mi_results["MI_Score"] > mi_threshold]["Feature"].tolist()

            # Excel
            excel_sel_path = os.path.join(self.work_dir, "cell_features_selected.xlsx")
            excel_mi_path  = os.path.join(self.work_dir, "mutual_information_scores.xlsx")
            df_selected    = df[["Cell_Label", "X", "Y"] + selected_features + ["IDA_Label"]]
            df_selected.to_excel(excel_sel_path, index=False)
            mi_results.to_excel(excel_mi_path, index=False)

            # 
            top5 = mi_results.head(5)["Feature"].tolist()
            summary = (
                f"Feature Selection (Mutual Information) Complete!\n\n"
                f"Total cells: {len(df)}  |  IDA: {ida_count}  |  Normal: {normal_count}\n\n"
                f"Selected features (MI > {mi_threshold}): {len(selected_features)} / {len(feature_cols)}\n\n"
                f"Top 5 features:\n"
                + "\n".join(f"  {i+1}. {f}" for i, f in enumerate(top5))
                + f"\n\nSaved:\n"
                f"  cell_features_selected.xlsx\n"
                f"  mutual_information_scores.xlsx"
            )
            self.detectText.setText(summary)
            print(summary)

            # Simpan untuk generatePDF
            self.df_selected   = df_selected
            self.mi_results    = mi_results
            self.selected_features = selected_features

            # ── Visualisasi: bounding box warna sesuai IDA_Label ──
            copy_rbc = self.raw_image.copy()
            for row_idx, info in enumerate(self.cell_info):
                x, y, w, h = info["bbox"]
                original_label = row_idx + 1
                # Cek apakah label ini ada di df_selected
                row_data = df[df["Cell_Label"] == original_label]
                if row_data.empty:
                    color = (128, 128, 128)  # Abu = ditolak QC
                elif row_data["IDA_Label"].values[0] == 1:
                    color = (0, 0, 255)      # Merah = IDA
                else:
                    color = (0, 255, 0)      # Hijau = Normal
                cv.rectangle(copy_rbc, (x, y), (x + w, y + h), color, 5)

                # Label angka
                center_x = x + w // 2
                center_y = y + h // 2
                label_text = str(original_label)
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale, thickness = 0.6, 2
                (lw, lh), _ = cv.getTextSize(label_text, font, font_scale, thickness)
                padding = 4
                lx = max(0, min(center_x - lw // 2 - padding,
                                copy_rbc.shape[1] - lw - 2 * padding))
                ly = max(lh + 2 * padding,
                         min(center_y - lh // 2 - padding, copy_rbc.shape[0]))
                cv.rectangle(copy_rbc,
                             (lx, ly - lh - padding),
                             (lx + lw + 2 * padding, ly + padding),
                             (0, 0, 0), -1)
                cv.putText(copy_rbc, label_text,
                           (lx + padding, ly - padding),
                           font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)

            self.detectResultPath = os.path.join(self.work_dir, "detect_result.png")
            cv.imwrite(self.detectResultPath, copy_rbc)

            pixmap = QPixmap(self.detectResultPath)
            labelW = self.detectIm.width()
            labelH = self.detectIm.height()
            self.detectIm.setPixmap(
                pixmap.scaled(labelW, labelH, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            self.detectIm.setAlignment(Qt.AlignCenter)

            # ── Preview top-8 sel di visualIm ──
            for i in range(8):
                self.visualIm[i].clear()
            slot = 0
            for i, (cell_img, x, y) in enumerate(self.extracted_cells):
                if slot >= 8:
                    break
                h_img, w_img = cell_img.shape[:2]
                if h_img == 0 or w_img == 0:
                    continue
                rgb = cv.cvtColor(cell_img, cv.COLOR_BGR2RGB) if cell_img.ndim == 3 else cell_img
                q_img = QImage(rgb.data, w_img, h_img, rgb.strides[0], QImage.Format_RGB888)
                pix = QPixmap.fromImage(q_img)
                lw2 = self.visualIm[slot].width()
                lh2 = self.visualIm[slot].height()
                self.visualIm[slot].setPixmap(
                    pix.scaled(lw2, lh2, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                self.visualIm[slot].setAlignment(Qt.AlignCenter)
                slot += 1

        except Exception as e:
            self.detectText.setText(f"Feature selection failed: {e}")
            print(f"[ERROR] Feature selection: {e}")

    def generatePDF(self):
        # Jumlah sel yang lolos quality filter (dari feature extraction)
        if hasattr(self, "df_features") and not self.df_features.empty:
            quality_cell_count = len(self.df_features)
        elif hasattr(self, "cellsPath"):
            quality_cell_count = len(glob.glob(os.path.join(self.cellsPath, "*")))
        else:
            quality_cell_count = 0

        cell_count = len(self.cell_info) if hasattr(self, "cell_info") else 0
        csv_path   = os.path.join(self.work_dir, "cell_features.csv")

        pdf = PDFWithHeaderFooter()
        pdf.generate_result(
            imagePath=self.imagePath,
            detectPath=self.detectResultPath,
            cells=cell_count,
            mal=quality_cell_count,
            parPath=csv_path,
        )
        self.detectText.setText("Report generated in PDF format.")

    def setStyles(self):
        button_style = (
            "QPushButton {border:1px; border-radius: 10px; color: rgb(0,0,0); background-color: rgb(214, 222, 255);}"
            "QPushButton:hover {background-color: rgb(35, 56, 148); color: rgb(255,255,255); border: 1px;}"
            "QPushButton:checked {background-color: rgb(4, 21, 98); color: rgb(255,255,255);border: 1px;}"
        )
        menu_style = (
            "QPushButton {border:1px; color: rgb(225,225,225); background-color: rgb(17, 70, 143)}"
            "QPushButton:hover {background-color: rgb(35, 56, 148); color: rgb(255,255,255); border: 1px;}"
            "QPushButton:checked {background-color: rgb(4, 21, 98); color: rgb(255,255,255);border: 1px;}"
        )
        motor_style = (
            "QPushButton {border:1px; border-radius: 8px; color: rgb(255,255,255); background-color: rgb(34, 139, 34);}"
            "QPushButton:hover {background-color: rgb(0, 100, 0); border: 1px;}"
            "QPushButton:pressed {background-color: rgb(0, 60, 0); border: 1px;}"
        )
        self.mainPage.setStyleSheet(menu_style)
        self.segmentPage.setStyleSheet(button_style)
        self.detectPage.setStyleSheet(button_style)
        self.aboutPage.setStyleSheet(menu_style)

        # Style tombol motor (jika widget ditemukan)
        for btn_name in ["btn_fast_up", "btn_fast_down", "btn_fine_up", "btn_fine_down"]:
            btn = self.findChild(QPushButton, btn_name)
            if btn:
                btn.setStyleSheet(motor_style)

    def moveMainPage(self):
        self.stackedWidget.setCurrentIndex(0)
        self.inputIm.clear()
        self.imageSource[0].setChecked(False)
        self.imageSource[1].setChecked(False)

    def moveSegmentPage(self):
        self.stackedWidget.setCurrentIndex(1)
        for i in range(6):
            self.clusterIm[i].clear()
            self.selectCluster[i].setChecked(False)

    def moveExtractPage(self):
        self.stackedWidget.setCurrentIndex(2)
        self.extractedIm.clear()
        self.rbcValText.setText("No red blood cell detected")

    def moveDetectPage(self):
        self.stackedWidget.setCurrentIndex(3)
        self.detectText.setText("No cell detected")
        self.detectIm.clear()
        for i in range(8):
            self.visualIm[i].clear()

    def moveAboutPage(self):
        self.stackedWidget.setCurrentIndex(4)

    def closeApp(self):
        # Cleanup GPIO saat aplikasi ditutup
        if GPIO_AVAILABLE:
            GPIO.cleanup()
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()