import sys
import os
import time
import datetime
import glob
import shutil
import numpy as np
import cv2 as cv
import pandas as pd
import imageio
from PIL import Image
from sklearn.feature_selection import mutual_info_classif
from fpdf import FPDF

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QFileDialog, QPushButton,
    QRadioButton, QStackedWidget, QVBoxLayout, QCheckBox, QSpinBox,
    QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import uic

# Modul custom kamu
from segmentyanes import *
from sensor import *
from feature_extraction import run_feature_extraction
import resources_rc

# --- INISIALISASI KOMUNIKASI SERIAL USB (RASPI ke ESP32) ---
import serial
SERIAL_AVAILABLE = False
esp_serial = None

# Sistem mencoba mencari ESP32 di port USB standar Linux
try:
    esp_serial = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
    SERIAL_AVAILABLE = True
    print("✅ ESP32 Terhubung via /dev/ttyUSB0!")
    time.sleep(2) # Waktu untuk ESP32 reset sejenak
except Exception:
    try:
        esp_serial = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        SERIAL_AVAILABLE = True
        print("✅ ESP32 Terhubung via /dev/ttyACM0!")
        time.sleep(2)
    except Exception as e:
        print(f"❌ ESP32 Tidak Terdeteksi. Program jalan dalam Mode Simulasi Motor.")

# Coba import picamera2
try:
    from picamera2.previews.qt import QPicamera2
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except Exception:
    QPicamera2 = None
    Picamera2 = None
    PICAM_AVAILABLE = False

motor_position = 0 

# --- CLASS REPORT PDF ---
class PDFWithHeaderFooter(FPDF):
    def __init__(self, base_dir):
        super().__init__()
        self.base_dir = base_dir
        self.add_font("Poppins", "", os.path.join(self.base_dir, "add-on/Poppins-Bold.ttf"), uni=True)
        self.add_font("Inter", "", os.path.join(self.base_dir, "add-on/Inter_18pt-SemiBold.ttf"), uni=True)
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    def header(self):
        self.set_font("Poppins", "", 20)
        self.set_fill_color(4, 21, 98)
        self.set_xy(0, 16)
        self.cell(165, 3, fill=True)
        
        logo_path = os.path.join(self.base_dir, "add-on/logo.png")
        if os.path.exists(logo_path):
            self.image(logo_path, x=170, y=10, w=30)
            
        self.ln(10)
        self.cell(0, 10, "Segmentation and Detection Report", ln=True, align="L")
        self.ln(5)

    def footer(self):
        self.set_y(-20)
        self.set_font("Poppins", "", 15)
        self.set_text_color(100)
        self.cell(0, 10, "MalaScope, 2026", align="L")
        self.set_xy(65, 281)
        self.set_fill_color(4, 21, 98)
        self.cell(170, 2, fill=True)

    def generate_result(self, imagePath, detectPath, cells, mal, parPath, output_path, patient_name):
        self.add_page()
        self.set_font("Inter", size=12)
        self.set_xy(18, 40)
        self.set_text_color(0)
        self.cell(170, 10, f"Patient Name / ID: {patient_name.replace('_', ' ')}", ln=True)
        
        self.set_xy(18, 50)
        self.set_text_color(120)
        self.cell(170, 10, f"Report generated on {self.timestamp}", ln=True)

        if os.path.exists(imagePath): self.image(imagePath, x=18, y=70, w=88, h=49.5)
        if os.path.exists(detectPath): self.image(detectPath, x=100, y=70, w=88, h=49.5)

        self.set_font_size(10)
        self.set_text_color(150)
        self.set_xy(18, 123)
        self.multi_cell(170, 5, "Green bounding boxes indicate normal red blood cells, while red bounding boxes indicate IDA/malaria-infected cells.")

        self.set_text_color(0)
        self.set_xy(18, 135)
        self.set_font_size(14)
        self.cell(170, 10, f"Total red blood cells detected: {cells}", border=1, ln=True)
        self.set_x(18)
        self.cell(170, 10, f"Infected/Abnormal cells detected: {mal}", border=1, ln=True)

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
            self.multi_cell(170, 6, "Based on our system's detection results, the patient is identified as having abnormalities (IDA) and requires further clinical evaluation.", border=1, fill=True)
        else:
            self.set_xy(18, 170)
            self.set_fill_color(0, 255, 0)
            self.multi_cell(170, 6, "Our system's detection results indicate normal cells in the patient.", border=1, fill=True)

        self.output(output_path)


# --- MAIN GUI CLASS ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("Main_Program.ui", self)

        # 1. SETUP FOLDER MASTER DATA PASIEN
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.master_data_dir = os.path.join(self.base_dir, "DATA_PASIEN")
        os.makedirs(self.master_data_dir, exist_ok=True)
        
        self.current_raw_dir = None
        self.current_clust_dir = None
        self.current_sep_dir = None
        self.current_res_dir = None
        self.current_patient = "Anonim"

        # 2. INISIALISASI KAMERA
        self.using_picam = PICAM_AVAILABLE
        self.picam2 = None
        self.qpicamera2 = None

        if PICAM_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                self.picam2.configure(self.picam2.create_preview_configuration({"size": (480, 270)}))
                self.qpicamera2 = QPicamera2(self.picam2, width=480, height=270, keep_ar=True)
            except Exception:
                self.using_picam = False
                self.picam2 = None
                self.qpicamera2 = None

        if not self.using_picam:
            self.cap = cv.VideoCapture(0)
            self.timer = QTimer()
            self.timer.timeout.connect(self._update_frame)

        # 3. MENGHUBUNGKAN ELEMEN UI
        self.stackedWidget = self.findChild(QStackedWidget, "stackedWidget")
        self.nameInput = self.findChild(QLineEdit, "nameInput")

        self.mainPage = self.findChild(QPushButton, "mainBtn")
        self.segmentPage = self.findChild(QPushButton, "rbcBtn")
        self.detectPage = self.findChild(QPushButton, "malBtn")
        self.aboutPage = self.findChild(QPushButton, "abtBtn")
        self.close_app = self.findChild(QPushButton, "closeBtn")

        self.distVal = self.findChild(QLabel, "distVal")
        self.imageSource = [self.findChild(QRadioButton, "camInput"), self.findChild(QRadioButton, "fileInput")]
        self.getButton = self.findChild(QPushButton, "getBtn")
        self.inputIm = self.findChild(QLabel, "rawImage")
        self.kmeansButton = self.findChild(QPushButton, "kmeansBtn")
        self.layout = QVBoxLayout()

        self.clusterText = self.findChild(QLabel, "clustText")
        self.selectCluster = [self.findChild(QCheckBox, f"clust{i}") for i in range(1, 7)]
        self.clusterIm = [self.findChild(QLabel, f"clust{i}Im") for i in range(1, 7)]
        
        self.extractButton = self.findChild(QPushButton, "extBtn")
        self.extractedIm = self.findChild(QLabel, "cellsExtract")
        self.rbcValText = self.findChild(QLabel, "rbcText")
        self.sepOverlap = self.findChild(QPushButton, "overlapBtn")
        self.saveCells = self.findChild(QPushButton, "saveBtn")
        self.detectButton = self.findChild(QPushButton, "detectBtn")

        self.detectText = self.findChild(QLabel, "detectText")
        self.detectIm = self.findChild(QLabel, "detectIm")
        self.visualIm = [self.findChild(QLabel, f"vizImage_{i}") for i in range(1, 9)]
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

        # 4. KONEKSI TOMBOL
        if self.btn_fast_up: self.btn_fast_up.clicked.connect(self.fast_up)
        if self.btn_fast_down: self.btn_fast_down.clicked.connect(self.fast_down)
        if self.btn_fine_up: self.btn_fine_up.clicked.connect(self.fine_up)
        if self.btn_fine_down: self.btn_fine_down.clicked.connect(self.fine_down)

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
            try: self.picam2.start()
            except Exception: pass

        # 5. INISIALISASI SENSOR & TIMER
        self.sensor = MagnificationSensor()
        self.update_position()  
        
        self.sensor_timer = QTimer()
        self.sensor_timer.timeout.connect(self.update_sensor_value)

    # --- FUNGSI UPDATE SENSOR (REAL-TIME) ---
    def update_sensor_value(self):
        distance = self.sensor.read_distance()
        if self.distVal:
            self.distVal.setText(f"Lens to Object Dist : {distance:.1f} mm")
        
    def _create_session_folders(self):
        if self.nameInput and self.nameInput.text().strip() != "":
            self.current_patient = self.nameInput.text().strip().replace(" ", "_")
        else:
            self.current_patient = "Anonim"
            
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session_folder_name = f"{self.current_patient}_{timestamp}"
        session_path = os.path.join(self.master_data_dir, session_folder_name)
        
        self.current_raw_dir = os.path.join(session_path, "0_raw_image")
        self.current_clust_dir = os.path.join(session_path, "1_clustering_image")
        self.current_sep_dir = os.path.join(session_path, "2_separated_cells")
        self.current_res_dir = os.path.join(session_path, "3_results")
        
        for folder in [self.current_raw_dir, self.current_clust_dir, self.current_sep_dir, self.current_res_dir]:
            os.makedirs(folder, exist_ok=True)

    # --- KONTROL MOTOR MENGGUNAKAN SERIAL (KE ESP32) ---
    def send_command_to_esp(self, direction_char, steps):
        global motor_position
        if SERIAL_AVAILABLE:
            # Format pesan misal: "U100\n" atau "D50\n"
            pesan = f"{direction_char}{steps}\n"
            esp_serial.write(pesan.encode('utf-8'))
        
        # Update simulasi nilai posisi di GUI
        dir_val = 1 if direction_char == 'U' else -1
        motor_position += (dir_val * steps)
        self.update_position()

    def fast_up(self):
        steps = self.spinBox_fast.value()
        self.label_position.setText(f"Memerintahkan ESP32 Naik {steps} steps...")
        self.send_command_to_esp('U', steps)

    def fast_down(self):
        steps = self.spinBox_fast.value()
        self.label_position.setText(f"Memerintahkan ESP32 Turun {steps} steps...")
        self.send_command_to_esp('D', steps)

    def fine_up(self):
        steps = self.spinBox_fine.value()
        self.label_position.setText(f"Memerintahkan ESP32 Naik (Fine) {steps} steps...")
        self.send_command_to_esp('U', steps)

    def fine_down(self):
        steps = self.spinBox_fine.value()
        self.label_position.setText(f"Memerintahkan ESP32 Turun (Fine) {steps} steps...")
        self.send_command_to_esp('D', steps)

    def update_position(self):
        if self.label_position:
            self.label_position.setText(f"Position: {motor_position} step")

    # --- KONTROL KAMERA ---
    def cameraInputToggled(self, checked):
        if checked:
            self.sensor_timer.start(500)
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
            self.sensor_timer.stop()
            self.distVal.setText("Camera is not active")
            if self.using_picam and self.qpicamera2 is not None and self.qpicamera2.parent():
                self.layout.removeWidget(self.qpicamera2)
                self.qpicamera2.setParent(None)
            else:
                if hasattr(self, "timer") and self.timer.isActive():
                    self.timer.stop()
                if hasattr(self, "cap") and self.cap.isOpened():
                    try: self.cap.release()
                    except Exception: pass
            self.inputIm.clear()

    def takeImage(self):
        self._create_session_folders()
        self.imagePath = None
        self.fileValue = False
        save_name = f"raw_{self.current_patient}.jpg"
        
        if self.imageSource[0].isChecked():
            if self.using_picam and self.picam2 is not None:
                cfg = self.picam2.create_still_configuration(main={"size": (480, 270)})
                self.imagePath = os.path.join(self.current_raw_dir, save_name)
                
                self.picam2.switch_mode_and_capture_file(
                    cfg, self.imagePath, signal_function=self.on_capture_done
                )
                return
            else:
                if not hasattr(self, "cap") or not self.cap.isOpened():
                    self.cap = cv.VideoCapture(0)
                ret, frame = self.cap.read()
                if ret:
                    self.imagePath = os.path.join(self.current_raw_dir, save_name)
                    cv.imwrite(self.imagePath, frame)
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
                self.imagePath = os.path.join(self.current_raw_dir, save_name)
                self.fileValue = True
                shutil.copy(selected_files[0], self.imagePath)
                self.displayImage(self.imagePath)

    def on_capture_done(self, picam2):
        self.imagePath = os.path.join(self.current_raw_dir, f"raw_{self.current_patient}.jpg")
        time.sleep(0.5) 
        if os.path.exists(self.imagePath):
            image = Image.open(self.imagePath)
            image.save(self.imagePath)
            self.displayImage(self.imagePath)
            if self.qpicamera2.parent():
                self.layout.removeWidget(self.qpicamera2)
                self.qpicamera2.setParent(None)
        else:
            print(f"❌ ERROR: Gambar gagal disave ke {self.imagePath}")

    def _update_frame(self):
        if not hasattr(self, "cap"): return
        ret, frame = self.cap.read()
        if not ret: return
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.inputIm.setPixmap(pixmap.scaled(self.inputIm.width(), self.inputIm.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.inputIm.setAlignment(Qt.AlignCenter)

    def displayImage(self, imagePath):
        self.pixmap = QPixmap(imagePath)
        self.inputIm.setPixmap(self.pixmap.scaled(self.inputIm.width(), self.inputIm.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.inputIm.setAlignment(Qt.AlignCenter)

    # --- PENGOLAHAN CITRA ---
    def kmeansProcess(self):
        if not self.current_clust_dir:
            self.clusterText.setText("Silakan Get Image dulu!")
            return
            
        self.moveSegmentPage()
        self.segmentPage.setChecked(True)
        self.clusterText.setText("Please wait, doing k-means clustering...")
        QApplication.processEvents()

        if os.path.exists(self.imagePath):
            self.raw_image = imageio.imread(self.imagePath)
            self.raw_image = cv.cvtColor(self.raw_image, cv.COLOR_BGR2RGB)
            self.hsv_clean_image, _ = convert_hsv_circular(self.raw_image, v_thresh=20)
            kmeans_val = 6
            self.segmented_images, self.labels_full = kmeans_segmentation(
                self.hsv_clean_image, kmeans_val, use_preprocessing=True, v_thresh=20
            )

            for idx, segment_image in enumerate(self.segmented_images):
                clusterPath = os.path.join(self.current_clust_dir, f"cluster_{idx+1}.jpg")
                cv.imwrite(clusterPath, cv.cvtColor(segment_image, cv.COLOR_RGB2BGR))
                
                self.pixmap = QPixmap(clusterPath)
                self.clusterIm[idx].setPixmap(self.pixmap.scaled(self.clusterIm[idx].width(), self.clusterIm[idx].height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.clusterIm[idx].setAlignment(Qt.AlignCenter)

            self.clusterText.setText("k-means clustering done.")

    def extractCells(self):
        self.moveExtractPage()
        QApplication.processEvents()

        self.selected_cluster = [i for i, checkbox in enumerate(self.selectCluster) if checkbox.isChecked()]
        rgb_clean_image = cv.cvtColor(self.hsv_clean_image, cv.COLOR_HSV2RGB)

        self.rbc_only_image, self.filtered_mask, self.binary_mask = remove_unwanted_cells_extended(
            self.segmented_images, self.selected_cluster, rgb_clean_image
        )
      
        rbc_only_image_gray = cv.cvtColor(self.rbc_only_image, cv.COLOR_RGB2GRAY)
        edge_map, contour_edge = sobel_edge_detect(rbc_only_image_gray)
        cells_detected = draw_bounding_boxes(self.rbc_only_image, contour_edge)
        
        contours_for_label, _ = extract_contours(rbc_only_image_gray, edge_map)
        for idx, contour in enumerate(contours_for_label, start=1):
            x, y, w, h = cv.boundingRect(contour)
            center_x, center_y = x + w // 2, y + h // 2
            label_text = str(idx)
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale, thickness = 0.6, 2
            (lw, lh), _ = cv.getTextSize(label_text, font, font_scale, thickness)
            padding = 4
            lx = max(0, min(center_x - lw // 2 - padding, cells_detected.shape[1] - lw - 2 * padding))
            ly = max(lh + 2 * padding, min(center_y - lh // 2 - padding, cells_detected.shape[0]))
            cv.rectangle(cells_detected, (lx, ly - lh - padding), (lx + lw + 2 * padding, ly + padding), (0, 0, 0), -1)
            cv.putText(cells_detected, label_text, (lx + padding, ly - padding), font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)
          
        detectPath = os.path.join(self.current_res_dir, "detect_cells_initial.jpg")
        cv.imwrite(detectPath, cells_detected)

        self.pixmap = QPixmap(detectPath)
        self.extractedIm.setPixmap(self.pixmap.scaled(self.extractedIm.width(), self.extractedIm.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
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

        self.rbcValText.setText(f"{len(self.extracted_cells)} Red Blood Cells detected. Click Separate Cells if overlapping.")

    def separateOverlap(self):
        self.rbcValText.setText("Separating overlapping cells using BO-FRS + GMM...")
        QApplication.processEvents()

        opened_mask = bounded_opening(self.filtered_mask, num_openings=3)
        bofrs_results = bounded_opening_frs(opened_mask, num_openings=3)
        cropped_cells, bounding_boxes, cell_masks = separate_overlapping_rbc_with_gmm(bofrs_results, self.rbc_only_image)

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
            center_x, center_y = x + w // 2, y + h // 2
            label_text = str(idx)
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale, thickness = 0.6, 2
            (lw, lh), _ = cv.getTextSize(label_text, font, font_scale, thickness)
            padding = 4
            lx = max(0, min(center_x - lw // 2 - padding, copy_rbc.shape[1] - lw - 2 * padding))
            ly = max(lh + 2 * padding, min(center_y - lh // 2 - padding, copy_rbc.shape[0]))
            cv.rectangle(copy_rbc, (lx, ly - lh - padding), (lx + lw + 2 * padding, ly + padding), (0, 0, 0), -1)
            cv.putText(copy_rbc, label_text, (lx + padding, ly - padding), font, font_scale, (255, 255, 255), thickness, cv.LINE_AA)
            
        sepPath = os.path.join(self.current_res_dir, "after_sep.jpg")
        cv.imwrite(sepPath, copy_rbc)

        self.pixmap = QPixmap(sepPath)
        self.extractedIm.setPixmap(self.pixmap.scaled(self.extractedIm.width(), self.extractedIm.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.extractedIm.setAlignment(Qt.AlignCenter)
        self.rbcValText.setText(f"Separation completed! {len(self.extracted_cells)} individual cells detected.")

    def saveExtractedCells(self):
        self.cell_info = []
        for idx, (cells_image, x, y) in enumerate(self.extracted_cells):
            h, w = cells_image.shape[:2]
            filename = os.path.join(self.current_sep_dir, f"cell_{idx}.png")
            cv.imwrite(filename, cells_image)
            self.cell_info.append({"filename": f"cell_{idx}.png", "bbox": [x, y, w, h]})

        extracted_cells_count = len(self.extracted_cells)
        self.rbcValText.setText("Saving cells and extracting features, please wait...")
        QApplication.processEvents()

        excel_path = os.path.join(self.current_res_dir, f"features_{self.current_patient}.xlsx")
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
                self.rbcValText.setText(f"{extracted_cells_count} cells saved. {total_passed} quality cells. Saved to results folder.")
            else:
                self.rbcValText.setText("Feature extraction returned no results.")
        except Exception as e:
            self.rbcValText.setText(f"Feature extraction failed: {e}")

    def detectCells(self):
        self.moveDetectPage()
        self.detectPage.setChecked(True)
        self.detectText.setText("Running feature selection (Mutual Information)...")
        QApplication.processEvents()

        if not hasattr(self, "df_features") or self.df_features.empty:
            self.detectText.setText("No feature data found.\nPlease run Extract → Separate → Save first.")
            return

        try:
            df = self.df_features.copy()
            area_threshold    = df["Area"].quantile(0.33)
            cp_ratio_threshold = df["CP_Ratio"].quantile(0.67)
            df["IDA_Label"] = ((df["Area"] < area_threshold) & (df["CP_Ratio"] > cp_ratio_threshold)).astype(int)
            ida_count    = int(df["IDA_Label"].sum())
            normal_count = len(df) - ida_count

            exclude_cols  = ["Cell_Label", "X", "Y", "IDA_Label"]
            feature_cols  = [c for c in df.columns if c not in exclude_cols]
            X_feat = df[feature_cols].fillna(0)
            y_feat = df["IDA_Label"]
            mi_scores = mutual_info_classif(X_feat, y_feat, random_state=42)

            mi_results = pd.DataFrame({"Feature": feature_cols, "MI_Score": mi_scores}).sort_values("MI_Score", ascending=False).reset_index(drop=True)
            mi_threshold      = 0.01
            selected_features = mi_results[mi_results["MI_Score"] > mi_threshold]["Feature"].tolist()

            excel_sel_path = os.path.join(self.current_res_dir, f"features_selected_{self.current_patient}.xlsx")
            excel_mi_path  = os.path.join(self.current_res_dir, f"mutual_info_{self.current_patient}.xlsx")
            df_selected    = df[["Cell_Label", "X", "Y"] + selected_features + ["IDA_Label"]]
            
            df_selected.to_excel(excel_sel_path, index=False)
            mi_results.to_excel(excel_mi_path, index=False)

            top5 = mi_results.head(5)["Feature"].tolist()
            summary = (
                f"Feature Selection Complete!\n"
                f"Total cells: {len(df)}  |  IDA: {ida_count}  |  Normal: {normal_count}\n"
                f"Top features:\n" + "\n".join(f"  {i+1}. {f}" for i, f in enumerate(top5))
            )
            self.detectText.setText(summary)
            self.df_selected   = df_selected

            copy_rbc = self.raw_image.copy()
            for row_idx, info in enumerate(self.cell_info):
                x, y, w, h = info["bbox"]
                original_label = row_idx + 1
                row_data = df[df["Cell_Label"] == original_label]
                if row_data.empty: color = (128, 128, 128)
                elif row_data["IDA_Label"].values[0] == 1: color = (0, 0, 255)
                else: color = (0, 255, 0)
                cv.rectangle(copy_rbc, (x, y), (x + w, y + h), color, 5)

            self.detectResultPath = os.path.join(self.current_res_dir, f"detect_result_{self.current_patient}.png")
            cv.imwrite(self.detectResultPath, copy_rbc)

            self.pixmap = QPixmap(self.detectResultPath)
            self.detectIm.setPixmap(self.pixmap.scaled(self.detectIm.width(), self.detectIm.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.detectIm.setAlignment(Qt.AlignCenter)

            for i in range(8): self.visualIm[i].clear()
            slot = 0
            for i, (cell_img, x, y) in enumerate(self.extracted_cells):
                if slot >= 8: break
                h_img, w_img = cell_img.shape[:2]
                if h_img == 0 or w_img == 0: continue
                rgb = cv.cvtColor(cell_img, cv.COLOR_BGR2RGB) if cell_img.ndim == 3 else cell_img
                q_img = QImage(rgb.data, w_img, h_img, rgb.strides[0], QImage.Format_RGB888)
                self.visualIm[slot].setPixmap(QPixmap.fromImage(q_img).scaled(self.visualIm[slot].width(), self.visualIm[slot].height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.visualIm[slot].setAlignment(Qt.AlignCenter)
                slot += 1

        except Exception as e:
            self.detectText.setText(f"Feature selection failed: {e}")

    def generatePDF(self):
        quality_cell_count = len(self.df_features) if hasattr(self, "df_features") else 0
        cell_count = len(self.cell_info) if hasattr(self, "cell_info") else 0
        
        pdf_path = os.path.join(self.current_res_dir, f"Report_{self.current_patient}.pdf")
        
        pdf = PDFWithHeaderFooter(self.base_dir)
        pdf.generate_result(
            imagePath=self.imagePath,
            detectPath=self.detectResultPath,
            cells=cell_count,
            mal=quality_cell_count,
            parPath=self.current_sep_dir,
            output_path=pdf_path,
            patient_name=self.current_patient
        )
        self.detectText.setText(f"Report generated in PDF format at {self.current_res_dir}")

    # --- FUNGSI UI LAINNYA ---
    def setStyles(self):
        button_style = "QPushButton {border:1px; border-radius: 10px; color: rgb(0,0,0); background-color: rgb(214, 222, 255);} QPushButton:hover {background-color: rgb(35, 56, 148); color: rgb(255,255,255);} QPushButton:checked {background-color: rgb(4, 21, 98); color: rgb(255,255,255);}"
        menu_style = "QPushButton {border:1px; color: rgb(225,225,225); background-color: rgb(17, 70, 143)} QPushButton:hover {background-color: rgb(35, 56, 148); color: rgb(255,255,255);} QPushButton:checked {background-color: rgb(4, 21, 98); color: rgb(255,255,255);}"
        motor_style = "QPushButton {border:1px; border-radius: 8px; color: rgb(255,255,255); background-color: rgb(34, 139, 34);} QPushButton:hover {background-color: rgb(0, 100, 0);} QPushButton:pressed {background-color: rgb(0, 60, 0);}"
        
        self.mainPage.setStyleSheet(menu_style)
        self.segmentPage.setStyleSheet(button_style)
        self.detectPage.setStyleSheet(button_style)
        self.aboutPage.setStyleSheet(menu_style)
        for btn_name in ["btn_fast_up", "btn_fast_down", "btn_fine_up", "btn_fine_down"]:
            btn = self.findChild(QPushButton, btn_name)
            if btn: btn.setStyleSheet(motor_style)

    def moveMainPage(self):
        self.stackedWidget.setCurrentIndex(0)
    def moveSegmentPage(self):
        self.stackedWidget.setCurrentIndex(1)
    def moveExtractPage(self):
        self.stackedWidget.setCurrentIndex(2)
    def moveDetectPage(self):
        self.stackedWidget.setCurrentIndex(3)
    def moveAboutPage(self):
        self.stackedWidget.setCurrentIndex(4)

    def closeEvent(self, event):
        if self.using_picam and self.picam2 is not None:
            try: self.picam2.stop()
            except Exception: pass
        else:
            if hasattr(self, "timer") and self.timer.isActive(): self.timer.stop()
            if hasattr(self, "sensor_timer") and self.sensor_timer.isActive(): self.sensor_timer.stop()
            if hasattr(self, "cap") and self.cap.isOpened():
                try: self.cap.release()
                except Exception: pass
        if esp_serial is not None and esp_serial.is_open:
            esp_serial.close()
        super().closeEvent(event)

    def closeApp(self):
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
