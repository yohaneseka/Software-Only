import time
import board
import busio
import digitalio
from adafruit_vl53l0x import VL53L0X

# 1. Bangunkan si VLX lewat Pin XSHUT (GPIO 4)
xshut = digitalio.DigitalInOut(board.D4)
xshut.direction = digitalio.Direction.OUTPUT
xshut.value = True # Set ke HIGH untuk menyalakan
time.sleep(0.2)

# 2. Inisialisasi I2C dan Sensor
i2c = busio.I2C(board.SCL, board.SDA)

try:
    sensor = VL53L0X(i2c)
    print("✅ SENSOR TERDETEKSI!")
    
    while True:
        print(f"Jarak saat ini: {sensor.range} mm")
        time.sleep(0.5)
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("Coba cek kabel SDA/SCL atau apakah Pin 7 sudah ke XSHUT?")
