import time
import board
import busio
from adafruit_vl53l0x import VL53L0X

# Inisialisasi I2C
i2c = busio.I2C(board.SCL, board.SDA)

# Panggil sensor (Langsung deteksi)
try:
    sensor = VL53L0X(i2c)
    print("✅ Hardware OK! Jarak kebaca.")
    while True:
        print(f"Distance: {sensor.range} mm")
        time.sleep(0.5)
except Exception as e:
    print(f"❌ Gagal: {e}")
