import time
import board
import busio

try:
    import adafruit_vl53l0x
    SENSOR_LIB_READY = True
    print("📡 Library VL53L0X terinstal dengan baik.")
except ImportError:
    SENSOR_LIB_READY = False
    print("❌ Library adafruit-vl53l0x tidak ditemukan!")

class MagnificationSensor:
    def __init__(self):
        self.sensor = None
        self.is_connected = False
        self.last_distance = float("nan")
        self._initialize_sensor()

    def _initialize_sensor(self):
        if not SENSOR_LIB_READY:
            return

        try:
            # Menggunakan I2C standar (Pin 3 dan 5)
            i2c = busio.I2C(board.SCL, board.SDA)
            self.sensor = adafruit_vl53l0x.VL53L0X(i2c)
            self.is_connected = True
            print("✅ Hardware VL53L0X terdeteksi dan aktif.")
        except Exception as e:
            print(f"❌ Gagal konek hardware VL53L0X: {e}")
            self.is_connected = False

    def read_distance(self) -> float:
        if not self.is_connected or self.sensor is None:
            return float('nan')
        try:
            # Mengambil jarak dalam satuan mm
            distance = self.sensor.range
            self.last_distance = distance
            return float(distance)
        except Exception:
            return float('nan')

if __name__ == "__main__":
    s = MagnificationSensor()
    while True:
        print(f"Jarak: {s.read_distance()} mm")
        time.sleep(0.5)
