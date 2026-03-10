import time
import board
import busio
import digitalio # <--- WAJIB UNTUK XSHUT
try:
    import adafruit_vl53l0x
    LIBRARY_INSTALLED = True
    print("📡 VL53L0X library found")
except ImportError:
    LIBRARY_INSTALLED = False
    print("❌ Library adafruit_vl53l0x NOT found!")

class MagnificationSensor:
    def __init__(self, xshut_pin=board.D4): # Default GPIO 4 (Pin 7)
        self.sensor = None
        self.is_connected = False
        self.last_distance = float("nan")
        
        # 1. Inisialisasi Pin XSHUT (Hardware 1-ke-1)
        try:
            self.xshut = digitalio.DigitalInOut(xshut_pin)
            self.xshut.direction = digitalio.Direction.OUTPUT
            # Langkah Reset: Matikan lalu Nyalakan
            self.xshut.value = False
            time.sleep(0.1)
            self.xshut.value = True
            time.sleep(0.1)
            print(f"✅ XSHUT pin (GPIO {xshut_pin}) initialized and set to HIGH")
        except Exception as e:
            print(f"❌ Failed to control XSHUT pin: {e}")

        # 2. Coba koneksi ke Sensor
        self._initialize_sensor()

    def _initialize_sensor(self):
        if not LIBRARY_INSTALLED:
            print("⚠️ Cannot initialize: Library missing.")
            return

        try:
            # Setup I2C
            i2c = busio.I2C(board.SCL, board.SDA)
            
            # CEK HARDWARE: Apakah ada device di alamat 0x29?
            # Ini adalah bagian "cek apakah sensor ada atau tidak"
            while not i2c.try_lock():
                pass
            devices = i2c.scan()
            i2c.unlock()

            if 0x29 not in devices:
                print("❌ HARDWARE NOT FOUND: Sensor is not detected on I2C bus (Check wiring/XSHUT)")
                self.is_connected = False
                return

            # Jika hardware ditemukan, baru inisialisasi library
            self.sensor = adafruit_vl53l0x.VL53L0X(i2c)
            self.is_connected = True
            print("✅ VL53L0X hardware detected and initialized successfully")

        except Exception as e:
            print(f"❌ Initialization Error: {e}")
            self.is_connected = False

    def read_distance(self) -> float:
        if not self.is_connected or self.sensor is None:
            return float('nan')
        try:
            return self.sensor.range
        except Exception as e:
            print(f"❌ Read error: {e}")
            return float('nan')

if __name__ == "__main__":
    sensor = MagnificationSensor()
    if sensor.is_connected:
        try:
            while True:
                dist = sensor.read_distance()
                print(f"Distance: {dist} mm")
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("Stop")
    else:
        print("Sistem berhenti karena sensor tidak terdeteksi.")
