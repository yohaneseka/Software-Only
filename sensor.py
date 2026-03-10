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
        """Initialize the VL53L0X sensor with Hardware Check"""
        if not SENSOR_AVAILABLE:
            print("📡 Library missing - Check installation of adafruit-blinka")
            return

        try:
            # BAGIAN PERBAIKAN: Inisialisasi Pin XSHUT (Gaya Kating 1-ke-1)
            # Hubungkan XSHUT VLX ke Pin 7 (GPIO 4) Raspberry Pi
            import digitalio
            xshut = digitalio.DigitalInOut(board.D4)
            xshut.direction = digitalio.Direction.OUTPUT
            xshut.value = True # Membangunkan sensor
            time.sleep(0.2)

            # Inisialisasi I2C
            i2c = busio.I2C(board.SCL, board.SDA)
            
            # BAGIAN PERBAIKAN: Cek apakah hardware terdeteksi di kabel
            # Scan semua alamat I2C yang aktif
            while not i2c.try_lock():
                pass
            found_devices = i2c.scan()
            i2c.unlock()

            if 0x29 not in found_devices:
                print("❌ HARDWARE TIDAK DITEMUKAN! Periksa kabel SDA/SCL/XSHUT.")
                self.is_connected = False
                return

            # Jika hardware ditemukan, baru jalankan driver-nya
            self.sensor = adafruit_vl53l0x.VL53L0X(i2c)
            self.is_connected = True
            print("✅ VL53L0X hardware detected and initialized!")

        except Exception as e:
            print(f"❌ Failed to initialize VL53L0X sensor: {e}")
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

