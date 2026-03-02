import time
try:
    import board
    import busio
    import adafruit_vl53l0x
    SENSOR_AVAILABLE = True
    print("📡 VL53L0X sensor modules available")
except (ImportError, NotImplementedError, RuntimeError) as e:
    SENSOR_AVAILABLE = False

class MagnificationSensor:

    def __init__(self):
        """
        Initialize magnification sensor

        Args:
            reference_distance (float): Reference distance in mm for calibration
            reference_magnification (float): Magnification at reference distance
        """
        self.sensor = None
        self.is_connected = False
        self.last_distance = float("nan")

        # Try to initialize sensor
        self._initialize_sensor()

    def _initialize_sensor(self):
        """Initialize the VL53L0X sensor"""
        if not SENSOR_AVAILABLE:
            print("📡 Sensor not available - using simulation mode")
            return

        try:
            # Initialize I2C and sensor
            i2c = busio.I2C(board.SCL, board.SDA)
            self.sensor = adafruit_vl53l0x.VL53L0X(i2c)
            self.is_connected = True
            print("✅ VL53L0X sensor initialized successfully")

            # Test read
            test_distance = self.sensor.range
            print(f"📏 Initial sensor reading: {test_distance}mm")

        except (Exception, NotImplementedError, RuntimeError) as e:
            print(f"❌ Failed to initialize VL53L0X sensor: {e}")
            self.is_connected = False
            self.sensor = None

    def read_distance(self) -> float:
        try:
            distance = self.sensor.range
            self.last_distance = distance
            return distance
        except Exception as e:
            print(f"❌ Sensor read error: {e}")
            return float('nan')


if __name__ == "__main__":
    print("\nStarting continuous test for MagnificationSensor...\n")
    sensor = MagnificationSensor()

    try:
        while True:
            distance = sensor.read_distance()
            print(f"Distance: {distance:.2f} mm")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopped by user")
