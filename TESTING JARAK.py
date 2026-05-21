import time
import board
import busio
import adafruit_vl53l0x

# 1. Inisialisasi jalur komunikasi I2C di Raspberry Pi
i2c = busio.I2C(board.SCL, board.SDA)

# 2. Inisialisasi sensor VL53L0X
try:
    sensor = adafruit_vl53l0x.VL53L0X(i2c)
    print("✅ Sensor VL53L0X terdeteksi. Mulai membaca data...")
except Exception as e:
    print(f"❌ Gagal menemukan sensor. Cek kabel SDA/SCL. Error: {e}")
    exit()

# 3. Looping untuk membaca jarak dan menghitung perbesaran
while True:
    try:
        # Ambil data jarak dari sensor (dalam milimeter)
        distance = sensor.range
        
        # Validasi jarak (mencegah pembagian dengan nol atau angka ngawur)
        if 0 < distance < 1000: 
            # Rumus kalibrasi dokumen TA kamu
            magnification = 60000.0 / distance
            
            # Print hasil persis seperti format di screenshot kamu
            print(f"Jarak sensor: {distance:.2f} mm | Estimasi perbesaran: {magnification:.1f}x")
        else:
            print(" out of range ")
            
    except Exception as e:
        print(f"Gagal membaca data: {e}")
        
    time.sleep(0.5) # Jeda setengah detik biar terminal nggak nge-spam terlalu cepat
