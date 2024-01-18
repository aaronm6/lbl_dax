import serial

#TODO: Make error handling
class pfeifferPressureSensor:
    # Create sensor object and make connection to specified serial port
    # Always check if serial port is correct if usb was unplugged
    def __init__(self, serialPort):
        # Communication constants (see Pfeiffer Manual for more details)
        self.ETX = chr(3)    # \x03
        self.CR = chr(13)    # \r
        self.LF = chr(10)    # \n
        self.ENQ = chr(5)    # \x05
        self.ACK = chr(6)    # \x06
        self.NAK = chr(21)   # \x15

        # Serial connection initialization
        self.serial = serial.Serial(serialPort)

    def getPfeifferPressure(self, channel):
        command = "PRX" + self.CR + self.LF
        self.serial.write(str.encode(command))
        self.serial.readline()
        self.serial.write(str.encode(self.ENQ))
        res = self.serial.readline().decode()
        pressure = res.split(",")[2*channel - 1]
        return pressure

    def close(self):
        self.serial.close()


# with psycopg.connect("dbname=sensor_readings user=postgres password=LZ4850") as conn:
#     conn.autocommit = True
#     with conn.cursor() as cur:
#         startTime = datetime.now()
#         currentTime = datetime.now()
#         readingHours = 3
#         differenceTime = currentTime - startTime
#         s = serial.Serial(serialPort)
#         while (differenceTime.total_seconds() < 3600*readingHours):
#             if (differenceTime.total_seconds() % 600):
#                 print("10 minutes have passed")
#             s.write(str.encode("PRX" + CR + LF))
#             res = s.readline()
#             s.write(str.encode(ENQ))
#             res = s.readline().decode()
#             pressure = res.split(",")[1]
#             cur.execute(
#                 "INSERT INTO temppressure (timestamp, pressure) VALUES (%s, %s)",
#                 (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), pressure))
#             time.sleep(5)
#         cur.close()
#     conn.close()