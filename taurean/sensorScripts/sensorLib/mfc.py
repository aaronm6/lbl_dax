import serial

#TODO: Make error handling
class mfc:
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
        try:
            self.serial = serial.Serial(serialPort)
        except:
            raise TimeoutError("Can't connect to mfc, check device connection and com port")

    def getPfeifferPressure(self, channel):
        command = "TESTING" + self.CR + self.LF
        self.serial.write(str.encode(command))
        self.serial.readline()
        self.serial.write(str.encode(self.ENQ))
        res = self.serial.readline().decode()
        print(res)
        return res

    def close(self):
        self.serial.close()

mfcConn = mfc("COM3")
mfcConn.getPfeifferPressure("1")
