import serial

#TODO: Make error handling
class mfc:
    # Create sensor object and make connection to specified serial port
    # Always check if serial port is correct if usb was unplugged
    def __init__(self, serialPort):
        # Serial connection initialization
        try:
            self.serial = serial.Serial(serialPort)
            self.serial.timeout = 1.0
        except:
            raise TimeoutError("Can't connect to mfc, check device connection and com port")

    def getFlowRate(self):
        command = "@@@254F?;FF"
        print("TEST")
        self.serial.write(str.encode(command))
        print("test")
        res = self.serial.readline().decode()
        print(res)
        return res

    def close(self):
        self.serial.close()

mfcConn = mfc("COM4")
mfcConn.getFlowRate()
mfcConn.close()
