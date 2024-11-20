import serial, time

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

    def setMaxFlowRate(self, newRate):
        #AZ.02P1=500 flow rate change cmd
        command = "AZ.02P1="+str(newRate)+"\r\n"
        self.serial.write(str.encode(command))
        print("test")
        time.sleep(1)
        res = self.serial.readline()
        print(res)
        while (res):
            res = self.serial.readline()
            print(res)

    def getFlowRate(self):
        command = "AZR\r\n"
        print("TEST")
        self.serial.write(str.encode(command))
        print("test")
        res = self.serial.readline().decode()
        while (not res):
            res = self.serial.readline().decode()

        temp = res.split(",")
        # print(temp)
        # print(temp[3].replace(" ", ""))
        flowRate = temp[3].replace(" ", "")

        # Code to get units of flow rate if needed later
        # command = "AZ.00V\r\n"
        # self.serial.write(str.encode(command))
        # print("test")

        # res = self.serial.readline().decode()
        # units = ""
        # while (res):
        #     if ("Units" in res):
        #         temp = res.replace(" ", "").split("Units")
        #         units = temp[1].replace("\r\n", "")
        #     if ("Time Base" in res):
        #         temp = res.replace(" ", "").split("TimeBase")
        #         units = units + "/" + temp[1].replace("\r\n", "")
        #     res = self.serial.readline().decode()
        # print(units)

        return flowRate

    def close(self):
        self.serial.close()

# mfcConn = mfc("/dev/ttyUSB0")
# # mfcConn.setMaxFlowRate(1500)
# mfcConn.getFlowRate()
# mfcConn.close()
