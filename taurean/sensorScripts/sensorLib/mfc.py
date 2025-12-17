import serial, time, psycopg
from datetime import datetime, timezone

#TODO: Make error handling
class mfc:
    # Create sensor object and make connection to specified serial port
    # Always check if serial port is correct if usb was unplugged
    def __init__(self, serialPort):
        # Serial connection initialization
        try:
            self.serial = serial.Serial(port=serialPort, baudrate=9600, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE)
            self.serial.timeout = 0.3
        except:
            raise TimeoutError("Can't connect to mfc, check device connection and com port")

    def setMaxFlowRate(self, newRate):
        #AZ.02P1=500 flow rate change cmd
        command = "AZ.02P1="+str(newRate)+"00\r\n"
        self.serial.write(str.encode(command))
        time.sleep(1)
        res = self.serial.read(2048)
        return newRate

    def getFlowRate(self):
        command = "AZR\r\n"
        buffer = self.serial.write(str.encode(command))
        res = self.serial.read(2048).decode()

        while (res == ""):
            buffer = self.serial.write(str.encode(command))
            time.sleep(0.1)
            res = self.serial.read(2048).decode()
        temp = res.split(",")
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

""" while True:
    mfcConn = mfc("/dev/ttyUSB0")
# # mfcConn.setMaxFlowRate(1500)
    flowRate = mfcConn.getFlowRate(0)
    currentTime = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    mfcConn.close()
    databaseLogin = ("dbname=sensor_readings user=postgres password=LZ4850")
    databaseConn = psycopg.connect(databaseLogin)
    databaseConn.autocommit = True
    databaseCur = databaseConn.cursor()
    print(flowRate)
    query = "insert into mfc values ('" + currentTime + "', '" + flowRate + "', '0', '0')"
    databaseCur.execute(query)
    databaseCur.close()
    databaseConn.close() """