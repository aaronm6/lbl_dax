import socket
from datetime import datetime
import time
#TODO: Make error handling 
class omegaLoadSensor:
    # Create sensor object and make connection to specified host and port
    def __init__(self, host, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect((host,int(port)))
        except:
            raise TimeoutError("Can't connect to Omega Pressure, check device connection and IP address")

    def getOmegaSensorLoad(self):
        command = '*X01\r\n'
        self.socket.sendall(str.encode(command))
        pressure = self.socket.recv(256).decode()
        return pressure

    def close(self):
        self.socket.close()

# Old Test Loop Code
host = "192.168.1.201"
port = "2000"
while True:
    try:
        s = omegaLoadSensor(host, port)
        strain = s.getOmegaSensorLoad()
        currentTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("Current Time: " + currentTime)
        testString = "Current Load: " + str(float(strain)) + " kg"
        print(testString)
        time.sleep(2)
    except KeyboardInterrupt:
        break
s.close()