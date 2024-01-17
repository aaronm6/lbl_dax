import socket, time
from datetime import datetime

host = "192.168.1.200"        #Omega Sensor IP Address
port = 2000                 #Omega Sensor Port Number

#Create socket and connect to cryo-con
def connectOmegaSensor(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Socket Initialized")
    s.connect((host, port))
    print("Connected")
    return s

def getOmegaSensorReading(s):
    command = '*G110\r'
    s.sendall(str.encode(command))
    currentTemp = s.recv(1024).decode()
    return currentTemp.replace("\r", "")

# while True:
#     try:
#         s = connect(host, port)
#         temp = getCurrentTemp("a", s)
#         setPoint = getSetPoint("1", s)
#         outputPower = getCurrentOutputPower("1", s)
#         currentTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         print("Current Time: " + currentTime)
#         print("Current Temp: %sK" % temp)
#         print("Current Setpoint: %s" % setPoint)
#         print("Current Output Power:%s" % outputPower)
#         print("------------------------------------------")
#         time.sleep(2)
#     except KeyboardInterrupt:
#         break
# s.close()
