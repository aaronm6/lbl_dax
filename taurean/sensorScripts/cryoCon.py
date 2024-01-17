import socket, time
from datetime import datetime

host = "192.168.1.5"        #Cryo-con ip address
port = 2000                 #Cryo-con port number

#Create socket and connect to cryo-con
def connectCryoCon(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Socket Initialized")
    s.connect((host, port))
    print("Connected")
    return s

def getCryoConTemp(channel, s):
    command = 'input? ' + channel + '\r\n'
    s.send(str.encode(command))
    currentTemp = s.recv(1024).decode()
    return currentTemp.replace("\r\n", "")

def getCryoConSetPoint(loop, s):
    command = 'loop ' + loop + ':setpt?\r\n'
    s.send(str.encode(command))
    setPoint = s.recv(1024).decode()
    return setPoint.replace("\r\n", "")

def getCryoConOutputPower(loop, s):
    command = 'loop ' + loop + ':htrread?\r\n'
    s.send(str.encode(command))
    outputPower = s.recv(1024).decode()
    return outputPower.replace("\r\n", "")

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
