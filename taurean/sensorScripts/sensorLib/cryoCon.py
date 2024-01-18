import socket

#TODO: Make error handling
class cryoCon:
    # Create sensor object and make connection to specified host and port
    def __init__(self, host, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host,int(port)))

    # Get temperature reading for a specific channel
    def getCryoConTemp(self, channel):
        command = 'input? ' + channel + '\r\n'
        self.socket.send(str.encode(command))
        currentTemp = self.socket.recv(1024).decode()
        return currentTemp.replace("\r\n", "")

    # Get set point for a specific loop
    def getCryoConSetPoint(self, loop):
        command = 'loop ' + loop + ':setpt?\r\n'
        self.socket.send(str.encode(command))
        setPoint = self.socket.recv(1024).decode()
        return setPoint.replace("\r\n", "").replace("K", "")

    # Get output power for a specific loop
    def getCryoConOutputPower(self, loop):
        command = 'loop ' + loop + ':htrread?\r\n'
        self.socket.send(str.encode(command))
        outputPower = self.socket.recv(1024).decode()
        return outputPower.replace("\r\n", "").replace(" ", ""). replace("%", "")

    def close(self):
        self.socket.close()
# Old Test Loop Code
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
