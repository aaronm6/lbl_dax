import socket

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
        pressure = self.socket.recv(1024).decode()
        return pressure

    def close(self):
        self.socket.close()