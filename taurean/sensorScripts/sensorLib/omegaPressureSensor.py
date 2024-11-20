import socket

#TODO: Make error handling 
class omegaPressureSensor:
    # Create sensor object and make connection to specified host and port
    def __init__(self, host, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect((host,int(port)))
        except:
            raise TimeoutError("Can't connect to Omega Pressure, check device connection and IP address")

    def getOmegaSensorPressure(self):
        command = '*G110\r'
        self.socket.sendall(str.encode(command))
        pressure = self.socket.recv(1024).decode()
        return str((float(pressure)+0.01)*517.149326) #converts from psi to torr

    def close(self):
        self.socket.close()

# dev = omegaPressureSensor("192.168.1.200", 2000)
# print(dev.getOmegaSensorPressure())
# dev.close()