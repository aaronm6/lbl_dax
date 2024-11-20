from sensorLib import cryoCon, omegaPressureSensor, pfeifferPressureSensor, strainGauge
import json

# Parse configuration details from cfg.json
with open("cfg.json") as json_data:
    data = json.load(json_data)
    takeInterval = data["data_output_settings"]["time_between_measurements"]
    sensorDetails = data["sensor_details"]

try:
    print("Attempting to connect to cryo con")
    cryoConDev = cryoCon.cryoCon(sensorDetails["cryo_con"]["ip_address"], sensorDetails["cryo_con"]["port"])
except socket.timeout:
    raise TimeoutError("Cryo Con couldn't be connected to, check connections and ip addresses and try again later")

cryoConLoops = sensorDetails["cryo_con"]["loops"]
print(cryoConLoops)
temp = "160"
for loop in cryoConLoops:
    cryoConDev.setCryoConSetPoint(loop, temp)
    print("Loop " + loop + " set point changed to "+ temp + " K")