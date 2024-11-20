import psycopg, time, json, socket
from datetime import datetime, timezone
from sensorLib import cryoCon, omegaPressureSensor, pfeifferPressureSensor, strainGauge

# Make insert query for postgres 
def makeInsertQuery(tableName, columnList, valueList):
    columnString = "("
    valueString = "("

    for i in range(len(columnList)):
        columnString = columnString + columnList[i]
        valueString = valueString + "'" + valueList[i] + "'"
        if (i < len(columnList) - 1):
            columnString = columnString + ", "
            valueString = valueString + ", "
    columnString = columnString + ")"
    valueString = valueString + ")"
    
    query = ("INSERT INTO " + tableName + " " + columnString 
        + " VALUES " + valueString)
    return query


def collectCryoCon(sensorDetails, cfgTableData, cursor):
    try:
        print("Attempting to connect to cryo con")
        cryoConDev = cryoCon.cryoCon(sensorDetails["cryo_con"]["ip_address"], sensorDetails["cryo_con"]["port"])
    except socket.timeout:
        raise TimeoutError("Cryo Con couldn't be connected to, check connections and ip addresses and try again later")

    print("Connection successful")

    # Get channel and loop details
    cryoConChannels = sensorDetails["cryo_con"]["channels"]
    cryoConLoops = sensorDetails["cryo_con"]["loops"]

    #Get table details
    cryoConTable = cfgTableData["database_details"]["cryo_con"]["table_name"]
    cryoConColumns = cfgTableData["database_details"]["cryo_con"]["column_names"]

    cryoConVals = []
    currentTime = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    cryoConVals.append(currentTime)

    #Get data
    print(cryoConLoops)
    for loop in cryoConLoops:
        if (cryoConDev.getCryoConTemp(loop) == "......."):
            #print("One or more loop(s) are disconnected, check cryo con.")
            cryoConVals.append('0')
        else:
            cryoConVals.append(cryoConDev.getCryoConTemp(loop))

    for channel in cryoConChannels:
        cryoConVals.append(cryoConDev.getCryoConSetPoint(channel))
        outputPower = cryoConDev.getCryoConOutputPower(channel)
        if outputPower == "0":
            cryoConVals.append("False")
        else:
            cryoConVals.append("True")
        cryoConVals.append(outputPower)
    print(cryoConVals)
    cursor.execute(makeInsertQuery(cryoConTable, cryoConColumns, cryoConVals))
    cryoConDev.close()

def collectOmegaPressure(sensorDetails, cursor):
    try:
        omegaPressureDev = omegaPressureSensor.omegaPressureSensor(sensorDetails["omega_pressure"]["ip_address"], sensorDetails["omega_pressure"]["port"])
    except socket.timeout:
        raise TimeoutError("Couldn't connect to omega pressure readout, check connections and ip addresses and try again later")
    
    omegaPressureTable = data["database_details"]["omega_pressure"]["table_name"]
    omegaPressureColumns = data["database_details"]["omega_pressure"]["column_names"]

    currentTime = datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%S')

def collectPfeiffer(sensorDetails, cursor):
    try:
        pfeifferDev = pfeifferPressureSensor.pfeifferPressureSensor(sensorDetails["pfeiffer_pressure"]["serial_port"])
    except socket.timeout:
        raise TimeoutError("Couldn't connect to pfeiffer pressure readout, check connections and serial port address and try again later")

    pfeifferChannels = sensorDetails["pfeiffer_pressure"]["channels"]

    pfeifferTable = data["database_details"]["pfeiffer_pressure"]["table_name"]
    pfeifferColumns = data["database_details"]["pfeiffer_pressure"]["column_names"]

    currentTime = datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%S')

def collectStrainGauge(sensorDetails, cursor):
    try:
        loadSensorDev1 = strainGauge.strainGauge(sensorDetails["load_sensor_1"]["ip_address"], sensorDetails["load_sensor_1"]["port"])
        loadSensorDev2 = strainGauge.strainGauge(sensorDetails["load_sensor_2"]["ip_address"], sensorDetails["load_sensor_2"]["port"])
    except socket.timeout:
        raise TimeoutError("Couldn't connect to one or more strain gauges, check connections and ip addresses and try again later")

    loadSensorTable = data["database_details"]["load_sensors"]["table_name"]
    loadSensorColumns = data["database_details"]["load_sensors"]["column_names"]

    currentTime = datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%S')

# Parse configuration details from cfg.json
with open("cfg.json") as json_data:
    data = json.load(json_data)
    takeInterval = data["data_output_settings"]["time_between_measurements"]
    sensorDetails = data["sensor_details"]

    # Make database login string
    credentials = data["database_details"]["credentials"]
    databaseLogin = ("dbname=" + credentials["database"] + " user=" + 
        credentials["username"] + " password=" + credentials["password"])
    databaseConn = psycopg.connect(databaseLogin)
    databaseConn.autocommit = True
    databaseCur = databaseConn.cursor()

if sensorDetails["cryo_con"]["enabled"] != "True":
    print("Cryo Con is disabled in cfg, please change this if this is an error")

if sensorDetails["omega_pressure"]["enabled"] != "True":
    print("Cryo Con is disabled in cfg, please change this if this is an error")

if sensorDetails["pfeiffer_pressure"]["enabled"] != "True":
    print("Cryo Con is disabled in cfg, please change this if this is an error")

if sensorDetails["load_sensor_1"]["enabled"] != "True":
    print("Cryo Con is disabled in cfg, please change this if this is an error")

if sensorDetails["cryo_con"]["enabled"] != "True":
    print("Cryo Con is disabled in cfg, please change this if this is an error")

counter = 0
try:
    while 1:
        if sensorDetails["cryo_con"]["enabled"] == "True":
            collectCryoCon(sensorDetails, data, databaseCur)
        time.sleep(takeInterval)
except Exception as e:
    print(e)
    # Shutdown when exception is triggered
    databaseConn.close()
    databaseCur.close()
    print("Shutting down sensor data gathering")



"""         cryoConVals = []
        omegaPressureVals = []
        pfeifferVals = []
        loadSensorVals = []

        currentTime = datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%S')

        #TODO: Make loops and channels configurable in cfg.json later
        # Get cryo con vals and send them to database

        # Read set point, output power, and current temperature
        cryoConVals.append(currentTime)
        for loop in cryoConLoops:
            if (cryoConDev.getCryoConTemp(loop) == "......."):
                #print("One or more loop(s) are disconnected, check cryo con.")
                cryoConVals.append('0')
            else:
                cryoConVals.append(cryoConDev.getCryoConTemp(loop))
        for channel in cryoConChannels:
            cryoConVals.append(cryoConDev.getCryoConSetPoint(channel))
            cryoConVals.append(cryoConDev.getCryoConOutputPower(channel))
        databaseCur.execute(makeInsertQuery(cryoConTable, cryoConColumns, cryoConVals))
        cryoConVals = []

        # Read pressure
        omegaPressureVals.append(currentTime)
        omegaPressureVals.append(omegaPressureDev.getOmegaSensorPressure())
        databaseCur.execute(makeInsertQuery(omegaPressureTable, omegaPressureColumns, omegaPressureVals))
        omegaPressureVals = []

        # Read vacuum
        pfeifferVals.append(currentTime)
        for channel in pfeifferChannels:
            pfeifferVals.append(pfeifferDev.getPfeifferPressure(channel))
        print(pfeifferVals)
        databaseCur.execute(makeInsertQuery(pfeifferTable, pfeifferColumns, pfeifferVals))
        pfeifferVals = []
        # Read load sensors
        loadSensorVals.append(currentTime)
        loadSensorVals.append(loadSensorDev1.getOmegaSensorLoad())
        loadSensorVals.append('0')
        #loadSensorVals.append(loadSensorDev1.getOmegaSensorLoad())
        databaseCur.execute(makeInsertQuery(loadSensorTable, loadSensorColumns, loadSensorVals))
        loadSensorVals = []


        counter = counter + 1
        time.sleep(takeInterval)

        if counter % 5 == 0:
            print("5 submissions made")
"""