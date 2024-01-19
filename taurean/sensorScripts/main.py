import psycopg, time, json, socket
from datetime import datetime
from sensorLib import cryoCon, omegaPressureSensor, pfeifferPressureSensor

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


# Parse configuration details from cfg.json
with open("cfg.json") as json_data:
    data = json.load(json_data)
    takeInterval = data["data_output_settings"]["time_between_measurements"]

    # Set up all serial and ethernet based connections
    sensorDetails = data["sensor_details"]
    try:
        cryoConDev = cryoCon.cryoCon(sensorDetails["cryo_con"]["ip_address"], sensorDetails["cryo_con"]["port"])
        omegaPressureDev = omegaPressureSensor.omegaPressureSensor(sensorDetails["omega_pressure"]["ip_address"], sensorDetails["omega_pressure"]["port"])
        pfeifferDev = pfeifferPressureSensor.pfeifferPressureSensor(sensorDetails["pfeiffer_pressure"]["serial_port"])
    except socket.timeout:
        raise TimeoutError("One or more devices can't be connected to")

    # Get channel and loop details
    cryoConChannels = sensorDetails["cryo_con"]["channels"]
    cryoConLoops = sensorDetails["cryo_con"]["loops"]
    pfeifferChannels = sensorDetails["pfeiffer_pressure"]["channels"]

    # Make database login string
    credentials = data["database_details"]["credentials"]
    databaseLogin = ("dbname=" + credentials["database"] + " user=" + 
        credentials["username"] + " password=" + credentials["password"])
    databaseConn = psycopg.connect(databaseLogin)
    databaseConn.autocommit = True
    databaseCur = databaseConn.cursor()

    # Get table names
    cryoConTable = data["database_details"]["cryo_con"]["table_name"]
    omegaPressureTable = data["database_details"]["omega_pressure"]["table_name"]
    pfeifferTable = data["database_details"]["pfeiffer_pressure"]["table_name"]

    # Get column names
    cryoConColumns = data["database_details"]["cryo_con"]["column_names"]
    omegaPressureColumns = data["database_details"]["omega_pressure"]["column_names"]
    pfeifferColumns = data["database_details"]["pfeiffer_pressure"]["column_names"]

counter = 0
try:
    while 1:
        cryoConVals = []
        omegaPressureVals = []
        pfeifferVals = []

        currentTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        #TODO: Make loops and channels configurable in cfg.json later
        # Get cryo con vals and send them to database

        # Read set point, output power, and current temperature
        cryoConVals.append(currentTime)
        for loop in cryoConLoops:
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
        databaseCur.execute(makeInsertQuery(pfeifferTable, pfeifferColumns, pfeifferVals))
        pfeifferVals = []

        counter = counter + 1
        time.sleep(takeInterval)

        if counter % 5 == 0:
            print("5 submissions made")

except Exception as e:
    print(e)
    # Shutdown when exception is triggered
    cryoConDev.close()
    pfeifferDev.close()
    omegaPressureDev.close()
    databaseConn.close()
    databaseCur.close()
    print("Shutting down sensor data gathering")