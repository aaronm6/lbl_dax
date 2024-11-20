import multiprocessing, time, sys, psycopg, json, socket
import numpy as np
from datetime import datetime, timezone
from sensorLib import cryoCon, omegaPressureSensor, pfeifferPressureSensor, strainGauge, mfc

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
        cryoConDev = cryoCon.cryoCon(sensorDetails["cryo_con"]["ip_address"], sensorDetails["cryo_con"]["port"])
    except socket.timeout:
        raise TimeoutError("Cryo Con couldn't be connected to, check connections and ip addresses and try again later")

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

def changeCryoConSetpt(sensorDetails, channel, temperature):
    try:
        cryoConDev = cryoCon.cryoCon(sensorDetails["cryo_con"]["ip_address"], sensorDetails["cryo_con"]["port"])
    except socket.timeout:
        raise TimeoutError("Cryo Con couldn't be connected to, check connections and ip addresses and try again later")
    setpt = cryoConDev.setCryoConSetPoint(channel, temperature)
    print("Temperature set point has been changed to " + str(temperature) + " for channel " + channel + ".")

def collectOmegaPressure(sensorDetails, cfgTableData, cursor):
    try:
        omegaPressureDev = omegaPressureSensor.omegaPressureSensor(sensorDetails["omega_pressure"]["ip_address"], sensorDetails["omega_pressure"]["port"])
    except socket.timeout:
        raise TimeoutError("Couldn't connect to omega pressure readout, check connections and ip addresses and try again later")
    
    omegaPressureTable = cfgTableData["database_details"]["omega_pressure"]["table_name"]
    omegaPressureColumns = cfgTableData["database_details"]["omega_pressure"]["column_names"]

    omegaPressureVals = []
    currentTime = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    omegaPressureVals.append(currentTime)
    omegaPressureVals.append(omegaPressureDev.getOmegaSensorPressure())

    cursor.execute(makeInsertQuery(omegaPressureTable, omegaPressureColumns, omegaPressureVals))

    omegaPressureDev.close()

def collectPfeiffer(sensorDetails, cfgTableData, cursor):
    try:
        pfeifferDev = pfeifferPressureSensor.pfeifferPressureSensor(sensorDetails["pfeiffer_pressure"]["serial_port"])
    except socket.timeout:
        raise TimeoutError("Couldn't connect to pfeiffer pressure readout, check connections and serial port address and try again later")

    pfeifferChannels = sensorDetails["pfeiffer_pressure"]["channels"]

    pfeifferTable = cfgTableData["database_details"]["pfeiffer_pressure"]["table_name"]
    pfeifferColumns = cfgTableData["database_details"]["pfeiffer_pressure"]["column_names"]

    pfeifferVals = []
    currentTime = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    pfeifferVals.append(currentTime)
    pfeifferVals.append(pfeifferDev.getPfeifferPressure())

    cursor.execute(makeInsertQuery(pfeifferTable, pfeifferColumns, pfeifferVals))
    pfeifferDev.close()

def collectMfc(sensorDetails, cfgTableData, cursor, startTime):
    try:
        mfcDev = mfc.mfc(sensorDetails["mfc"]["serial_port"])
    except socket.timeout:
        raise TimeoutError("Couldn't connect to pfeiffer pressure readout, check connections and serial port address and try again later")

    mfcTable = cfgTableData["database_details"]["mfc"]["table_name"]
    mfcColumns = cfgTableData["database_details"]["mfc"]["column_names"]

    mfcVals = []
    mfcVals.append(0)
    mfcVals.append(mfcDev.getFlowRate())
    currentTime = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    mfcVals[0] = currentTime
    query = "select timestamp, current_flow_rate from mfc where timestamp between '" + startTime + "' and '" + currentTime + "'"
    cursor.execute(query)
    flowRates = cursor.fetchall()

    if len(flowRates) > 1:
        timeArr = []
        flowArr = []
        startOfIntegration = flowRates[0][0].timestamp()
        for row in flowRates:
            timeArr.append(row[0].timestamp() - startOfIntegration)
            flowArr.append(row[1])
        totalVolume = np.trapz(flowArr, timeArr)
        scalingFactor = 0.00586
        mfcVals.append(str(totalVolume))
        #TODO: FIND SCALING FACTOR FOR XENON VOLUME TO MASS
        mfcVals.append(str(totalVolume * scalingFactor))
    else:
        mfcVals.append("0")
        mfcVals.append("0")
    cursor.execute(makeInsertQuery(mfcTable, mfcColumns, mfcVals))
    mfcDev.close()

def changeFlowRate(sensorDetails, flowRate):
    try:
        mfcDev = mfc.mfc(sensorDetails["mfc"]["serial_port"])
    except socket.timeout:
        raise TimeoutError("Couldn't connect to pfeiffer pressure readout, check connections and serial port address and try again later")

    mfcDev.setMaxFlowRate(flowRate)
    print("Max flow rate has been changed to " + str(flowRate) + ".")

def collectStrainGauge(sensorDetails, cfgTableData, cursor):
    try:
        loadSensorDev1 = strainGauge.strainGauge(sensorDetails["load_sensor_1"]["ip_address"], sensorDetails["load_sensor_1"]["port"])
        loadSensorDev2 = strainGauge.strainGauge(sensorDetails["load_sensor_2"]["ip_address"], sensorDetails["load_sensor_2"]["port"])
    except socket.timeout:
        raise TimeoutError("Couldn't connect to one or more strain gauges, check connections and ip addresses and try again later")

    loadSensorTable = cfgTableData["database_details"]["load_sensors"]["table_name"]
    loadSensorColumns = cfgTableData["database_details"]["load_sensors"]["column_names"]

    loadSensorVals = []
    currentTime = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    loadSensorVals.append(currentTime)
    loadSensorVals.append(loadSensorDev1.getStrainGaugeLoad())
    loadSensorVals.append(loadSensorDev2.getStrainGaugeLoad())
    cursor.execute(makeInsertQuery(loadSensorTable, loadSensorColumns, loadSensorVals))
    #print(makeInsertQuery(loadSensorTable, loadSensorColumns, loadSensorVals))
    loadSensorDev1.close()
    loadSensorDev2.close()

def monitoringLoop():
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
        print("Omega pressure sensor is disabled in cfg, please change this if this is an error")

    if sensorDetails["pfeiffer_pressure"]["enabled"] != "True":
        print("Pfeiffer pressure sensor in cfg, please change this if this is an error")

    if sensorDetails["load_sensor_1"]["enabled"] != "True":
        print("Strain gauge for bottle 1 is disabled in cfg, please change this if this is an error")

    if sensorDetails["load_sensor_2"]["enabled"] != "True":
        print("Strain gauge for bottle 2 is disabled in cfg, please change this if this is an error")

    if sensorDetails["mfc"]["enabled"] != "True":
        print("MFC is disabled in cfg, please change this if this is an error")

    startTime = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    try:
        while 1:
            if sensorDetails["cryo_con"]["enabled"] == "True":
                collectCryoCon(sensorDetails, data, databaseCur)

            if sensorDetails["omega_pressure"]["enabled"] == "True":
                collectOmegaPressure(sensorDetails, data, databaseCur)

            if sensorDetails["pfeiffer_pressure"]["enabled"] == "True":
                collectPfeiffer(sensorDetails, data, databaseCur)

            if sensorDetails["mfc"]["enabled"] == "True":
                collectMfc(sensorDetails, data, databaseCur, startTime)

            if (sensorDetails["load_sensor_1"]["enabled"] == "True" and sensorDetails["load_sensor_1"]["enabled"] == "True"):
                collectStrainGauge(sensorDetails, data, databaseCur)

            time.sleep(takeInterval)
    except Exception as e:
        print(e)
        type, value, traceback = sys.exc_info()
        print('Error opening %s: %s' % (value.filename, value.strerror))
        # Shutdown when exception is triggered
        databaseConn.close()
        databaseCur.close()
        print("Shutting down sensor data gathering")

def main():
    # Create main menu to do tasks
    # Main Task List
    # 1. Start monitoring
    # 2. Stop monitoring
    # 3. Change cryo con setpoint
    # 4. Change flow rate setpoint
    # 5. Exit
    bg_process = multiprocessing.Process(target=monitoringLoop)

    while True:
        time.sleep(0.1)
        print("\nMenu:")
        if bg_process.is_alive():
            print("Monitoring loop is currently running.")
        else:
            print("Monitoring loop is off.")
        print("1. Start monitoring")
        print("2. Stop montioring")
        print("3. Change cryo con setpoint")
        print("4. Change flow rate setpoint")
        print("5. Exit")

        choice = input("Enter your number choice: ")

        if choice == "1":
            if not bg_process.is_alive():
                bg_process = multiprocessing.Process(target=monitoringLoop)
                bg_process.start()
                print("Background task started.")
            else:
                print("Background task is already running.")
        elif choice == "2":
            if bg_process.is_alive():
                bg_process.terminate()
                print("Background task stopped.")
            else:
                print("Background task is not running.")
        elif choice == "3":
            print("\nEnter the heater channel and new setpoint temperature in the following format (no quotations): \"heater_channel, temp\"")
            print("i.e. \"1, 120\"")
            while True:
                newSetting = input("Please enter the new heater + setpoint temp or go back to the main menu by typing \"Back\":")
                if newSetting.lower() == "back":
                    print("Going back to main menu.")
                    break
                else:
                    newSetting = ''.join(newSetting.split())
                    if "," in newSetting:
                        values = newSetting.split(",")
                        if len(values) == 2:
                            channel = values[0]
                            temp = values[1]
                            if (channel != '1' and channel != '2'):
                                print("Invalid input, please try again")
                            else:
                                try: 
                                    float(temp)
                                    with open("cfg.json") as json_data:
                                        data = json.load(json_data)
                                        sensorDetails = data["sensor_details"]
                                        changeCryoConSetpt(sensorDetails, channel, temp)
                                        break
                                except ValueError:
                                    print("Invalid input, please try again")
                        else:
                            print("Invalid input, please try again")
                    else:
                        print("Invalid input, please try again")
        elif choice == "4":
            while True:
                newFlowRate = input("\nEnter the flow rate you want in slpm:")
                if newFlowRate.lower() == "back":
                    print("Going back to main menu.")
                    break
                else:
                    try:
                        float(newFlowRate)
                        with open("cfg.json") as json_data:
                                data = json.load(json_data)
                                sensorDetails = data["sensor_details"]
                                changeFlowRate(sensorDetails, newFlowRate)
                        break
                    except ValueError:
                        print("Invalid input, please try again")
        elif choice == "5":
            if bg_process.is_alive():
                bg_process.terminate()
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()