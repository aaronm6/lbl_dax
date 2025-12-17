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


def collectCryoCon(sensorDetails, cfgTableData, cursor, cryoConDev):
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
    #print(cryoConLoops)
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
    #print(cryoConVals)
    cursor.execute(makeInsertQuery(cryoConTable, cryoConColumns, cryoConVals))

def collectCryoConTpc(sensorDetails, cfgTableData, cursor, cryoConTpcDev):
    #Get table details
    cryoConTpcTable = cfgTableData["database_details"]["cryo_con_tpc"]["table_name"]
    cryoConTpcColumns = cfgTableData["database_details"]["cryo_con_tpc"]["column_names"]

    cryoConTpcVals = []
    currentTime = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    cryoConTpcVals.append(currentTime)

    #Get data
    temperature = cryoConTpcDev.getCryoConTemp("C")
    if (temperature == '-------'):
        temperature = 10000
    cryoConTpcVals.append(temperature)
    cursor.execute(makeInsertQuery(cryoConTpcTable, cryoConTpcColumns, cryoConTpcVals))
    return temperature

def changeCryoConSetpt(sensorDetails, channel, temperature, cryoConDev):
    setpt = cryoConDev.setCryoConSetPoint(channel, temperature)
    print("Temperature set point has been changed to " + str(temperature) + " for channel " + channel + ".")

def collectOmegaPressure(sensorDetails, cfgTableData, cursor, omegaPressureDev):
    omegaPressureTable = cfgTableData["database_details"]["omega_pressure"]["table_name"]
    omegaPressureColumns = cfgTableData["database_details"]["omega_pressure"]["column_names"]

    omegaPressureVals = []
    currentTime = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    omegaPressureVals.append(currentTime)
    pressure = omegaPressureDev.getOmegaSensorPressure()
    omegaPressureVals.append(pressure)

    cursor.execute(makeInsertQuery(omegaPressureTable, omegaPressureColumns, omegaPressureVals))
    return pressure

def collectPfeiffer(sensorDetails, cfgTableData, cursor, pfeifferDev):
    pfeifferChannels = sensorDetails["pfeiffer_pressure"]["channels"]

    pfeifferTable = cfgTableData["database_details"]["pfeiffer_pressure"]["table_name"]
    pfeifferColumns = cfgTableData["database_details"]["pfeiffer_pressure"]["column_names"]

    pfeifferVals = []
    currentTime = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    pfeifferVals.append(currentTime)
    pfeifferVals.append(pfeifferDev.getPfeifferPressure(pfeifferChannels))
    #print(pfeifferVals)
    cursor.execute(makeInsertQuery(pfeifferTable, pfeifferColumns, pfeifferVals))

def collectMfc(sensorDetails, cfgTableData, cursor, startTime, mfcDev):
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
        totalVolume = np.trapz(flowArr, timeArr)/60
        scalingFactor = 0.00586
        mfcVals.append(str(totalVolume))
        #TODO: FIND SCALING FACTOR FOR XENON VOLUME TO MASS
        mfcVals.append(str(totalVolume * scalingFactor))
    else:
        mfcVals.append("0")
        mfcVals.append("0")
    cursor.execute(makeInsertQuery(mfcTable, mfcColumns, mfcVals))

def changeFlowRate(sensorDetails, flowRate, mfcDev):
    try:
        mfcDev = mfc.mfc(sensorDetails["mfc"]["serial_port"])
    except socket.timeout:
        raise TimeoutError("Couldn't connect to pfeiffer pressure readout, check connections and serial port address and try again later")

    mfcDev.setMaxFlowRate(flowRate)
    print("Max flow rate has been changed to " + str(flowRate) + ".")

def collectStrainGauge(sensorDetails, cfgTableData, cursor, loadSensorDev1, loadSensorDev2):
    loadSensorTable = cfgTableData["database_details"]["load_sensors"]["table_name"]
    loadSensorColumns = cfgTableData["database_details"]["load_sensors"]["column_names"]

    loadSensorVals = []
    currentTime = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    loadSensorVals.append(currentTime)
    loadSensorVals.append(loadSensorDev1.getStrainGaugeLoad())
    loadSensorVals.append(loadSensorDev2.getStrainGaugeLoad())
    cursor.execute(makeInsertQuery(loadSensorTable, loadSensorColumns, loadSensorVals))
    #print(makeInsertQuery(loadSensorTable, loadSensorColumns, loadSensorVals))

def monitoringLoop(alert_flag):
    # Parse configuration details from cfg.json
    with open("cfg.json") as json_data:
        data = json.load(json_data)
        takeInterval = data["data_output_settings"]["time_between_measurements"]
        sensorDetails = data["sensor_details"]
        print("TEST")
        # Make database login string
        credentials = data["database_details"]["credentials"]
        databaseLogin = ("dbname=" + credentials["database"] + " user=" + 
            credentials["username"] + " password=" + credentials["password"])
        databaseConn = psycopg.connect(databaseLogin)
        databaseConn.autocommit = True
        databaseCur = databaseConn.cursor()

    if sensorDetails["cryo_con"]["enabled"] != "True":
        print("Cryo Con is disabled in cfg, please change this if this is an error")

    if sensorDetails["cryo_con_tpc"]["enabled"] != "True":
        print("TPC Cryo Con is disabled in cfg, please change this if this is an error")

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

    #initialize all sockets
    if sensorDetails["cryo_con"]["enabled"] == "True":
        try:
            cryoConDev = cryoCon.cryoCon(sensorDetails["cryo_con"]["ip_address"], sensorDetails["cryo_con"]["port"])
        except socket.timeout:
            raise TimeoutError("Cryo Con couldn't be connected to, check connections and ip addresses and try again later")

    if sensorDetails["cryo_con_tpc"]["enabled"] == "True":
        try:
            cryoConTpcDev = cryoCon.cryoCon(sensorDetails["cryo_con_tpc"]["ip_address"], sensorDetails["cryo_con_tpc"]["port"])
        except socket.timeout:
            raise TimeoutError("TPC Cryo Con couldn't be connected to, check connections and ip addresses and try again later")

    if sensorDetails["omega_pressure"]["enabled"] == "True":
        try:
            omegaPressureDev = omegaPressureSensor.omegaPressureSensor(sensorDetails["omega_pressure"]["ip_address"], sensorDetails["omega_pressure"]["port"])
        except socket.timeout:
            raise TimeoutError("Couldn't connect to omega pressure readout, check connections and ip addresses and try again later")
    
    if sensorDetails["pfeiffer_pressure"]["enabled"] == "True":
        try:
            pfeifferDev = pfeifferPressureSensor.pfeifferPressureSensor(sensorDetails["pfeiffer_pressure"]["serial_port"])
        except socket.timeout:
            raise TimeoutError("Couldn't connect to pfeiffer pressure readout, check connections and serial port address and try again later")

    if sensorDetails["mfc"]["enabled"] == "True":
        try:
            mfcDev = mfc.mfc(sensorDetails["mfc"]["serial_port"])
        except socket.timeout:
            raise TimeoutError("Couldn't connect to MFC readout, check connections and serial port address and try again later")

    if (sensorDetails["load_sensor_1"]["enabled"] == "True" and sensorDetails["load_sensor_1"]["enabled"] == "True"):
        try:
            loadSensorDev1 = strainGauge.strainGauge(sensorDetails["load_sensor_1"]["ip_address"], sensorDetails["load_sensor_1"]["port"])
            loadSensorDev2 = strainGauge.strainGauge(sensorDetails["load_sensor_2"]["ip_address"], sensorDetails["load_sensor_2"]["port"])
        except socket.timeout:
            raise TimeoutError("Couldn't connect to one or more strain gauges, check connections and ip addresses and try again later")


    temperatureThreshold = float(sensorDetails["cryo_con_tpc"]["alert_threshold"])
    pressureThreshold = float(sensorDetails["omega_pressure"]["alert_threshold"])

    counter = 0
    try:
        while 1:
            pressure = 0
            temperature = 0

            if sensorDetails["cryo_con"]["enabled"] == "True":
                collectCryoCon(sensorDetails, data, databaseCur, cryoConDev)

            if sensorDetails["cryo_con_tpc"]["enabled"] == "True":
                temperature = float(collectCryoConTpc(sensorDetails, data, databaseCur, cryoConTpcDev))

            if sensorDetails["omega_pressure"]["enabled"] == "True":
                pressure = float(collectOmegaPressure(sensorDetails, data, databaseCur, omegaPressureDev))

            if sensorDetails["pfeiffer_pressure"]["enabled"] == "True":
                collectPfeiffer(sensorDetails, data, databaseCur, pfeifferDev)

            if sensorDetails["mfc"]["enabled"] == "True":
                collectMfc(sensorDetails, data, databaseCur, startTime, mfcDev)

            if (sensorDetails["load_sensor_1"]["enabled"] == "True" and sensorDetails["load_sensor_1"]["enabled"] == "True"):
                collectStrainGauge(sensorDetails, data, databaseCur, loadSensorDev1, loadSensorDev2)

    except Exception as e:
        print(e)
        type, value, traceback = sys.exc_info()
        print('Error opening %s: %s' % (value.filename, value.strerror))
        # Shutdown when exception is triggered
        databaseConn.close()
        databaseCur.close()

        if sensorDetails["cryo_con"]["enabled"] == "True":
            cryoConDev.close()

        if sensorDetails["omega_pressure"]["enabled"] == "True":
            omegaPressureDev.close()

        if sensorDetails["pfeiffer_pressure"]["enabled"] == "True":
            pfeifferDev.close()

        if sensorDetails["mfc"]["enabled"] == "True":
            mfcDev.close()

        if (sensorDetails["load_sensor_1"]["enabled"] == "True" and sensorDetails["load_sensor_1"]["enabled"] == "True"):
            loadSensorDev1.close()
            loadSensorDev2.close()
        
        print("Shutting down sensor data gathering")
    
    databaseConn.close()
    databaseCur.close()

def main():
    # Create main menu to do tasks
    # Main Task List
    # 1. Start monitoring
    # 2. Stop monitoring
    # 3. Turn on email alerts
    # 4. Turn off email alerts
    # 5. Change cryo con setpoint
    # 6. Change flow rate setpoint
    # 7. Exit
    alert_flag = '0'
    bg_process = multiprocessing.Process(target=monitoringLoop, args=(alert_flag))
    print("Alert system is inactive")

    while True:
        time.sleep(0.1)
        print("\nMenu:")
        if bg_process.is_alive():
            print("Monitoring loop is currently running.")
        else:
            print("Monitoring loop is off.")
        
        print("1. Start monitoring")
        print("2. Stop montioring")
        print("3. Turn on email alerts")
        print("4. Turn off email alerts")
        print("5. Change cryo con setpoint")
        print("6. Change flow rate setpoint")
        print("7. Exit")

        choice = input("Enter your number choice: ")

        if choice == "1":
            if not bg_process.is_alive():
                bg_process = multiprocessing.Process(target=monitoringLoop, args=(alert_flag))
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
            alert_flag = '1'
            print("Alert system is active, please restart monitoring loop to send changes")
        elif choice == "4":
            alert_flag = '0'
            print("Alert system is inactive, please restart monitoring loop to send changes")
        elif choice == "5":
            print("\nEnter the heater channel and new setpoint temperature in the following format (no quotations): \"heater_channel, temp\"")
            print("i.e. \"1, 120\"")
            while True:
                newSetting = input("Please enter the new heater + setpoint temp or go back to the main menu by typing \"Back\":")
                if newSetting.lower() == "back":
                    print("Going back to main menu.")
                    break
                else:
                    wasAlive = False
                    if bg_process.is_alive():
                        bg_process.terminate()
                        wasAlive = True
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
                                        try:
                                            cryoConDev = cryoCon.cryoCon(sensorDetails["cryo_con"]["ip_address"], sensorDetails["cryo_con"]["port"])
                                        except socket.timeout:
                                            raise TimeoutError("Cryo Con couldn't be connected to, check connections and ip addresses and try again later")
                                        changeCryoConSetpt(sensorDetails, channel, temp, cryoConDev)
                                        cryoConDev.close()
                                        if wasAlive:
                                            bg_process = multiprocessing.Process(target=monitoringLoop, args=(alert_flag))
                                            bg_process.start()
                                        break
                                except ValueError:
                                    print("Invalid input, please try again")
                        else:
                            print("Invalid input, please try again")
                    else:
                        print("Invalid input, please try again")
                    if wasAlive:
                        bg_process.start()
        elif choice == "6":
            while True:
                newFlowRate = input("\nEnter the flow rate you want in slpm:")
                if newFlowRate.lower() == "back":
                    print("Going back to main menu.")
                    break
                else:
                    wasAlive = False
                    if bg_process.is_alive():
                        bg_process.terminate()
                        wasAlive = True
                    try:
                        float(newFlowRate)
                        with open("cfg.json") as json_data:
                                data = json.load(json_data)
                                sensorDetails = data["sensor_details"]
                                if sensorDetails["mfc"]["enabled"] == "True":
                                    try:
                                        mfcDev = mfc.mfc(sensorDetails["mfc"]["serial_port"])
                                    except socket.timeout:
                                        raise TimeoutError("Couldn't connect to MFC readout, check connections and serial port address and try again later")
                                changeFlowRate(sensorDetails, newFlowRate, mfcDev)
                                mfcDev.close()
                                if wasAlive:
                                    bg_process = multiprocessing.Process(target=monitoringLoop, args=(alert_flag))
                                    bg_process.start()
                        break
                    except ValueError:
                        print("Invalid input, please try again")
                    if wasAlive:
                        bg_process.start()
        elif choice == "7":
            if bg_process.is_alive():
                bg_process.terminate()
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()