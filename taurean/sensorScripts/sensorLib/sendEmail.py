import os, psycopg, time
from datetime import datetime

def sendAlert(temperature, pressure, temperatureThreshold, pressureThreshold):
    #Initialize variables
    alertFlag = False
    message = ""
    subject = ""

    dateTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

    if ((temperature > temperatureThreshold) and (pressure > pressureThreshold)):
        alertFlag = True
        message = ("TPC temperature is above " + str(temperatureThreshold)
            + " K and TPC pressure is above " + str(pressureThreshold) + " bara."
            + "\r\nCurrent TPC temperature is " + str(temperature) + " K and "
            + "current TPC pressure is " + str(pressure) + " bara."
            + "\r\nAlert detected at " + dateTime + "."
            + "\r\nPlease check the lab setup at 070A-2263 as soon as possible.")
        subject = " ALERT: TPC Temperature and Pressure Above Threshold"
    elif (temperature > temperatureThreshold):
        alertFlag = True
        message = ("TPC temperature is above " + str(temperatureThreshold) 
            + " K.\r\nCurrent TPC temperature is "
            + str(temperature) + " K."
            + "\r\nAlert detected at " + dateTime + "."
            + "\r\nPlease check the lab setup at 070A-2263 as soon as possible.")
        subject = " ALERT: TPC Temperature Above Threshold"
    elif (pressure > pressureThreshold):
        alertFlag = True
        message = ("TPC pressure is above " + str(pressureThreshold) 
            + " bara. \r\nCurrent TPC pressure is "
            + str(pressure) + " bara."
            + "\r\nAlert detected at " + dateTime + "."
            + "\r\nPlease check the lab setup at 070A-2263 as soon as possible.")
        subject = " ALERT: TPC Pressure Above Threshold"
    #Check if alert flag was triggered
    if alertFlag:
        os.system("sendmail taureanzhang@lbl.gov <<EOF"
        +"\r\nTo: taureanzhang@lbl.gov"
        +"\r\nSubject: "+subject
        +"\r\nFrom: hydrox-slowctrl@darkmatter.gov"
        +"\r\n"+message
        +"\r\n")

        os.system("sendmail aaronm@lbl.gov <<EOF"
        +"\r\nTo: aaronm@lbl.gov"
        +"\r\nSubject: "+subject
        +"\r\nFrom: hydrox-slowctrl@darkmatter.gov"
        +"\r\n"+message
        +"\r\n")

    return alertFlag

pressureQuery = "select pressure from omega_pressure order by timestamp desc limit 1"
tempQuery = "SELECT tempreture FROM tpc_temp order by timestamp desc limit 1"

databaseLogin = ("dbname=sensor_readings" + " user=postgres" + " password=LZ4850")
databaseConn = psycopg.connect(databaseLogin)
databaseCur = databaseConn.cursor()

try:
    while True:
        databaseCur.execute(pressureQuery)
        pressure = databaseCur.fetchone()[0]
        databaseCur.execute(tempQuery)
        temperature = databaseCur.fetchone()[0]
        status = sendAlert(temperature, pressure, 179, 2.1)
        if status:
            print("Email alert triggered, going to 15 minute alert cooldown")
            time.sleep(900)
        time.sleep(5)
except KeyboardInterrupt:
    databaseConn.close()
    databaseCur.close()
    print("DB connection closed")


#Test Cases:

#Temperature above threshold
#print(sendAlert(100, 1, 80, 2))

#Pressure above threshold
#print(sendAlert(60, 2.5, 80, 2))

#Both above threshold
#print(sendAlert(100, 2.5, 80, 2))

#Nothing above threshold
#print(sendAlert(100, 2.5, 120, 3))