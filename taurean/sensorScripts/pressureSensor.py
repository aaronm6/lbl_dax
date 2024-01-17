import serial, psycopg, time
from datetime import datetime

serialPort = "COM3"

ETX = chr(3)    # \x03
CR = chr(13)    # \r
LF = chr(10)    # \n
ENQ = chr(5)    # \x05
ACK = chr(6)    # \x06
NAK = chr(21)   # \x15

with psycopg.connect("dbname=sensor_readings user=postgres password=LZ4850") as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        startTime = datetime.now()
        currentTime = datetime.now()
        readingHours = 3
        differenceTime = currentTime - startTime
        s = serial.Serial(serialPort)
        while (differenceTime.total_seconds() < 3600*readingHours):
            if (differenceTime.total_seconds() % 600):
                print("10 minutes have passed")
            s.write(str.encode("PRX" + CR + LF))
            res = s.readline()
            s.write(str.encode(ENQ))
            res = s.readline().decode()
            pressure = res.split(",")[1]
            cur.execute(
                "INSERT INTO temppressure (timestamp, pressure) VALUES (%s, %s)",
                (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), pressure))
            time.sleep(5)
        cur.close()
    conn.close()

s = serial.Serial(serialPort)
s.write(str.encode("PRX" + CR + LF))
res = s.readline()
print(res)
s.write(str.encode(ENQ))
res = s.readline().decode()
print(res)