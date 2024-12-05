from sensorLib import cryoCon
from datetime import datetime, timezone
import time, psycopg

# Old Test Loop Code
host = "192.168.1.203"
port = "5000"
s = cryoCon.cryoCon(host, port)


while True:
    databaseLogin = ("dbname=sensor_readings user=postgres password=LZ4850")
    databaseConn = psycopg.connect(databaseLogin)
    databaseConn.autocommit = True
    databaseCur = databaseConn.cursor()

    try:
        temp = s.getCryoConTemp("C")
        #print(s.setCryoConSetPoint("1", "250"))
        currentTime = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        query = "insert into tpc_temp values ('" + currentTime + "', '" + temp + "')"
        databaseCur.execute(query)
        time.sleep(5)
    except KeyboardInterrupt:
        break

    databaseCur.close()
s.close()