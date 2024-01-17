import psycopg, time
from datetime import datetime
from lib import cryoCon
from lib import omegaSensor

with psycopg.connect("dbname=sensor_readings user=postgres password=LZ4850") as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        s = omegaSensor.connectOmegaSensor("192.168.1.200", 2000)
        reading = omegaSensor.getOmegaSensorReading(s)
        cur.execute(
            "INSERT INTO sensor_data (cryo_con_temp, date) VALUES (%s, %s)",
            (reading, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        cur.close()
    conn.close()