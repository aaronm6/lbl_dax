Contained within this folder are all the sensor scripts plus main running function
for the detector setup.

main.py runs the sensor logging, with all current results sent to a postgres
database

cfg.json describes the configuration of each sensor and the postgres database,
make sure to verify all the port and ip addresses for each individual sensor

To run:
    - python main.py (At this point, you should be able to see data being put
    into your database)

Libraries:
    - cryoCon.py is for the temperature sensor
    - omegaPressureSensor.py is for the pressure sensor with an Omega readout
    - vacuumSensor.py is for the Pfeiffer pressure sensor

TODO:
    - Add load sensor library
    - Add error handling for initialization of sensors
    - Add error handling for reading data from sensors
    - Add cfg functionality for the channels and loops for cryo con and pfeiffer sensors
    - Create additional script to generate json/csv data from database or setup grafana page