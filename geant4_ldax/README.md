# HydroX Geant4 Simulation

A Simple TPC is simulated where the dopant fraction of hydrogen or helium can be set via macro commands. There also exists the option to build EJ200 backing detectors for neutrons

## Docker Image setup 

- First download docker desktop for your computer. Note that these instructions have been tested on MacOS Ventura 13.0
- The only required edit should be to add your name and replace where you see instnaces of "danielkodroff".
- Go into docker folder and do  `docker-compose build` to creater the docker image. You only need to do this once.
- The YAML file allows for read/write mounting of the folder container the simulation into the container. 
    - Pay attention to the relative path of the Geant directory is with respect to this. 
    - Note also that the entry under `services` and `image` must also correspond to the dockerfile and container you built.
- Now enter the docker container using the yaml file to mount the Geant4 sims package (and whatever else you want). You should be in the docker container to run this command.   
   - Do `docker-compose run --rm dockerfile` to enter the container and start a terminal. This utilize the yaml file (dockerfile refers to entry under services in yaml file)
- Upon entering the container:
   - You can enable root by doing `root.exe`
   - You can edit files using emacs
   - Now the simulations can be built and ran following the instructions below.

## Building the Simulation

After installing Geant4 and ROOT, or building the included docker image (Geant4 10.5.1 and Python 3.6.9), build the simulation like

    cd Geant4_Sim
    mkdir build
    cd build
    cmake ..
    make

## Running the GEANT Simulations

The simulation can be run, in the build directory, using the command `./Geant4Simulation ../Macros/runMac.mac`, given the exmaple macro `runMac.mac`. The macro commands provided in that example are below. 

First off, `/run/beamOn` sets how many events to simulate. And `/run/initialize` constructs the detector geometry and must be called before the generator commands but after the detector commands.

First the detector configuration can be built

1. `/mydet/WhichGeometry ` --> There is only one detector configuration so `HydroX` is the default command
2. `/mydet/WhichDopant` --> The dopant in the LXe can be `H`, `He`, or `None`
3. `/mydet/DopantFraction` --> Set the dopant fraction by mass

There are afew types of commands to pass to the event generator.

1. To simulate single particle with mono-energetic energy use the below command
    - `/myapp/generator particle <particleType> <Energy> <Unit>`
    - `particleType` can be e-, gamma, alpha, neutron, geantino, mu-, etc to name few
    - `Energy` and `Unit` could be 10. keV, for exmaple (exceptable units are eV, keV, MeV, GeV)
    - An example output name would `Particle_e-_10.00keV.root` 
2. To simulate an radioactive isotope do
    -  `/myapp/generator Ion <Z> <A>` where `Z` and `A` provide information to define the isotope.
    - An example output name would `Ion_Cr51_0keV.root`. Here 0 keV means the ion is at rest before decaying.
3. To simulate a particle spectra do
    - `/myapp/generator Spectrum <particleType> <file>` where `particleType` is the same as above and `file` corresponds to a txt file that can be read into a TH1 to be Monte Carlo Sampled.
    - File must be two columns (space or tab delimited) where the first colum is energy and the second is the rate. The energies should be monotonically increasing. The assumed energy unit is keV.
    - An example output name would `Spectrum_e-.root`. Where the energy spectrum would be a beta spectrum.
4. To simulate the tritium beta spectra do
    - `/myapp/generator Tritium`
    - Samples full tritium beta spectra out to 18.6 keV endpoint with electron.
    - An example out name would be `Tritium_e-_spectrum.root`

To set the position of the particle use one of the following options.

1. To set particle to be randomly distributed within a sphere some radius do
`/myapp/setParticlePosition RandomSphere <Radius> <unit>`

2. To set particle to originate from a point do 
`/myapp/setParticlePosition Point <X> <Y> <Z> <unit>`

3. To set the particles to be uniformly distributed in a spherical shell
`/myapp/setParticlePosition RandomSphereShell <InnerRadius> <OuterRadius> <unit>`

4. To default the particle to the origin do
`/myapp/setParticlePosition`

To set the particle angle (works for particles, not radioactive decays)

1. `/myapp/setParticleAngles <Phi [deg]> <Theta [deg]> <dummy 0>`

### Root File Output

The output is a root file with a single TTree called `Events`` where there are as many events in the root file corresponds as the beamOn value.

The `Events` TTree is broken down into Branches categorized as relating to either the primary particle that was created or the tracks/steps the primary (and potentially subsequent child) particles take. Each branch is vector allowing for the creation of multiple primary particles created (in the case of `primaries` branches) and the occurrence of many tracks and steps taken by the primary particle and its child particles (in the case of the the `tracks` branches.)

- `primaries.particleName`       --> (string) primary particles names.
- `primaries.volumeName`         --> (string) volume in which primary particle is generated. See below for details
- `primaries.pid`                --> (int) PDG code for primary particle
- `primaries.time_ns`            --> (int) Time of creation for primary particle. Initalized to zero.
- `primaries.energy_keV`         --> (float) Starting energy of primary particle.
- `primaries.totalEnergy_keV`    --> (float) Starting total energy (kinetic and rest) of primary particle.
- `primaries.totalMomentum_keV`  --> (float) Starting total momentum of primary particle.
- `primaries.position_X_mm`      --> (float) Starting position of primary. Defaults to global origin.
- `primaries.position_Y_mm`      --> (float) Starting position of primary. Defaults to global origin.
- `primaries.position_Z_mm`      --> (float) Starting position of primary. Defaults to global origin.
- `primaries.direction_X`        --> (float) Starting momentum unit vector. Randomly sampled over surface of sphere.
- `primaries.direction_Y`        --> (float) Starting momentum unit vector. Randomly sampled over surface of sphere.
- `primaries.direction_Z`        --> (float) Starting momentum unit vector. Randomly sampled over surface of sphere.
- `tracks.trackID`               --> (int) Unique ID of given particle's track. A track is made up of a series of steps.
- `tracks.stepNumber`            --> (int) Step number for a given particle's track.
- `tracks.parentID`              --> (int) Track ID of parent particle.
- `tracks.detectorName`          --> (string) Detector name in which step occurred. See below for details.
- `tracks.detectorID`            --> (int) Unique ID for each detector. See below for details
- `tracks.particleName`          --> (string) Name of particle for a given track undergoing steps. 
- `tracks.particleID`            --> (int) PDG code for for particle in track.
- `tracks.creationProcess`       --> (string) GEANT4 process by which a given particle (unique track) was created.
- `tracks.particleMass_keV`      --> (float) Mass of particle in keV.
- `tracks.stepTime_ns`           --> (float) Time at which step occurred.
- `tracks.preStepEnergy_keV`     --> (float) Energy of particle prior to step.
- `tracks.energyDeposition_keV`  --> (float) Energy deposited by particle during step.
- `tracks.stepLength_um`         --> (float) Length of step taken in microns.
- `tracks.preStepMomentumX_keV`  --> (float) Momentum of particle in X-direction before step.
- `tracks.preStepMomentumY_keV`  --> (float) Momentum of particle in Y-direction before step.
- `tracks.preStepMomentumY_keV`  --> (float) Momentum of particle in Z-direction before step.
- `tracks.postStepMomentumX_keV` --> (float) Momentum of particle in X-direction after step.
- `tracks.postStepMomentumY_keV` --> (float) Momentum of particle in Y-direction after step.
- `tracks.postStepMomentumZ_keV` --> (float) Momentum of particle in Z-direction after step.
- `tracks.x_global_um`           --> (float) Position of particle in global coordinates.
- `tracks.y_global_um`           --> (float) Position of particle in global coordinates.
- `tracks.z_global_um`           --> (float) Position of particle in global coordinates.
