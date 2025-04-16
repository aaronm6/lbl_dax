#include "MyPrimaryGeneratorMessenger.hh"
#include "G4SystemOfUnits.hh"
#include "G4AntiNeutrinoE.hh"
#include <sstream>

MyPrimaryGeneratorMessenger::MyPrimaryGeneratorMessenger(MyPrimaryGeneratorAction* action)
: G4UImessenger(),
  fAction(action),
  setParticleAnglesCmd(0),
  setParticlePositionCmd(0),
  setParticleGeneratorCmd(0)
{
    setParticleAnglesCmd = new G4UIcmdWith3Vector("/myapp/setParticleAngles", this);
    setParticleAnglesCmd->SetGuidance("Set phi and theta angles for particles");
    setParticleAnglesCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

    setParticlePositionCmd = new G4UIcmdWithAString("/myapp/setParticlePosition", this);
    setParticlePositionCmd->SetGuidance("Set particle position");
    setParticlePositionCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

    setParticleGeneratorCmd = new G4UIcmdWithAString("/myapp/generator", this);
    setParticleGeneratorCmd->SetGuidance("Set the particle generator type");
    setParticleGeneratorCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
}

MyPrimaryGeneratorMessenger::~MyPrimaryGeneratorMessenger() {
    delete setParticleAnglesCmd;
    delete setParticlePositionCmd;
    delete setParticleGeneratorCmd;
}

void MyPrimaryGeneratorMessenger::SetNewValue(G4UIcommand* command, G4String newValue) {
  
    if (command == setParticleAnglesCmd) {
        fAction->SetParticleAngles(setParticleAnglesCmd->GetNew3VectorValue(newValue));
    }
    else if (command == setParticlePositionCmd){
        std::istringstream iss(newValue.data());
        std::string posCMD;
        double innerRadius, outerRadius, x, y, z;
        std::string unit;
        iss >> posCMD;
        if (posCMD == "RandomSphere"){
            iss >> outerRadius >> unit;
            fAction->setPositionType(fAction->positionType::RandomSphere, 0., outerRadius, 0., 0., 0., unit=unit);
        }
        else if (posCMD == "Point"){
            iss >> x >> y >> z >> unit;
            fAction->setPositionType(fAction->positionType::Point, 0., 0, x, y, z, unit=unit);
        }
        else if (posCMD == "RandomSphereShell"){
            iss >> innerRadius >> outerRadius >> unit;
            fAction->setPositionType(fAction->positionType::RandomSphereShell, innerRadius, outerRadius, x, y, z, unit=unit);
        }
        else{
            fAction->setPositionType(fAction->positionType::Point, 0., 0., 0., 0., 0., unit="cm");
        }
    }
    else if (command == setParticleGeneratorCmd){
        std::istringstream iss(newValue.data());
        std::string generatorName;
        iss >> generatorName;
        if (generatorName == "Particle"){
            fAction->setGeneratorType(fAction->generatorType::Particle);
            G4String particleName;
            iss >> particleName;
            fAction->setParticleType(particleName);
            double particleEnergy;
            std::string energyUnit;
            iss >> particleEnergy >> energyUnit;
            fAction->SetParticleEnergyAndUnit(particleEnergy, energyUnit);
        }
        else if (generatorName == "Ion"){
            fAction->setGeneratorType(fAction->generatorType::Ion);
            int Z, A;
            if (iss >> Z >> A) {
                fAction->SetIonZA(Z, A);
            }
            else {
                throw std::runtime_error{"Ion Z, A not set properly"};
            }
        }
        else if (generatorName == "Spectrum"){
            fAction->setGeneratorType(fAction->generatorType::Spectrum);
            G4String particleName;
            iss >> particleName;
            fAction->setParticleType(particleName);
            std::string spectrumFilePath;
            iss >> spectrumFilePath;
            // Some function to pass text file, read into TH1 and sample
            fAction->setEnergySpectrum(spectrumFilePath);
        }
        else if (generatorName == "Tritium"){
            fAction->setGeneratorType(fAction->generatorType::Tritium);
        }
        else{
            throw std::runtime_error {"Generator Name " + generatorName + " does not match options"};
        }

    }
    // Done reading macro commands
    else{
        std::cout << "Macro Command [" << command->GetTitle() << "] with value [" << newValue << "] Doesn't Exist" << std::endl;
    }
}
