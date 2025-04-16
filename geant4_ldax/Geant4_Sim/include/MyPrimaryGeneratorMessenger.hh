#ifndef MYPARTICLEGENERATORMESSENGER_HH
#define MYPARTICLEGENERATORMESSENGER_HH

#include "G4UImessenger.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWith3Vector.hh"
#include "MyPrimaryGeneratorAction.hh"
#include "MyDetectorConstruction.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"

class MyPrimaryGeneratorMessenger : public G4UImessenger {
public:
    MyPrimaryGeneratorMessenger(MyPrimaryGeneratorAction* action);
    virtual ~MyPrimaryGeneratorMessenger();

    virtual void SetNewValue(G4UIcommand* command, G4String newValue);

private:
    MyPrimaryGeneratorAction* fAction;

    G4UIcmdWith3Vector* setParticleAnglesCmd;

    G4UIcmdWithAString* setParticlePositionCmd;

    G4UIcmdWithAString* setParticleGeneratorCmd;

};

#endif
