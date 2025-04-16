#ifndef MyDetectorConstMessenger_h
#define MyDetectorConstMessenger_h 1

class MyDetectorConstruction;
class G4UIdirectory;
class G4UIcmdWithABool;
class G4UIcmdWithAString;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;

#include "globals.hh"
#include "G4UImessenger.hh"

class MyDetectorConstMessenger: public G4UImessenger
{
  public:
    MyDetectorConstMessenger(MyDetectorConstruction* fDet);
    ~MyDetectorConstMessenger();

    virtual void SetNewValue(G4UIcommand * command, G4String newValues);
    virtual G4String GetCurrentValue(G4UIcommand * command);

  private:
    MyDetectorConstruction * fDetector;

    G4UIdirectory *             fMydetDirectory;
    G4UIcmdWithAString*         fWhichGeometryCMD;
    G4UIcmdWithAString*         fWhichDopantCMD;
    G4UIcmdWithADouble*         fDopantFractionCMD;
    
};

#endif


