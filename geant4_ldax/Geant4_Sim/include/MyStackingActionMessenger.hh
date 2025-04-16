#ifndef MyStackingActionMessenger_h
#define MyStackingActionMessenger_h 1

class MyStackingAction;
class G4UIdirectory;
class G4UIcmdWithABool;
class G4UIcmdWithADoubleAndUnit;

#include "globals.hh"
#include "G4UImessenger.hh"

class MyStackingActionMessenger: public G4UImessenger
{
  public:
    MyStackingActionMessenger(MyStackingAction* StackAct);
    ~MyStackingActionMessenger();

    virtual void SetNewValue(G4UIcommand * command, G4String newValues);

  private:
    MyStackingAction * fStackingAction;

    G4UIdirectory *             fStackActDirectory;
    
};

#endif


