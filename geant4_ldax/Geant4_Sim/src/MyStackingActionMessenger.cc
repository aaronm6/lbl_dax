#include "MyStackingActionMessenger.hh"
#include "MyStackingAction.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4ios.hh"

MyStackingActionMessenger::MyStackingActionMessenger(MyStackingAction* StackAct)
:fStackingAction(StackAct)
{
  fStackActDirectory = new G4UIdirectory("/mystackact/");
  fStackActDirectory->SetGuidance("Adjust Stacking action");

}

MyStackingActionMessenger::~MyStackingActionMessenger()
{
  delete fStackActDirectory;

}

void MyStackingActionMessenger::SetNewValue(G4UIcommand * command, G4String newValue)
{
  std::cout << "Macro Command [" << command->GetTitle() << "] with value [" << newValue << "] Doesn't Exist" << std::endl;

}