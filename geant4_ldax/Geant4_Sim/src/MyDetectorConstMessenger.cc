#include "MyDetectorConstMessenger.hh"
#include "MyDetectorConstruction.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4ios.hh"

MyDetectorConstMessenger::MyDetectorConstMessenger(MyDetectorConstruction* fDet)
:fDetector(fDet)
{
  fMydetDirectory = new G4UIdirectory("/mydet/");
  fMydetDirectory->SetGuidance("My detector setup control commands.");

  fWhichGeometryCMD = new G4UIcmdWithAString("/mydet/WhichGeometry", this);
  fWhichGeometryCMD->SetGuidance("Which geometry setup to use");
  fWhichGeometryCMD->SetDefaultValue("HydroX");

  fWhichDopantCMD = new G4UIcmdWithAString("/mydet/WhichDopant", this);
  fWhichDopantCMD->SetGuidance("Which dopant to use");
  fWhichDopantCMD->SetDefaultValue("None");

  fDopantFractionCMD = new G4UIcmdWithADouble("/mydet/DopantFraction", this);
  fDopantFractionCMD->SetGuidance("What dopant fraction (by mass) to use");
  fDopantFractionCMD->SetDefaultValue(0.);

}

MyDetectorConstMessenger::~MyDetectorConstMessenger()
{
  delete fMydetDirectory;
  delete fWhichGeometryCMD;
  delete fWhichDopantCMD;
  delete fDopantFractionCMD;

}

void MyDetectorConstMessenger::SetNewValue(G4UIcommand * command, G4String newValue)
{
 
  if (command == fWhichGeometryCMD){
    fDetector->SetWhichGeometry(newValue);
  }
  else if (command == fWhichDopantCMD){
    fDetector->SetWhichDopant(newValue);
  }
  else if (command == fDopantFractionCMD){
    fDetector->SetDopantFraction(fDopantFractionCMD->GetNewDoubleValue(newValue));
  }
  // Done reading macro commands
  else{
    std::cout << "Macro Command [" << command->GetCommandPath() << "] with value [" << newValue << "] Doesn't Exist" << std::endl;
  }

}

G4String MyDetectorConstMessenger::GetCurrentValue(G4UIcommand * command)
{
  G4String cv;
  if ( command==fDopantFractionCMD ){
    cv = fDopantFractionCMD->ConvertToString(fDetector->GetDopantFraction()); 
  }

  return cv;
}