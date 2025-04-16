#include "Randomize.hh"

#include "MyDetectorConstruction.hh"
#include "MyPhysicsList.hh"
#include <algorithm>

#include "MyPrimaryGeneratorAction.hh"
#include "MyEventAction.hh"
#include "MyRunAction.hh"
#include "MySteppingAction.hh"
#include "MyStackingAction.hh"

#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"
#include "G4SystemOfUnits.hh"

char* getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
    return std::find(begin, end, option) != end;
}

int main(int argc, char** argv) {
    
    // Create the run manager
    auto runManager = new G4RunManager;

    // Set the detector construction
    auto detectorConstruction = new MyDetectorConstruction;
    runManager->SetUserInitialization(detectorConstruction);

    // Set the physics list
    runManager->SetUserInitialization(new MyPhysicsList);

    // Initiate the generator action
    MyPrimaryGeneratorAction* generatorAction = MyPrimaryGeneratorAction::Instance();
    
    // Initiate the run action
    MyRunAction* runAction = new MyRunAction(generatorAction);
    
    // Initiate the event action
    MyEventAction* eventAction = new MyEventAction(runAction);

    // Initiate the stacking action to occur as tracks are created
    MyStackingAction* stackAction = new MyStackingAction();

    // Set the generator action
    runManager->SetUserAction(generatorAction);

    // Set the run action
    runManager->SetUserAction(runAction);

    // Set up the event action
    runManager->SetUserAction(eventAction);

    // Set up the stacking action
    runManager->SetUserAction(stackAction);

    // Set and initalize the stepping action
    runManager->SetUserAction(new MySteppingAction(runAction));

    // Initialize the run manager
    // CALL THIS VIA MACRO WITH /run/initialize before /run/beamOn
    //runManager->Initialize();

    // Initialize visualization
    G4VisManager* visManager = new G4VisExecutive;
    visManager->Initialize();

    // Get the pointer to the User Interface manager
    G4UImanager* UImanager = G4UImanager::GetUIpointer(); 

    if (argc!=1) { 
        // batch mode  
        G4String command = "/control/execute ";
        G4String fileName = argv[1];
        UImanager->ApplyCommand(command+fileName);
    }
    else { 
        // interactive mode : define UI session
        G4UIExecutive* ui = new G4UIExecutive(argc, argv);
        UImanager->ApplyCommand("/control/execute init_vis.mac"); 
        ui->SessionStart();
        delete ui;
    }

    // Job termination
    delete visManager;
    delete runManager;

    return 0;
}