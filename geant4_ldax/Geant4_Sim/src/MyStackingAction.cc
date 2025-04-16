#include "MyStackingAction.hh"
#include "MyStackingActionMessenger.hh"

#include "G4ClassificationOfNewTrack.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4DecayProducts.hh"
#include "G4SystemOfUnits.hh"

MyStackingAction::MyStackingAction()
    : G4UserStackingAction(),
    fMessenger(0)
    { 
        fMessenger = new MyStackingActionMessenger(this);
    }   

MyStackingAction::~MyStackingAction() 
{ 
    delete fMessenger;
}

G4ClassificationOfNewTrack MyStackingAction::ClassifyNewTrack(const G4Track* aTrack)
{

    G4ClassificationOfNewTrack result(fUrgent);

    // Reset time for immediate daughter track(s)
    // This has the effect of having all daughters of radioactive decay start at t=0
    if (aTrack->GetCreatorProcess()){
        if (aTrack->GetCreatorProcess()->GetProcessName() == "RadioactiveDecay" && aTrack->GetParentID() == 1){
            (const_cast<G4Track*>(aTrack))->SetGlobalTime(0.0);
        }
    }

    return result;

}

void MyStackingAction::NewStage(){

    stackManager->ReClassify();

}

void MyStackingAction::PrepareNewEvent(){

    //std::cout << "New Event Stack" << std::endl;

}
