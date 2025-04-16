#ifndef MYSTACKINGACTION_HH
#define MYSTACKINGACTION_HH

#include "G4String.hh"
#include "G4UserStackingAction.hh"
#include "G4StackManager.hh"

class G4Track;
class MyStackingActionMessenger;

class MyStackingAction : public G4UserStackingAction {
public:
    MyStackingAction(); 
    virtual ~MyStackingAction();

    G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* aTrack);

    void NewStage();
    void PrepareNewEvent();

private:
    MyStackingActionMessenger* fMessenger;

};

#endif
