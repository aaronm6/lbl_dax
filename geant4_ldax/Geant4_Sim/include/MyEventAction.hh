#ifndef MYEVENTACTION_HH
#define MYEVENTACTION_HH

#include "TFile.h"
#include "TTree.h"
#include "G4UserEventAction.hh"
#include "MyRunAction.hh"
#include "G4String.hh"
#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"

class G4Event;

class MyEventAction : public G4UserEventAction {
public:
    MyEventAction(MyRunAction* runAction); // Updated constructor
    virtual ~MyEventAction();

    virtual void BeginOfEventAction(const G4Event* anEvent);
    virtual void EndOfEventAction(const G4Event* anEvent);

    void addPrimary(G4PrimaryVertex* primaryVertex, G4PrimaryParticle* g4primary);

private:
    MyRunAction* fRunAction;

};

#endif
