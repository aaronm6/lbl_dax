#ifndef MYPHYSICSLIST_HH
#define MYPHYSICSLIST_HH

#include "globals.hh"
#include "G4VModularPhysicsList.hh"

class MyPhysicsList : public G4VModularPhysicsList {
public:
    MyPhysicsList();
    virtual ~MyPhysicsList();

    virtual void ConstructParticle();
    virtual void ConstructProcess();

};

#endif
