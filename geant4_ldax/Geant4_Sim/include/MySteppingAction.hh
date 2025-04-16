#ifndef MYSTEPPINGACTION_HH
#define MYSTEPPINGACTION_HH

#include "MyRunAction.hh"
#include "G4UserSteppingAction.hh"

class G4Step;

class MySteppingAction : public G4UserSteppingAction {
public:
    MySteppingAction(MyRunAction* runAction);
    virtual ~MySteppingAction();

    virtual void UserSteppingAction(const G4Step* step);

private:
    MyRunAction* fRunAction;
};

#endif
