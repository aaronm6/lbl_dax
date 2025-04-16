#ifndef MY_PARTICLE_USER_INFORMATION_HH
#define MY_PARTICLE_USER_INFORMATION_HH

#include "G4VUserPrimaryParticleInformation.hh"
#include "G4Types.hh"

class MyParticleUserInformation : public G4VUserPrimaryParticleInformation {
public:
    MyParticleUserInformation(G4int id, G4int kind, G4double energy);
    virtual ~MyParticleUserInformation();

    virtual void Print() const;
    G4int GetPrimaryID() const { return fPrimaryID; }
    void SetPrimaryID(G4int id) { fPrimaryID = id; }
    G4int GetPrimaryKind() const { return fPrimaryKind; }
    void SetPrimaryKind(G4int kind) { fPrimaryKind = kind; }
    G4double GetPrimaryEnergy() const { return fPrimaryEnergy; }
    void SetPrimaryEnergy(G4double energy) { fPrimaryEnergy = energy; }

private:
    G4int fPrimaryID;
    G4int fPrimaryKind;
    G4double fPrimaryEnergy;
};

#endif
