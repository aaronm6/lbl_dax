#include "MyParticleUserInformation.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"

MyParticleUserInformation::MyParticleUserInformation(G4int id, G4int kind, G4double energy)
    : fPrimaryID(id), fPrimaryKind(kind), fPrimaryEnergy(energy) {}

MyParticleUserInformation::~MyParticleUserInformation() {}

void MyParticleUserInformation::Print() const {
    std::cout << "Event ID: " << fPrimaryID << ", particle code: " << fPrimaryKind << ", energy: " << G4BestUnit(fPrimaryEnergy,"Energy") << std::endl;
}
