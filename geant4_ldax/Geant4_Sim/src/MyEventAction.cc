#include "MyEventAction.hh"
#include "MyRunAction.hh"
#include "G4Event.hh"
#include "G4RunManager.hh"
#include "MyPrimaryGeneratorAction.hh"
#include "G4SystemOfUnits.hh"
#include "G4TransportationManager.hh"

MyEventAction::MyEventAction(MyRunAction* runAction) 
    : G4UserEventAction(), 
    fRunAction(runAction) {}

MyEventAction::~MyEventAction() { }

// Reset values for each begninning of each event
void MyEventAction::BeginOfEventAction(const G4Event* anEvent){

    // Loop through Primaries stored in G4Event
    // and fill vectors for primary info in ttree
    G4int numberOfPrimaryVertex = anEvent->GetNumberOfPrimaryVertex();
    for (G4int i = 0; i < numberOfPrimaryVertex; ++i){

        G4PrimaryVertex* primaryVertex(anEvent->GetPrimaryVertex(i));
        G4int numberOfParticle = primaryVertex->GetNumberOfParticle();

        for (G4int j = 0; j < numberOfParticle; ++j){
            G4PrimaryParticle* primary(primaryVertex->GetPrimary(j));
            addPrimary(primaryVertex, primary);
        }
    }

}

void MyEventAction::EndOfEventAction(const G4Event* anEvent){

    // Fill the tree only at the end of the event 
    fRunAction->fTree->Fill();

    // Clear the vectors after the event
    fRunAction->primaryName.clear();
    fRunAction->primaryVolumeName.clear();
    fRunAction->primaryPid.clear();
    fRunAction->primaryTime_ns.clear();
    fRunAction->primaryEnergy_keV.clear();
    fRunAction->primaryTotalEnergy_keV.clear();
    fRunAction->primaryTotalMomentum_keV.clear();
    fRunAction->primaryPosition_X_mm.clear();
    fRunAction->primaryPosition_Y_mm.clear();
    fRunAction->primaryPosition_Z_mm.clear();
    fRunAction->primaryDirection_X.clear();
    fRunAction->primaryDirection_Y.clear();
    fRunAction->primaryDirection_Z.clear();

    fRunAction->fTrackID.clear();
    fRunAction->fTrackStepNum.clear();
    fRunAction->fTrackParentID.clear();
    fRunAction->fDetectorName.clear();
    fRunAction->fDetectorID.clear();
    fRunAction->fParticleName.clear();
    fRunAction->fParticleID.clear();
    fRunAction->fParticleCreatorProcesses.clear();
    fRunAction->fParticleMass.clear();
    fRunAction->fStepTime_ns.clear();
    fRunAction->fPreStepEnergy.clear();
    fRunAction->fEnergyDep.clear();
    fRunAction->fPreStepMomentumX.clear();
    fRunAction->fPreStepMomentumY.clear();
    fRunAction->fPreStepMomentumZ.clear();
    fRunAction->fPostStepMomentumX.clear();
    fRunAction->fPostStepMomentumY.clear();
    fRunAction->fPostStepMomentumZ.clear();
    fRunAction->fStepLength.clear();
    fRunAction->fXglobal.clear();
    fRunAction->fYglobal.clear();
    fRunAction->fZglobal.clear();
    fRunAction->fXlocal.clear();
    fRunAction->fYlocal.clear();
    fRunAction->fZlocal.clear();
    fRunAction->implantDepth.clear();

}

void MyEventAction::addPrimary(G4PrimaryVertex* primaryVertex, G4PrimaryParticle* g4primary){

    fRunAction->primaryName.push_back(g4primary->GetParticleDefinition()->GetParticleName());

    // Need to get volume name
    G4Navigator* g4Navigator = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
    const G4VPhysicalVolume* pv = g4Navigator->LocateGlobalPointAndSetup(primaryVertex->GetPosition(), nullptr, false, true);

    fRunAction->primaryVolumeName.emplace_back(pv->GetName());

    fRunAction->primaryPid.emplace_back(g4primary->GetPDGcode());

    fRunAction->primaryTime_ns.emplace_back(primaryVertex->GetT0() / ns);

    fRunAction->primaryEnergy_keV.emplace_back(g4primary->GetKineticEnergy() / keV);

    fRunAction->primaryTotalEnergy_keV.emplace_back(g4primary->GetTotalEnergy() / keV);

    fRunAction->primaryTotalMomentum_keV.emplace_back(g4primary->GetTotalMomentum() / keV);

    fRunAction->primaryPosition_X_mm.emplace_back(primaryVertex->GetX0() / mm);
    fRunAction->primaryPosition_Y_mm.emplace_back(primaryVertex->GetY0() / mm);
    fRunAction->primaryPosition_Z_mm.emplace_back(primaryVertex->GetZ0() / mm);

    fRunAction->primaryDirection_X.emplace_back(g4primary->GetMomentumDirection().x());
    fRunAction->primaryDirection_Y.emplace_back(g4primary->GetMomentumDirection().y());
    fRunAction->primaryDirection_Z.emplace_back(g4primary->GetMomentumDirection().z());
}
