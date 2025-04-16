#include "MySteppingAction.hh"
#include "MyPrimaryGeneratorAction.hh"
#include "MyParticleUserInformation.hh"
#include <string>  // required for std::stoi
#include <algorithm> // For std::min
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "G4PrimaryParticle.hh"
#include "G4UserLimits.hh"
#include "G4Box.hh"
#include "G4VProcess.hh" // getting creation process

MySteppingAction::MySteppingAction(MyRunAction* runAction)
: G4UserSteppingAction(),
  fRunAction(runAction) {}

MySteppingAction::~MySteppingAction() {}

void MySteppingAction::UserSteppingAction(const G4Step* step) {
    G4Track* track = step->GetTrack();
    G4String particleName = track->GetDefinition()->GetParticleName();
    G4String volumeName = track->GetVolume()->GetName();
    G4ParticleDefinition* particleDefinition = track->GetDefinition();
    G4int Z = particleDefinition->GetAtomicNumber();
    G4int A = particleDefinition->GetAtomicMass();

    G4ThreeVector preStepMomentum = step->GetPreStepPoint()->GetMomentum() / keV;
    G4ThreeVector postStepMomentum = step->GetPostStepPoint()->GetMomentum() / keV;

    G4ThreeVector globalPosition = step->GetPostStepPoint()->GetPosition();
    G4ThreeVector globalPosition_preStep = step->GetPreStepPoint()->GetPosition();
    
    G4String creatorProcessName;
    if (!track->GetCreatorProcess()){
        creatorProcessName = "primary";
    }
    else{
        creatorProcessName = track->GetCreatorProcess()->GetProcessName();
    }

    // Record steps for particles in defined volumes
    // or if the particle is a recoiling nucleus
    if ( (volumeName.contains("TPC") || volumeName.contains("PTFE") || volumeName.contains("ICV") || volumeName.contains("OCV") || volumeName.contains("LiqScintDet")) 
        && step->GetTotalEnergyDeposit() / keV > 0.){

        // Store info about this track itself
        fRunAction->fTrackID.emplace_back(track->GetTrackID());
        fRunAction->fTrackParentID.emplace_back(track->GetParentID());
        fRunAction->fTrackStepNum.emplace_back(track->GetCurrentStepNumber());
        fRunAction->fParticleCreatorProcesses.emplace_back(creatorProcessName);

        // Store particle mass
        fRunAction->fParticleMass.emplace_back(track->GetDynamicParticle()->GetMass() / keV);

        // Store energy info
        fRunAction->fStepTime_ns.emplace_back(track->GetGlobalTime() / ns);
        fRunAction->fEnergyDep.emplace_back(step->GetTotalEnergyDeposit() / keV);
        fRunAction->fPreStepEnergy.emplace_back(step->GetPreStepPoint()->GetKineticEnergy() / keV);  

        // step length for dE/dx
        fRunAction->fStepLength.emplace_back(step->GetStepLength() / um);

        // Store momentum info
        fRunAction->fPreStepMomentumX.emplace_back(preStepMomentum.x());
        fRunAction->fPreStepMomentumY.emplace_back(preStepMomentum.y());
        fRunAction->fPreStepMomentumZ.emplace_back(preStepMomentum.z());

        fRunAction->fPostStepMomentumX.emplace_back(postStepMomentum.x());
        fRunAction->fPostStepMomentumY.emplace_back(postStepMomentum.y());
        fRunAction->fPostStepMomentumZ.emplace_back(postStepMomentum.z());

        // store particle name and PDG code
        fRunAction->fParticleName.emplace_back(particleName);
        fRunAction->fParticleID.emplace_back(track->GetDefinition()->GetPDGEncoding());

        // global coordinates using world origin
        fRunAction->fXglobal.emplace_back(globalPosition.x() / um);
        fRunAction->fYglobal.emplace_back(globalPosition.y() / um);
        fRunAction->fZglobal.emplace_back(globalPosition.z() / um);

        // Get the touchable history
        const G4TouchableHistory* theTouchable = (const G4TouchableHistory*)(track->GetTouchable());
        
        // Transform the global coordinates to local coordinates
        // e.g. local to the detector by shifting origin to be center of the this logocal volume
        G4ThreeVector localPosition = theTouchable->GetHistory()->GetTopTransform().TransformPoint(globalPosition);

        // Local cooordinates using origin of volume in which step occurred
        fRunAction->fXlocal.emplace_back(localPosition.x() / um);
        fRunAction->fYlocal.emplace_back(localPosition.y() / um);
        fRunAction->fZlocal.emplace_back(localPosition.z() / um);

        // Store volume name
        fRunAction->fDetectorName.emplace_back(volumeName);

        // Get Material of interaction
        //G4Material* pre_material = step->GetPreStepPoint()->GetMaterial(); 
        //G4Material* post_material = step->GetPostStepPoint()->GetMaterial(); 
        //std::cout << pre_material->GetName() << " hit in " << volumeName << " by " << particleName << std::endl;
        //std::cout << post_material->GetName() << " hit in " << volumeName << " by " << particleName << std::endl;

        // -------------------------------------- // 
        // Don't write tracks until end of event  //
        // -------------------------------------- // 

        // OPTION TO IGNORE SIXTH PANEL EVENTS
        // Kill events that hit that sixth panel
        //if (volumeName.contains("Si6")){
        //    std::cout << "Kill event after hitting Si6!" << std::endl;
        //    track->SetTrackStatus(fStopAndKill);
        //}
    
    }
    else{
        //std::cout << particleName << " has " << step->GetPreStepPoint()->GetKineticEnergy() / keV << " keV in " <<
        //volumeName << " and was created by " << creatorProcessName << 
        //" and deposited " << step->GetTotalEnergyDeposit() / keV << " keV" << std::endl;
    }

}
