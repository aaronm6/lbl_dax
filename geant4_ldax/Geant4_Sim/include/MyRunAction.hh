#ifndef MYRUNACTION_HH
#define MYRUNACTION_HH

#include "TFile.h"
#include "TTree.h"
#include "G4UserRunAction.hh"
#include "G4String.hh"

class G4Run;
class MyPrimaryGeneratorAction; // Forward declaration of MyPrimaryGeneratorAction

class MyRunAction : public G4UserRunAction {
public:
    MyRunAction(MyPrimaryGeneratorAction* generatorAction); // Updated constructor
    virtual ~MyRunAction();

    virtual void BeginOfRunAction(const G4Run* aRun);
    virtual void EndOfRunAction(const G4Run* aRun);

    TFile* fFile;
    TTree* fTree;
    
    //vector of track info
    std::vector<float> fXglobal, fYglobal, fZglobal;
    std::vector<float> fXlocal, fYlocal, fZlocal;
    std::vector<float> implantDepth;
    std::vector<float> fParticleMass, fStepTime_ns, fEnergyDep, fPreStepEnergy,fStepLength;
    std::vector<float> fPreStepMomentumX, fPreStepMomentumY, fPreStepMomentumZ;
    std::vector<float> fPostStepMomentumX, fPostStepMomentumY, fPostStepMomentumZ;
    std::vector<int> fDetectorID, fParticleID, fTrackID, fTrackParentID, fTrackStepNum;
    std::vector<std::string> fDetectorName, fParticleName, fParticleCreatorProcesses;

    // Vector of primary info
    // e.g. if one or more primaries exist like in radioactive decays
    std::vector<std::string> primaryName;
    std::vector<std::string> primaryVolumeName;
    std::vector<int32_t> primaryPid;
    std::vector<float> primaryTime_ns;
    std::vector<float> primaryEnergy_keV;
    std::vector<float> primaryTotalEnergy_keV;
    std::vector<float> primaryTotalMomentum_keV;
    std::vector<float> primaryPosition_X_mm;
    std::vector<float> primaryPosition_Y_mm;
    std::vector<float> primaryPosition_Z_mm;
    std::vector<float> primaryDirection_X;
    std::vector<float> primaryDirection_Y;
    std::vector<float> primaryDirection_Z;

private:
    MyPrimaryGeneratorAction* fGeneratorAction; // Add a pointer to MyPrimaryGeneratorAction
};

#endif
