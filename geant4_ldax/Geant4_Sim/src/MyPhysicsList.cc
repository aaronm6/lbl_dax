#include "MyPhysicsList.hh"

#include "G4ParticleDefinition.hh"
#include "G4ParticlePropertyData.hh"
#include "G4EmLivermorePhysics.hh"
#include "G4DecayPhysics.hh"
#include "G4RadioactiveDecayPhysics.hh"
#include "G4HadronElasticPhysicsHP.hh"
#include "G4SystemOfUnits.hh"
#include "G4StepLimiterPhysics.hh"
#include "G4ProductionCuts.hh"
#include "G4Region.hh"
#include "G4RegionStore.hh"
#include "G4Decay.hh"
#include "G4ProcessManager.hh"
#include "G4Triton.hh"
#include "G4RadioactiveDecay.hh"
#include "G4RadioactiveDecayBase.hh"

MyPhysicsList::MyPhysicsList() 
:   G4VModularPhysicsList()
{  
    RegisterPhysics(new G4EmLivermorePhysics());
    RegisterPhysics(new G4DecayPhysics());
    RegisterPhysics(new G4RadioactiveDecayPhysics());
    RegisterPhysics(new G4HadronElasticPhysicsHP());

    G4StepLimiterPhysics* stepLimitPhys = new G4StepLimiterPhysics();
    RegisterPhysics(stepLimitPhys);
    // defaultCutValue = 1.0 * um;
    // SetDefaultCutValue(1 * um);
    
    // Read macro commands to set sterile Nu mass
    //std::cout << "Neutrino Mass must be set here!" << std::endl;
    //G4AntiNeutrinoE::SetNeutrinoMass(500.*keV);

}

MyPhysicsList::~MyPhysicsList() {
}

void MyPhysicsList::ConstructParticle() {
    G4VModularPhysicsList::ConstructParticle();
}

void MyPhysicsList::ConstructProcess() {
    G4VModularPhysicsList::ConstructProcess();

    // Make tritium unstable, so it can be decayed
    G4ParticleDefinition* tr = G4Triton::Definition();
    tr->SetPDGStable(false);

    // Remove G4Decay process, which requires a registered decay table
    G4VProcess* decay = 0;
    G4ProcessManager* pman = tr->GetProcessManager();
    G4ProcessVector* pvec = pman->GetAtRestProcessVector();
    for (G4int i=0; i<pvec->size() && decay==0; i++) {
        if ((*pvec)[i]->GetProcessName() == "Decay") decay = (*pvec)[i];
    }
    if (decay) pman->RemoveProcess(decay);

    // Attach RDM, which is a rest-discrete process
    #if G4VERSION_NUMBER < 1060             /* See G4RadioactiveDecayPhysics.cc */
    decay = new G4RadioactiveDecay();
    #else
    decay = new G4RadioactiveDecayBase();
    #endif

    tr->GetProcessManager()->AddProcess(decay, 1000, -1, 1000);
    
    // Add Decay Process
    G4Decay* theDecayProcess = new G4Decay();
    auto particleIterator = GetParticleIterator();
    particleIterator->reset();
    while ((*particleIterator)()) {
        G4ParticleDefinition* particle = particleIterator->value();
        G4ProcessManager* pmanager = particle->GetProcessManager();
        if (theDecayProcess->IsApplicable(*particle)) {
            pmanager->AddProcess(theDecayProcess);
        }
    }
}