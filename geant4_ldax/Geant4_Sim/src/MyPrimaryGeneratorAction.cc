#include "MyPrimaryGeneratorAction.hh"
#include "MyParticleUserInformation.hh"
#include "MyPrimaryGeneratorMessenger.hh"

#include "Randomize.hh"
#include "TH1.h" 
#include "TGraph.h" 
#include <random>
#include <iomanip>  // Required for setprecision

#include "G4Event.hh"
#include "G4IonTable.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include "G4PrimaryParticle.hh"
#include "G4RandomDirection.hh"
#include "G4GeneralParticleSource.hh"
#include "G4SPSEneDistribution.hh"
#include "G4Electron.hh"

MyPrimaryGeneratorAction* MyPrimaryGeneratorAction::instance = 0;

MyPrimaryGeneratorAction* MyPrimaryGeneratorAction::Instance() {
    if (instance == 0) {
        instance = new MyPrimaryGeneratorAction();
    }
    return instance;
}

MyPrimaryGeneratorAction::MyPrimaryGeneratorAction()
: G4VUserPrimaryGeneratorAction(),
  fParticleGun(0),
  fEventID(0),
  messenger(0),
  fDirection(0),
  gSpectrum{ nullptr },
  hSpectrum{ nullptr }

{
    messenger = new MyPrimaryGeneratorMessenger(this);
    fParticleGun = new G4ParticleGun(1);
    fGeneralParticleSource = new G4GeneralParticleSource();
    pTable = G4ParticleTable::GetParticleTable();
    G4cout << "MyPrimaryGeneratorAction()" << G4endl;

    CLHEP::HepRandom::setTheEngine(new CLHEP::MTwistEngine); 
}

MyPrimaryGeneratorAction::~MyPrimaryGeneratorAction() {
    delete fParticleGun;
    delete messenger;
    delete fGeneralParticleSource;
}

void MyPrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent) {
   
    switch (fGeneratorType){
        case generatorType::Particle:{

            fParticleGun->SetParticleDefinition(fParticleDef);
            fParticleGun->SetParticleEnergy(fParticleEnergy);

            break;
        }
        case generatorType::Ion:{

            fParticleGun->SetParticleDefinition(fIon);
            fParticleGun->SetParticleEnergy(0.0 * keV); // at rest

            break;
        }
        case generatorType::Spectrum:{
            
            fParticleGun->SetParticleDefinition(fParticleDef);
            // sample the energy spectrum
            double Eprim = hSpectrum->GetRandom();
            fParticleGun->SetParticleEnergy(Eprim * keV);
            
            break;
        }
        case generatorType::Tritium:{
            
            fParticleGun->SetParticleDefinition(G4Electron::Definition());
            // sample the energy spectrum
            G4double Ebeta = GetTritiumElectronEnergy();;
            fParticleGun->SetParticleEnergy(Ebeta * keV);
            
            break;
        }
    }

    // Common to all generators
    if (!useSpecifiedAngles){
        fDirection = G4RandomDirection();
    }
    fParticleGun->SetParticleMomentumDirection(fDirection);
    fParticleGun->SetParticlePosition( setParticlePosition(fPositionType) ); 
    fParticleGun->SetParticleTime(0. * ns); // set to start at t=0
    fParticleGun->GeneratePrimaryVertex(anEvent); // generatre primary vertex to start event
    
    // Record some information
    G4PrimaryParticle* primaryParticle = anEvent->GetPrimaryVertex()->GetPrimary(0);

    fPrimaryEnergy = primaryParticle->GetKineticEnergy();
    fPrimaryKind = primaryParticle->GetParticleDefinition()->GetPDGEncoding();
    fParticleName = primaryParticle->GetParticleDefinition()->GetParticleName();
    
    primaryParticle->SetUserInformation(new MyParticleUserInformation(fEventID, primaryParticle->GetPDGcode(), primaryParticle->GetKineticEnergy()));

    fEventID++;

}

// Set user provided angles, default is random momentum 
void MyPrimaryGeneratorAction::SetParticleAngles(G4ThreeVector const& newAngles) {

    fParticleAngles = newAngles;
    useSpecifiedAngles = true;

    double phi = fParticleAngles.x();
    double theta = fParticleAngles.y();
    // convert degrees to radians for use in trigonometric functions
    phi = phi * M_PI / 180.;
    theta = theta * M_PI / 180.;
    double dx = std::sin(theta) * std::cos(phi);
    double dy = std::sin(theta) * std::sin(phi);
    double dz = std::cos(theta);
    
    G4cout << "Setting particle direction to (" << dx << ", " << dy << ", " << dz << ")" << G4endl;

    fDirection = G4ThreeVector(dx, dy, dz);
}

// Set particle position type
void MyPrimaryGeneratorAction::setPositionType(positionType whichPositionType, double innerradius=0., double outerradius=0., double x=0., double y=0., double z=0., std::string unit="cm"){
    fPositionType = whichPositionType;
    fPrimaryParticleOuterRadius = innerradius * G4UIcommand::ValueOf((G4String)unit);
    fPrimaryParticleInnerRadius = outerradius * G4UIcommand::ValueOf((G4String)unit);
    fPrimaryParticleX = x * G4UIcommand::ValueOf((G4String)unit);
    fPrimaryParticleY = y * G4UIcommand::ValueOf((G4String)unit);
    fPrimaryParticleZ = z * G4UIcommand::ValueOf((G4String)unit);
}

// Set particle position
G4ThreeVector MyPrimaryGeneratorAction::setParticlePosition(positionType whichPos){

    switch (whichPos){
        case positionType::RandomSphere:{
            //double phi = G4UniformRand() * 2. * M_PI;
            //double theta = G4UniformRand() * M_PI;
            G4ThreeVector randomDir = G4RandomDirection();
            fPrimaryParticleX = fPrimaryParticleOuterRadius * randomDir.x(); //std::sin(theta) * std::cos(phi);
            fPrimaryParticleY = fPrimaryParticleOuterRadius * randomDir.y(); //std::sin(theta) * std::sin(phi);
            fPrimaryParticleZ = fPrimaryParticleOuterRadius * randomDir.z(); //std::cos(theta);
            fParticlePosTypeStr = "randomSphere";
            break;
        }
        case positionType::RandomSphereShell:{
            G4ThreeVector randomDir = G4RandomDirection();
            double u =  G4UniformRand();
            G4double Radius = pow( u*pow(fPrimaryParticleOuterRadius,3.) + (1-u)*pow(fPrimaryParticleInnerRadius,3 ), 1./3.);
            fPrimaryParticleX = Radius * randomDir.x(); //std::sin(theta) * std::cos(phi);
            fPrimaryParticleY = Radius * randomDir.y(); //std::sin(theta) * std::sin(phi);
            fPrimaryParticleZ = Radius * randomDir.z(); //std::cos(theta);
            fParticlePosTypeStr = "randomSphereShell";
            break;
        }
        case positionType::Point:{
            fParticlePosTypeStr = "Point";
            break;
        }
    }

    return G4ThreeVector(fPrimaryParticleX, fPrimaryParticleY, fPrimaryParticleZ);

}

// Set Ion info for Ion generator
void MyPrimaryGeneratorAction::SetIonZA(G4int Z, G4int A) {
    G4cout << "Setting ion to Z = " << Z << " A = " << A << G4endl;
    fIon = pTable->GetIonTable()->GetIon(Z, A, 0);
}

// Set Sterile Neutrino Mass
void MyPrimaryGeneratorAction::SetSterileNuMass(double NuMass){
    G4cout << "Setting Sterile Neutrino mass to " << NuMass << " keV" << G4endl;
    fSterileNuMass = NuMass*keV;
}

// Set info for output file
G4String MyPrimaryGeneratorAction::GetParticleInfo() {
    
    std::stringstream ss;

    std::stringstream angles;
    if (useSpecifiedAngles) {
        angles << fParticleAngles.x() << "phi_" << fParticleAngles.y() << "theta";
    }

    switch (fGeneratorType){
        case generatorType::Particle:{
            fGeneratorName = "Particle";
            break;
        }
        case generatorType::Ion:{
            fGeneratorName = "Ion";
            G4ParticleDefinition * anti_nu_e_Def = pTable->FindParticle("anti_nu_e");
            G4double nuMass = anti_nu_e_Def->GetPDGMass();
            if (nuMass > 0.){
                fParticleEnergyStr = std::to_string(int(nuMass/keV)) + "keVSterileNu";
            }
            else{
                fParticleEnergyStr = "0keV";
            }
            break;
        }
        case generatorType::Spectrum:{
            fGeneratorName = "Spectrum";
            fParticleEnergyStr = "";
            break;
        }
        case generatorType::Tritium:{
            fGeneratorName = "Tritium";
            fParticleEnergyStr = "spectrum";
            break;
        }
    }

    ss << fGeneratorName << "_" << fParticleName << "_" << fParticleEnergyStr << "_" << fParticlePosTypeStr << angles.str();
    return ss.str();
    
}

// Get Which Generator is being used
void MyPrimaryGeneratorAction::setGeneratorType(generatorType whichGen){
    fGeneratorType = whichGen;
}

// Set particle type for Particle generator 
void MyPrimaryGeneratorAction::setParticleType(G4String whichParticle){

    G4String gpsParticleList = "alpha,anti_nu_e,anti_nu_mu,anti_nu_tau,"
                                "chargedgeantino,deuteron,e+,e-,gamma,geantino,mu+,mu-,neutron,nu_e,"
                                "nu_mu,nu_tau,pi+,pi0,pi-,tau+,tau-";
    
    G4String gpsmatch;
    std::stringstream gpsParticlesSS(gpsParticleList);
    G4bool gpsIsGood = false;
    while ( getline(gpsParticlesSS, gpsmatch, ',') && !gpsIsGood ){
        if (gpsmatch == whichParticle){
            gpsIsGood = true;
        }
    }
    if (!gpsIsGood){
        throw std::runtime_error{"Particle Name " + whichParticle + " does not match options."};
    }
    
    G4cout << "Using Particle Type " << whichParticle << G4endl;
    fParticleDef = pTable->FindParticle(whichParticle);
    fParticleType = whichParticle;
}

// Set particle energy for particle generator
void MyPrimaryGeneratorAction::SetParticleEnergyAndUnit(double whatEnergy, std::string whichUnit){

    std::string fUnit = "";
    if (whichUnit.find("keV") != std::string::npos){
        fUnit = "keV";
    }
    else if (whichUnit.find("MeV") != std::string::npos){
        fUnit = "MeV";
    }
    else if (whichUnit.find("MeV") != std::string::npos){
        fUnit = "MeV";
    }
    else if (whichUnit.find("GeV") != std::string::npos){
        fUnit = "GeV";
    }
    else if (whichUnit == "eV"){
        fUnit = "GeV";
    }
    else{
        std::cout << "Provided Unit " << whichUnit << " could not be found. Default to keV." << std::endl;
    }

    fParticleEnergy = ( whatEnergy * G4UIcommand::ValueOf((G4String)fUnit) );

    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << fParticleEnergy / keV << "keV";
    fParticleEnergyStr = stream.str();
}

// Read in spectrum into TF1
void MyPrimaryGeneratorAction::setEnergySpectrum(std::string spectrumFilePath){

    std::ifstream spectrumFile(spectrumFilePath.c_str());
    if (!spectrumFile)
    {
       throw std::runtime_error("Cannot open the energy spectrum text file.");
    }

    // Read in each column into energy or rate
    std::vector<double> energies, rates;
    double energy, rate;
    while (spectrumFile >> energy >> rate)
    {
        energies.emplace_back(energy);
        rates.emplace_back(rate);
    }
    int nEntries = (int)energies.size();
    double Emin = *std::min_element(energies.begin(), energies.end());
    double Emax = *std::max_element(energies.begin(), energies.end());

    gSpectrum = new TGraph(nEntries, &energies[0], &rates[0]);
    hSpectrum = new TH1D("mySpectrum","",nEntries, Emin, Emax);
    for (int j = 0; j < nEntries; j++){
        double x,y;
        gSpectrum->GetPoint(j,x,y);
        hSpectrum->SetBinContent(j+1,y);
    }

    //double Emin = *std::min_element(energies.begin(), energies.end());
    //double Emax = *std::max_element(energies.begin(), energies.end());
    // linear interpolation by default (assumed energies are closely spaced)
    //fSpectrumInterp = std::make_unique<ROOT::Math::Interpolator>(energies, rates, ROOT::Math::Interpolation::kLINEAR);
    //fSpectrum = std::make_unique<TF1>("mySpectrum", this, &MyPrimaryGeneratorAction::EvalSpectrum, Emin, Emax, 0, "MyPrimaryGeneratorAction::EvalSpectrum", "EvalSpectrum");
    //fSpectrum->SetNpx(1000);

}

// Analytics form of tritium beta spectrum
//	The Q value comes from Sz. Nagy et al., "On the Q-value of the tritium
//	beta-decay", Europhys. Lett., 74 (3), p. 404 (2006).
G4double MyPrimaryGeneratorAction::GetTritiumElectronEnergy(){

    G4double Q = 18.5898;
    G4double m_e = 511.;
    G4double pi = 3.14159265358979312;
    G4double a = 1. / 137;
    G4double Z = 2.;

    G4double xmax = Q;
    G4double ymax = 1.1e7;

    G4double xTry = xmax * G4UniformRand();
    G4double yTry = ymax * G4UniformRand();

    G4double B = sqrt(xTry * xTry + 2 * xTry * m_e) / (xTry + m_e);
    G4double x = (2 * pi * Z * a) * (xTry + m_e) / sqrt(xTry * xTry + 2 * xTry * m_e);

    while (yTry > (sqrt(2 * xTry * m_e) * (xTry + m_e) * (Q - xTry) * (Q - xTry) * x * (1. / (1 - exp(-x))) * (1.002037 - 0.001427 * (B))))
    {
        xTry = xmax * G4UniformRand();
        yTry = ymax * G4UniformRand();
        B = sqrt(xTry * xTry + 2 * xTry * m_e) / (xTry + m_e);
        x = (2 * pi * Z * a) * (xTry + m_e) / sqrt(xTry * xTry + 2 * xTry * m_e);
    }

    return (xTry);

}
