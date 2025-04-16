#ifndef MYDETECTORCONSTRUCTION_HH
#define MYDETECTORCONSTRUCTION_HH

#include "globals.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4LogicalVolume.hh"

class G4VPhysicalVolume;
class MyDetectorConstMessenger;

class MyDetectorConstruction : public G4VUserDetectorConstruction 
{
public:
    MyDetectorConstruction();
    virtual ~MyDetectorConstruction();

    virtual G4VPhysicalVolume* Construct();

    void SetWhichGeometry(std::string whichGeom);
    inline std::string GetWhichGeometry(){ return fWhichGeometry; }

    void SetWhichDopant(std::string whichDopant);
    inline std::string GetWhichDopant(){ return fWhichDopant; }

    void SetDopantFraction(double dopantFrac);
    inline double GetDopantFraction(){ return fDopantFraction; }

private:
    MyDetectorConstMessenger* fMessenger;

    // G4double fMaxStepSize;
    G4LogicalVolume* worldLogVol;

    // control whether to build stuff
    std::string fWhichGeometry;

    std::string fWhichDopant;
    double fDopantFraction;

};

#endif
