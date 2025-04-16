#include "MyDetectorConstruction.hh"
#include "MyDetectorConstMessenger.hh"

#include "G4RunManager.hh"

#include "G4Box.hh"
#include "G4Orb.hh"
#include "G4Tubs.hh"
#include "G4Cons.hh"
#include "G4Sphere.hh"
#include "G4UnionSolid.hh"
#include "G4Ellipsoid.hh"
#include "G4SubtractionSolid.hh"

#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"

#include "G4Material.hh"
#include "G4Element.hh"

#include "G4NistManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4UserLimits.hh"
#include "G4Region.hh"
#include "G4VisAttributes.hh"
#include <sstream>


MyDetectorConstruction::MyDetectorConstruction() 
:   G4VUserDetectorConstruction(),
    fMessenger(0),
    fWhichGeometry("HydroX"),
    fWhichDopant("He"),
    fDopantFraction(0.)
{
    fMessenger = new MyDetectorConstMessenger(this);
    G4cout << "MyDetectorConstruction()" << G4endl;
}

MyDetectorConstruction::~MyDetectorConstruction() {
    delete fMessenger;
}

G4VPhysicalVolume* MyDetectorConstruction::Construct() {
    
    // Define the world volume
    G4double worldSizeXY = 1000.0 * cm;
    G4double worldSizeZ = 1000.0 * cm;
    G4Material* worldMaterial = G4NistManager::Instance()->FindOrBuildMaterial("G4_Galactic");  // vacuum

    G4Box* worldBox = new G4Box("World", worldSizeXY, worldSizeXY, worldSizeZ);
    worldLogVol = new G4LogicalVolume(worldBox, worldMaterial, "World");

    // Visibility attributes
    G4VisAttributes* WorldVisAtt = new G4VisAttributes(G4Colour::Red());
    WorldVisAtt->SetVisibility(false);
    worldLogVol->SetVisAttributes(WorldVisAtt);

    // Create world
    G4VPhysicalVolume* worldPhysVol = new G4PVPlacement(0, G4ThreeVector(), worldLogVol, "World", 0, false, 0);

    // ------------------------------- //
    //  Create Materials and Elements  //
    // ------------------------------- //
    G4Element* el_H = new G4Element("Hydrogen","H", 1, 1.008*g/mole);
    G4Element* el_He = new G4Element("Helium","He", 2, 2.*g/mole);
    G4Element* el_Xe = new G4Element("Xenon","Xe", 54, 131.*g/mole);

    G4Element* el_C = new G4Element("Carbon","C", 6, 12.011*g/mole);
    G4Element* el_F = new G4Element("Fluorine","F", 9, 18.99*g/mole);

    G4Element* el_Mn = new G4Element("Manganese","Mn", 25, 54.94*g/mole);
    G4Element* el_Si = new G4Element("Silicon","Si", 14, 28.09*g/mole);
    G4Element* el_Cr = new G4Element("Chromium","Cr", 24, 52.*g/mole);
    G4Element* el_Ni = new G4Element("Nickel","Ni", 28, 58.7*g/mole);
    G4Element* el_Fe = new G4Element("Iron","Fe", 26, 55.85*g/mole);

    G4double He_density = 0.0001786*g/cm3;
    G4double H2_density = 0.08*kg/m3;
    G4double LXe_density = 2.8*g/cm3;
    G4double Teflon_density = 2.2*g/cm3;
    G4double SS_density = 8.02*g/cm3;
    G4double EJ301_density = 0.874*g/cm3;

    G4Material* mat_H2 = new G4Material("mat_H2", H2_density, 1);
    mat_H2->AddElement(el_H, 2);

    G4Material* mat_PTFE = new G4Material("PTFE", Teflon_density, 2);
    mat_PTFE->AddElement(el_C, 0.240183);
    mat_PTFE->AddElement(el_F, 0.759817);

    G4Material* mat_SS = new G4Material("StainlessSteel", SS_density, 5);
    mat_SS->AddElement(el_Mn, 0.02);
    mat_SS->AddElement(el_Si, 0.01);
    mat_SS->AddElement(el_Cr, 0.19);
    mat_SS->AddElement(el_Ni, 0.10);
    mat_SS->AddElement(el_Fe, 0.68);

    G4Material* mat_EJ301 = new G4Material("EJ301", EJ301_density, 2);
    mat_EJ301->AddElement(el_H, 1-0.452);
    mat_EJ301->AddElement(el_C, 0.452);

    // ---------------------------- //
    //       Using HydroX Setup      //
    // ---------------------------- //
    if (fWhichGeometry == "HydroX"){
        std::cout << "Using HydroX setup!" << std::endl;

        // Make Xenon cocktail
        G4Material* mat_dopedLXe;
        double mass_frac = fDopantFraction;
        if (fWhichDopant == "He"){
            mat_dopedLXe = new G4Material("doped_LXe", LXe_density, 2);
            mat_dopedLXe->AddElement(el_He, mass_frac);
            mat_dopedLXe->AddElement(el_Xe, 1-mass_frac);
        }
        else if (fWhichDopant == "H2"){
            mat_dopedLXe = new G4Material("doped_LXe", LXe_density, 2);
            mat_dopedLXe->AddMaterial(mat_H2, mass_frac);
            mat_dopedLXe->AddElement(el_Xe, 1-mass_frac);
        }
        else{
            mat_dopedLXe = new G4Material("doped_LXe", LXe_density, 1);
            mat_dopedLXe->AddElement(el_Xe, 1);
        }
        

        // Make TPC Volume
        G4double tpc_r = 2.3*cm;
        G4double tpc_h = 2.3*cm;
        G4Tubs* tpcVol = new G4Tubs("tpcVol", 0., tpc_r, tpc_h/2, 0.*deg, 360.*deg);
        G4LogicalVolume* tpcLogVol = new G4LogicalVolume(tpcVol, mat_dopedLXe, "tpcVol");

        new G4PVPlacement(new G4RotationMatrix(), G4ThreeVector(0., 0., 0.), tpcLogVol, "TPC", worldLogVol, false, 0.);

        // Make Teflon Surrounding TPC
        G4double teflon_r_inner = tpc_r;
        G4double teflon_thickness = 1.0*cm;
        G4double teflon_r_outer = teflon_r_inner+teflon_thickness;
        G4double teflon_h = tpc_h;

        G4Tubs* teflonVol = new G4Tubs("teflonVol", teflon_r_inner, teflon_r_outer, teflon_h/2, 0.*deg, 360.*deg);
        G4LogicalVolume* teflonLogVol = new G4LogicalVolume(teflonVol, mat_PTFE, "teflonVol");

        new G4PVPlacement(new G4RotationMatrix(), G4ThreeVector(0., 0., 0.), teflonLogVol, "PTFE", worldLogVol, false, 0.);

        // Making ICV Stainless Steel Can
        G4double icv_r_inner = 4*cm;
        G4double icv_thickness = 1.5*cm;
        G4double icv_r_outer = icv_r_inner+icv_thickness;
        G4double icv_h = (7.5*2.54)*cm;

        G4Tubs* icvRingVol = new G4Tubs("icvVol", icv_r_inner, icv_r_outer, icv_h/2, 0.*deg, 360.*deg);     
        G4LogicalVolume* icvLogVol = new G4LogicalVolume(icvRingVol, mat_SS, "icvVol");

        new G4PVPlacement(new G4RotationMatrix(), G4ThreeVector(0., 0., 0.), icvLogVol, "ICV", worldLogVol, false, 0.);


        // Making OCV Stainless Steel Can
        G4double ocv_r_inner = (9.75*2.54)*cm;
        G4double ocv_thickness = 1.5*cm;
        G4double ocv_r_outer = ocv_r_inner+ocv_thickness;
        G4double ocv_h = (20.5*2.54)*cm;
        G4double ocv_offset = (11.5*2.54)*cm;

        G4Tubs* ocvRingVol = new G4Tubs("ocvVol", ocv_r_inner, ocv_r_outer, ocv_h/2, 0.*deg, 360.*deg);
        G4LogicalVolume* ocvLogVol = new G4LogicalVolume(ocvRingVol, mat_SS, "ocvVol");

        new G4PVPlacement(new G4RotationMatrix(), G4ThreeVector(0., 0., 0.), ocvLogVol, "OCV", worldLogVol, false, 0.);

        // Build Array of cylndrical EJ301 detectors
        // if neutron beam goes in +X, place detectors 1 m from TPC center (origin) at +/- 45, 90, 135, 180 degrees
        G4double liquidScint_r = 5*cm;
        G4double liquidScint_h = 5*cm;

        G4Tubs* liqScintVol = new G4Tubs("liqScintVol", 0., liquidScint_r, liquidScint_h/2, 0.*deg, 360.*deg);
        G4LogicalVolume* liquidScintVol = new G4LogicalVolume(liqScintVol, mat_EJ301, "liquidScintVol");
        
        G4double liqScint_offset = 100*cm;

        // 0 degrees
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateY(90*deg)), G4ThreeVector(liqScint_offset, 0., 0.), liquidScintVol, "LiqScintDet0deg", worldLogVol, false, 0.);
        
        // 15 degrees
        double s1 = (sqrt(3)-1)/(2*sqrt(2));
        double l1 = (sqrt(3)+1)/(2*sqrt(2));
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(90*deg).rotateY(-75*deg)), G4ThreeVector(liqScint_offset*l1, liqScint_offset*s1, 0.), liquidScintVol, "LiqScintDet15deg", worldLogVol, false, 0.);
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(90*deg).rotateY(75*deg)), G4ThreeVector(liqScint_offset*l1, -liqScint_offset*s1, 0.), liquidScintVol, "LiqScintDet15deg", worldLogVol, false, 1.);

        // 30 degrees
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(90*deg).rotateY(-60*deg)), G4ThreeVector(liqScint_offset*sqrt(3)/2, liqScint_offset/2, 0.), liquidScintVol, "LiqScintDet30deg", worldLogVol, false, 0.);
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(90*deg).rotateY(60*deg)), G4ThreeVector(liqScint_offset*sqrt(3)/2, -liqScint_offset/2, 0.), liquidScintVol, "LiqScintDet30deg", worldLogVol, false, 1.);
       
        // 45 degrees
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(90*deg).rotateY(-45*deg)), G4ThreeVector(liqScint_offset/sqrt(2), liqScint_offset/sqrt(2), 0.), liquidScintVol, "LiqScintDet45deg", worldLogVol, false, 0.);
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(90*deg).rotateY(45*deg)), G4ThreeVector(liqScint_offset/sqrt(2), -liqScint_offset/sqrt(2), 0.), liquidScintVol, "LiqScintDet45deg", worldLogVol, false, 1.);
        
        // 60 degrees
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(90*deg).rotateY(-30*deg)), G4ThreeVector(liqScint_offset/2, liqScint_offset*sqrt(3)/2, 0.), liquidScintVol, "LiqScintDet60deg", worldLogVol, false, 0.);
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(90*deg).rotateY(30*deg)), G4ThreeVector(liqScint_offset/2, -liqScint_offset*sqrt(3)/2, 0.), liquidScintVol, "LiqScintDet60deg", worldLogVol, false, 1.);

        // 75 degrees
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(90*deg).rotateY(-15*deg)), G4ThreeVector(liqScint_offset*s1, liqScint_offset*l1, 0.), liquidScintVol, "LiqScintDet75deg", worldLogVol, false, 0.);
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(90*deg).rotateY(15*deg)), G4ThreeVector(liqScint_offset*s1, -liqScint_offset*l1, 0.), liquidScintVol, "LiqScintDet75deg", worldLogVol, false, 1.);

        // 90 degrees
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(90*deg)), G4ThreeVector(0., liqScint_offset, 0.), liquidScintVol, "LiqScintDet90deg", worldLogVol, false, 0.);
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(90*deg)), G4ThreeVector(0., -liqScint_offset, 0.), liquidScintVol, "LiqScintDet90deg", worldLogVol, false, 1.);

        // 105 degrees
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(90*deg).rotateY(15*deg)), G4ThreeVector(-liqScint_offset*s1, liqScint_offset*l1, 0.), liquidScintVol, "LiqScintDet105deg", worldLogVol, false, 0.);
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(90*deg).rotateY(-15*deg)), G4ThreeVector(-liqScint_offset*s1, -liqScint_offset*l1, 0.), liquidScintVol, "LiqScintDet105deg", worldLogVol, false, 1.);

        // 120 degrees
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(90*deg).rotateY(30*deg)), G4ThreeVector(-liqScint_offset/2, liqScint_offset*sqrt(3)/2, 0.), liquidScintVol, "LiqScintDet120deg", worldLogVol, false, 0.);
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(90*deg).rotateY(-30*deg)), G4ThreeVector(-liqScint_offset/2, -liqScint_offset*sqrt(3)/2, 0.), liquidScintVol, "LiqScintDet120deg", worldLogVol, false, 1.);

        // 135 degrees
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(-90*deg).rotateY(45*deg)), G4ThreeVector(-liqScint_offset/sqrt(2), -liqScint_offset/sqrt(2), 0.), liquidScintVol, "LiqScintDet135deg", worldLogVol, false, 0.);
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(-90*deg).rotateY(-45*deg)), G4ThreeVector(-liqScint_offset/sqrt(2), liqScint_offset/sqrt(2), 0.), liquidScintVol, "LiqScintDet135deg", worldLogVol, false, 1.);
        
        // 150 degrees
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(90*deg).rotateY(60*deg)), G4ThreeVector(-liqScint_offset*sqrt(3)/2, liqScint_offset/2, 0.), liquidScintVol, "LiqScintDet150deg", worldLogVol, false, 0.);
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(90*deg).rotateY(-60*deg)), G4ThreeVector(-liqScint_offset*sqrt(3)/2, -liqScint_offset/2, 0.), liquidScintVol, "LiqScintDet150deg", worldLogVol, false, 1.);

        // 165 degrees
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(90*deg).rotateY(75*deg)), G4ThreeVector(-liqScint_offset*l1, liqScint_offset*s1, 0.), liquidScintVol, "LiqScintDet165deg", worldLogVol, false, 0.);
        new G4PVPlacement(new G4RotationMatrix(G4RotationMatrix().rotateX(90*deg).rotateY(-75*deg)), G4ThreeVector(-liqScint_offset*l1, -liqScint_offset*s1, 0.), liquidScintVol, "LiqScintDet165deg", worldLogVol, false, 1.);


    }

    return worldPhysVol;

}

// Get Detector parameters from Messenger
void MyDetectorConstruction::SetWhichGeometry(std::string whichGeom){
    if (whichGeom == "HydroX"){
        fWhichGeometry = whichGeom;
    }
    else{
        fWhichGeometry = whichGeom;
    }
}
void MyDetectorConstruction::SetWhichDopant(std::string whichDopant){
    if (whichDopant == "He" || whichDopant == "helium"){
        fWhichDopant = "He";
    }
    else if (whichDopant == "H2" || whichDopant == "hydrogen"){
        fWhichDopant = "H2";
    }
    else {
        fWhichDopant = "None";
    }
}

void MyDetectorConstruction::SetDopantFraction(double dopantFrac){
    if (dopantFrac < 0){
        fDopantFraction = 0.;
    }
    if (dopantFrac > 1.){
        fDopantFraction = 1.;
    }
    else{
        fDopantFraction = dopantFrac;
    }
}