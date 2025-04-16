cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
# cmake_minimum_required(VERSION 3.16...3.21)
project(Geant4Simulation)

option(WITH_GEANT4_UIVIS "Build example with Geant4 UI and Vis drivers" ON)
if(WITH_GEANT4_UIVIS)
  find_package(Geant4 REQUIRED ui_all vis_all)
else()
  find_package(Geant4 REQUIRED)
endif()

find_package(ROOT REQUIRED)

include(${Geant4_USE_FILE})
include_directories(${PROJECT_SOURCE_DIR}/include ${ROOT_INCLUDE_DIRS})
add_definitions(${ROOT_CXX_FLAGS})

file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)

add_executable(Geant4Simulation main.cc ${sources} ${headers})
target_link_libraries(Geant4Simulation ${Geant4_LIBRARIES} ${ROOT_LIBRARIES})
