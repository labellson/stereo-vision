cmake_minimum_required(VERSION 2.8.11)
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set( CMAKE_BUILD_TYPE DEBUG )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x" )# --sysroot=${CMAKE_FIND_ROOT_PATH}" )
if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES arm*)
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lpthread" )
endif()
#link_directories(
#    $ENV{HOME}/Rasp/root/lib/arm-linux-gnueabihf
#    $ENV{HOME}/Rasp/root/lib
#    $ENV{HOME}/Rasp/root/usr/lib/arm-linux-gnueabihf
#    $ENV{HOME}/Rasp/root/usr/lib
#    $ENV{HOME}/Rasp/root/usr/local/lib
#)   
find_package(OpenCV)

add_subdirectory (calibrateStereo)
if(OPENCV_NONFREE_FOUND)
    add_subdirectory (SurfObjectDetector)
endif()
add_subdirectory (stereo_matchSGBM)
add_subdirectory (stereo_matchBM)
add_subdirectory (stereo_matchBM_threads)
add_subdirectory (lib)
add_subdirectory (VideoCapture)
