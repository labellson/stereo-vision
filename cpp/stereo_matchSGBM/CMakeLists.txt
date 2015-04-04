project( StereoMatchSGBM )
find_package( OpenCV REQUIRED )
add_executable( stereo_matchSGBM stereo_matchSGBM.cpp )
target_link_libraries( stereo_matchSGBM ${OpenCV_LIBS} SCalibData )
