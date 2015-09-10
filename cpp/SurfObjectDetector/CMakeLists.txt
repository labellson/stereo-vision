project( SurfObjectDetector )
find_package( OpenCV REQUIRED )
add_executable( surf_object_detector surf_object_detector.cpp )
target_link_libraries( surf_object_detector ${OpenCV_LIBS} )
