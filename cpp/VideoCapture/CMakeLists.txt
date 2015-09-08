project( VideoCapture )
find_package( OpenCV REQUIRED )
add_executable( VideoCapture VideoCapture.cpp )
target_link_libraries( VideoCapture ${OpenCV_LIBS} )
