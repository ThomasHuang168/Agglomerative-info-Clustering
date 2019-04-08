cmake_minimum_required(VERSION 2.8)

project( AIC )
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONs OFF)

include_directories(.)
find_package( OpenCV REQUIRED )
add_executable( AIC cart_demo.cpp )
target_link_libraries( AIC ${OpenCV_LIBS} )
