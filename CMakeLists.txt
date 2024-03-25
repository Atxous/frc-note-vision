cmake_minimum_required(VERSION 3.0.0)
project(frc-note-vision VERSION 0.1.0)

# set the directory for opencv bin

set(OpenCV_DIR "C:/Users/jl255/Desktop/dependencies/opencv/build/bin")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCES filterByEllipseColor.cpp thresholding.cpp)  # Specify the extension .cpp for color_constancy2

add_executable(frc-note-vision ${SOURCES})

target_link_libraries(frc-note-vision ${OpenCV_LIBS})

# Add the directory containing calculations.h to the include directories
target_include_directories(frc-note-vision PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
