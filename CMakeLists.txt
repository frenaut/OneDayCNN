cmake_minimum_required(VERSION 2.8.3)
project(one_day_cnn)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(Eigen3 REQUIRED)
find_package(PNG REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR} ${PNG_INCLUDE_DIR})

file(GLOB_RECURSE project_SOURCES "src/*.cpp")
file(GLOB_RECURSE project_HEADERS "include/*.hpp")

set (project_INCLUDE_DIRS "")
foreach (_headerFile ${project_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list (APPEND project_INCLUDE_DIRS ${_dir})
endforeach()
list(REMOVE_DUPLICATES project_INCLUDE_DIRS)

include_directories(${project_INCLUDE_DIRS})
add_executable (cnn_test ${project_SOURCES})
target_link_libraries(cnn_test ${PNG_LIBRARY})