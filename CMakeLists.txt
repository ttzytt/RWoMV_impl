cmake_minimum_required (VERSION 3.2)
project (RWoMV_impl)
set (CMAKE_CXX_STANDARD 20)

set(eigen3_dir "/usr/include/eigen3")
find_package(Eigen3 REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/ext
    $ENV{eigen3_dir}
)

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/ext/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/ext/*.h
)


add_executable(RWoMV_impl ${SOURCES})
target_link_libraries(RWoMV_impl Eigen3::Eigen)
