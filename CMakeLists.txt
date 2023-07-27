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
# OpenMP
FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3 -march=native -ggdb3")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb3 -ASAN_OPTIONS=alloc_dealloc_mismatch=0")

add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)

