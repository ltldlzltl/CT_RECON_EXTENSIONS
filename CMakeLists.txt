###
 # @Description: 
 # @Author: Tianling Lyu
 # @Date: 2019-12-04 11:36:05
 # @LastEditors: Tianling Lyu
 # @LastEditTime: 2019-12-05 17:36:59
###
cmake_minimum_required (VERSION 3.8)
project (CT_RECON_EXTENSIONS LANGUAGES CXX CUDA)
# common settings
set (CMAKE_CXX_COMPILER "g++")
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CUDA_STANDARD 11)
set (CMAKE_CUDA_STANDARD_REQUIRED ON)
set (CMAKE_CUDA_ARCHITECTURES 61 75)
set (CMAKE_CXX_EXTENSIONS OFF)
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/temp)
set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set (CMAKE_POSITION_INDEPENDENT_CODE ON)
set (CMAKE_CXX_FLAGS "-Wall -Wextra")
set (CMAKE_CXX_FLAGS_RELEASE "-O3")
# user settings
option (USE_CUDA "Do you want to build GPU code?" ON)
option (USE_TF "Do you want to build Tensorflow extension library?" ON)
option (USE_NP "Do you want to build numpy extension library?" ON)
option (DEBUG "Debug mode?" OFF)
if (DEBUG)
    set (CMAKE_BUILD_TYPE Debug)
else ()
    set (CMAKE_BUILD_TYPE Release)
endif()
# include path
include_directories (${PROJECT_SOURCE_DIR})
# sub-directories
if (USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    add_definitions("-DUSE_CUDA")
    add_subdirectory("cuda")
endif ()
add_subdirectory("src")
set (TEMP ${SRC_TEMP})
if (USE_CUDA)
    set(TEMP ${TEMP} ${CUDA_TEMP})
endif ()

# build shared library
add_library(ctext SHARED ${TEMP})
# Tensorflow extension
if (USE_TF)
    add_subdirectory("tensorflow")
    set (TF_TEMP ${TEMP} ${TF_TEMP})
    add_library (tensorflow_ctext SHARED ${TF_TEMP})
    target_link_libraries (tensorflow_ctext ${TF_LIB})
endif ()
# numpy extension
if (USE_NP)
    add_subdirectory("numpy_ext")
    set (NP_TEMP ${TEMP} ${NPEXT_TEMP})
    add_library (numpy_ctext SHARED ${NP_TEMP})
endif()