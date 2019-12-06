###
 # @Description: compile tensorflow operators
 # This file was writen according to https://github.com/PatWie/tensorflow-cmake/blob/master/cmake/modules/FindTensorFlow.cmake
 # Much thanks to the contributers. 
 # @Author: Tianling Lyu
 # @Date: 2019-12-04 15:36:50
 # @LastEditors: Tianling Lyu
 # @LastEditTime: 2019-12-05 17:48:32
###

# check system
if(APPLE)
  message(WARNING "This CMakeLists.txt is not tested on APPLE")
endif()

if(WIN32)
  message(FATAL_ERROR "This CMakeLists.txt does not support windows.")
endif()

set(PYTHON_EXECUTABLE "python" CACHE STRING "specify the python version TensorFlow is installed on.")

# tensorflow information
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import tensorflow as tf; print(tf.__version__); print(tf.sysconfig.get_include()); print(tf.sysconfig.get_lib());"
    OUTPUT_VARIABLE TF_INFORMATION_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE retcode)

if ("${retcode}" STREQUAL "0")
    message (STATUS "Detecting Tensorflow info - done")
else ()
    message (FATAL_ERROR "Detecting TensorFlow info - failed  \n Did you installed TensorFlow?")
endif ()

string (REPLACE "\n" ";" TF_INFORMATION_LIST ${TF_INFORMATION_STRING})
list (GET TF_INFORMATION_LIST 0 VER)
list (GET TF_INFORMATION_LIST 1 INC)
list (GET TF_INFORMATION_LIST 2 LIB)

# version dependent settings
if ("${VER}" VERSION_LESS "1.13")
    message (FATAL_ERROR "Tensorflow version ${VER} is too old. Only versions after 1.13 are supported now.")
endif ()

set (TF_LIB ${LIB}/libtensorflow_framework.so PARENT_SCOPE)

if ("${VER}" VERSION_GREATER "1.14" OR "${VER}" VERSION_EQUAL "1.14")
    if (NOT EXISTS ${INC}/third_party/gpus/cuda)
        include (FindCUDA)
        file (MAKE_DIRECTORY ${INC}/third_party/gpus/cuda)
        execute_process (COMMAND "ln" -s ${CUDA_INCLUDE_DIRS} ${INC}/third_party/gpus/cuda/include)
    endif ()
    add_definitions (-DTFVER_1_14)
    set (TF_LIB ${LIB}/libtensorflow_framework.so.1 PARENT_SCOPE)
endif ()

if ("${VER}" VERSION_GREATER "1.15" OR "${VER}" VERSION_EQUAL "1.15")
    add_definitions (-D_GLIBCXX_USE_CXX11_ABI=0)
else ()
    add_definitions (-D_GLIBCXX_USE_CXX11_ABI=1)
endif ()

if ("${VER}" VERSION_GREATER "2.0.0a0" OR "${VER}" VERSION_EQUAL "2.0.0a0")
    set (TF_LIB ${LIB}/libtensorflow_framework.so.2 PARENT_SCOPE)
endif ()

# flags
if (USE_CUDA)
    add_definitions (-DGOOGLE_CUDA)
    add_definitions (-DNDEBUG)
endif ()
include_directories (/usr/local ${INC})
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w --expt-relaxed-constexpr -O3")

# compile objects
add_library (bp_par_2d_ops OBJECT bp_par_2d_ops.cu)
add_library (bp_par_2d_sv_ops OBJECT bp_par_2d_sv_ops.cu)
add_library (fp_par_2d_ops OBJECT fp_par_2d_ops.cu)
add_library (ramp_filter_ops OBJECT ramp_filter_ops.cu)

# to make the final line shorter
set (TARGET1 $<TARGET_OBJECTS:bp_par_2d_ops>)
set (TARGET2 $<TARGET_OBJECTS:bp_par_2d_sv_ops>)
set (TARGET3 $<TARGET_OBJECTS:fp_par_2d_ops>)
set (TARGET4 $<TARGET_OBJECTS:ramp_filter_ops>)

# push objects to parent
set (TF_TEMP ${TARGET1} ${TARGET2} ${TARGET3} ${TARGET4} PARENT_SCOPE)