# use nvcc
include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

# compile objects
add_library(common OBJECT common.cu)
add_library(bp_fan_2d_angle_npext OBJECT bp_fan_2d_angle_npext.cu)
add_library(filter_npext OBJECT filter_npext.cu)
add_library(fan_weighting_npext OBJECT fan_weighting_npext.cu)

# to make the final line shorter
set (TARGET1 $<TARGET_OBJECTS:common>)
set (TARGET2 $<TARGET_OBJECTS:bp_fan_2d_angle_npext>)
set (TARGET3 $<TARGET_OBJECTS:filter_npext>)
set (TARGET4 $<TARGET_OBJECTS:fan_weighting_npext>)

# push objects to parent
set (NPEXT_TEMP ${TARGET1} ${TARGET2} ${TARGET3} ${TARGET4} PARENT_SCOPE)