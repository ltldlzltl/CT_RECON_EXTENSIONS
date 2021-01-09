# use nvcc
include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

# compile objects
add_library(bp_fan_2d_angle_npext OBJECT bp_fan_2d_angle_npext.cu)

# to make the final line shorter
set (TARGET1 $<TARGET_OBJECTS:bp_fan_2d_angle_npext>)

# push objects to parent
set (NPEXT_TEMP ${TARGET1}  PARENT_SCOPE)