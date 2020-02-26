# use nvcc
include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

# compile objects
add_library(bp_par_2d_cu OBJECT bp_par_2d.cu)
add_library(bp_par_2d_sv_cu OBJECT bp_par_2d_sv.cu)
add_library(fp_par_2d_cu OBJECT fp_par_2d.cu)
add_library(filter_cu OBJECT filter.cu)

# to make the final line shorter
set (TARGET1 $<TARGET_OBJECTS:bp_par_2d_cu>)
set (TARGET2 $<TARGET_OBJECTS:bp_par_2d_sv_cu>)
set (TARGET3 $<TARGET_OBJECTS:fp_par_2d_cu>)
set (TARGET4 $<TARGET_OBJECTS:filter_cu>)

# push objects to parent
set (CUDA_TEMP ${TARGET1} ${TARGET2} ${TARGET3} ${TARGET4} PARENT_SCOPE)