# compile objects
add_library(bp_par_2d OBJECT bp_par_2d.cc)
add_library(bp_par_2d_sv OBJECT bp_par_2d_sv.cc)
add_library(fp_par_2d OBJECT fp_par_2d.cc)
add_library(filter OBJECT filter.cc)

# to make the final line shorter
set (TARGET1 $<TARGET_OBJECTS:bp_par_2d>)
set (TARGET2 $<TARGET_OBJECTS:bp_par_2d_sv>)
set (TARGET3 $<TARGET_OBJECTS:fp_par_2d>)
set (TARGET4 $<TARGET_OBJECTS:filter>)

# push objects to parent
set (SRC_TEMP ${TARGET1} ${TARGET2} ${TARGET3} ${TARGET4} PARENT_SCOPE)