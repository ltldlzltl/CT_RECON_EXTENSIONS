# compile objects
add_library(bp_par_2d OBJECT bp_par_2d.cc)
add_library(bp_par_2d_sv OBJECT bp_par_2d_sv.cc)
add_library(fp_par_2d OBJECT fp_par_2d.cc)
add_library(filter OBJECT filter.cc)
add_library(fan_weighting OBJECT fan_weighting.cc)
add_library(bp_fan_2d OBJECT bp_fan_2d.cc)
add_library(bp_fan_2d_angle OBJECT bp_fan_2d_angle.cc)
add_library(helical_interp OBJECT helical_interpolation.cc)

# to make the final line shorter
set (TARGET1 $<TARGET_OBJECTS:bp_par_2d>)
set (TARGET2 $<TARGET_OBJECTS:bp_par_2d_sv>)
set (TARGET3 $<TARGET_OBJECTS:fp_par_2d>)
set (TARGET4 $<TARGET_OBJECTS:filter>)
set (TARGET5 $<TARGET_OBJECTS:fan_weighting>)
set (TARGET6 $<TARGET_OBJECTS:bp_fan_2d>)
set (TARGET7 $<TARGET_OBJECTS:bp_fan_2d_angle>)
set (TARGET8 $<TARGET_OBJECTS:helical_interp>)

# push objects to parent
set (SRC_TEMP ${TARGET1} ${TARGET2} ${TARGET3} ${TARGET4} 
    ${TARGET5} ${TARGET6} ${TARGET7} ${TARGET8} PARENT_SCOPE)