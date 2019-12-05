# tensorflow paths
execute_process (COMMAND "python" "-c" "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_include())" OUTPUT_VARIABLE INC)
execute_process (COMMAND "python" "-c" "import tensorflow as tf; import sys; sys.stdout.write(tf.sysconfig.get_lib())" OUTPUT_VARIABLE LIB)

# flags
if (USE_CUDA)
    add_definitions (-DGOOGLE_CUDA)
    add_definitions (-DNDEBUG)
endif ()
include_directories (/usr/local ${INC})
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -O3")

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
set (TF_LIB ${LIB} PARENT_SCOPE)
set (SRC_TEMP ${TARGET1} ${TARGET2} ${TARGET3} ${TARGET4} PARENT_SCOPE)