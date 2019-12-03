### 
# @Description: bash script to compile the op
 # @Author: Tianling Lyu
 # @Date: 2019-11-21 17:26:11
 # @LastEditors: Tianling Lyu
 # @LastEditTime: 2019-12-03 10:47:39
 ###

# some useful symbols
export TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
export TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

# compile CUDA related code
nvcc -c -std=c++11 -o temp/fp_par_2d.cu.o cuda/fp_par_2d.cu -I. \
    -I/usr/local/cuda/include -x cu -Xcompiler -fPIC \
    --expt-relaxed-constexpr -O3
nvcc -c -std=c++11 -o temp/fp_par_2d_ops.o \
    tensorflow/fp_par_2d_ops.cc -I/usr/local/cuda/include \
    -I/usr/local -I$TF_INC -I. -x cu -Xcompiler -fPIC \
    --expt-relaxed-constexpr -expt-relaxed-constexpr -O3 \
    -D GOOGLE_CUDA -DNDEBUG
nvcc -c -std=c++11 -o temp/bp_par_2d.cu.o cuda/bp_par_2d.cu -I. \
    -I/usr/local/cuda/include -x cu -Xcompiler -fPIC \
    --expt-relaxed-constexpr -O3
nvcc -c -std=c++11 -o temp/bp_par_2d_ops.o \
    tensorflow/bp_par_2d_ops.cc -I/usr/local/cuda/include \
    -I/usr/local -I$TF_INC -I. -x cu -Xcompiler -fPIC \
    --expt-relaxed-constexpr -expt-relaxed-constexpr -O3 \
    -D GOOGLE_CUDA -DNDEBUG
nvcc -c -std=c++11 -o temp/filter.cu.o cuda/filter.cu -I. \
    -I/usr/local/cuda/include -x cu -Xcompiler -fPIC \
    --expt-relaxed-constexpr -O3
nvcc -c -std=c++11 -o temp/ramp_filter_ops.o \
    tensorflow/ramp_filter_ops.cc -I/usr/local/cuda/include \
    -I/usr/local -I$TF_INC -I. -x cu -Xcompiler -fPIC \
    --expt-relaxed-constexpr -expt-relaxed-constexpr -O3 \
    -D GOOGLE_CUDA -DNDEBUG

# compile c++ code
g++ -c -std=c++11 -o temp/fp_par_2d.o src/fp_par_2d.cc -I. -I/usr/local/cuda/include -fPIC -O3
g++ -c -std=c++11 -o temp/bp_par_2d.o src/bp_par_2d.cc -I. -I/usr/local/cuda/include -fPIC -O3

# link all .o into .so
g++ -shared -std=c++11 -o lib/libctrecon.so temp/bp_par_2d_ops.o \
    temp/bp_par_2d.o temp/bp_par_2d.cu.o temp/fp_par_2d_ops.o \
    temp/fp_par_2d.o temp/fp_par_2d.cu.o temp/ramp_filter_ops.o \
    temp/filter.o temp/filter.cu.o -L$TF_LIB -ltensorflow_framework \
    -L /usr/local/cuda/lib64 -lcudart -fPIC -O3
