### 
# @Description: bash script to compile the op
 # @Author: Tianling Lyu
 # @Date: 2019-11-21 17:26:11
 # @LastEditors: Tianling Lyu
 # @LastEditTime: 2019-11-30 11:11:34
 ###

# some useful symbols
export TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
export TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

# compile CUDA code
nvcc -c -std=c++11 -o temp/fp_par_2d.cu.o cuda/fp_par_2d.cu -I. -I/usr/local/cuda/include -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -O3
nvcc -c -std=c++11 -o temp/fp_par_2d_ops.cu.o tensorflow/fp_par_2d_ops.cu.cc -I/usr/local/cuda/include -I/usr/local -I$TF_INC -I. -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -O3 -D GOOGLE_CUDA -DNDEBUG
nvcc -c -std=c++11 -o temp/bp_par_2d.cu.o cuda/bp_par_2d.cu -I. -I/usr/local/cuda/include -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -O3
nvcc -c -std=c++11 -o temp/bp_par_2d_ops.cu.o tensorflow/fp_par_2d_ops.cu.cc -I/usr/local/cuda/include -I/usr/local -I$TF_INC -I. -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -O3 -D GOOGLE_CUDA -DNDEBUG

# compile c++ code
g++ -c -std=c++11 -o temp/fp_par_2d.o src/fp_par_2d.cc -I. -I/usr/local/cuda/include -fPIC -O3
g++ -c -std=c++11 -o temp/bp_par_2d.o src/bp_par_2d.cc -I. -I/usr/local/cuda/include -fPIC -O3
g++ -std=c++11 -shared -o lib/libctrecon.so \
    tensorflow/fp_par_2d_ops.cc tensorflow/bp_par_2d_ops.cc \
    temp/fp_par_2d.o temp/fp_par_2d.cu.o temp/fp_par_2d_ops.cu.o \
    temp/bp_par_2d.o temp/bp_par_2d.cu.o temp/fp_par_2d_ops.cu.o \
    -I. -I $TF_INC -D GOOGLE_CUDA \
    -L $TF_LIB -ltensorflow_framework -L /usr/local/cuda/lib64 -lcudart \
    -fPIC -D_GLIBCXX_USE_CXX11_ABI=1 -O3
