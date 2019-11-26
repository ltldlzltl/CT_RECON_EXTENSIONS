<!--
 * @Description: develop log describing problems and solves
 * @Author: Tianling Lyu
 * @Date: 2019-11-21 11:36:02
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-11-21 14:25:15
 -->

# 11/21/19
## 1. Eigen::GpuDevice incomplete type when compiling fp_par_2d_ops.cc
This happens when implementing member function of GpuDevice specializations in .h file. It seems that those functions must be compiled using nvcc not g++. 

Solve: Implement those functions in a .cu file and leave only the declarations in .h file. Compile the .cu file with nvcc

## 2. error: constexpr function return is non-constant in absl/strings/string_view.h when compiling fp_par_2d_ops.cu.cc with nvcc
see: https://github.com/tensorflow/tensorflow/issues/22766 . It happened when I was using tensorflow==1.13.1. Seems to be a bug of Tensorflow. 

Solve: Add -DNDEBUG flag to the compile instruction. Note that this flag will disable all assertations. Some guys said that it would be better to use flag -Dconstexpr= instead of -DNDEBUG, in my experiment, however, it introduces a lot of new problems. It's said that this bug was fixed in later versions of Tensorflow. 

## 3. error: cuda_kernel_helper.h: No such file or directory in tensorflow-gpu==1.14.0
Tensorflow deprecated cuda_kernel_helper.h and move the contents to gpu_kernel_helper.h in versino 1.14.0.

Solve: change "cuda_kernel_helper.h" to "gpu_kernel_helper.h"

## 4. error: third_party/gpus/cuda/include/cuda_fp16.h: No such file or directory in tensorflow-gpu==1.14.0
see https://github.com/tensorflow/tensorflow/issues/31912 . The cuda_fp16.h file can be found in /usr/local/cuda/include, but it seems that it not in the tensorflow folder. This problem also happens in version 1.15.0. 

Solve: not solved yet. Some guys said it might work to do some copies and pastes but I wonder. I just roll back to 1.13.1. Now the problem is the code is not suitable for all versions of Tensorflow. 

## 5.  error: no matching function for call to ‘GetNodeAttr(const tensorflow::NodeDef&, tensorflow::StringPiece&, double*&)’ return GetNodeAttr(def(), attr_name, value); when compiling fp_par_2d_ops.cc
It seems that tensorflow does not support using double type directly in operator attributes. Float, int, bool and string are definitely supported. There is a overloading version with tensorflow::DataType but I'm not sure how to use that now. 

Solve: modify all "double" to "float" and all "vector\<double>" to "vector\<float>"

## 6. undefined symbol: _ZN10tensorflow12OpDefBuilder4AttrESs when loading module in python
see https://github.com/google/sentencepiece/issues/293 . No idea why it happens. 

Solve: set -D_GLIBCXX_USE_CXX11_ABI=1 when compiling