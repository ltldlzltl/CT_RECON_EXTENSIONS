<!--
 * @Description: develop log describing problems and solves
 * @Author: Tianling Lyu
 * @Date: 2019-11-21 11:36:02
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-12-05 22:10:01
 -->

# 11/21/19
## 1. Eigen::GpuDevice incomplete type when compiling fp_par_2d_ops.cc
This happens when implementing member function of GpuDevice specializations in .h file. It seems that those functions must be compiled using nvcc not g++. 

**Solve:** Implement those functions in a .cu file and leave only the declarations in .h file. Compile the .cu file with nvcc

## 2. error: constexpr function return is non-constant in absl/strings/string_view.h when compiling fp_par_2d_ops.cu.cc with nvcc
see: https://github.com/tensorflow/tensorflow/issues/22766 . It happened when I was using tensorflow==1.13.1. Seems to be a bug of Tensorflow. 

**Solve:** Add -DNDEBUG flag to the compile instruction. Note that this flag will disable all assertations. Some guys said that it would be better to use flag -Dconstexpr= instead of -DNDEBUG, in my experiment, however, it introduces a lot of new problems. It's said that this bug was fixed in later versions of Tensorflow. 

## 3. error: cuda_kernel_helper.h: No such file or directory in tensorflow-gpu==1.14.0
Tensorflow deprecated cuda_kernel_helper.h and move the contents to gpu_kernel_helper.h in versino 1.14.0.

**Solve:** change "cuda_kernel_helper.h" to "gpu_kernel_helper.h"

## 4. error: third_party/gpus/cuda/include/cuda_fp16.h: No such file or directory in tensorflow-gpu==1.14.0
see https://github.com/tensorflow/tensorflow/issues/31912 . The cuda_fp16.h file can be found in /usr/local/cuda/include, but it seems that it not in the tensorflow folder. This problem also happens in version 1.15.0. 

**Solve:** not solved yet. Some guys said it might work to do some copies and pastes but I wonder. I just roll back to 1.13.1. Now the problem is the code is not suitable for all versions of Tensorflow. 

## 5.  error: no matching function for call to ‘GetNodeAttr(const tensorflow::NodeDef&, tensorflow::StringPiece&, double*&)’ return GetNodeAttr(def(), attr_name, value); when compiling fp_par_2d_ops.cc
It seems that tensorflow does not support using double type directly in operator attributes. Float, int, bool and string are definitely supported. There is a overloading version with tensorflow::DataType but I'm not sure how to use that now. 

**Solve:** modify all "double" to "float" and all "vector\<double>" to "vector\<float>"

## 6. undefined symbol: _ZN10tensorflow12OpDefBuilder4AttrESs when loading module in python
see https://github.com/google/sentencepiece/issues/293 . No idea why it happens. 

**Solve:** set -D_GLIBCXX_USE_CXX11_ABI=1 when compiling

# 12/02/19

## 1. passing '...' as ‘this’ argument discards qualifiers when compiling ramp_filter_ops.cc
forgot to add 'const' identifier over functions in ramp_filter.h

**Solve:** add 'const' to those functions

## 2. Eigen::GpuDevice incomplete type when compiling ramp_filter_ops.cu.cc
It happened again even I used nvcc to compile the file. It seems that the problem is I have to define `EIGEN_USE_GPU` before including "tensorflow/core/util/cuda_kernel_helper.h". This problem does not happen in other .cu.cc files because `EIGEN_USE_GPU` has already been defined in their corresponding .h files.

**Solve:** move `#define EIGEN_USE_GPU` to the front of `#include "tensorflow/core/util/cuda_kernel_helper.h"`

## 3. the calculation results of `int n_elements = param_.na * MAX(param_.nx, param_.ny);` are wrong
It happens due to the definition of `MAX(x, y)`. I defined it as `#define MAX(x, y) (x>y) ? x : y` in the original version, which was translated into `int n_elements = param.na*(param.nx>param.ny) ? param.nx : param.ny;` while compiling. 

**Solve:** be more careful on the macros, use `#define MAX(x, y) ((x) > (y) ? (x) : (y))` instead.

# 12/03/19

## 1. rethinking 11/21/19-1 and 12/02/19-2
Maybe the functions do not need to be compiled with nvcc, the problem happens simply because `EIGEN_USE_GPU` is not defined before `cuda_kernel_helper.h`. I am going to test that. 
### TEST1
**INSTRUCTION:**
```bash
g++ -std=c++11 -shared -o lib/libbp_par_2d.so tensorflow bp_par_2d_ops.cc temp/bp_par_2d.o temp/bp_par_2d.cu.o -I. -I$TF_INC -D GOOGLE_CUDA -L$TF_LIB -ltensorflow_framework -L/usr/local/cuda/lib64 -lcudart -fPIC -D_GLIBCXX_USE_CXX11_ABI=1 -O3
```
**RESULT:** Failed to compile. Cannot find the definition of `blockIdx` in many related .h files. It seems that the file still need to be compiled with nvcc. I am going to see what will happen if I use nvcc to replace g++ here. 
### TEST2
**INSTRUCTION:**
```bash
nvcc -std=c++11 -shared -o lib/libbp_par_2d.so tensorflow/bp_par_2d_ops.cc temp/bp_par_2d.o temp/bp_par_2d.cu.o -I. -I$TF_INC -I/usr/local -D GOOGLE_CUDA -L$TF_LIB -ltensorflow_framework -L/usr/local/cuda/lib64 -lcudart -x cu -Xcompiler -fPIC -expt-relaxed-constexpr -DNDEBUG -D_GLIBCXX_USE_CXX11_ABI=1 -O3
```
Added some useful flags to the instruction like '-expt-relaxed-constexpr' and '-DNDEBUG', which would otherwise lead to problems in compiling. 

**RESULT:** Some strange errors in linking, e.g. 'temp/bp_par_2d.o(1): error: unrecognized token'. Those errors still occur after I recompile bp_par_2d.o with nvcc. No idea why it happens. Going to see what if I compile .o with nvcc and link those .o files with g++. 
### TEST3
**INSTRUCTION:**
```bash
nvcc -std=c++11 -c -o temp/bp_par_2d_ops.o tensorflow/bp_par_2d_ops.cc -I. -I$TF_INC -I/usr/local -D GOOGLE_CUDA -x cu -Xcompiler -fPIC -expt-relaxed-constexpr -DNDEBUG -D_GLIBCXX_USE_CXX11_ABI=1 -O3
g++ -shared -std=c++11 -o lib/libbp_par_2d.so temp/bp_par_2d_ops.o temp/bp_par_2d.o temp/bp_par_2d.cu.o -L$TF_LIB -ltensorflow_framework -L /usr/local/cuda/lib64 -lcudart -fPIC -O3
```
**RESULT:** It seems to work. *BE CAREFUL, DO NOT INCLUDE HEADERS INSIDE NAMESPACES.*

# 12/05/19
## 1. Rethinking 11/21/19-4 error: third_party/gpus/cuda/include/cuda_fp16.h: No such file or directory in tensorflow-gpu>=1.14.0
We do not have to copy the files. An easier way is to add a soft link in third_party/gpus to /usr/local/cuda. 

**RESULT:** It works. 

## 2. Undefined symbols for tensorflow-gpu>=1.15
The CMakeLists.txts work fine with tensorflow-gpu<=1.14 but not with 1.15 and 2.0. I got the following error when loading the library. 
```
tensorflow.python.framework.errors_impl.NotFoundError: build/lib/libtensorflow_ctext.so: undefined symbol: _ZN10tensorflow12OpDefBuilder4AttrENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```
I tried ```nm build/lib/libtensorflow_ctext.so``` on 1.14 version and 2.0 version, both shared libraries have this undefined symbol in the middle. 
```
U _ZN10tensorflow12OpDefBuilder4AttrENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```
It seems that the program is going to find this symbol in the linked Tensorflow framework library libtensorflow_framework.so. I searched libtensorflow_framework.so.2 for similar symbols and found several of them. 
```
0000000000cacc50 T _ZN10tensorflow12OpDefBuilder10DeprecatedEiSs
0000000000cace00 T _ZN10tensorflow12OpDefBuilder10SetShapeFnESt8functionIFNS_6StatusEPNS_15shape_inference16InferenceContextEEE
0000000000cacb20 T _ZN10tensorflow12OpDefBuilder13ControlOutputESs
0000000000cac980 T _ZN10tensorflow12OpDefBuilder13SetIsStatefulEv
0000000000cac970 T _ZN10tensorflow12OpDefBuilder14SetIsAggregateEv
0000000000cac960 T _ZN10tensorflow12OpDefBuilder16SetIsCommutativeEv
0000000000cac990 T _ZN10tensorflow12OpDefBuilder27SetAllowsUninitializedInputEv
0000000000cacb50 T _ZN10tensorflow12OpDefBuilder3DocESs
0000000000caca90 T _ZN10tensorflow12OpDefBuilder4AttrESs
0000000000cacac0 T _ZN10tensorflow12OpDefBuilder5InputESs
0000000000cacaf0 T _ZN10tensorflow12OpDefBuilder6OutputESs
0000000000cac830 T _ZN10tensorflow12OpDefBuilderC1ESs
0000000000cac830 T _ZN10tensorflow12OpDefBuilderC2ESs
0000000000c702d0 W _ZN10tensorflow12OpDefBuilderD1Ev
0000000000c702d0 W _ZN10tensorflow12OpDefBuilderD2Ev
```
The symbol ```_ZN10tensorflow12OpDefBuilder4AttrESs``` looks very similar but different in the last several letters. 

**Solve:** I used ```nm -C``` instruction to look inside the .so files and found that in Tensorflow>=1.15.0, the function is defined as
```
0000000000caca90 T tensorflow::OpDefBuilder::Attr(std::string)
```
while in Tensorflow<=1.14.0, the function is defined as
```
0000000000c96ed0 T tensorflow::OpDefBuilder::Attr(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)
```
So, they use different settings on _GLIBCXX_USE_CXX11_ABI when compiling the shared library. 

In order to be consistant and avoid those undefined symbol problems, I need to define ```-D_GLIBCXX_USE_CXX11_ABI=1``` for early versions of Tensorflow and define ```-D_GLIBCXX_USE_CXX11_ABI=0``` for later versions.

Things have changed when testing on 01/06/21. Tensorflow-gpu=1.14.0 works with ```-D_GLIBCXX_USE_CXX11_ABI=0```. Other versions have not been tested. 