/*
 * @Description: 2-D forward projection structures for tensorflow
 * @Author: Tianling Lyu
 * @Date: 2019-11-19 11:37:41
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-12-09 11:49:05
 */

#ifndef TENSORFLOW_CORE_USER_OPS_FP_PAR_2D_OPS_H_
#define TENSORFLOW_CORE_USER_OPS_FP_PAR_2D_OPS_H_

//#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
//#include "tensorflow/core/framework/types.h"
//#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "include/fp_par_2d.h"

namespace tensorflow
{ 
// Functor for forward projection preparation
template <typename Device>
struct LaunchFpPar2DPrepOp {
    bool operator()(OpKernelContext* ctx, double* sincostbl, double* buffer1, 
        int* buffer2, const ct_recon::ParallelProjection2DPrepare* prep)
    {
        return prep->calculate_on_cpu(sincostbl, buffer1, buffer2);
    }
}; // struct LaunchFpPar2DPrepOp

// Functor for forward projection calculation
template <typename Device, typename T>
struct LaunchFpPar2DOp {
    bool operator()(OpKernelContext* ctx, const T* img, T* proj, 
        const double* sincostbl, const double* buffer1, const int* buffer2, 
        const ct_recon::ParallelProjection2D<T>* fp, const int nbatch, 
        const unsigned int sizeimg, const unsigned int sizeproj)
    {
        bool result = true;
        const T* img_ptr = img;
        T* proj_ptr = proj;
        // iterate in batch
        for (int ibatch = 0; ibatch < nbatch; ++ibatch) {
            result &= fp->calculate_on_cpu(img_ptr, proj_ptr, sincostbl, 
                buffer1, buffer2);
            img_ptr += sizeimg;
            proj_ptr += sizeproj;
        }
        return result;
    }
}; // struct LaunchFpPar2DOp

// Functor for forward projection gradient preparation
template <typename Device>
struct LaunchFpPar2DGradPrepOp {
    bool operator()(OpKernelContext* ctx, double* buffer1, double* buffer2, 
        int* buffer3, const ct_recon::ParallelProjection2DGradPrepare* prep)
    {
        return prep->calculate_on_cpu(buffer1, buffer2, buffer3);
    }
}; // struct LaunchFpPar2DGradPrepOp

// Functor for forward projection gradient calculation
template <typename Device, typename T>
struct LaunchFpPar2DGradOp {
    bool operator()(OpKernelContext* ctx, const T* proj, T* grad, 
        const double* buffer1, const double* buffer2, const int* buffer3, 
        ct_recon::ParallelProjection2DGrad<T> *fp_grad, const int nbatch, 
        const unsigned int sizeimg, const unsigned int sizeproj)
    {
        bool result = true;
        T* grad_ptr = grad;
        const T* proj_ptr = proj;
        // iterate in batch
        for (int ibatch = 0; ibatch < nbatch; ++ibatch) {
            result &= fp_grad->calculate_on_cpu(proj_ptr, grad_ptr, buffer1, 
                buffer2, buffer3);
            grad_ptr += sizeimg;
            proj_ptr += sizeproj;
        }
        return result;
    }
}; // struct LaunchFpPar2DGradOp
} // namespace tensorflow

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#ifdef TFVER_1_14
#include "tensorflow/core/util/gpu_kernel_helper.h"
#else
#include "tensorflow/core/util/cuda_kernel_helper.h"
#endif
namespace tensorflow
{
// partial specializations for GPU devices
template <>
struct LaunchFpPar2DPrepOp<Eigen::GpuDevice> {
    bool operator()(OpKernelContext* ctx, double* sincostbl, double* buffer1, 
        int* buffer2, const ct_recon::ParallelProjection2DPrepare* prep)
    {
        auto device = ctx->eigen_gpu_device();
        return prep->calculate_on_gpu(sincostbl, buffer1, buffer2,
                                    device.stream());
    }
}; // struct LaunchFpPar2DPrepOp<Eigen::GpuDevice, T>

template <typename T>
struct LaunchFpPar2DOp<Eigen::GpuDevice, T> {
    bool operator()(OpKernelContext* ctx, const T* img, T* proj, 
        const double* sincostbl, const double* buffer1, const int* buffer2, 
        const ct_recon::ParallelProjection2D<T>* fp, const int nbatch, 
        const unsigned int sizeimg, const unsigned int sizeproj)
    {
        auto device = ctx->eigen_gpu_device();
        bool result = true;
        const T *img_ptr = img;
        T *proj_ptr = proj;
        // iterate in batch
        for (int ibatch = 0; ibatch < nbatch; ++ibatch)
        {
            result &= fp->calculate_on_gpu(img_ptr, proj_ptr, sincostbl,
                                        buffer1, buffer2, device.stream());
            img_ptr += sizeimg;
            proj_ptr += sizeproj;
        }
        return result;
    }
}; // struct LaunchFpPar2DOp<Eigen::GpuDevice, T>

template <>
struct LaunchFpPar2DGradPrepOp<Eigen::GpuDevice> {
    bool operator()(OpKernelContext* ctx, double* buffer1, double* buffer2, 
        int* buffer3, const ct_recon::ParallelProjection2DGradPrepare* prep)
    {
        auto device = ctx->eigen_gpu_device();
        return prep->calculate_on_gpu(buffer1, buffer2, buffer3,
                                    device.stream());
    }
}; // struct LaunchFpPar2DGradPrepOp

// Functor for forward projection gradient calculation
template <typename T>
struct LaunchFpPar2DGradOp<Eigen::GpuDevice, T> {
    bool operator()(OpKernelContext* ctx, const T* proj, T* grad, 
        const double* buffer1, const double* buffer2, const int* buffer3, 
        ct_recon::ParallelProjection2DGrad<T> *fp_grad, const int nbatch, 
        const unsigned int sizeimg, const unsigned int sizeproj)
    {
        auto device = ctx->eigen_gpu_device();
        bool result = true;
        T *grad_ptr = grad;
        const T *proj_ptr = proj;
        // iterate in batch
        for (int ibatch = 0; ibatch < nbatch; ++ibatch)
        {
            result &= fp_grad->calculate_on_gpu(proj_ptr, grad_ptr, buffer1,
                                                buffer2, buffer3, device.stream());
            grad_ptr += sizeimg;
            proj_ptr += sizeproj;
        }
        return result;
    }
}; // struct LaunchFpPar2DGradOp
} // namespace tensorflow
#endif // GOOGLE_CUDA

#endif