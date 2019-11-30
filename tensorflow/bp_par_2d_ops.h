/*
 * @Description: 2-D backprojection structures for tensorflow
 * @Author: Tianling Lyu
 * @Date: 2019-11-26 15:56:29
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-11-30 11:27:56
 */

#ifndef TENSORFLOW_CORE_USER_OPS_BP_PAR_2D_OPS_H_
#define TENSORFLOW_CORE_USER_OPS_BP_PAR_2D_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"

#include "include/bp_par_2d.h"

namespace tensorflow
{

// Functor for backprojection preparation
template <typename Device>
struct LaunchBpPar2DPrepOp {
    bool operator()(OpKernelContext* ctx, double* buffer1, double* buffer2, 
        int* buffer3, const ct_recon::ParallelBackprojection2DPrepare* prep)
    {
        return prep->calculate_on_cpu(buffer1, buffer2, buffer3);
    }
}; // struct LaunchFpPar2DPrepOp

// Functor for backprojection calculation
template <typename Device, typename T>
struct LaunchBpPar2DOp {
    bool operator()(OpKernelContext* ctx, const T* proj, T* img, 
        const double* buffer1, const double* buffer2, const int* buffer3, 
        const ct_recon::ParallelBackprojection2D<T>* bp, const int nbatch, 
        const unsigned int sizeimg, const unsigned int sizeproj)
    {
        bool result = true;
        const T* proj_ptr = proj;
        T* img_ptr = img;
        // iterate in batch
        for (int ibatch = 0; ibatch < nbatch; ++ibatch) {
            result &= bp->calculate_on_cpu(proj_ptr, img_ptr, buffer1, 
                buffer2, buffer3);
            img_ptr += sizeimg;
            proj_ptr += sizeproj;
        }
        return result;
    }
}; // struct LaunchBpPar2DOp

// Functor for backprojection gradient preparation
template <typename Device>
struct LaunchBpPar2DGradPrepOp {
    bool operator()(OpKernelContext* ctx, double* buffer1, double* buffer2, 
        bool* buffer3, const ct_recon::ParallelBackprojection2DGradPrep* prep)
    {
        return prep->calculate_on_cpu(buffer1, buffer2, buffer3);
    }
}; // struct LaunchBpPar2DGradPrepOp

// Functor for backprojection gradient calculation
template <typename Device, typename T>
struct LaunchBpPar2DGradOp {
    bool operator()(OpKernelContext* ctx, const T* img, T* grad, 
        const double* buffer1, const double* buffer2, const bool* buffer3, 
        ct_recon::ParallelBackprojection2DGrad<T> *bp_grad, const int nbatch, 
        const unsigned int sizeimg, const unsigned int sizeproj)
    {
        bool result = true;
        T* grad_ptr = grad;
        const T* img_ptr = img;
        // iterate in batch
        for (int ibatch = 0; ibatch < nbatch; ++ibatch) {
            result &= bp_grad->calculate_on_cpu(img_ptr, grad_ptr, buffer1, 
                buffer2, buffer3);
            grad_ptr += sizeproj;
            img_ptr += sizeimg;
        }
        return result;
    }
}; // struct LaunchBpPar2DGradOp

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
// partial specializations for GPU devices
template <>
struct LaunchBpPar2DPrepOp<Eigen::GpuDevice> {
    bool operator()(OpKernelContext* ctx, double* buffer1, double* buffer2, 
        int* buffer3, const ct_recon::ParallelBackprojection2DPrepare* prep);
}; // struct LaunchBpPar2DPrepOp<Eigen::GpuDevice, T>

template <typename T>
struct LaunchBpPar2DOp<Eigen::GpuDevice, T> {
    bool operator()(OpKernelContext* ctx, const T* proj, T* img, 
        const double* buffer1, const double* buffer2, const int* buffer3, 
        const ct_recon::ParallelBackprojection2D<T>* bp, const int nbatch, 
        const unsigned int sizeimg, const unsigned int sizeproj);
}; // struct LaunchBpPar2DOp<Eigen::GpuDevice, T>

template <>
struct LaunchBpPar2DGradPrepOp<Eigen::GpuDevice> {
    bool operator()(OpKernelContext* ctx, double* buffer1, double* buffer2, 
        bool* buffer3, const ct_recon::ParallelBackprojection2DGradPrep* prep);
}; // struct LaunchBpPar2DGradPrepOp

// Functor for forward projection gradient calculation
template <typename T>
struct LaunchBpPar2DGradOp<Eigen::GpuDevice, T> {
    bool operator()(OpKernelContext* ctx, const T* img, T* grad, 
        const double* buffer1, const double* buffer2, const bool* buffer3, 
        ct_recon::ParallelBackprojection2DGrad<T> *bp_grad, const int nbatch, 
        const unsigned int sizeimg, const unsigned int sizeproj);
}; // struct LaunchBpPar2DGradOp
#endif // GOOGLE_CUDA

} // namespace tensorflow

#endif