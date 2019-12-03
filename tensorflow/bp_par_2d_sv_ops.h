/*
 * @Description: 2-D single-view parallel bp structures for tensorflow
 * @Author: Tianling Lyu
 * @Date: 2019-12-03 11:55:08
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-12-03 12:02:54
 */

#ifndef TENSORFLOW_CORE_USER_OPS_BP_PAR_2D_SV_OPS_H_
#define TENSORFLOW_CORE_USER_OPS_BP_PAR_2D_SV_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"

#include "include/bp_par_2d_sv.h"
namespace tensorflow
{

// Functor for SingleViewBp preparation
template <typename Device>
struct LaunchBpPar2DSVPrepOp {
    bool operator()(OpKernelContext* ctx, double* buffer1, double* buffer2, 
        int* buffer3, const ct_recon::ParallelSingleViewBp2DPrepare* prep)
    {
        return prep->calculate_on_cpu(buffer1, buffer2, buffer3);
    }
}; // struct LaunchFpPar2DSVPrepOp

// Functor for SingleViewBp calculation
template <typename Device, typename T>
struct LaunchBpPar2DSVOp {
    bool operator()(OpKernelContext* ctx, const T* proj, T* img, 
        const double* buffer1, const double* buffer2, const int* buffer3, 
        const ct_recon::ParallelSingleViewBp2D<T>* bp, const int nbatch, 
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
}; // struct LaunchBpPar2DSVOp
} // namespace tensorflow

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/util/cuda_kernel_helper.h"
namespace tensorflow
{
// partial specializations for GPU devices
template <>
struct LaunchBpPar2DSVPrepOp<Eigen::GpuDevice> {
    bool operator()(OpKernelContext* ctx, double* buffer1, double* buffer2, 
        int* buffer3, const ct_recon::ParallelSingleViewBp2DPrepare* prep)
    {
        auto device = ctx->eigen_gpu_device();
        return prep->calculate_on_gpu(buffer1, buffer2, buffer3,
            device.stream());
    }
}; // struct LaunchBpPar2DSVPrepOp<Eigen::GpuDevice, T>

template <typename T>
struct LaunchBpPar2DSVOp<Eigen::GpuDevice, T> {
    bool operator()(OpKernelContext* ctx, const T* proj, T* img, 
        const double* buffer1, const double* buffer2, const int* buffer3, 
        const ct_recon::ParallelSingleViewBp2D<T>* bp, const int nbatch, 
        const unsigned int sizeimg, const unsigned int sizeproj)
    {
        auto device = ctx->eigen_gpu_device();
        bool result = true;
        const T *proj_ptr = proj;
        T *img_ptr = img;
        // iterate in batch
        for (int ibatch = 0; ibatch < nbatch; ++ibatch)
        {
            result &= bp->calculate_on_gpu(proj_ptr, img_ptr, buffer1, 
                                        buffer2, buffer3, device.stream());
            img_ptr += sizeimg;
            proj_ptr += sizeproj;
        }
        return result;
    }
}; // struct LaunchBpPar2DSVOp<Eigen::GpuDevice, T>
} // namespace tensorflow
#undef EIGEN_USE_GPU
#endif // GOOGLE_CUDA

#endif