/*
 * @Author: Tianling Lyu
 * @Date: 2021-02-05 10:34:40
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2021-02-05 12:05:58
 * @FilePath: /CT_RECON_EXTENSIONS/tensorflow/fan_weighting_ops.h
 */

#ifndef TENSORFLOW_CORE_USER_OPS_FAN_WEIGHTING_OPS_H_
#define TENSORFLOW_CORE_USER_OPS_FAN_WEIGHTING_OPS_H_

#include "tensorflow/core/framework/op_kernel.h"

#include "include/fan_weighting.h"

namespace tensorflow
{

// Functor for fan weighting
template <typename Device, typename T>
struct LaunchFanWOp {
    bool operator()(OpKernelContext* ctx, const T* proj_in, T* proj_out,
        const ct_recon::FanWeighting<T>* fw, const int nbatch, 
        const unsigned int sizeproj)
    {
        bool result = true;
        const T* in_ptr = proj_in;
        T* out_ptr = proj_out;
        // iterate in batches
        for (int ibatch = 0; ibatch < nbatch; ++ibatch) {
            result &= fw->calculate_on_cpu(in_ptr, out_ptr);
            in_ptr += sizeproj;
            out_ptr += sizeproj;
        }
        return result;
    }
}; // struct LaunchFanWOp

// Functor for fan weighting gradient
template <typename Device, typename T>
struct LaunchFanWGradOp {
    bool operator()(OpKernelContext* ctx, const T* proj_in, T* proj_out,
        const ct_recon::FanWeightingGrad<T>* fw, const int nbatch, 
        const unsigned int sizeproj)
    {
        bool result = true;
        const T* in_ptr = proj_in;
        T* out_ptr = proj_out;
        // iterate in batches
        for (int ibatch = 0; ibatch < nbatch; ++ibatch) {
            result &= fw->calculate_on_cpu(in_ptr, out_ptr);
            in_ptr += sizeproj;
            out_ptr += sizeproj;
        }
        return result;
    }
}; // struct LaunchFanWOp
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
template <typename T>
struct LaunchFanWOp<Eigen::GpuDevice, T> {
    bool operator()(OpKernelContext* ctx, const T* proj_in, T* proj_out,
        const ct_recon::FanWeighting<T>* fw, const int nbatch, 
        const unsigned int sizeproj)
    {
        auto device = ctx->eigen_gpu_device();
        bool result = true;
        const T *in_ptr = proj_in;
        T* out_ptr = proj_out;
        // iterate in batches
        for (int ibatch = 0; ibatch < nbatch; ++ibatch)
        {
            result &= fw->calculate_on_gpu(in_ptr, out_ptr, device.stream());
            in_ptr += sizeproj;
            out_ptr += sizeproj;
        }
        return result;
    }
}; // struct LaunchFanWOp

template <typename T>
struct LaunchFanWGradOp<Eigen::GpuDevice, T> {
    bool operator()(OpKernelContext* ctx, const T* proj_in, T* proj_out,
        const ct_recon::FanWeightingGrad<T>* fw, const int nbatch, 
        const unsigned int sizeproj)
    {
        auto device = ctx->eigen_gpu_device();
        bool result = true;
        const T *in_ptr = proj_in;
        T* out_ptr = proj_out;
        // iterate in batches
        for (int ibatch = 0; ibatch < nbatch; ++ibatch)
        {
            result &= fw->calculate_on_gpu(in_ptr, out_ptr, device.stream());
            in_ptr += sizeproj;
            out_ptr += sizeproj;
        }
        return result;
    }
}; // struct LaunchFanWGradOp

} // namespace tensorflow
#undef EIGEN_USE_GPU
#endif // GOOGLE_CUDA

#endif // TENSORFLOW_CORE_USER_OPS_FAN_WEIGHTING_OPS_H_