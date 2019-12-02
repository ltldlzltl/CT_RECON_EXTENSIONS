/*
 * @Description: GPU functors for ramp_filter_ops.h
 * @Author: Tianling Lyu
 * @Date: 2019-12-02 10:37:52
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-12-02 14:06:39
 */

#include "tensorflow/ramp_filter_ops.h"

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow
{
// partial specializations for GPU devices
template <typename T>
bool LaunchRampFilterPrepOp<Eigen::GpuDevice, T>::operator()(
    OpKernelContext* ctx, T* filter, 
    const ct_recon::RampFilterPrep<T>* prep) 
{
    auto device = ctx->eigen_gpu_device();
    return prep->calculate_on_gpu(filter, device.stream());
}

template struct LaunchRampFilterPrepOp<Eigen::GpuDevice, float>;
template struct LaunchRampFilterPrepOp<Eigen::GpuDevice, double>;

template <typename T>
bool LaunchRampFilterOp<Eigen::GpuDevice, T>::operator()(
    OpKernelContext* ctx, const T* in, const T* filter, T* out, 
    const ct_recon::RampFilter<T>* filt_, const int nbatch, 
    const unsigned int sizeproj)
{
    auto device = ctx->eigen_gpu_device();
    bool result = true;
    const T* in_ptr = in;
    T* out_ptr = out;
    for (int ibatch = 0; ibatch < nbatch; ++ibatch) {
        result &= filt_->calculate_on_gpu(in_ptr, filter, out_ptr, 
            device.stream());
        in_ptr += sizeproj;
        out_ptr += sizeproj;
    }
    return result;
}

template struct LaunchRampFilterOp<Eigen::GpuDevice, float>;
template struct LaunchRampFilterOp<Eigen::GpuDevice, double>;

template <typename T>
bool LaunchRampFilterGradOp<Eigen::GpuDevice, T>::operator()(
    OpKernelContext* ctx, const T* in, const T* filter, T* out, 
    const ct_recon::RampFilterGrad<T>* filt_, 
    const int nbatch, const unsigned int sizeproj)
{
    auto device = ctx->eigen_gpu_device();
    bool result = true;
    const T* in_ptr = in;
    T* out_ptr = out;
    for (int ibatch = 0; ibatch < nbatch; ++ibatch) {
        result &= filt_->calculate_on_gpu(in_ptr, filter, out_ptr, 
            device.stream());
        in_ptr += sizeproj;
        out_ptr += sizeproj;
    }
    return result;
}

template struct LaunchRampFilterGradOp<Eigen::GpuDevice, float>;
template struct LaunchRampFilterGradOp<Eigen::GpuDevice, double>;

} // namespace tensorflow

#endif // if GOOGLE_CUDA