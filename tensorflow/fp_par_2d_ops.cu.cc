/*
 * @Description: GPU functors for fp_par_2d_ops.h
 * @Author: Tianling Lyu
 * @Date: 2019-11-21 10:55:07
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-11-21 11:31:21
 */

#include "tensorflow/fp_par_2d_ops.h"

#if GOOGLE_CUDA

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/tensor_format.h"

#define EIGEN_USE_GPU

namespace tensorflow
{
// partial specializations for GPU devices
bool LaunchFpPar2DPrepOp<Eigen::GpuDevice>::operator()(OpKernelContext *ctx, double *sincostbl, double *buffer1,
                                                       int *buffer2, const ct_recon::ParallelProjection2DPrepare *prep)
{
    auto device = ctx->eigen_gpu_device();
    return prep->calculate_on_gpu(sincostbl, buffer1, buffer2,
                                  device.stream());
}

template <typename T>
bool LaunchFpPar2DOp<Eigen::GpuDevice, T>::operator()(OpKernelContext *ctx, const T *img, T *proj,
                                                      const double *sincostbl, const double *buffer1, const int *buffer2,
                                                      const ct_recon::ParallelProjection2D<T> *fp, const int nbatch,
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

template struct LaunchFpPar2DOp<Eigen::GpuDevice, float>;
template struct LaunchFpPar2DOp<Eigen::GpuDevice, double>;

bool LaunchFpPar2DGradPrepOp<Eigen::GpuDevice>::operator()(OpKernelContext *ctx, double *buffer1, double *buffer2,
                                                           int *buffer3, const ct_recon::ParallelProjection2DGradPrepare *prep)
{
    auto device = ctx->eigen_gpu_device();
    return prep->calculate_on_gpu(buffer1, buffer2, buffer3,
                                  device.stream());
}

// Functor for forward projection gradient calculation
template <typename T>
bool LaunchFpPar2DGradOp<Eigen::GpuDevice, T>::operator()(OpKernelContext *ctx, const T *proj, T *grad,
                                                          const double *buffer1, const double *buffer2, const int *buffer3,
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

template struct LaunchFpPar2DGradOp<Eigen::GpuDevice, float>;
template struct LaunchFpPar2DGradOp<Eigen::GpuDevice, double>;
} // namespace tensorflow
#endif // GOOGLE_CUDA