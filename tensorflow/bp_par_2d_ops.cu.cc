/*
 * @Description: GPU functors for bp_par_2d_ops.h
 * @Author: Tianling Lyu
 * @Date: 2019-11-28 15:14:53
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-11-28 15:19:58
 */

#include "tensorflow/bp_par_2d_ops.h"

#if GOOGLE_CUDA

#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/tensor_format.h"

#define EIGEN_USE_GPU

namespace tensorflow
{
// partial specializations for GPU devices
bool LaunchBpPar2DPrepOp<Eigen::GpuDevice>::operator()(OpKernelContext *ctx, double *buffer1, double *buffer2,
                                                       int *buffer3, const ct_recon::ParallelBackprojection2DPrep *prep)
{
    auto device = ctx->eigen_gpu_device();
    return prep->calculate_on_gpu(buffer1, buffer2, buffer3,
                                  device.stream());
}

template <typename T>
bool LaunchBpPar2DOp<Eigen::GpuDevice, T>::operator()(OpKernelContext *ctx, const T *proj, T *img,
                                                      const double *buffer1, const double *buffer2, const int *buffer3,
                                                      const ct_recon::ParallelBackprojection2D<T> *bp, const int nbatch,
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

template struct LaunchBpPar2DOp<Eigen::GpuDevice, float>;
template struct LaunchBpPar2DOp<Eigen::GpuDevice, double>;

bool LaunchBpPar2DGradPrepOp<Eigen::GpuDevice>::operator()(OpKernelContext *ctx, double *buffer1, double *buffer2,
                                                           bool *buffer3, const ct_recon::ParallelBackprojection2DGradPrepare *prep)
{
    auto device = ctx->eigen_gpu_device();
    return prep->calculate_on_gpu(buffer1, buffer2, buffer3,
                                  device.stream());
}

// Functor for backprojection gradient calculation
template <typename T>
bool LaunchBpPar2DGradOp<Eigen::GpuDevice, T>::operator()(OpKernelContext *ctx, const T *img, T *grad,
                                                          const double *buffer1, const double *buffer2, const bool *buffer3,
                                                          ct_recon::ParallelBackprojection2DGrad<T> *bp_grad, const int nbatch,
                                                          const unsigned int sizeimg, const unsigned int sizeproj)
{
    auto device = ctx->eigen_gpu_device();
    bool result = true;
    T *grad_ptr = grad;
    const T *img_ptr = img;
    // iterate in batch
    for (int ibatch = 0; ibatch < nbatch; ++ibatch)
    {
        result &= bp_grad->calculate_on_gpu(img_ptr, grad_ptr, buffer1,
                                            buffer2, buffer3, device.stream());
        grad_ptr += sizeproj;
        img_ptr += sizeimg;
    }
    return result;
}

template struct LaunchBpPar2DGradOp<Eigen::GpuDevice, float>;
template struct LaunchBpPar2DGradOp<Eigen::GpuDevice, double>;
} // namespace tensorflow
#endif // GOOGLE_CUDA