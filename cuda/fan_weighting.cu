/*
 * @Description: 
 * @Author: Tianling Lyu
 * @Date: 2021-01-09 08:47:49
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2021-01-09 09:08:13
 */

 #include "include/fan_weighting.h"

#include <cstdio>
#include "cuda/cuda_common.h"

namespace ct_recon {
    #ifdef USE_CUDA
    template <typename T>
    __global__ void FlatWeightingKernel(const T* in, T* out, 
        const FanWeightingParam param, const int n_elements)
    {
        double cents = static_cast<double>(param.ns-1) / 2;
        for (int thread_id : CudaGridRangeX<int>(n_elements)) {
            int is = thread_id % param.ns;
            double s = param.ds * (static_cast<double>(is) - cents);
            double w = param.dso * fabs(cos(atan2(s, param.dsd))) / param.dsd;
            out[thread_id] = w * in[thread_id];
        }
        return;
    }

    template <typename T>
    __global__ void FanWeightingKernel(const T* in, T* out, 
        const FanWeightingParam param, const int n_elements)
    {
        double cents = static_cast<double>(param.ns-1) / 2;
        for (int thread_id : CudaGridRangeX<int>(n_elements)) {
            int is = thread_id % param.ns;
            double s = param.ds * (static_cast<double>(is) - cents);
            double w = param.dso * fabs(cos(s / param.dsd)) / param.dsd;
            out[thread_id] = w * in[thread_id];
        }
        return;
    }

    template <typename T>
    __global__ void FlatWeightingGradKernel(const T* in, T* out, 
        const FanWeightingParam param, const int n_elements)
    {
        double cents = static_cast<double>(param.ns-1) / 2;
        for (int thread_id : CudaGridRangeX<int>(n_elements)) {
            int is = thread_id % param.ns;
            double s = param.ds * (static_cast<double>(is) - cents);
            double w = param.dsd / param.dso * fabs(cos(atan2(s, param.dsd)));
            out[thread_id] = in[thread_id] * w;
        }
        return;
    }

    template <typename T>
    __global__ void FanWeightingGradKernel(const T* in, T* out, 
        const FanWeightingParam param, const int n_elements)
    {
        double cents = static_cast<double>(param.ns-1) / 2;
        for (int thread_id : CudaGridRangeX<int>(n_elements)) {
            int is = thread_id % param.ns;
            double s = param.ds * (static_cast<double>(is) - cents);
            double w = param.dsd / param.dso * fabs(cos(s / param.dsd));
            out[thread_id] = in[thread_id] * w;
        }
        return;
    }

    template <>
    bool FanWeighting<float>::calculate_on_gpu(const float* in, 
        float* out, cudaStream_t stream) const
    {
        int n_elements = param_.ns*param_.nrow;
        CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
        if (param_.type == "fan") {
            FanWeightingKernel<float>
                <<<config.block_count, config.thread_per_block, 0, stream>>>
                (in, out, param_, n_elements);
        } else if (param_.type == "flat") {
            FlatWeightingKernel<float>
                <<<config.block_count, config.thread_per_block, 0, stream>>>
                (in, out, param_, n_elements);
        } else {
            return false;
        }
        cudaError_t err = cudaDeviceSynchronize();
        return err==cudaSuccess;
    }

    template <>
    bool FanWeighting<double>::calculate_on_gpu(const double* in, 
        double* out, cudaStream_t stream) const
    {
        int n_elements = param_.ns*param_.nrow;
        CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
        if (param_.type == "fan") {
            FanWeightingKernel<double>
                <<<config.block_count, config.thread_per_block, 0, stream>>>
                (in, out, param_, n_elements);
        } else if (param_.type == "flat") {
            FlatWeightingKernel<double>
                <<<config.block_count, config.thread_per_block, 0, stream>>>
                (in, out, param_, n_elements);
        } else {
            return false;
        }
        cudaError_t err = cudaDeviceSynchronize();
        return err==cudaSuccess;
    }

    template <>
    bool FanWeightingGrad<float>::calculate_on_gpu(const float* in, 
        float* out, cudaStream_t stream) const
    {
        int n_elements = param_.ns*param_.nrow;
        CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
        if (param_.type == "fan") {
            FanWeightingGradKernel<float>
                <<<config.block_count, config.thread_per_block, 0, stream>>>
                (in, out, param_, n_elements);
        } else if (param_.type == "flat") {
            FlatWeightingGradKernel<float>
                <<<config.block_count, config.thread_per_block, 0, stream>>>
                (in, out, param_, n_elements);
        } else {
            return false;
        }
        cudaError_t err = cudaDeviceSynchronize();
        return err==cudaSuccess;
    }

    template <>
    bool FanWeightingGrad<double>::calculate_on_gpu(const double* in, 
        double* out, cudaStream_t stream) const
    {
        int n_elements = param_.ns*param_.nrow;
        CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
        if (param_.type == "fan") {
            FanWeightingGradKernel<double>
                <<<config.block_count, config.thread_per_block, 0, stream>>>
                (in, out, param_, n_elements);
        } else if (param_.type == "flat") {
            FlatWeightingGradKernel<double>
                <<<config.block_count, config.thread_per_block, 0, stream>>>
                (in, out, param_, n_elements);
        } else {
            return false;
        }
        cudaError_t err = cudaDeviceSynchronize();
        return err==cudaSuccess;
    }

    #endif // USE_CUDA
} // namespace ct_recon