/*
 * @Description: GPU implementation of filter.h
 * @Author: Tianling Lyu
 * @Date: 2019-11-30 20:10:31
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2021-03-11 14:31:15
 */

#include "include/filter.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdio>
#include "cuda/cuda_common.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#define M_PI_4 M_PI/4
#endif

namespace ct_recon {
#ifdef USE_CUDA
__global__ void RampFilterPrepParKernel(double* filter, const FilterParam param, 
    const int n_elements)
{
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        int ipos = thread_id - param.ns;
        if (ipos == 0) filter[thread_id] = 0.25 / (param.ds * param.ds);
        else if (ipos % 2 == 0) filter[thread_id] = 0;
        else filter[thread_id] = -1.0 / (param.ds*param.ds*M_PI*M_PI*ipos*ipos);
    }
    return;
}

__global__ void RampFilterPrepFanKernel(double* filter, const FilterParam param, 
    const int n_elements)
{
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        int ipos = thread_id - param.ns;
        if (ipos == 0) filter[thread_id] = 0.25 / (param.ds * param.ds);
        else if (ipos % 2 == 0) filter[thread_id] = 0;
        else
        {
            double sin_angle = sin(ipos * param.ds / param.dsd);
            filter[thread_id] = -1.0 / (M_PI*M_PI * param.dsd*param.dsd * 
                sin_angle*sin_angle);
        }
    }
    return;
}

bool RampFilterPrep::calculate_on_gpu(double* filter, cudaStream_t stream) const
{
    int n_elements = 2*param_.ns+1;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    if (param_.type == 0 || param_.type == 2) {
        RampFilterPrepParKernel
            <<<config.block_count, config.thread_per_block, 0, stream>>>
            (filter, param_, n_elements);
    } else if (param_.type == 1) {
        RampFilterPrepFanKernel
            <<<config.block_count, config.thread_per_block, 0, stream>>>
            (filter, param_, n_elements);
    } else {
        return false;
    }
    cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <typename T>
__global__ void RampFilterKernel(const T* in, const double* filter, T* out, 
    const FilterParam param, const int n_elements)
{
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        int irow = thread_id / param.ns;
        int is = thread_id % param.ns;
        int ipos, ipos2;
        double sum = 0;
        const T* in_ptr = in + irow*param.ns;
        for (ipos = -int(param.ns); ipos < int(param.ns); ++ipos) {
            ipos2 = is - ipos;
            if (ipos2 >= 0 && ipos2 < param.ns) {
                sum += in_ptr[ipos2] * filter[ipos];
            }
        }
        out[thread_id] = sum * param.ds;
    }
    return;
}

template <>
bool RampFilter<float>::calculate_on_gpu(const float* in, const double* filter, 
    float* out, cudaStream_t stream) const
{
    int n_elements = param_.ns*param_.nrow;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    RampFilterKernel<float>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (in, filter+param_.ns, out, param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <>
bool RampFilter<double>::calculate_on_gpu(const double* in, const double* filter, 
    double* out, cudaStream_t stream) const
{
    int n_elements = param_.ns*param_.nrow;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    RampFilterKernel<double>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (in, filter+param_.ns, out, param_, n_elements);
    cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <typename T>
__global__ void RampFilterGradKernel(const T* in, const double* filter, T* out, 
    const FilterParam param, const int n_elements)
{
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        int irow = thread_id / param.ns;
        int is = thread_id % param.ns;
        int ipos, ipos2;
        double sum = 0;
        const T* in_ptr = in + irow*param.ns;
        for (ipos = -int(param.ns); ipos < int(param.ns); ++ipos) {
            ipos2 = is + ipos;
            if (ipos2 >= 0 && ipos2 < param.ns) {
                if (filter[ipos] > 0 || filter[ipos] < 0)
                    sum += in_ptr[ipos2] * filter[ipos];
            }
        }
        out[thread_id] = sum * param.ds;
    }
    return;
}

template <>
bool RampFilterGrad<float>::calculate_on_gpu(const float* in, 
    const double* filter, float* out, cudaStream_t stream) const
{
    int n_elements = param_.ns*param_.nrow;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    RampFilterGradKernel<float>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (in, filter+param_.ns, out, param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <>
bool RampFilterGrad<double>::calculate_on_gpu(const double* in, 
    const double* filter, double* out, cudaStream_t stream) const
{
    int n_elements = param_.ns*param_.nrow;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    RampFilterGradKernel<double>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (in, filter+param_.ns, out, param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}
#endif
} // namespace ct_recon