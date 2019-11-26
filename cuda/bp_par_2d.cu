/*
 * @Description: GPU implementation of fp_par_2d.h
 * @Author: Tianling Lyu
 * @Date: 2019-11-24 10:36:05
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-11-24 11:08:16
 */

#include "include/bp_par_2d.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include "cuda/cuda_common.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#define M_PI_4 M_PI/4
#endif

namespace ct_recon
{

__global__ void ParallelBackprojection2DPixDrivenPrepKernel(double* xcos, 
    double* ysin, const ParallelBackprojection2DParam param, 
    const int n_elements)
{
    unsigned int length = param.nx > param.ny ? param.nx : param.ny;
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        int ia = thread_id % length;
        int ipos = thread_id / length;
        double angle = param.orbit_start + ia * param.orbit;
        // calculate x*cos(angle)
        if (ipos < param.nx) {
            double centx = static_cast<double>(param.nx-1) / 2 + 
                param.offset_x;
            double posx = (ipos-centx) * param.dx;
            xcos[ia + ipos*param.na] = posx * cos(angle);
        }
        // calculate y*sin(angle)
        if (ipos < param.ny) {
            double centy = static_cast<double>(param.ny-1) / 2 + 
                param.offset_y;
            double posy = (param.ny-1-centy-ipos) * param.ny;
            ysin[ia + ipos*param.na] = posy * sin(angle);
        }
    }
    return;
}

bool ParallelBackprojection2DPixDrivenPrep::calculate_on_gpu(double* xcos,
    double* ysin, int* buffer, cudaStream_t stream) const
{
    CudaLaunchConfig config = GetCudaLaunchConfig(param_.na);
    ParallelBackprojection2DPixDrivenPrepKernel
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (xcos, ysin, param_, param_.na);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <typename T>
__global__ void ParallelBackprojection2DPixDrivenKernel(const T* proj, 
    T* img, const double* xcos, const double* ysin, 
    ParallelBackprojection2DParam param, const int n_elements)
{
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        int ix = n_elements % param.nx;
        int iy = n_elements / param.nx;
        double cents = (static_cast<double>(param.ns-1)) / 2 + 
            param.offset_s;
        double s, is1, is2, u;
        double sum = 0;
        double *xcos_ptr = xcos + ix * param.na;
        double *ysin_ptr = ysin + iy * param.na;
        const T* proj_ptr = proj;
        // backprojection
        for (unsigned int ia = 0; ia < param.na; ++ia) {
            s = (xcos_ptr[ia] + ysin_ptr[ia]) / param.ds + cents;
            if (s >= 0 && s <= param.ns-1) {
                // linear interpolation
                is1 = floor(s);
                is2 = ceil(s);
                u = s - is1;
                sum += (1-u) * proj_ptr[is1] + u * proj_ptr[is2];
            }
            proj_ptr += param.ns;
        }
        // write to image
        img[thread_id] = sum * param.orbit;
    }
    return;
}

template <>
bool ParallelBackprojection2DPixDriven<float>::calculate_on_gpu(const float* proj, 
	float* img, const double* xcos, const double* ysin, const int* buffer, 
    cudaStream_t stream) const
{
    int n_elements = param_.nx*param_.ny;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    ParallelBackprojection2DPixDrivenKernel<float>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (proj, img, xcos, ysin, param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <>
bool ParallelBackprojection2DPixDriven<double>::calculate_on_gpu(const double* proj, 
	double* img, const double* xcos, const double* ysin, const int* buffer, 
    cudaStream_t stream) const
{
    int n_elements = param_.nx*param_.ny;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    ParallelBackprojection2DPixDrivenKernel<float>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (proj, img, xcos, ysin, param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

} // namespace ct_recon