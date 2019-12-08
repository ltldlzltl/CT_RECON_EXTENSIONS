/*
 * @Description: GPU implementation of bp_par_2d_sv.h
 * @Author: Tianling Lyu
 * @Date: 2019-12-03 11:42:04
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-12-08 14:51:04
 */

 #include "include/bp_par_2d_sv.h"

 #define _USE_MATH_DEFINES
#include <cmath>
#include "cuda/cuda_common.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#define M_PI_4 M_PI/4
#endif

#define MAX(x, y) (((x)>(y)) ? (x) : (y))

namespace ct_recon
{
#ifdef USE_CUDA
__global__ void ParallelSingleViewBp2DPixDrivenPrepKernel(double* xcos, 
    double* ysin, const ParallelSingleViewBp2DParam param, 
    const int n_elements)
{
    unsigned int length = MAX(param.nx, param.ny);
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        int ia = thread_id % param.na;
        int ipos = thread_id / param.na;
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
            double posy = (centy-ipos) * param.dy;
            ysin[ia + ipos*param.na] = posy * sin(angle);
        }
    }
    return;
}

bool ParallelSingleViewBp2DPixDrivenPrep::calculate_on_gpu(double* xcos,
    double* ysin, int* buffer, cudaStream_t stream) const
{
    int n_elements = param_.na * MAX(param_.nx, param_.ny);
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    ParallelSingleViewBp2DPixDrivenPrepKernel
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (xcos, ysin, param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <typename T>
__global__ void ParallelSingleViewBp2DPixDrivenKernel(const T* proj, 
    T* img, const double* xcos, const double* ysin, 
    ParallelSingleViewBp2DParam param, const int n_elements)
{
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        int ix = thread_id % param.nx;
        int iy = thread_id / param.nx;
        double cents = (static_cast<double>(param.ns-1)) / 2 + 
            param.offset_s;
        double s, u;
        int is1, is2;
        double sum = 0;
        const double *xcos_ptr = xcos + ix * param.na;
        const double *ysin_ptr = ysin + iy * param.na;
        const T* proj_ptr = proj;
        T* img_ptr = img + thread_id * param.na;
        // SingleViewBp
        for (unsigned int ia = 0; ia < param.na; ++ia) {
            s = (xcos_ptr[ia] + ysin_ptr[ia]) / param.ds + cents;
            if (s >= 0 && s <= param.ns-1) {
                // linear interpolation
                is1 = floor(s);
                is2 = ceil(s);
                u = s - is1;
                *img_ptr = (1-u) * proj_ptr[is1] + u * proj_ptr[is2];
            } else {
                *img_ptr = 0;
            }
            ++img_ptr;
            proj_ptr += param.ns;
        }
        // write to image
        img[thread_id] = sum * param.orbit;
    }
    return;
}

template <>
bool ParallelSingleViewBp2DPixDriven<float>::calculate_on_gpu(const float* proj, 
	float* img, const double* xcos, const double* ysin, const int* buffer, 
    cudaStream_t stream) const
{
    int n_elements = this->param_.nx*this->param_.ny;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    ParallelSingleViewBp2DPixDrivenKernel<float>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (proj, img, xcos, ysin, this->param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <>
bool ParallelSingleViewBp2DPixDriven<double>::calculate_on_gpu(const double* proj, 
	double* img, const double* xcos, const double* ysin, const int* buffer, 
    cudaStream_t stream) const
{
    int n_elements = this->param_.nx*this->param_.ny;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    ParallelSingleViewBp2DPixDrivenKernel<double>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (proj, img, xcos, ysin, this->param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}
#endif
} // namespace ct_recon