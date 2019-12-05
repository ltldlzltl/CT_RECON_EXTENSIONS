/*
 * @Description: GPU implementation of bp_par_2d.h
 * @Author: Tianling Lyu
 * @Date: 2019-11-24 10:36:05
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-12-02 17:04:14
 */

#include "include/bp_par_2d.h"

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

__global__ void ParallelBackprojection2DPixDrivenPrepKernel(double* xcos, 
    double* ysin, const ParallelBackprojection2DParam param, 
    const int n_elements)
{
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

bool ParallelBackprojection2DPixDrivenPrep::calculate_on_gpu(double* xcos,
    double* ysin, int* buffer, cudaStream_t stream) const
{
    int n_elements = param_.na * MAX(param_.nx, param_.ny);
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    ParallelBackprojection2DPixDrivenPrepKernel
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (xcos, ysin, param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <typename T>
__global__ void ParallelBackprojection2DPixDrivenKernel(const T* proj, 
    T* img, const double* xcos, const double* ysin, 
    ParallelBackprojection2DParam param, const int n_elements)
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
    int n_elements = this->param_.nx*this->param_.ny;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    ParallelBackprojection2DPixDrivenKernel<float>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (proj, img, xcos, ysin, this->param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <>
bool ParallelBackprojection2DPixDriven<double>::calculate_on_gpu(const double* proj, 
	double* img, const double* xcos, const double* ysin, const int* buffer, 
    cudaStream_t stream) const
{
    int n_elements = this->param_.nx*this->param_.ny;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    ParallelBackprojection2DPixDrivenKernel<double>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (proj, img, xcos, ysin, this->param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

__global__ void ParallelBackprojection2DPixDrivenGradPrepKernel(double* begins, 
    double* offsets, bool* usex, const ParallelBackprojection2DParam param, 
    const int n_elements)
{
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        unsigned int ia = thread_id, is;
        double angle = param.orbit_start + ia * param.orbit;
        double sin_angle = sin(angle);
        double cos_angle = cos(angle);
        double* begin_ptr = begins + ia * param.ns;
        // useful constants
        const double cents = (static_cast<double>(param.ns-1)) / 2 + 
            param.offset_s;
        const double centx = (static_cast<double>(param.nx-1)) / 2 + 
            param.offset_x;
        const double centy = (static_cast<double>(param.ny-1)) / 2 + 
            param.offset_y;
        // adjust angle to range [0, 2*pi)
        while (angle < 0) angle += 2*M_PI;
        while (angle >= 2*M_PI) angle -= 2*M_PI;
        bool b_usex = !((angle >= M_PI_4 && angle < 3*M_PI_4) ||
            (angle >= 5*M_PI_4 && angle < 7*M_PI_4));
        usex[ia] = b_usex;
        double offset1, offset2, begin;
        if (b_usex) {
            offset1 = param.ds / (cos_angle * param.dx);
            offset2 = param.dy * sin_angle / (cos_angle * param.dx);
            begin = centx - centy * offset2 - cents * offset1;
        } else {
            offset1 = param.ds / (sin_angle * param.dy);
            offset2 = param.dx * cos_angle / (sin_angle * param.dy);
            begin = centy - centx * offset2 - cents * offset1;
        }
        offsets[ia] = offset2;
        for (is = 0; is < param.ns; ++is) {
            *begin_ptr = begin;
            begin += offset1;
            //if (ia == 0)
            //    printf("is=%d, begin=%f, offset1=%f, offset2=%f\n", is, *begin_ptr, offset1, offset2);
            ++begin_ptr;
        }
    }
    return;
}

bool ParallelBackprojection2DPixDrivenGradPrep::calculate_on_gpu(double* begins,
    double* offsets, bool* usex, cudaStream_t stream) const
{
    int n_elements = param_.na;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    ParallelBackprojection2DPixDrivenGradPrepKernel
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (begins, offsets, usex, param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <typename T>
__global__ void ParallelBackprojection2DPixDrivenGradKernel(const T* img, 
    T* grad, const double* begins, const double* offsets, const bool* usex, 
    const ParallelBackprojection2DParam param, const int n_elements)
{
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        // variables
        const int ia = thread_id / param.ns;
        const int is = thread_id % param.ns;
        int i, j, unit1, unit2, range1, range2;
        double sum = 0.0, u, left, right;
        bool b_usex = usex[ia];
        double offset = offsets[ia];
        double pos = begins[thread_id];
        // to eliminate the difference between using x and y direction
        unit1 = b_usex ? 1 : param.nx;
        unit2 = b_usex ? param.nx : 1;
        range1 = b_usex ? param.nx : param.ny;
        range2 = b_usex ? param.ny : param.nx;
        left = (is == 0) ? pos : begins[thread_id - 1];
        right = (is == param.ns-1) ? pos : begins[thread_id + 1];
        if (right < left) {
            double temp = right;
            right = left;
            left = temp;
        }
        double length = (is == 0) ? begins[thread_id + 1] - pos : 
            pos - begins[thread_id - 1];
        // accumulate gradient
        for (i = 0; i < range2; ++i) {
            if (is == 0 && ia == 0)
                printf("iy=%d, sum=%f, pos=%f, left=%f, right=%f\n", i, sum, pos, left, right);
            for (j = ceil(left); j <= right; ++j) {
                if (j < 0 || j >= range1) continue;
                u = fabs((j - pos) / length);
                sum += (1-u) * img[j*unit1 + i*unit2];
            }
            pos += offset;
            left += offset;
            right += offset;
        }
        // write to destination
        grad[thread_id] = sum * param.orbit;
    }
    return;
}

template <>
bool ParallelBackprojection2DPixDrivenGrad<float>::calculate_on_gpu(const float* img, 
	float* grad, const double* begins, const double* offsets, const bool* usex, 
    cudaStream_t stream) const
{
    int n_elements = this->param_.ns*this->param_.na;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    ParallelBackprojection2DPixDrivenGradKernel<float>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (img, grad, begins, offsets, usex, param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <>
bool ParallelBackprojection2DPixDrivenGrad<double>::calculate_on_gpu(const double* img, 
	double* grad, const double* begins, const double* offsets, const bool* usex, 
    cudaStream_t stream) const
{
    int n_elements = this->param_.ns*this->param_.na;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    ParallelBackprojection2DPixDrivenGradKernel<double>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (img, grad, begins, offsets, usex, this->param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

} // namespace ct_recon