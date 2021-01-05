/*
 * @Description: GPU implementation of bp_fan_2d.h
 * @Author: Tianling Lyu
 * @Date: 2021-01-05 11:00:28
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2021-01-05 15:58:58
 */

 #include "include/bp_fan_2d.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include "cuda/cuda_common.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#define M_PI_4 M_PI/4
#endif

#define MAX(x, y) (((x)>(y)) ? (x) : (y))

namespace ct_recon {
#ifdef USE_CUDA
// CUDA functions
// set all elements in the array to 0
template <typename T>
__global__ void ClearElements(T* ptr, const int n_elements)
{
    for (int i : CudaGridRangeX<int>(n_elements)) {
        ptr[i] = 0;
    }
    return
}

// bp prepare Kernel1 to calculate xpos
__global__ void FanBackprojection2DPixDrivenKernelX(double* xpos, 
    const FanBackprojection2DParam param, const int nx)
{
    for (int ix : CudaGridRangeX<int>(nx)) {
        const double centx = (static_cast<double>(param.nx-1)) / 2 + 
            param.offset_x;
        xpos[ix] = (static_cast<double>(ix) - centx) * param.dx;
    }
    return;
}

// bp prepare Kernel2 to calculate ypos
__global__ void FanBackprojection2DPixDrivenKernelY(double* ypos, 
    const FanBackprojection2DParam param, const int ny)
{
    for (int iy : CudaGridRangeX<int>(ny)) {
        const double centy = (static_cast<double>(param.ny-1)) / 2 + 
            param.offset_y;
        ypos[iy] = (static_cast<double>(iy) - centy) * param.dx;
    }
    return;
}

// bp prepare Kernel3 to calculate sin(angle) and cos(angle)
__global__ void FanBackprojection2DPixDrivenKernelA(double* sincostbl, 
    const FanBackprojection2DParam param, const int na)
{
    for (int ia : CudaGridRangeX<int>(na)) {
        const double angle = param.orbit_start + param.orbit*static_cast<double>(ia);
        sincostbl[2*ia] = sin(angle);
        sincostbl[2*ia+1] = cos(angle);
    }
    return;
}

// bp kernel
template <typename T>
__global__ void FanBackprojection2DPixDrivenKernel(const T* proj, T* img, 
    const double* xpos, const double* ypos, const double* sincostbl, 
    const FanBackprojection2DParam param, const int n_elements)
{
    const T factor = round(abs(param.na*param.orbit) / M_PI);
    const double cents = (static_cast<double>(param.ns-1)) / 2 + 
        param.offset_s;
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        int ix = thread_id % param.nx;
        int iy = thread_id / param.nx;
        double s, u, d_loop, r_loop, w;
        int is1, is2;
        double sum = 0;
        const double x = xpos[ix]
        const double y = ypos[iy];
        const T* proj_ptr = proj;
        double* a_ptr = sincostbl;
        // backprojection
        for (unsigned int ia = 0; ia < param.na; ++ia) {
            d_loop = param.dso + x*a_ptr[0] - y*a_ptr[1];
            r_loop = x*a_ptr[1] + y*a_ptr[0];
            w = param.dsd*param.dsd / (d_loop*d_loop + r_loop*r_loop);
            s = atan2(r_loop, d_loop) * param.dsd / param.ds + cents;
            if (s >= 0 && s <= param.ns-1) {
                // linear interpolation
                is1 = floor(s);
                is2 = ceil(s);
                u = s - is1;
                sum += ((1-u) * proj_ptr[is1] + u * proj_ptr[is2]) * w;
            }
            proj_ptr += param.ns;
            a_ptr += 2;
        }
        // write to image
        img[thread_id] = sum * param.orbit / factor;
    }
    return;
}

// bp gradient kernel implementation 1
// compute parallelly on angles, no atomic operation needed
template <typename T>
__global__ void FanBackprojection2DPixDrivenGradKernel1(const T* img, T* grad, 
    const double* xpos, const double* ypos, const double* sincostbl, 
    const FanBackprojection2DParam param, const int n_elements)
{
    const T factor = param.orbit / round(abs(param.na*param.orbit) / M_PI);
    const double cents = (static_cast<double>(param.ns-1)) / 2 + 
        param.offset_s;
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        int ia = thread_id;
        double s, u, d_loop, r_loop, w, x, y;
        int is1, is2, ix, iy;
        double sum = 0;
        const double sinangle = sincostbl[2*ia];
        const double cosangle = sincostbl[2*ia+1];
        const T* img_ptr = img;
        T* grad_ptr = grad + ia*param.ns;

        for (iy = 0; iy < param.ny; ++iy) {
            y = ypos[iy];
            for (ix = 0; ix < param.nx; ++ix) {
                x = xpos[ix];
                d_loop = param.dso + x*sinangle - y*cosangle;
                r_loop = x*cosangle + y*sinangle;
                w = param.dsd*param.dsd / (d_loop*d_loop + r_loop*r_loop);
                s = atan2(r_loop, d_loop) * param.dsd / param.ds + cents;
                if (s >= 0 && s <= param.ns-1) {
                    // linear interpolation
                    is1 = static_cast<unsigned int>(floor(s));
                    is2 = static_cast<unsigned int>(ceil(s));
                    u = s - is1;
                    grad_ptr[is1] += static_cast<T>(w * (1-u)) * (*img_ptr) * factor;
                    grad_ptr[is2] += static_cast<T>(w * u) * (*img_ptr) * factor;
                }
                ++img_ptr;
            }
        }
    }
    return;
}

// bp gradient kernel implementation 2
// parallel on x and y, use atomic add
template <typename T>
__global__ void FanBackprojection2DPixDrivenGradKernel2(const T* img, T* grad, 
    const double* xpos, const double* ypos, const double* sincostbl, 
    const FanBackprojection2DParam param, const int n_elements)
{
    const T factor = param.orbit / round(abs(param.na*param.orbit) / M_PI);
    const double cents = (static_cast<double>(param.ns-1)) / 2 + 
        param.offset_s;
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        int ix = thread_id % param.nx;
        int iy = thread_id / param.nx;
        T value = img[ix+iy*param.nx];
        double s, u, d_loop, r_loop, w;
        int is1, is2;
        double sum = 0;
        const double x = xpos[ix]
        const double y = ypos[iy];
        const T* proj_ptr = grad;
        double* a_ptr = sincostbl;
        // backprojection
        for (unsigned int ia = 0; ia < param.na; ++ia) {
            d_loop = param.dso + x*a_ptr[0] - y*a_ptr[1];
            r_loop = x*a_ptr[1] + y*a_ptr[0];
            w = param.dsd*param.dsd / (d_loop*d_loop + r_loop*r_loop);
            s = atan2(r_loop, d_loop) * param.dsd / param.ds + cents;
            if (s >= 0 && s <= param.ns-1) {
                // linear interpolation
                is1 = floor(s);
                is2 = ceil(s);
                u = s - is1;
                atomicAdd(proj_ptr+is1, static_cast<T>(w * (1-u)) * value * factor);
                atomicAdd(proj_ptr+is2, static_cast<T>(w * u) * value * factor);
            }
            proj_ptr += param.ns;
            a_ptr += 2;
        }
    }
    return;
}

// C functions
bool FanBackprojection2DPixDrivenPrep::calculate_on_gpu(double* xpos,
    double* ypos, double* sincostbl, cudaStream_t stream) const
{
    CudaLaunchConfig config;
    config = GetCudaLaunchConfig(this->param_.nx);
    FanBackprojection2DPixDrivenKernelX
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (xpos, this->param_, this->param_.nx);
    cudaError_t err = cudaDeviceSynchronize();
    config = GetCudaLaunchConfig(this->param_.ny);
    FanBackprojection2DPixDrivenKernelY
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (ypos, this->param_, this->param_.ny);
    err = cudaDeviceSynchronize();
    config = GetCudaLaunchConfig(this->param_.na);
    FanBackprojection2DPixDrivenKernelA
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (sincostbl, this->param_, this->param_.na);
	err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <>
bool FanBackprojection2DPixDriven::calculate_on_gpu(const float* proj, 
	float* img, const double* xpos, const double* ypos, const double* sincostbl, 
    cudaStream_t stream) const
{
    int n_elements = this->param_.nx * this->param_.ny;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    FanBackprojection2DPixDrivenKernel<float>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (proj, img, xpos, ypos, this->param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <>
bool FanBackprojection2DPixDriven::calculate_on_gpu(const double* proj, 
	double* img, const double* xpos, const double* ypos, const double* sincostbl, 
    cudaStream_t stream) const
{
    int n_elements = this->param_.nx * this->param_.ny;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    FanBackprojection2DPixDrivenKernel<double>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (proj, img, xpos, ypos, this->param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

bool FanBackprojection2DPixDrivenGradPrep::calculate_on_gpu(double* xpos,
    double* ypos, double* sincostbl, cudaStream_t stream) const
{
    CudaLaunchConfig config;
    config = GetCudaLaunchConfig(this->param_.nx);
    FanBackprojection2DPixDrivenKernelX
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (xpos, this->param_, this->param_.nx);
    cudaError_t err = cudaDeviceSynchronize();
    config = GetCudaLaunchConfig(this->param_.ny);
    FanBackprojection2DPixDrivenKernelY
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (ypos, this->param_, this->param_.ny);
    err = cudaDeviceSynchronize();
    config = GetCudaLaunchConfig(this->param_.na);
    FanBackprojection2DPixDrivenKernelA
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (sincostbl, this->param_, this->param_.na);
	err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <>
bool FanBackprojection2DPixDrivenGrad::calculate_on_gpu(const float* img, 
	float* grad, const double* xpos, const double* ypos, const double* sincostbl, 
    cudaStream_t stream) const
{
    int n_elements = this->param_.nx * this->param_.ny;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    if (FAN_BP_PIX_DRIVEN_KERNEL == 1) {
        FanBackprojection2DPixDrivenGradKernel1<float>
            <<<config.block_count, config.thread_per_block, 0, stream>>>
            (img, grad, xpos, ypos, this->param_, n_elements);
    } else if (FAN_BP_PIX_DRIVEN_KERNEL == 2) {
        FanBackprojection2DPixDrivenGradKernel2<float>
            <<<config.block_count, config.thread_per_block, 0, stream>>>
            (img, grad, xpos, ypos, this->param_, n_elements);
    } else {
        return false;
    }
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <>
bool FanBackprojection2DPixDrivenGrad::calculate_on_gpu(const double* img, 
	double* grad, const double* xpos, const double* ypos, const double* sincostbl, 
    cudaStream_t stream) const
{
    int n_elements = this->param_.nx * this->param_.ny;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    if (FAN_BP_PIX_DRIVEN_KERNEL == 1) {
        FanBackprojection2DPixDrivenGradKernel1<double>
            <<<config.block_count, config.thread_per_block, 0, stream>>>
            (img, grad, xpos, ypos, this->param_, n_elements);
    } else if (FAN_BP_PIX_DRIVEN_KERNEL == 2) {
        FanBackprojection2DPixDrivenGradKernel2<double>
            <<<config.block_count, config.thread_per_block, 0, stream>>>
            (img, grad, xpos, ypos, this->param_, n_elements);
    } else {
        return false;
    }
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

#endif // #ifdef USE_CUDA
} // namespace ct_recon