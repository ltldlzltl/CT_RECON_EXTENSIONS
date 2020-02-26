/*
 * @Description: GPU implementation of fp_par_2d.h
 * @Author: Tianling Lyu
 * @Date: 2019-11-13 14:42:30
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2019-12-09 11:50:37
 */

#include "include/fp_par_2d.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include "cuda/cuda_common.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#define M_PI_4 M_PI/4
#endif
#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif
#ifndef MIN
#define MIN(x, y) ((x) > (y) ? (y) : (x))
#endif

namespace ct_recon
{
//#if USE_CUDA
__global__ void ParallelProjection2DRayCastingPrepareKernel(double* sincostbl, 
    double* begins, int* nsteps, const ParallelProjection2DParam param, 
    const double step_size, const int n_elements)
{
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        int ia = thread_id;
        double angle = param.orbit_start + ia * param.orbit;
        double sin_angle = sin(angle);
        double cos_angle = cos(angle);
        sincostbl[2*ia] = sin_angle;
        sincostbl[2*ia + 1] = cos_angle;
        // useful constants
        const double x_center = static_cast<double>(param.nx-1) / 2.0 + param.offset_x;
        const double y_center = static_cast<double>(param.ny-1) / 2.0 + param.offset_y;
        const double fov_squ = param.fov*param.fov;
        const double s_begin = -(static_cast<double>(param.ns-1) / 2.0 + param.offset_s)
            * param.ds;
        const bool usex = sin_angle > (1.0/1.414);
        const double step_x = step_size * sin_angle;
        const double step_y = step_size * cos_angle;
        // variables
        double s = s_begin, x1, y1, x2, y2;
        double half_subtense_squ, half_subtense;
        double* begin_ptr = begins + 2*ia*param.ns;
        int* nstep_ptr = nsteps + ia*param.ns;
        for (int is = 0; is < param.ns; ++is) {
            half_subtense_squ = fov_squ - s*s;
            if (half_subtense_squ > 0) {
                half_subtense = sqrt(half_subtense_squ);
                // intersection points
                x1 = (s*cos_angle - half_subtense*sin_angle) / param.dx + x_center;
                y1 = -(s*sin_angle + half_subtense*cos_angle) / param.dy + y_center;
				x2 = (s*cos_angle + half_subtense*sin_angle) / param.dx + x_center;
				y2 = -(s*sin_angle - half_subtense*cos_angle) / param.dy + y_center;
				if (usex) *nstep_ptr = static_cast<int>((x2 - x1) / step_x);
				else *nstep_ptr = static_cast<int>((y2 - y1) / step_y);
                // store results
                begin_ptr[0] = x1;
                begin_ptr[1] = y1;
            } else {
                // not intersected with FoV
                *nstep_ptr = 0;
                begin_ptr[0] = 0;
                begin_ptr[1] = 0;
            }
            // next channel
            begin_ptr += 2;
            ++nstep_ptr;
            s += param.ds;
        }
    }
    return;
}

bool ParallelProjection2DRayCastingPrepare::calculate_on_gpu(double* sincostbl,
    double* begins, int* nsteps, cudaStream_t stream) const
{
    CudaLaunchConfig config = GetCudaLaunchConfig(param_.na);
    ParallelProjection2DRayCastingPrepareKernel
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (sincostbl, begins, nsteps, param_, step_size_, param_.na);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <typename T>
__global__ void ParallelProjection2DRayCastingKernel(const T* img, T* proj, 
    const double* sincostbl, const double* begins, const int* nsteps, 
    const ParallelProjection2DParam param, const double step_size, 
    const int n_elements)
{
    for (int thread_id : CudaGridRangeX(n_elements)) {
        int ia = thread_id / param.ns;
        int is = thread_id % param.ns;
        double sin_angle = sincostbl[2*ia];
        double cos_angle = sincostbl[2*ia + 1];
        double step_x = step_size * sin_angle;
        double step_y = step_size * cos_angle;
        int pos = is + ia * param.ns;
        double x = begins[2*pos];
        double y = begins[2*pos + 1];
        double sum = 0;
        int ix1, ix2, iy1, iy2;
        double u, v;
        for (int ray_index = 0; ray_index < nsteps[pos]; ++ray_index) {
            if (x >= 0 && x <= param.nx-1 && y >= 0 && y <= param.ny-1){
                // 2-D linear interpolation
                ix1 = floor(x);
                ix2 = ceil(x); // use ceil instead of ix1+1 to suit to x==nx-1
                u = x - ix1;
                iy1 = floor(y);
                iy2 = ceil(y);
                v = y - iy1;
                sum += (1-v) * ((1-u)*img[ix1+iy1*param.nx] + u*img[ix2+iy1*param.nx])
                    + v * ((1-u)*img[ix1+iy2*param.nx] + u*img[ix2+iy2*param.nx]);
            }
            x += step_x;
            y += step_y;
        }
        proj[pos] = sum * step_size * param.dx;
    }
    return;
}

template <>
bool ParallelProjection2DRayCasting<float>::calculate_on_gpu(const float* img, 
	float* proj, const double* sincostbl, const double* begins, 
	const int* nsteps, cudaStream_t stream) const
{
    int n_elements = param_.na*param_.ns;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    ParallelProjection2DRayCastingKernel<float>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (img, proj, sincostbl, begins, nsteps, param_, step_size_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <>
bool ParallelProjection2DRayCasting<double>::calculate_on_gpu(const double* img, 
	double* proj, const double* sincostbl, const double* begins, 
	const int* nsteps, cudaStream_t stream) const
{
    int n_elements = param_.na*param_.ns;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    ParallelProjection2DRayCastingKernel<double>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (img, proj, sincostbl, begins, nsteps, param_, step_size_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

__global__ void ParallelProjection2DRayDrivenPrepareKernel(double* sincostbl, 
    double* beginoffset, int* usex, const ParallelProjection2DParam param, 
    const int n_elements)
{
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        int ia = thread_id, is;
        double angle = param.orbit_start + ia * param.orbit;
        double sin_angle = sin(angle);
        double cos_angle = cos(angle);
        sincostbl[2*ia] = sin_angle;
        sincostbl[2*ia + 1] = cos_angle;
		while (angle < 0) angle += 2 * M_PI;
		while (angle >= 2 * M_PI) angle -= 2 * M_PI;
        bool b_usex = !((angle >= M_PI_4 && angle < 3*M_PI_4) || 
            (angle >= 5*M_PI_4 && angle < 7*M_PI_4));
        usex[ia] = b_usex ? 1 : 0;
        // useful constants
        const double centx = static_cast<double>(param.nx-1) / 2.0 + param.offset_x;
        const double centy = static_cast<double>(param.ny-1) / 2.0 + param.offset_y;
        const double cents = static_cast<double>(param.ns-1) / 2.0 + param.offset_s;
        // variables
        double begin, offset, offset2;
        double* begin_ptr = beginoffset + 2*ia*param.ns;
        if (b_usex) {
            offset = param.dy * sin_angle / (cos_angle * param.dx);
            offset2 = param.ds / (cos_angle * param.dx);
            begin = centx - centy*offset - cents*offset2;
            for (is = 0; is < param.ns; ++is) {
                begin_ptr[0] = begin; 
                begin_ptr[1] = offset;
                // next channel
                begin += offset2;
                begin_ptr += 2;
            }
        } else {
            offset = param.dx * cos_angle / (sin_angle * param.dy);
            offset2 = -param.ds / (sin_angle * param.dy);
            begin = centy - centx*offset - cents*offset2;
            for (is = 0; is < param.ns; ++is) {
                begin_ptr[0] = begin;
                begin_ptr[1] = offset;
                // next channel
                begin += offset2;
                begin_ptr += 2;
            }
        }
    }
    return;
}

bool ParallelProjection2DRayDrivenPrepare::calculate_on_gpu(double* sincostbl,
    double* beginoffset, int* usex, cudaStream_t stream) const
{
    CudaLaunchConfig config = GetCudaLaunchConfig(param_.na);
    ParallelProjection2DRayDrivenPrepareKernel
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (sincostbl, beginoffset, usex, param_, param_.na);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <typename T>
__global__ void ParallelProjection2DRayDrivenKernel(const T* img, T* proj, 
    const double* sincostbl, const double* beginoffset, const int* usex, 
    const ParallelProjection2DParam param, const int n_elements)
{
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        int ia = thread_id / param.ns;
        int is = thread_id % param.ns;
        bool b_usex = usex[ia];
        int pos = is + ia * param.ns;
        double begin = beginoffset[2*pos];
        double offset = beginoffset[2*pos+1];
        T sum = 0;
        if (b_usex) {
            double x = begin, u;
            int ix1, ix2, iy;
            for (iy = 0; iy < param.ny; ++iy) {
                if (x >= 0 && x <= param.nx-1) {
                    // linear interpolation
                    ix1 = static_cast<int>(floor(x));
                    ix2 = static_cast<int>(ceil(x));
                    u = x - ix1;
                    sum += (1-u) * img[ix1 + iy*param.nx] + 
                        u * img[ix2 + iy*param.nx];
                }
                // next row
                x += offset;
            }
            proj[pos] = sum * fabs(param.dy / sincostbl[2*ia + 1]);
        } else {
            double y = begin, u;
            int ix, iy1, iy2;
            for (ix = 0; ix < param.nx; ++ix) {
                if (y >= 0 && y <= param.ny) {
                    //linear interpolation
                    iy1 = static_cast<int>(floor(y));
                    iy2 = static_cast<int>(ceil(y));
                    u = y - iy1;
                    sum += (1-u) * img[ix + iy1*param.nx] + 
                        u * img[ix + iy2*param.nx];
                }
                // next column
                y += offset;
            }
            proj[pos] = sum * fabs(param.dx / sincostbl[2*ia]);
        }
    }
    return;
}

template <>
bool ParallelProjection2DRayDriven<float>::calculate_on_gpu(const float* img, 
	float* proj, const double* sincostbl, const double* beginoffset, 
	const int* usex, cudaStream_t stream) const
{
    int n_elements = param_.na*param_.ns;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    ParallelProjection2DRayDrivenKernel<float>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (img, proj, sincostbl, beginoffset, usex, param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <>
bool ParallelProjection2DRayDriven<double>::calculate_on_gpu(const double* img, 
	double* proj, const double* sincostbl, const double* beginoffset, 
	const int* usex, cudaStream_t stream) const
{
    int n_elements = param_.na*param_.ns;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    ParallelProjection2DRayDrivenKernel<double>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (img, proj, sincostbl, beginoffset, usex, param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

__global__ void ParallelProjection2DRayDrivenGradPrepKernel(double* weights, 
    double* pos, int* usex, const ParallelProjection2DParam param, 
    const int n_elements)
{
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        // declare variables
        int ia = thread_id;
        double cos_angle, sin_angle;
        // we use nx for both nx and ny here, so please make sure nx==ny
        double* pos_ptr = pos + 2*ia*param.nx;
        // useful constants
        const double centx = static_cast<double>(param.nx-1) / 2.0 
            + param.offset_x;
        const double centy = static_cast<double>(param.ny-1) / 2.0 
            + param.offset_y;
        const double cents = static_cast<double>(param.ns-1) / 2.0 
            + param.offset_s;
        // begin calculation
        double angle = param.orbit_start + ia * param.orbit;
        sin_angle = sin(angle);
        cos_angle = cos(angle);
        bool b_usex = !((angle >= M_PI_4 && angle < 3*M_PI_4) || 
            (angle >= 5*M_PI_4 && angle < 7*M_PI_4));
        usex[ia] = b_usex;
        weights[ia] = fabs(b_usex ? param.dy / cos_angle : param.dx / sin_angle);
        double temp1 = param.dx * cos_angle / param.ds;
        double temp2 = -param.dy * sin_angle / param.ds;
        double offset1 = b_usex ? temp1 : temp2;
        double offset2 = b_usex ? temp2 : temp1;
        double begin = cents - centx * temp1 - centy * temp2;
        // again, we use nx here for both nx and ny to reduce code amount
        for (unsigned int i = 0; i < param.nx; ++i) {
            pos_ptr[0] = begin;
            pos_ptr[1] = offset1;
            begin += offset2;
            pos_ptr += 2;
        }
    }
    return;
}

bool ParallelProjection2DRayDrivenGradPrep::calculate_on_gpu(double* weights,
    double* pos, int* usex, cudaStream_t stream) const
{
    CudaLaunchConfig config = GetCudaLaunchConfig(param_.na);
    ParallelProjection2DRayDrivenGradPrepKernel
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (weights, pos, usex, param_, param_.na);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <typename T>
__global__ void ParallelProjection2DRayDrivenGradKernel(const T* proj, T* img, 
    const double* weights, const double* pos, const int* usex, 
    const ParallelProjection2DParam param, const int n_elements)
{
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        // declare variables
        int iy = thread_id / param.nx;
        int ix = thread_id % param.nx;
        unsigned int ia, is;
        bool b_usex;
        double begin, offset, mid, left, right, temp;
        int i_hori, i_vert;
        const double* weight_ptr = weights;
        const double* pos_ptr = pos;
        const int* usex_ptr = usex;
        const T* proj_ptr = proj;
        // begin calculation
        double sum = 0.0, tempsum;
        for (ia = 0; ia < param.na; ++ia) {
            b_usex = *usex_ptr;
            i_hori = b_usex ? ix : iy;
            i_vert = b_usex ? iy : ix;
            // calculate corresponding ray range
            begin = pos_ptr[i_vert<<1];
            offset = pos_ptr[(i_vert<<1) | 1];
            mid = begin + offset * i_hori;
            left = i_hori==0 ? mid : mid-offset;
            right = i_hori==param.nx-1 ? mid : mid+offset;
            // make sure left <= right
            if (left > right) {
                temp = left;
                left = right;
                right = temp;
            }
            // accumulate values within the range
            tempsum = 0.0;
            for (is = ceil(left); is <= right; ++is) {
                tempsum += (1-abs((mid-static_cast<double>(is)) / offset))
                    * proj_ptr[is];
            }
            sum += tempsum * (*weight_ptr);
            // next angle
            ++usex_ptr;
            ++weight_ptr;
            proj_ptr += param.ns;
            pos_ptr += 2*param.nx;
        }
        // write to image
        img[ix + iy*param.nx] = sum;
    }
    return;
}

template <>
bool ParallelProjection2DRayDrivenGrad<float>::calculate_on_gpu(const float* proj, 
	float* img, const double* weights, const double* pos, const int* usex, 
    cudaStream_t stream) const
{
    int n_elements = param_.nx*param_.ny;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    ParallelProjection2DRayDrivenGradKernel<float>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (proj, img, weights, pos, usex, param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <>
bool ParallelProjection2DRayDrivenGrad<double>::calculate_on_gpu(const double* proj, 
	double* img, const double* weights, const double* pos, const int* usex, 
    cudaStream_t stream) const
{
    int n_elements = param_.nx*param_.ny;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    ParallelProjection2DRayDrivenGradKernel<double>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (proj, img, weights, pos, usex, param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

__global__ void ParallelProjection2DDisDrivenPrepKernel(double* sincostbl, 
    double* beginoffset, int* usex, const ParallelProjection2DParam param, 
    const int n_elements)
{
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        int ia = thread_id, is;
        double angle = param.orbit_start + ia * param.orbit;
        double sin_angle = sin(angle);
        double cos_angle = cos(angle);
        sincostbl[2*ia] = sin_angle;
        sincostbl[2*ia + 1] = cos_angle;
		while (angle < 0) angle += 2 * M_PI;
		while (angle >= 2 * M_PI) angle -= 2 * M_PI;
        bool b_usex = !((angle >= M_PI_4 && angle < 3*M_PI_4) || 
            (angle >= 5*M_PI_4 && angle < 7*M_PI_4));
        usex[ia] = b_usex ? 1 : 0;
        // useful constants
        const double centx = static_cast<double>(param.nx-1) / 2.0 + param.offset_x;
        const double centy = static_cast<double>(param.ny-1) / 2.0 + param.offset_y;
        const double cents = static_cast<double>(param.ns-1) / 2.0 + param.offset_s;
        // variables
        double begin, offset, offset2;
        double* begin_ptr = beginoffset + 3*ia*param.ns;
        if (b_usex) {
            offset = param.dy * sin_angle / (cos_angle * param.dx);
            offset2 = param.ds / (cos_angle * param.dx);
            begin = centx - centy*offset - cents*offset2;
            for (is = 0; is < param.ns; ++is) {
                begin_ptr[0] = begin; 
                begin_ptr[1] = offset;
                begin_ptr[2] = offset2 / 2;
                // next channel
                begin += offset2;
                begin_ptr += 3;
            }
        } else {
            offset = param.dx * cos_angle / (sin_angle * param.dy);
            offset2 = -param.ds / (sin_angle * param.dy);
            begin = centy - centx*offset - cents*offset2;
            for (is = 0; is < param.ns; ++is) {
                begin_ptr[0] = begin;
                begin_ptr[1] = offset;
                begin_ptr[2] = offset2 / 2;
                // next channel
                begin += offset2;
                begin_ptr += 3;
            }
        }
    }
    return;
}

bool ParallelProjection2DDisDrivenPrep::calculate_on_gpu(double* sincostbl,
    double* beginoffset, int* usex, cudaStream_t stream) const
{
    CudaLaunchConfig config = GetCudaLaunchConfig(param_.na);
    ParallelProjection2DDisDrivenPrepKernel
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (sincostbl, beginoffset, usex, param_, param_.na);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <typename T>
__global__ void ParallelProjection2DDisDrivenKernel(const T* img, T* proj, 
    const double* sincostbl, const double* beginoffset, const int* usex, 
    const ParallelProjection2DParam param, const int n_elements)
{
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        int ia = thread_id / param.ns;
        int ix, iy;
        bool b_usex = usex[ia];
        double begin = beginoffset[3*thread_id];
        double offset = beginoffset[3*thread_id+1];
        double offset2 = beginoffset[3*thread_id+2] / 2;
        double left = begin - offset2;
        double right = begin + offset2;
        if (left > right) {
            double temp = left;
            left = right;
            right = temp;
        }
        double length;
        T sum = 0, tsum, lsum;
        if (b_usex) {
            for (iy = 0; iy < param.ny; ++iy) {
                tsum = 0;
                lsum = 0;
                for (ix = floor(left); ix < ceil(right); ++ix) {
                    if (ix < 0 || ix >= param.nx-1) continue;
                    length = MIN(ix+1, right) - MAX(ix, left);
                    tsum += length * (img[ix+iy*param.nx]
                        + img[ix+1+iy*param.nx]) / 2;
                    lsum += length;
                }
                if (lsum > 0)
                    sum += tsum / lsum;
                // next row
                left += offset;
                right += offset;
            }
            proj[thread_id] = sum * fabs(param.dy / sincostbl[2*ia + 1]);
        } else {
            for (ix = 0; ix < param.nx; ++ix) {
                tsum = 0;
                lsum = 0;
                for (iy = floor(left); iy < ceil(right); ++iy) {
                    if (iy < 0 || iy >= param.ny-1) continue;
                    length = MIN(iy+1, right) - MAX(iy, left);
                    tsum += length * (img[ix+iy*param.nx]
                        + img[ix+(iy+1)*param.nx]) / 2;
                    lsum += length;
                }
                if (lsum > 0)
                    sum += tsum / lsum;
                // next row
                left += offset;
                right += offset;
            }
            proj[thread_id] = sum * fabs(param.dx / sincostbl[2*ia]);
        }
    }
    return;
}

template <>
bool ParallelProjection2DDisDriven<float>::calculate_on_gpu(const float* img, 
	float* proj, const double* sincostbl, const double* beginoffset, 
	const int* usex, cudaStream_t stream) const
{
    int n_elements = param_.na*param_.ns;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    ParallelProjection2DDisDrivenKernel<float>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (img, proj, sincostbl, beginoffset, usex, param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <>
bool ParallelProjection2DDisDriven<double>::calculate_on_gpu(const double* img, 
	double* proj, const double* sincostbl, const double* beginoffset, 
	const int* usex, cudaStream_t stream) const
{
    int n_elements = param_.na*param_.ns;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    ParallelProjection2DDisDrivenKernel<double>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (img, proj, sincostbl, beginoffset, usex, param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

__global__ void ParallelProjection2DDisDrivenGradPrepKernel(double* weights, 
    double* pos, int* usex, const ParallelProjection2DParam param, 
    const int n_elements)
{
    // the same as the Ray-Driven one
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        // declare variables
        int ia = thread_id;
        double cos_angle, sin_angle;
        // we use nx for both nx and ny here, so please make sure nx==ny
        double* pos_ptr = pos + 2*ia*param.nx;
        // useful constants
        const double centx = static_cast<double>(param.nx-1) / 2.0 
            + param.offset_x;
        const double centy = static_cast<double>(param.ny-1) / 2.0 
            + param.offset_y;
        const double cents = static_cast<double>(param.ns-1) / 2.0 
            + param.offset_s;
        // begin calculation
        double angle = param.orbit_start + ia * param.orbit;
        sin_angle = sin(angle);
        cos_angle = cos(angle);
        bool b_usex = !((angle >= M_PI_4 && angle < 3*M_PI_4) || 
            (angle >= 5*M_PI_4 && angle < 7*M_PI_4));
        usex[ia] = b_usex;
        weights[ia] = fabs(b_usex ? param.dy / cos_angle : param.dx / sin_angle);
        double temp1 = param.dx * cos_angle / param.ds;
        double temp2 = -param.dy * sin_angle / param.ds;
        double offset1 = b_usex ? temp1 : temp2;
        double offset2 = b_usex ? temp2 : temp1;
        double begin = cents - centx * temp1 - centy * temp2;
        // again, we use nx here for both nx and ny to reduce code amount
        for (unsigned int i = 0; i < param.nx; ++i) {
            pos_ptr[0] = begin;
            pos_ptr[1] = offset1;
            begin += offset2;
            pos_ptr += 2;
        }
    }
    return;
}

bool ParallelProjection2DDisDrivenGradPrep::calculate_on_gpu(double* weights,
    double* pos, int* usex, cudaStream_t stream) const
{
    CudaLaunchConfig config = GetCudaLaunchConfig(param_.na);
    ParallelProjection2DDisDrivenGradPrepKernel
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (weights, pos, usex, param_, param_.na);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <typename T>
__global__ void ParallelProjection2DDisDrivenGradKernel(const T* proj, T* img, 
    const double* weights, const double* pos, const int* usex, 
    const ParallelProjection2DParam param, const int n_elements)
{
    for (int thread_id : CudaGridRangeX<int>(n_elements)) {
        // declare variables
        int iy = thread_id / param.nx;
        int ix = thread_id % param.nx;
        unsigned int ia, is;
        bool b_usex;
        double begin, offset, mid, left, right, temp;
        int i_hori, i_vert;
        const double* weight_ptr = weights;
        const double* pos_ptr = pos;
        const int* usex_ptr = usex;
        const T* proj_ptr = proj;
        // begin calculation
        double sum = 0.0, tempsum, length;
        for (ia = 0; ia < param.na; ++ia) {
            b_usex = *usex_ptr;
            i_hori = b_usex ? ix : iy;
            i_vert = b_usex ? iy : ix;
            // calculate corresponding ray range
            begin = pos_ptr[i_vert<<1];
            offset = pos_ptr[(i_vert<<1) | 1];
            mid = begin + offset * i_hori;
            left = i_hori==0 ? mid : mid-offset;
            right = i_hori==param.nx-1 ? mid : mid+offset;
            // make sure left <= right
            if (left > right) {
                temp = left;
                left = right;
                right = temp;
            }
            // accumulate values within the range
            tempsum = 0.0;
            for (is = ceil(left - 0.5); is <= floor(right + 0.5); ++is)
            {
                length = MIN(static_cast<double>(is) + 0.5, right)
                     - MAX(static_cast<double>(is) - 0.5, left);
                tempsum += length * proj_ptr[is] / 2;
            }
            sum += tempsum * (*weight_ptr);
            // next angle
            ++usex_ptr;
            ++weight_ptr;
            proj_ptr += param.ns;
            pos_ptr += 2*param.nx;
        }
        // write to image
        img[ix + iy*param.nx] = sum;
    }
    return;
}

template <>
bool ParallelProjection2DDisDrivenGrad<float>::calculate_on_gpu(const float* proj, 
	float* img, const double* weights, const double* pos, const int* usex, 
    cudaStream_t stream) const
{
    int n_elements = param_.nx*param_.ny;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    ParallelProjection2DDisDrivenGradKernel<float>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (proj, img, weights, pos, usex, param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

template <>
bool ParallelProjection2DDisDrivenGrad<double>::calculate_on_gpu(const double* proj, 
	double* img, const double* weights, const double* pos, const int* usex, 
    cudaStream_t stream) const
{
    int n_elements = param_.nx*param_.ny;
    CudaLaunchConfig config = GetCudaLaunchConfig(n_elements);
    ParallelProjection2DDisDrivenGradKernel<double>
        <<<config.block_count, config.thread_per_block, 0, stream>>>
        (proj, img, weights, pos, usex, param_, n_elements);
	cudaError_t err = cudaDeviceSynchronize();
    return err==cudaSuccess;
}

//#endif

} // namespace ct_recon