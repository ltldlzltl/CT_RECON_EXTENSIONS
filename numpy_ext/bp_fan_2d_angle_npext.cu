/*
 * @Description: implement numpy extension library functions
 * @Author: Tianling Lyu
 * @Date: 2021-01-09 20:13:54
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2021-01-09 22:31:51
 */

#include "numpy_ext/bp_fan_2d_angle_npext.h"
#include "numpy_ext/common.h"
#ifdef USE_CUDA
#include "cuda_runtime.h"
#endif

namespace np_ext {

#define Container OpContainer<FanBpAngleNPExt, FanBp2DAngleAllocParam, FanBp2DAngleRunParam>

Container container_;

FanBpAngleNPExt::FanBpAngleNPExt(const FanBp2DAngleAllocParam& param, 
    int device)
    : param_(param.param), device_(device), allocated_(false), bp_prep_(param.param), 
    bp_(param.param), xpos_(nullptr), ypos_(nullptr), sincostbl_(nullptr), angles_(param.angles), 
    in_(nullptr), out_(nullptr)
{
#ifdef USE_CUDA
    stream_ = nullptr;
#endif
}

FanBpAngleNPExt::~FanBpAngleNPExt()
{
    if (device_ < 0)
    {
        // allocated on CPU
        if (xpos_ != nullptr)
            delete[] xpos_;
        if (ypos_ != nullptr)
            delete[] ypos_;
        if (sincostbl_ != nullptr)
            delete[] sincostbl_;
    }
    else
    {
        // allocated on GPU
#ifdef USE_CUDA
        if (xpos_ != nullptr)
            cudaFree(xpos_);
        if (ypos_ != nullptr)
            cudaFree(ypos_);
        if (sincostbl_ != nullptr)
            cudaFree(sincostbl_);
        if (in_ != nullptr)
            cudaFree(in_);
        if (out_ != nullptr)
            cudaFree(out_);
        if (stream_ != nullptr)
            cudaStreamDestroy(stream_);
#endif
    }
}

bool FanBpAngleNPExt::allocate() {
    if (allocated_) return true;
    if (device_ < 0) {
        xpos_ = new double[param_.nx];
        ypos_ = new double[param_.ny];
        sincostbl_ = new double[param_.na*2];
        allocated_ = true;
        return bp_prep_.calculate_on_cpu(angles_, xpos_, ypos_, sincostbl_);
    } else {
#ifdef USE_CUDA
        cudaError_t err;
        err = cudaSetDevice(device_);
        if (err != cudaSuccess) 
            throw std::runtime_error("Device not found!");
        err = cudaStreamCreate(&stream_);
        if (err != cudaSuccess) 
            throw std::runtime_error("Stream initialization failed!");
        err = cudaMalloc(&xpos_, param_.nx*sizeof(double));
        if (err != cudaSuccess)
            throw std::runtime_error("CUDA allocate xpos failed!");
        err = cudaMalloc(&ypos_, param_.ny*sizeof(double));
        if (err != cudaSuccess)
            throw std::runtime_error("CUDA allocate ypos failed!");
        err = cudaMalloc(&sincostbl_, param_.na*2*sizeof(double));
        if (err != cudaSuccess)
            throw std::runtime_error("CUDA allocate sincostbl failed!");
        err = cudaMalloc(&in_, param_.ns*param_.na*sizeof(double));
        if (err != cudaSuccess)
            throw std::runtime_error("CUDA allocate input array failed!");
        err = cudaMalloc(&out_, param_.nx*param_.ny*sizeof(double));
        if (err != cudaSuccess)
            throw std::runtime_error("CUDA allocate output array failed!");
        allocated_ == true;
        double* angles_gpu;
        err = cudaMalloc(&angles_gpu, param_.na*sizeof(double));
        if (err != cudaSuccess)
            throw std::runtime_error("CUDA allocate angles array failed!");
        err = cudaMemcpy(angles_gpu, angles_, param_.na*sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            cudaFree(angles_gpu);
            throw std::runtime_error("CUDA memcpy angles array failed!");
        }
        return bp_prep_.calculate_on_gpu(angles_gpu, xpos_, ypos_, sincostbl_, stream_);
#else
        return false;
#endif
    }
}

bool FanBpAngleNPExt::run(const FanBp2DAngleRunParam& param)
{
    if (device_ < 0) {
        // use CPU
        return bp_.calculate_on_cpu(param.in, param.out, xpos_, ypos_, sincostbl_);
    } else {
        // use GPU
#ifdef USE_CUDA
        cudaMemcpy(in_, param.in, param_.na*param_.ns*sizeof(double), cudaMemcpyHostToDevice);
        bool finish = bp_.calculate_on_gpu(in_, out_, xpos_, ypos_, sincostbl_, stream_);
        cudaMemcpy(param.out, out_, param_.nx*param_.ny*sizeof(double), cudaMemcpyDeviceToHost);
        return finish;
#else
        return false;
#endif
    }
}

} // namespace np_ext

int FanBp2DAngleCreate(double* angles, unsigned int ns, unsigned int na, 
    double ds, double offset_s, unsigned int nx, unsigned int ny, double dx, 
    double dy, double offset_x, double offset_y, double dso, double dsd, 
    double fov, int device)
{
    np_ext::FanBp2DAngleAllocParam param(ns, na, ds, offset_s, nx, ny, dx, dy, 
        offset_x, offset_y, dso, dsd, fov, angles);
    int handle = np_ext::container_.create(param, device);
    return handle;
}

bool FanBp2DAngleRun(int handle, double* in, double* out)
{
    np_ext::FanBp2DAngleRunParam param(in, out);
    return np_ext::container_.run(handle, param);
}

bool FanBp2DAngleDestroy(int handle)
{
    return np_ext::container_.erase(handle);
}