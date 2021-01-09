/*
 * @Description: classes and structs for bp_fan_2d_angle numpy extension
 * @Author: Tianling Lyu
 * @Date: 2021-01-09 18:40:00
 * @LastEditors: Tianling Lyu
 * @LastEditTime: 2021-01-09 21:50:40
 */

#ifndef _NP_EXT_BP_FAN_2D_ANGLE_H_
#define _NP_EXT_BP_FAN_2D_ANGLE_H_

#include "include/bp_fan_2d_angle.h"
#ifdef USE_CUDA
#include "cuda_runtime.h"
#endif

namespace np_ext {

struct FanBp2DAngleAllocParam
{
    ct_recon::FanBackprojection2DAngleParam param;
    double* angles;

    FanBp2DAngleAllocParam(unsigned int ns,
                           unsigned int na,
                           double ds,
                           double offset_s,
                           unsigned int nx,
                           unsigned int ny,
                           double dx,
                           double dy,
                           double offset_x,
                           double offset_y,
                           double dso,
                           double dsd,
                           double fov, 
                           double* angles)
        : param(ns, na, ds, offset_s, nx, ny, dx, dy, offset_x, offset_y, dso, 
            dsd, fov), 
        angles(angles)
    {}
};

struct FanBp2DAngleRunParam
{
    double* in;
    double* out;

    FanBp2DAngleRunParam(double* in, double* out)
        :in(in), out(out)
    {}
};

class FanBpAngleNPExt
{
public:
    FanBpAngleNPExt(const FanBp2DAngleAllocParam& param, 
        int device);
    ~FanBpAngleNPExt();

    bool allocate();
    bool run(const FanBp2DAngleRunParam& param);

private:
    double* xpos_;
    double* ypos_;
    double* sincostbl_;
    double* angles_;
    double* in_;
    double* out_;

    ct_recon::FanBackprojection2DAngleParam param_;
    int device_;
    bool allocated_;

    ct_recon::FanBackprojection2DAnglePixDrivenPrep bp_prep_;
    ct_recon::FanBackprojection2DAnglePixDriven<double> bp_;
#ifdef USE_CUDA
    cudaStream_t stream_;
#endif
};

} // namespace np_ext

#endif