/*
 * @Description: classes and structs for fan_weighting numpy extension
 * @Author: Tianling Lyu
 * @Date: 2021-01-10 19:42:42
 * @LastEditors: Please set LastEditors
 * @LastEditTime: 2021-01-11 18:10:19
 */

#ifndef _NP_EXT_FAN_WEIGHTING_H_
#define _NP_EXT_FAN_WEIGHTING_H_

#include "include/fan_weighting.h"
#ifdef USE_CUDA
#include "cuda_runtime.h"
#endif

namespace np_ext {

struct FanWeightingRunParam
{
    double* in; // pointer to input array
    double* out; // pointer to output array

    FanWeightingRunParam(double* in, double* out)
        : in(in), out(out)
    {}
};

class FanWeightingNPExt
{
public:
    FanWeightingNPExt(const ct_recon::FanWeightingParam& param);
    ~FanWeightingNPExt();

    bool allocate();
    bool run(const FanWeightingRunParam& param);

private:
#ifdef USE_CUDA
    double* inout_;
#endif

    ct_recon::FanWeightingParam param_;
    bool allocated_;

    ct_recon::FanWeighting<double> fw_;
};


} // namespace np_ext

#endif